#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
# Adapted from vllm/model_executor/models/qwen3_moe.py
# Copyright 2023 The vLLM team.
#
# This file is a part of the vllm-kunlun project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""
import os
from collections.abc import Iterable
from typing import Any, Optional, Union, Tuple, Set

import torch
import os
from torch import nn
from transformers import PretrainedConfig

from vllm_kunlun.ops.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm_kunlun.ops.activation import SiluAndMul
from vllm_kunlun.ops.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm_kunlun.ops.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm_kunlun.ops.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm_kunlun.ops.rotary_embedding import Split_Norm_Rope

logger = init_logger(__name__)


class Qwen3MoeMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3MoeSparseMoeBlock(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )
        self.quant_config = quant_config
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.quant_config is None:
            kunlun_linear_weights = self.gate.get_weights()
            final_hidden_states = self.experts(
                hidden_states=hidden_states, linear_weights=kunlun_linear_weights
            )
        else:
            kunlun_linear_weights = self.gate.get_weights()
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                linear_weights=kunlun_linear_weights,
            )

        if self.tp_size > 1:
            final_hidden_states = (
                self.experts.maybe_all_reduce_tensor_model_parallel(  # noqa E501
                    final_hidden_states
                )
            )

        return final_hidden_states.view(orig_shape)


class Qwen3MoeAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        if rope_scaling is not None:
            scaling_factor = rope_scaling["factor"]
            self.max_position_embeddings = int(
                self.max_position_embeddings * scaling_factor
            )

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if os.getenv("FUSED_QK_ROPE_OP") == "1":
            # Rope fusion operators
            q, k, v = Split_Norm_Rope(
                qkv,
                self.rotary_emb.cos_sin_cache,
                self.q_norm.weight,
                self.k_norm.weight,
                positions,
                self.max_position_embeddings,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            # Add qk-norm
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim
            )
            q_by_head = self.q_norm(q_by_head)
            q = q_by_head.view(q.shape)
            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(
                config=config, quant_config=quant_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Qwen3MoeModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3MoeDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states, residual)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        weights_to_quantize = {}

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    # Map to the parameter name in the model
                    name_mapped = name.replace(weight_name, param_name)

                    # Layer/PP skip judgment
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    if (
                        name_mapped.endswith(".bias") or name_mapped.endswith("_bias")
                    ) and name_mapped not in params_dict:
                        continue

                    # Get the param and target module
                    param = params_dict.get(name_mapped, None)
                    if param is None:
                        continue

                    # === Only when the target MoE layer has int8 weights and scales, and the name matches, the "streaming quantization" is performed ===
                    if self._should_stream_quantize(name_mapped):
                        # Note: Pass the mapped name_mapped instead of the original name
                        self._stream_quantize_moe_weight(
                            name_mapped,
                            param,
                            loaded_weight,
                            expert_id=expert_id,
                            shard_id=shard_id,
                        )
                        loaded_params.add(name_mapped)
                    else:
                        # Fallback: Normal weight loading (non-quantized)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                        loaded_params.add(name_mapped)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale"
                        )
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint "
                                f"(e.g. {name}), but not found the expected "
                                f"name in the model "
                                f"(e.g. {remapped_kv_scale_name}). "
                                "kv-scale is not loaded."
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
            # loaded_params.add(name)
        return loaded_params

    def _is_moe_weight(self, name: str) -> bool:
        """Check if the weight is MoE weight"""
        return name.endswith("w13_weight") or name.endswith("w2_weight")

    def _is_expert_complete(self, cache_key):
        cache = self._moe_weight_cache.get(cache_key)
        if cache is None:
            return False
        w13_ok = (0 in cache["w13_shards"]) and (1 in cache["w13_shards"])
        w2_ok = cache["w2_weight"] is not None
        return w13_ok and w2_ok

    @torch.no_grad()
    def _stream_quantize_moe_weight(
        self,
        param_name: str,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        *,
        expert_id,
        shard_id,
    ):

        rank = os.environ.get("RANK", "0")

        # Ensure expert_id is an integer
        try:
            expert_id = int(expert_id)
        except (ValueError, TypeError):
            if isinstance(expert_id, str):
                expert_id = int(expert_id)

        # Process shard_id
        if isinstance(shard_id, str):
            if shard_id in ("gate", "w1"):
                shard_id = 0
            elif shard_id in ("up", "w3"):
                shard_id = 1
            elif shard_id == "w2":
                shard_id = 0
            else:
                try:
                    shard_id = int(shard_id)
                except ValueError:
                    shard_id = 0
        else:
            shard_id = int(shard_id)

        # Initialize cache
        if not hasattr(self, "_moe_weight_cache"):
            self._moe_weight_cache = {}
            self._expert_batch_count = 0  # Batch counter

        module_path = ".".join(param_name.split(".")[:-1])
        cache_key = (module_path, expert_id)

        cache = self._moe_weight_cache.get(cache_key)
        if cache is None:
            cache = {
                "w13_shards": {},
                "w2_weight": None,
                "target_module": self.get_submodule(module_path),
                "done": False,
            }
            self._moe_weight_cache[cache_key] = cache

        if cache.get("done", False):
            return

        # Cache weights (keep original precision)
        if "w13_weight" in param_name:
            cache["w13_shards"][shard_id] = loaded_weight.clone()
        elif "w2_weight" in param_name:
            cache["w2_weight"] = loaded_weight.clone()

        # Check if complete
        if self._is_expert_complete(cache_key):
            # Quantize this expert
            self._quantize_expert_weights(cache_key)
            cache["done"] = True
            self._moe_weight_cache.pop(cache_key, None)

            # Force synchronization every 4 experts
            self._expert_batch_count += 1
            if self._expert_batch_count % 4 == 0:
                torch.cuda.synchronize()  # Force synchronization
                # print(f"[Rank {rank}] Completed batch of {self._expert_batch_count} experts")

    def _quantize_expert_weights(self, cache_key):
        """Quantize the complete weights of an expert (supports TP sharding)"""
        module_path, expert_id = cache_key
        cache = self._moe_weight_cache[cache_key]
        target_module = cache["target_module"]

        # Get TP config
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Get actual shapes
        E, twoN, H = target_module.w13_weight.shape
        _, H2, N = target_module.w2_weight.shape

        qmax = 127.0

        # Process w13_weight: concatenate gate and up
        gate_weight = cache["w13_shards"][0]  # [768, 2048]
        up_weight = cache["w13_shards"][1]  # [768, 2048]

        # TP sharding
        if tp_size > 1:
            # Calculate shard for each TP rank
            gate_per_rank = gate_weight.shape[0] // tp_size
            up_per_rank = up_weight.shape[0] // tp_size

            gate_start = tp_rank * gate_per_rank
            gate_end = (tp_rank + 1) * gate_per_rank
            up_start = tp_rank * up_per_rank
            up_end = (tp_rank + 1) * up_per_rank

            gate_weight = gate_weight[gate_start:gate_end, :]  # [192, 2048]
            up_weight = up_weight[up_start:up_end, :]  # [192, 2048]

        w13_complete = torch.cat([gate_weight, up_weight], dim=0)  # [384, 2048]

        # Quantize w13_weight
        w13_f = w13_complete.float()
        w13_abs_max = torch.amax(torch.abs(w13_f), dim=-1)  # [384]
        w13_scale_2d = torch.clamp(w13_abs_max, min=1e-6) / qmax  # [384]
        w13_scale_3d = w13_scale_2d.unsqueeze(-1)  # [384, 1]
        w13_q = torch.round(w13_f / w13_scale_3d).clamp_(-128, 127).to(torch.int8)

        # Write w13_weight
        target_module.w13_weight.data[expert_id, :, :].copy_(
            w13_q.to(target_module.w13_weight.device)
        )

        # Update w13_scale - pre-multiply 127
        s = getattr(target_module, "w13_weight_scale")
        s.data[expert_id, :].copy_((w13_scale_2d * 127.0).to(s.device))

        # Process w2_weight
        w2_weight = cache["w2_weight"]  # [2048, 768]

        # TP sharding for w2 weight
        if tp_size > 1:
            w2_per_rank = w2_weight.shape[1] // tp_size
            w2_start = tp_rank * w2_per_rank
            w2_end = (tp_rank + 1) * w2_per_rank
            w2_weight = w2_weight[:, w2_start:w2_end]  # [2048, 192]

        w2_f = w2_weight.float()  # [2048, 192]
        w2_abs_max = torch.amax(torch.abs(w2_f), dim=-1)  # [2048]
        w2_scale_2d = torch.clamp(w2_abs_max, min=1e-6) / qmax  # [2048]
        w2_scale_3d = w2_scale_2d.unsqueeze(-1)  # [2048, 1]
        w2_q = torch.round(w2_f / w2_scale_3d).clamp_(-128, 127).to(torch.int8)

        # Write w2_weight
        w2_param = getattr(target_module, "w2_weight")
        w2_param.data[expert_id, :, :].copy_(w2_q.to(w2_param.device))

        # Update w2_scale - pre-multiply 127
        w2_s = getattr(target_module, "w2_weight_scale")
        w2_s.data[expert_id, :].copy_((w2_scale_2d * 127.0).to(w2_s.device))

        # Clear cache
        cache["w13_shards"].clear()
        cache["w2_weight"] = None

    def _is_int8_moe_target_module(self, module_path: str) -> bool:
        """Check if a module_path is a FusedMoE target using INT8(W8A8).
        Determine by the actual existing parameters and dtype, not relying on quant_config names.
        """
        try:
            mod = self.get_submodule(module_path)
        except Exception:
            return False
        # Need to have both int8 weights and float32 scales, and dimensions come from CompressedTensorsW8A8 path
        if not (
            hasattr(mod, "w13_weight")
            and hasattr(mod, "w2_weight")
            and hasattr(mod, "w13_weight_scale")
            and hasattr(mod, "w2_weight_scale")
        ):
            return False
        try:
            return (
                mod.w13_weight.dtype == torch.int8
                and mod.w2_weight.dtype == torch.int8
                and mod.w13_weight_scale.dtype == torch.float32
                and mod.w2_weight_scale.dtype == torch.float32
            )
        except Exception:
            return False

    def _should_stream_quantize(self, param_name: str) -> bool:
        """Only when (1) the parameter name corresponds to the MoE weights we defined; and
                (2) the MoE layer is indeed the INT8 path (exists int8 weights + scales)
        Stream quantization is enabled; otherwise, it falls back to the default loading.
        """
        # First, determine if it is the MoE weight name we want to process (w13_weight / w2_weight)
        if not self._is_moe_weight(param_name):
            return False
        # Then, check if the module containing this param is the INT8 path
        module_path = ".".join(param_name.split(".")[:-1])
        return self._is_int8_moe_target_module(module_path)


class Qwen3MoeForCausalLM(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3MoeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, quant_config=quant_config
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_caches: list[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
