#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
#
# This file is a part of the vllm-kunlun project.
# Author: Chen Zhennan, Dong Xinyu
# Email: chenzhennan@baidu.com
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

import torch
from typing import Any, Literal, Optional, cast, Callable, Optional

from compressed_tensors.config import (
    CompressionFormat,
    SparsityCompressionConfig,
    SparsityStructure,
)
from compressed_tensors.quantization import ActivationOrdering, QuantizationStrategy
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter

# TODO: import position will be changed after 0.9.0
# vllm.model_executor.layers.fused_moe.fused_moe --> vllm.model_executor.layers.fused_moe

from vllm.model_executor.utils import set_weight_attrs
import re
import xtorch_ops


from safetensors.torch import load_file as safe_load_file


class CompressedTensorsMoEMethod(FusedMoEMethodBase):

    def get_moe_method(quant_config, layer) -> "CompressedTensorsMoEMethod":
        tsm = getattr(quant_config, "target_scheme_map", None) or {}
        linear_cfg = None
        for k in ("Linear", "FusedMoE", "MoE", "Moe", "Experts"):
            if k in tsm and isinstance(tsm[k], dict):
                linear_cfg = tsm[k]
                break
        if not linear_cfg:
            # print("target_scheme_map missing; fallback to INT8(W8A8) method")
            return CompressedTensorsW8A8Int8MoEMethod(quant_config)
        wq = linear_cfg.get("weights")
        aq = linear_cfg.get("input_activations")
        if not wq or not aq:
            # print("incomplete scheme; fallback to INT8(W8A8)")
            return CompressedTensorsW8A8Int8MoEMethod(quant_config)

        # Other branches are handled as needed; default fallback:
        return CompressedTensorsW8A8Int8MoEMethod(quant_config)


# copied from vllm 0.9.0
class CompressedTensorsW8A8Int8MoEMethod(CompressedTensorsMoEMethod):

    def __init__(
        self, quant_config: "CompressedTensorsConfig"  # type: ignore # noqa E501
    ):
        self.quant_config = quant_config

        # Directly create a default quantization config dictionary to avoid validation issues with QuantizationArgs
        # print("Creating default INT8 quantization config for MoE")

        # Create a default weight quantization config dictionary
        self.weight_quant = type(
            "WeightQuant",
            (),
            {
                "type": "int",
                "num_bits": 8,
                "strategy": "channel",
                "group_size": 128,
                "symmetric": True,
                "dynamic": False,
                "actorder": "none",
                "observer": None,
                "observer_kwargs": {},
                "block_structure": None,
            },
        )()

        # Create a default input activation quantization config dictionary
        self.input_quant = type(
            "InputQuant",
            (),
            {
                "type": "int",
                "num_bits": 8,
                "strategy": "token",
                "group_size": 128,
                "symmetric": True,
                "dynamic": True,
                "actorder": "none",
                "observer": None,
                "observer_kwargs": {},
                "block_structure": None,
            },
        )()

        # Change comparison method to directly compare strings
        per_channel = (
            self.weight_quant.strategy == "channel"
            and self.input_quant.strategy == "token"
        )
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}"
            )

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales."
            )

    def create_weights1(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Use float32 as a placeholder for weights to facilitate loading original weights from ckpt
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),  #  generally is torch.bfloat16
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Channel scale: float32 + 2D [E, out] (aligned with fused_moe/UT)
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size_per_partition, dtype=torch.float32
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        # Input scale can be dynamically calculated
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),  # directly use int8
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),  # directly use int8
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Scale factors
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size_per_partition, dtype=torch.float32
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        # Input scale can be dynamically calculated
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    @torch.no_grad()
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return
        # Convert original weights to float32 for more robust statistics
        w13_f = layer.w13_weight.float()
        w2_f = layer.w2_weight.float()

        # Each column (abs_max) -> per-column scale (out dimension is dim=1, column is dim=-1)
        qmax = 127.0
        w13_abs_max = torch.amax(torch.abs(w13_f), dim=-1)  # [E, 2N]
        w2_abs_max = torch.amax(torch.abs(w2_f), dim=-1)  # [E, H]

        w13_scale_2d = torch.clamp(w13_abs_max, min=1e-6) / qmax  # [E, 2N], float32
        w2_scale_2d = torch.clamp(w2_abs_max, min=1e-6) / qmax  # [E, H],  float32

        # Quantization: broadcast 3D scale and store back to 2D scale
        w13_scale_3d = w13_scale_2d.unsqueeze(-1)  # [E, 2N, 1]
        w2_scale_3d = w2_scale_2d.unsqueeze(-1)  # [E, H, 1]

        w13_q = torch.round(w13_f / w13_scale_3d).clamp_(-128, 127).to(torch.int8)
        w2_q = torch.round(w2_f / w2_scale_3d).clamp_(-128, 127).to(torch.int8)

        # Optional: If your fused/kernel expects scale pre-multiplied by 127 (to be consistent with some UT backends), uncomment the following two lines:
        w13_scale_2d = w13_scale_2d * 127.0
        w2_scale_2d = w2_scale_2d * 127.0

        # Write back parameters: weight int8; scale uses float32 + 2D
        replace_parameter(
            layer, "w13_weight", torch.nn.Parameter(w13_q, requires_grad=False)
        )
        replace_parameter(
            layer, "w2_weight", torch.nn.Parameter(w2_q, requires_grad=False)
        )
        replace_parameter(
            layer,
            "w13_weight_scale",
            torch.nn.Parameter(w13_scale_2d.contiguous(), requires_grad=False),
        )
        replace_parameter(
            layer,
            "w2_weight_scale",
            torch.nn.Parameter(w2_scale_2d.contiguous(), requires_grad=False),
        )

        # Brief check
        print(
            f"w13: {w13_q.shape}, w13_s: {w13_scale_2d.shape}, w2: {w2_q.shape}, w2_s: {w2_scale_2d.shape}"
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,  # Add this parameter
        expert_load_view: Optional[torch.Tensor] = None,  # Add this parameter
        logical_to_physical_map: Optional[torch.Tensor] = None,  # Add this parameter
        logical_replica_count: Optional[torch.Tensor] = None,  # Add this parameter
        linear_weights: Optional[torch.Tensor] = None,  # Add this parameter
    ) -> torch.Tensor:

        output = torch.empty_like(x)
        torch.ops._C.moe_ffn_per_token_block(
            x=x,
            inter_weight=layer.w13_weight,
            inter_scale=layer.w13_weight_scale,
            outer_weight=layer.w2_weight,
            outer_scale=layer.w2_weight_scale,
            top_k=top_k,
            global_num_experts=global_num_experts,
            linear_weights=linear_weights,
            expert_map=expert_map,
            activation=activation,
            output=output,
            use_expert_parallel=expert_map is not None,
            ep_size=expert_map.size(0) if expert_map is not None else 1,
            ep_rank=0,
        )
        return output


print(
    "[Monkey Patch Applied] >>> vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.CompressedTensorsMoEMethod \
      --> vllm_xpu.model_executor.layers.quantization.compressed_tensors_moe.py:CompressedTensorsMoEMethod"
)
