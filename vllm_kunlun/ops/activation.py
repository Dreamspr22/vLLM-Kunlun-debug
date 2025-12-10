# SPDX-License-Identifier: Apache-2.0
"""Custom activation functions."""
import torch
import torch.nn.functional as F
from vllm.model_executor.custom_op import CustomOp


@CustomOp.register("kunlun_silu_and_mul")
class SiluAndMul(CustomOp):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        torch.ops._C.swiglu(x, out)
        return out