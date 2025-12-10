# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import ReplicatedLinear as VllmReplicatedLinear

class ReplicatedLinear(VllmReplicatedLinear):
    """Replicated linear layer"""

    def get_weights(self):
        """get_weights"""
        if hasattr(self, 'kunlun_linear_weights'):
            return self.kunlun_linear_weights
        weights = torch.nn.Parameter(self.weight.to(torch.float32))
        self.register_parameter("kunlun_linear_weights", weights)
        return self.kunlun_linear_weights

    def get_weights_half(self):
        """get_weights_half"""
        if hasattr(self, 'kunlun_linear_weights_half'):
            return self.kunlun_linear_weights_half
        weights = torch.nn.Parameter(self.weight.to(torch.float16))