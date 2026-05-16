"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from math import inf
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_71: "f32[384, 512, 64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[32, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_71, [32, 12, 512, 64]);  bmm_71 = None
        mul_scalar: "f32[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(reshape_default, 0.3535533905932738);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default: "f32[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_scalar, [0, 2, 1, 3]);  mul_scalar = None
        clone_default: "f32[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(clone_default, [32, 512, 768]);  clone_default = None
        reshape_default_2: "f32[16384, 768]" = torch.ops.aten.reshape.default(reshape_default_1, [16384, 768]);  reshape_default_1 = None
        permute_default_1: "f32[768, 16384]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0])
        sum_dim_int_list: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(reshape_default_2, [0], True);  reshape_default_2 = None
        reshape_default_3: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list, [768]);  sum_dim_int_list = None
        return (permute_default_1, reshape_default_3)


def _default_make_inputs():
    return [
    torch.randn([384, 512, 64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
