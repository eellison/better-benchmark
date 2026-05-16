"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '128'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_707: "f32[32768, 128]", mul_697: "f32[256, 128, 128]", add_8: "f32[256, 128, 128]", primals_32: "f32[128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:359 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(mm_707, [256, 128, 128]);  mm_707 = None
        add_tensor: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_697, reshape_default);  mul_697 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:138 in forward, code: return input_tensor * self.weight + self.bias
        sum_dim_int_list: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 1], True)
        reshape_default_1: "f32[128]" = torch.ops.aten.reshape.default(sum_dim_int_list, [128]);  sum_dim_int_list = None
        mul_tensor: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_tensor, add_8);  add_8 = None
        mul_tensor_1: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_tensor, primals_32);  add_tensor = primals_32 = None
        sum_dim_int_list_1: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1], True);  mul_tensor = None
        reshape_default_2: "f32[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, [128]);  sum_dim_int_list_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:457 in forward, code: layer_outputs = self.dense(hidden_states)
        reshape_default_3: "f32[32768, 128]" = torch.ops.aten.reshape.default(mul_tensor_1, [32768, 128]);  mul_tensor_1 = None
        permute_default: "f32[128, 32768]" = torch.ops.aten.permute.default(reshape_default_3, [1, 0])
        sum_dim_int_list_2: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(reshape_default_3, [0], True);  reshape_default_3 = None
        reshape_default_4: "f32[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, [128]);  sum_dim_int_list_2 = None
        return (reshape_default_1, reshape_default_2, permute_default, reshape_default_4)


def _default_make_inputs():
    return [
    torch.randn([32768, 128], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
