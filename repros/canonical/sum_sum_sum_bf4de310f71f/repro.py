"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['256', '128', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['256', '128', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_66: "f32[32768, 768]", mul_199: "f32[256, 128, 768]", primals_15: "f32[768]", mul_4: "f32[256, 128, 768]", div_14: "f32[256, 128, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:421 in ff_chunk, code: x = self.lin1(input)
        reshape_default: "f32[256, 128, 768]" = torch.ops.aten.reshape.default(mm_66, [256, 128, 768]);  mm_66 = None
        add_tensor: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_199, reshape_default);  mul_199 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:481 in forward, code: sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)
        mul_tensor: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor, primals_15);  primals_15 = None
        mul_tensor_1: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_4);  mul_tensor = None
        sum_dim_int_list_1: "f32[256, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_4, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_tensor_1);  div_14 = sub_tensor_1 = None
        mul_tensor_5: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor, mul_4);  mul_4 = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 1]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/distilbert/modeling_distilbert.py:402 in forward, code: attn_output = self.out_lin(attn_output)
        reshape_default_1: "f32[32768, 768]" = torch.ops.aten.reshape.default(mul_tensor_4, [32768, 768]);  mul_tensor_4 = None
        permute_default: "f32[768, 32768]" = torch.ops.aten.permute.default(reshape_default_1, [1, 0])
        sum_dim_int_list_4: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(reshape_default_1, [0], True);  reshape_default_1 = None
        reshape_default_2: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [768]);  sum_dim_int_list_4 = None
        return (sum_dim_int_list_2, sum_dim_int_list_3, permute_default, reshape_default_2)


def _default_make_inputs():
    return [
    torch.randn([32768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([256, 128, 1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
