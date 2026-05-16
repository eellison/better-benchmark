"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '4096'], reduction_ranges=[]
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
    def forward(self, view_304: "f32[4096, 4096]", view_332: "f32[4096, 4096]", view_360: "f32[4096, 4096]", view_388: "f32[4096, 4096]", view_416: "f32[4096, 4096]", view_444: "f32[4096, 4096]", view_472: "f32[4096, 4096]", view_500: "f32[4096, 4096]", view_528: "f32[4096, 4096]", view_556: "f32[4096, 4096]", view_584: "f32[4096, 4096]", bmm_71: "f32[512, 512, 64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
        reshape_default: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list, [4096]);  sum_dim_int_list = None
        sum_dim_int_list_1: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_332, [0], True);  view_332 = None
        reshape_default_1: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, [4096]);  sum_dim_int_list_1 = None
        add_tensor: "f32[4096]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None
        sum_dim_int_list_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_360, [0], True);  view_360 = None
        reshape_default_2: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, [4096]);  sum_dim_int_list_2 = None
        add_tensor_1: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None
        sum_dim_int_list_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_388, [0], True);  view_388 = None
        reshape_default_3: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, [4096]);  sum_dim_int_list_3 = None
        add_tensor_2: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_3);  add_tensor_1 = reshape_default_3 = None
        sum_dim_int_list_4: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_416, [0], True);  view_416 = None
        reshape_default_4: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [4096]);  sum_dim_int_list_4 = None
        add_tensor_3: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_2, reshape_default_4);  add_tensor_2 = reshape_default_4 = None
        sum_dim_int_list_5: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_444, [0], True);  view_444 = None
        reshape_default_5: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, [4096]);  sum_dim_int_list_5 = None
        add_tensor_4: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_3, reshape_default_5);  add_tensor_3 = reshape_default_5 = None
        sum_dim_int_list_6: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_472, [0], True);  view_472 = None
        reshape_default_6: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, [4096]);  sum_dim_int_list_6 = None
        add_tensor_5: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_4, reshape_default_6);  add_tensor_4 = reshape_default_6 = None
        sum_dim_int_list_7: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_500, [0], True);  view_500 = None
        reshape_default_7: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, [4096]);  sum_dim_int_list_7 = None
        add_tensor_6: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_5, reshape_default_7);  add_tensor_5 = reshape_default_7 = None
        sum_dim_int_list_8: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_528, [0], True);  view_528 = None
        reshape_default_8: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, [4096]);  sum_dim_int_list_8 = None
        add_tensor_7: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_6, reshape_default_8);  add_tensor_6 = reshape_default_8 = None
        sum_dim_int_list_9: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_556, [0], True);  view_556 = None
        reshape_default_9: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, [4096]);  sum_dim_int_list_9 = None
        add_tensor_8: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_7, reshape_default_9);  add_tensor_7 = reshape_default_9 = None
        sum_dim_int_list_10: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_584, [0], True);  view_584 = None
        reshape_default_10: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, [4096]);  sum_dim_int_list_10 = None
        add_tensor_9: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_8, reshape_default_10);  add_tensor_8 = reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default_11: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_71, [8, 64, 512, 64]);  bmm_71 = None
        mul_scalar: "f32[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(reshape_default_11, 0.3535533905932738);  reshape_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:180 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_scalar, [0, 2, 1, 3]);  mul_scalar = None
        clone_default: "f32[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_12: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(clone_default, [8, 512, 4096]);  clone_default = None
        reshape_default_13: "f32[4096, 4096]" = torch.ops.aten.reshape.default(reshape_default_12, [4096, 4096]);  reshape_default_12 = None
        permute_default_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(reshape_default_13, [1, 0])
        sum_dim_int_list_11: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(reshape_default_13, [0], True);  reshape_default_13 = None
        reshape_default_14: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, [4096]);  sum_dim_int_list_11 = None
        add_tensor_10: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_9, reshape_default_14);  add_tensor_9 = reshape_default_14 = None
        return (permute_default_1, add_tensor_10)


def _default_make_inputs():
    return [
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512, 64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
