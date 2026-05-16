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
    def forward(self, view_296: "f32[4096, 4096]", view_324: "f32[4096, 4096]", view_352: "f32[4096, 4096]", view_380: "f32[4096, 4096]", view_408: "f32[4096, 4096]", view_436: "f32[4096, 4096]", view_464: "f32[4096, 4096]", view_492: "f32[4096, 4096]", view_520: "f32[4096, 4096]", view_548: "f32[4096, 4096]", view_576: "f32[4096, 4096]", bmm_68: "f32[512, 512, 64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
        reshape_default: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list, [4096]);  sum_dim_int_list = None
        sum_dim_int_list_1: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
        reshape_default_1: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, [4096]);  sum_dim_int_list_1 = None
        add_tensor: "f32[4096]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None
        sum_dim_int_list_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
        reshape_default_2: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, [4096]);  sum_dim_int_list_2 = None
        add_tensor_1: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None
        sum_dim_int_list_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
        reshape_default_3: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, [4096]);  sum_dim_int_list_3 = None
        add_tensor_2: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_3);  add_tensor_1 = reshape_default_3 = None
        sum_dim_int_list_4: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_408, [0], True);  view_408 = None
        reshape_default_4: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [4096]);  sum_dim_int_list_4 = None
        add_tensor_3: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_2, reshape_default_4);  add_tensor_2 = reshape_default_4 = None
        sum_dim_int_list_5: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_436, [0], True);  view_436 = None
        reshape_default_5: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, [4096]);  sum_dim_int_list_5 = None
        add_tensor_4: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_3, reshape_default_5);  add_tensor_3 = reshape_default_5 = None
        sum_dim_int_list_6: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
        reshape_default_6: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, [4096]);  sum_dim_int_list_6 = None
        add_tensor_5: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_4, reshape_default_6);  add_tensor_4 = reshape_default_6 = None
        sum_dim_int_list_7: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_492, [0], True);  view_492 = None
        reshape_default_7: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, [4096]);  sum_dim_int_list_7 = None
        add_tensor_6: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_5, reshape_default_7);  add_tensor_5 = reshape_default_7 = None
        sum_dim_int_list_8: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_520, [0], True);  view_520 = None
        reshape_default_8: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, [4096]);  sum_dim_int_list_8 = None
        add_tensor_7: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_6, reshape_default_8);  add_tensor_6 = reshape_default_8 = None
        sum_dim_int_list_9: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
        reshape_default_9: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, [4096]);  sum_dim_int_list_9 = None
        add_tensor_8: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_7, reshape_default_9);  add_tensor_7 = reshape_default_9 = None
        sum_dim_int_list_10: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
        reshape_default_10: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, [4096]);  sum_dim_int_list_10 = None
        add_tensor_9: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_8, reshape_default_10);  add_tensor_8 = reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default_11: "f32[8, 64, 512, 64]" = torch.ops.aten.reshape.default(bmm_68, [8, 64, 512, 64]);  bmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:182 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(reshape_default_11, [0, 2, 1, 3]);  reshape_default_11 = None
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
