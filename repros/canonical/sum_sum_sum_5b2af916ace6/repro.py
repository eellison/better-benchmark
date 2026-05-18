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
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, view_300: "f32[4096, 4096]", view_328: "f32[4096, 4096]", view_356: "f32[4096, 4096]", view_384: "f32[4096, 4096]", view_412: "f32[4096, 4096]", view_440: "f32[4096, 4096]", view_468: "f32[4096, 4096]", view_496: "f32[4096, 4096]", view_524: "f32[4096, 4096]", view_552: "f32[4096, 4096]", view_580: "f32[4096, 4096]", bmm_70: "f32[512, 64, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        sum_dim_int_list: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
        reshape_default: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list, [4096]);  sum_dim_int_list = None
        sum_dim_int_list_1: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_328, [0], True);  view_328 = None
        reshape_default_1: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, [4096]);  sum_dim_int_list_1 = None
        add_tensor: "f32[4096]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None
        sum_dim_int_list_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
        reshape_default_2: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, [4096]);  sum_dim_int_list_2 = None
        add_tensor_1: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None
        sum_dim_int_list_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_384, [0], True);  view_384 = None
        reshape_default_3: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, [4096]);  sum_dim_int_list_3 = None
        add_tensor_2: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_3);  add_tensor_1 = reshape_default_3 = None
        sum_dim_int_list_4: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_412, [0], True);  view_412 = None
        reshape_default_4: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [4096]);  sum_dim_int_list_4 = None
        add_tensor_3: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_2, reshape_default_4);  add_tensor_2 = reshape_default_4 = None
        sum_dim_int_list_5: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_440, [0], True);  view_440 = None
        reshape_default_5: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, [4096]);  sum_dim_int_list_5 = None
        add_tensor_4: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_3, reshape_default_5);  add_tensor_3 = reshape_default_5 = None
        sum_dim_int_list_6: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_468, [0], True);  view_468 = None
        reshape_default_6: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, [4096]);  sum_dim_int_list_6 = None
        add_tensor_5: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_4, reshape_default_6);  add_tensor_4 = reshape_default_6 = None
        sum_dim_int_list_7: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
        reshape_default_7: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, [4096]);  sum_dim_int_list_7 = None
        add_tensor_6: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_5, reshape_default_7);  add_tensor_5 = reshape_default_7 = None
        sum_dim_int_list_8: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_524, [0], True);  view_524 = None
        reshape_default_8: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, [4096]);  sum_dim_int_list_8 = None
        add_tensor_7: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_6, reshape_default_8);  add_tensor_6 = reshape_default_8 = None
        sum_dim_int_list_9: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_552, [0], True);  view_552 = None
        reshape_default_9: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, [4096]);  sum_dim_int_list_9 = None
        add_tensor_8: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_7, reshape_default_9);  add_tensor_7 = reshape_default_9 = None
        sum_dim_int_list_10: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_580, [0], True);  view_580 = None
        reshape_default_10: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, [4096]);  sum_dim_int_list_10 = None
        add_tensor_9: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_8, reshape_default_10);  add_tensor_8 = reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default_11: "f32[8, 64, 64, 512]" = torch.ops.aten.reshape.default(bmm_70, [8, 64, 64, 512]);  bmm_70 = None
        mul_scalar: "f32[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(reshape_default_11, 0.3535533905932738);  reshape_default_11 = None
        permute_default: "f32[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_scalar, [0, 1, 3, 2]);  mul_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:181 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_default_1: "f32[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        reshape_default_12: "f32[8, 512, 4096]" = torch.ops.aten.reshape.default(permute_default_1, [8, 512, 4096]);  permute_default_1 = None
        clone_default: "f32[8, 512, 4096]" = torch.ops.aten.clone.default(reshape_default_12, memory_format = torch.contiguous_format);  reshape_default_12 = None
        reshape_default_13: "f32[4096, 4096]" = torch.ops.aten.reshape.default(clone_default, [4096, 4096]);  clone_default = None
        permute_default_2: "f32[4096, 4096]" = torch.ops.aten.permute.default(reshape_default_13, [1, 0])
        sum_dim_int_list_11: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(reshape_default_13, [0], True);  reshape_default_13 = None
        reshape_default_14: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, [4096]);  sum_dim_int_list_11 = None
        add_tensor_10: "f32[4096]" = torch.ops.aten.add.Tensor(add_tensor_9, reshape_default_14);  add_tensor_9 = reshape_default_14 = None
        return (permute_default_2, add_tensor_10)


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
    torch.randn([512, 64, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
