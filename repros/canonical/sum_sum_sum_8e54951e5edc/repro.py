"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['1', '768'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, view_223: "f32[4096, 768]", view_245: "f32[4096, 768]", view_267: "f32[4096, 768]", view_289: "f32[4096, 768]", view_311: "f32[4096, 768]", view_333: "f32[4096, 768]", view_355: "f32[4096, 768]", view_377: "f32[4096, 768]", view_399: "f32[4096, 768]", view_421: "f32[4096, 768]", view_443: "f32[4096, 768]", getitem_144: "f32[8, 12, 512, 64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:378 in forward, code: self.query(hidden_states)
        sum_dim_int_list: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_223, [0], True);  view_223 = None
        reshape_default: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list, [768]);  sum_dim_int_list = None
        sum_dim_int_list_1: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_245, [0], True);  view_245 = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, [768]);  sum_dim_int_list_1 = None
        add_tensor: "f32[768]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None
        sum_dim_int_list_2: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_267, [0], True);  view_267 = None
        reshape_default_2: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, [768]);  sum_dim_int_list_2 = None
        add_tensor_1: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None
        sum_dim_int_list_3: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_289, [0], True);  view_289 = None
        reshape_default_3: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, [768]);  sum_dim_int_list_3 = None
        add_tensor_2: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_3);  add_tensor_1 = reshape_default_3 = None
        sum_dim_int_list_4: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
        reshape_default_4: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [768]);  sum_dim_int_list_4 = None
        add_tensor_3: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_2, reshape_default_4);  add_tensor_2 = reshape_default_4 = None
        sum_dim_int_list_5: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_333, [0], True);  view_333 = None
        reshape_default_5: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, [768]);  sum_dim_int_list_5 = None
        add_tensor_4: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_3, reshape_default_5);  add_tensor_3 = reshape_default_5 = None
        sum_dim_int_list_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_355, [0], True);  view_355 = None
        reshape_default_6: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, [768]);  sum_dim_int_list_6 = None
        add_tensor_5: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_4, reshape_default_6);  add_tensor_4 = reshape_default_6 = None
        sum_dim_int_list_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_377, [0], True);  view_377 = None
        reshape_default_7: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, [768]);  sum_dim_int_list_7 = None
        add_tensor_6: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_5, reshape_default_7);  add_tensor_5 = reshape_default_7 = None
        sum_dim_int_list_8: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_399, [0], True);  view_399 = None
        reshape_default_8: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, [768]);  sum_dim_int_list_8 = None
        add_tensor_7: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_6, reshape_default_8);  add_tensor_6 = reshape_default_8 = None
        sum_dim_int_list_9: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_421, [0], True);  view_421 = None
        reshape_default_9: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, [768]);  sum_dim_int_list_9 = None
        add_tensor_8: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_7, reshape_default_9);  add_tensor_7 = reshape_default_9 = None
        sum_dim_int_list_10: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_443, [0], True);  view_443 = None
        reshape_default_10: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, [768]);  sum_dim_int_list_10 = None
        add_tensor_9: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_8, reshape_default_10);  add_tensor_8 = reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:380 in forward, code: .transpose(1, 2)
        permute_default: "f32[8, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:379 in forward, code: .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        reshape_default_11: "f32[8, 512, 768]" = torch.ops.aten.reshape.default(permute_default, [8, 512, 768]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:378 in forward, code: self.query(hidden_states)
        reshape_default_12: "f32[4096, 768]" = torch.ops.aten.reshape.default(reshape_default_11, [4096, 768]);  reshape_default_11 = None
        permute_default_1: "f32[768, 4096]" = torch.ops.aten.permute.default(reshape_default_12, [1, 0])
        sum_dim_int_list_11: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(reshape_default_12, [0], True);  reshape_default_12 = None
        reshape_default_13: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, [768]);  sum_dim_int_list_11 = None
        add_tensor_10: "f32[768]" = torch.ops.aten.add.Tensor(add_tensor_9, reshape_default_13);  add_tensor_9 = reshape_default_13 = None
        return (permute_default_1, add_tensor_10)


def _default_make_inputs():
    return [
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 12, 512, 64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
