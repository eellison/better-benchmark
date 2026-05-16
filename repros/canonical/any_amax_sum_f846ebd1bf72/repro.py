"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=any, ranges=['8', '64', '512', '1'], reduction_ranges=[]
#   origins: ['aten.any.dim']
#   type=amax, ranges=['8', '64', '512', '1'], reduction_ranges=[]
#   origins: ['aten.amax.default']
#   type=sum, ranges=['8', '64', '512', '1'], reduction_ranges=[]
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
    def forward(self, bmm_22: "f32[512, 512, 512]", expand_2: "b8[8, 1, 512, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        reshape_default: "f32[8, 64, 512, 512]" = torch.ops.aten.reshape.default(bmm_22, [8, 64, 512, 512]);  bmm_22 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default, full_default_1);  expand_2 = full_default = full_default_1 = None
        add_tensor: "f32[8, 64, 512, 512]" = torch.ops.aten.add.Tensor(reshape_default, where_self);  reshape_default = where_self = None
        eq_scalar: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(add_tensor, -inf)
        logical_not_default: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default_2: "f32[8, 64, 512, 512]" = torch.ops.aten.full.default([8, 64, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        where_self_1: "f32[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_default_1, full_default_2, div_tensor);  logical_not_default_1 = full_default_2 = div_tensor = None
        expand_default: "f32[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_self_1, [8, 64, 512, 512]);  where_self_1 = None
        reshape_default_1: "f32[512, 512, 512]" = torch.ops.aten.reshape.default(expand_default, [512, 512, 512]);  expand_default = None
        return reshape_default_1


def _default_make_inputs():
    return [
    torch.randn([512, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, (512,), dtype=torch.bool, device='cuda').as_strided([8, 1, 512, 512], [0, 512, 1, 0]),  # expand_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
