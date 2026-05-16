"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=any, ranges=['32', '6', '128', '1'], reduction_ranges=[]
#   origins: ['aten.any.dim']
#   type=amax, ranges=['32', '6', '128', '1'], reduction_ranges=[]
#   origins: ['aten.amax.default']
#   type=sum, ranges=['32', '6', '128', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, bmm_default: "f32[192, 128, 128]", add_65: "f32[32, 6, 128, 128]"):
        # No stacktrace found for following nodes
        reshape_default: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_default, [32, 6, 128, 128]);  bmm_default = None
        add_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default, add_65);  reshape_default = add_65 = None
        eq_scalar: "b8[32, 6, 128, 128]" = torch.ops.aten.eq.Scalar(add_tensor, -inf)
        logical_not_default: "b8[32, 6, 128, 128]" = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim: "b8[32, 6, 128, 1]" = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1: "b8[32, 6, 128, 1]" = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default: "f32[32, 6, 128, 128]" = torch.ops.aten.full.default([32, 6, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_default: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor, amax_default);  add_tensor = amax_default = None
        exp_default: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        where_self: "f32[32, 6, 128, 128]" = torch.ops.aten.where.self(logical_not_default_1, full_default, div_tensor);  logical_not_default_1 = full_default = div_tensor = None
        expand_default: "f32[32, 6, 128, 128]" = torch.ops.aten.expand.default(where_self, [32, 6, 128, 128]);  where_self = None
        reshape_default_1: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(expand_default, [192, 128, 128]);  expand_default = None
        return reshape_default_1


def _default_make_inputs():
    return [
    torch.randn([192, 128, 128], dtype=torch.float32, device='cuda'),
    torch.randn([32, 6, 128, 128], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
