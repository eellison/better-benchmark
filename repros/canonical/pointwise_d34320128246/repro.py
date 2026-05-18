"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s1_g82
Pattern hash: d34320128246
Shape hash: e8e584ae
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[2, 209981]", _shape_param_0):
        # No stacktrace found for following nodes
        full_default: "f32[209981]" = torch.ops.aten.full.default([209981], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_int: "i64[209981]" = torch.ops.aten.select.int(arg0_1, 0, 0)
        select_int_1: "i64[209981]" = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        expand_default: "i64[209981]" = torch.ops.aten.expand.default(select_int_1, _shape_param_0);  _shape_param_0 = None
        full_default_1: "f32[10000]" = torch.ops.aten.full.default([10000], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter_add_default: "f32[10000]" = torch.ops.aten.scatter_add.default(full_default_1, 0, expand_default, full_default);  full_default_1 = expand_default = full_default = None
        pow_tensor_scalar: "f32[10000]" = torch.ops.aten.pow.Tensor_Scalar(scatter_add_default, -0.5);  scatter_add_default = None
        eq_scalar: "b8[10000]" = torch.ops.aten.eq.Scalar(pow_tensor_scalar, inf)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[10000]" = torch.ops.aten.where.self(eq_scalar, full_default_2, pow_tensor_scalar);  eq_scalar = full_default_2 = pow_tensor_scalar = None
        index_tensor: "f32[209981]" = torch.ops.aten.index.Tensor(where_self, [select_int]);  select_int = None
        index_tensor_1: "f32[209981]" = torch.ops.aten.index.Tensor(where_self, [select_int_1]);  where_self = select_int_1 = None
        mul_tensor: "f32[209981]" = torch.ops.aten.mul.Tensor(index_tensor, index_tensor_1);  index_tensor = index_tensor_1 = None
        return mul_tensor


def _default_make_inputs():
    return [
    torch.randint(0, 2, [2, 209981], dtype=torch.int64, device='cuda'),
    [209981],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
