"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s1_g102
Pattern hash: ed7f629de0d2
Shape hash: bc4aa1e6
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
    def forward(self, arg2_1: "i64[2, 209981]", mm: "f16[10000, 64]", arg3_1: "f32[209981]", arg4_1: "f32[64]", _shape_param_0):
        # No stacktrace found for following nodes
        select_int: "i64[209981]" = torch.ops.aten.select.int(arg2_1, 0, 0)
        index_tensor: "f16[209981, 64]" = torch.ops.aten.index.Tensor(mm, [select_int]);  mm = select_int = None
        select_int_1: "i64[209981]" = torch.ops.aten.select.int(arg2_1, 0, 1);  arg2_1 = None
        reshape_default: "f32[209981, 1]" = torch.ops.aten.reshape.default(arg3_1, [-1, 1]);  arg3_1 = None
        mul_tensor: "f32[209981, 64]" = torch.ops.aten.mul.Tensor(reshape_default, index_tensor);  reshape_default = index_tensor = None
        reshape_default_1: "i64[209981, 1]" = torch.ops.aten.reshape.default(select_int_1, [-1, 1]);  select_int_1 = None
        expand_default: "i64[209981, 64]" = torch.ops.aten.expand.default(reshape_default_1, _shape_param_0);  reshape_default_1 = _shape_param_0 = None
        full_default: "f32[10000, 64]" = torch.ops.aten.full.default([10000, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter_add_default: "f32[10000, 64]" = torch.ops.aten.scatter_add.default(full_default, 0, expand_default, mul_tensor);  full_default = expand_default = mul_tensor = None
        add_tensor: "f32[10000, 64]" = torch.ops.aten.add.Tensor(scatter_add_default, arg4_1);  scatter_add_default = arg4_1 = None
        _output_to_half_0: "f16[10000, 64]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float16);  add_tensor = None
        return _output_to_half_0


def _default_make_inputs():
    return [
    torch.randint(0, 10000, [2, 209981], dtype=torch.int64, device='cuda'),
    torch.randn([10000, 64], dtype=torch.float16, device='cuda'),
    torch.randn([209981], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    [209981, 64],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
