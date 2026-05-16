"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s1_g213
Pattern hash: 4589a681b729
Shape hash: f7364f3d
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[2, 209981]", arg4_1: "f32[10000, 64]", arg1_1: "f32[209981]"):
        # No stacktrace found for following nodes
        select_int: "i64[209981]" = torch.ops.aten.select.int(arg0_1, 0, 1)
        reshape_default: "i64[209981, 1]" = torch.ops.aten.reshape.default(select_int, [-1, 1]);  select_int = None
        expand_default: "i64[209981, 64]" = torch.ops.aten.expand.default(reshape_default, [209981, 64]);  reshape_default = None
        gather_default: "f32[209981, 64]" = torch.ops.aten.gather.default(arg4_1, 0, expand_default);  arg4_1 = expand_default = None
        reshape_default_1: "f32[209981, 1]" = torch.ops.aten.reshape.default(arg1_1, [-1, 1]);  arg1_1 = None
        mul_tensor: "f32[209981, 64]" = torch.ops.aten.mul.Tensor(gather_default, reshape_default_1);  gather_default = reshape_default_1 = None
        convert_element_type_default: "f16[209981, 64]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float16);  mul_tensor = None
        full_default: "f16[10000, 64]" = torch.ops.aten.full.default([10000, 64], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_int_1: "i64[209981]" = torch.ops.aten.select.int(arg0_1, 0, 0);  arg0_1 = None
        index_put_default: "f16[10000, 64]" = torch.ops.aten.index_put.default(full_default, [select_int_1], convert_element_type_default, True);  full_default = select_int_1 = convert_element_type_default = None
        permute_default: "f16[64, 10000]" = torch.ops.aten.permute.default(index_put_default, [1, 0]);  index_put_default = None
        return permute_default


def _default_make_inputs():
    return [
    torch.randint(0, 2, [2, 209981], dtype=torch.int64, device='cuda'),
    torch.randn([10000, 64], dtype=torch.float32, device='cuda'),
    torch.randn([209981], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
