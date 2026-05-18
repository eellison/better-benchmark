"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s8_g27
Pattern hash: dcb806fde724
Shape hash: 481f4497
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
    def forward(self, arg9_1: "f32[256, 1]", arg10_1: "f32[256, 1]", arg8_1: "f16[256, 1]", arg11_1: "f16[256, 1]", arg5_1: "f32[256, 1]", arg4_1: "f16[256, 1]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[256, 1]" = torch.ops.aten.add.Tensor(arg9_1, arg10_1);  arg9_1 = arg10_1 = None
        add_tensor_1: "f16[256, 1]" = torch.ops.aten.add.Tensor(arg8_1, arg11_1);  arg8_1 = arg11_1 = None
        mul_tensor: "f32[256, 1]" = torch.ops.aten.mul.Tensor(add_tensor, arg5_1);  add_tensor = arg5_1 = None
        convert_element_type_default: "f16[256, 1]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float16);  mul_tensor = None
        mul_tensor_1: "f16[256, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 6.0);  convert_element_type_default = None
        convert_element_type_default_1: "f32[256, 1]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float32);  mul_tensor_1 = None
        convert_element_type_default_2: "f32[256, 1]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        mul_tensor_2: "f32[256, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, convert_element_type_default_2);  convert_element_type_default_2 = None
        sub_tensor: "f32[256, 1]" = torch.ops.aten.sub.Tensor(1, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_3: "f32[256, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, sub_tensor);  convert_element_type_default_1 = sub_tensor = None
        convert_element_type_default_3: "f16[256, 1]" = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.float16);  mul_tensor_3 = None
        cat_default: "f16[256, 2]" = torch.ops.aten.cat.default([add_tensor_1, convert_element_type_default_3], 1);  add_tensor_1 = convert_element_type_default_3 = None
        return cat_default


def _default_make_inputs():
    return [
    torch.randn([256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256, 1], dtype=torch.float16, device='cuda'),
    torch.randn([256, 1], dtype=torch.float16, device='cuda'),
    torch.randn([256, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256, 1], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
