"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g21
Pattern hash: cdc0f023503e
Shape hash: 39bdd2a3
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
    def forward(self, getitem_124: "f32[1, 512, 1]", convert_element_type_220: "f32[1, 512, 768]", getitem_125: "f32[1, 512, 1]", arg207_1: "f32[768]", arg208_1: "f32[768]", arg209_1: "f32[30522]", arg1_1: "f32[30522, 768]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-12);  getitem_124 = None
        rsqrt_default: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_220, getitem_125);  convert_element_type_220 = getitem_125 = None
        mul_tensor: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg207_1);  mul_tensor = arg207_1 = None
        add_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg208_1);  mul_tensor_1 = arg208_1 = None
        convert_element_type_default: "f16[30522]" = torch.ops.prims.convert_element_type.default(arg209_1, torch.float16);  arg209_1 = None
        convert_element_type_default_1: "f16[30522, 768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float16);  arg1_1 = None
        convert_element_type_default_2: "f16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        reshape_default: "f16[512, 768]" = torch.ops.aten.reshape.default(convert_element_type_default_2, [512, 768]);  convert_element_type_default_2 = None
        permute_default: "f16[768, 30522]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        return (convert_element_type_default, reshape_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([30522], dtype=torch.float32, device='cuda'),
    torch.randn([30522, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
