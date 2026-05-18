"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s4_g10
Pattern hash: ad03364c504b
Shape hash: 76c6b820
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
    def forward(self, arg317_1: "bf16[80]", arg318_1: "bf16[80]", convolution_69: "bf16[8, 80, 7, 7]", arg319_1: "bf16[80]", arg320_1: "bf16[80]", arg327_1: "bf16[112]", arg328_1: "bf16[112]", convolution_71: "bf16[8, 112, 7, 7]", arg329_1: "bf16[112]", arg330_1: "bf16[112]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[80]" = torch.ops.prims.convert_element_type.default(arg317_1, torch.float32);  arg317_1 = None
        convert_element_type_default_1: "f32[80]" = torch.ops.prims.convert_element_type.default(arg318_1, torch.float32);  arg318_1 = None
        add_tensor: "f32[80]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[80]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        unsqueeze_default_2: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        sub_tensor: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_default_1);  convolution_69 = unsqueeze_default_1 = None
        mul_tensor_1: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "bf16[80, 1]" = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_default_5: "bf16[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "bf16[80, 1]" = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_default_7: "bf16[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "bf16[8, 80, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        convert_element_type_default_3: "f32[112]" = torch.ops.prims.convert_element_type.default(arg327_1, torch.float32);  arg327_1 = None
        convert_element_type_default_4: "f32[112]" = torch.ops.prims.convert_element_type.default(arg328_1, torch.float32);  arg328_1 = None
        add_tensor_2: "f32[112]" = torch.ops.aten.add.Tensor(convert_element_type_default_4, 1e-05);  convert_element_type_default_4 = None
        sqrt_default_1: "f32[112]" = torch.ops.aten.sqrt.default(add_tensor_2);  add_tensor_2 = None
        reciprocal_default_1: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_8: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_3, -1);  convert_element_type_default_3 = None
        unsqueeze_default_9: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        sub_tensor_1: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_default_9);  convolution_71 = unsqueeze_default_9 = None
        mul_tensor_4: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "bf16[112, 1]" = torch.ops.aten.unsqueeze.default(arg329_1, -1);  arg329_1 = None
        unsqueeze_default_13: "bf16[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "bf16[112, 1]" = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_default_15: "bf16[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_3: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None
        convert_element_type_default_5: "bf16[8, 112, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        return (convert_element_type_default_2, convert_element_type_default_5)


def _default_make_inputs():
    return [
    torch.randn([80], dtype=torch.bfloat16, device='cuda'),
    torch.randn([80], dtype=torch.bfloat16, device='cuda'),
    torch.randn([8, 80, 7, 7], dtype=torch.bfloat16, device='cuda'),
    torch.randn([80], dtype=torch.bfloat16, device='cuda'),
    torch.randn([80], dtype=torch.bfloat16, device='cuda'),
    torch.randn([112], dtype=torch.bfloat16, device='cuda'),
    torch.randn([112], dtype=torch.bfloat16, device='cuda'),
    torch.randn([8, 112, 7, 7], dtype=torch.bfloat16, device='cuda'),
    torch.randn([112], dtype=torch.bfloat16, device='cuda'),
    torch.randn([112], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
