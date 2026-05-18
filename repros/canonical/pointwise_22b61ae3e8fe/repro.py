"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s1_g10
Pattern hash: 22b61ae3e8fe
Shape hash: 9c454a41
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
    def forward(self, arg302_1: "bf16[192]", arg303_1: "bf16[192]", convolution_60: "bf16[128, 192, 17, 17]", arg304_1: "bf16[192]", arg305_1: "bf16[192]", arg317_1: "bf16[192]", arg318_1: "bf16[192]", convolution_63: "bf16[128, 192, 17, 17]", arg319_1: "bf16[192]", arg320_1: "bf16[192]", arg342_1: "bf16[192]", arg343_1: "bf16[192]", convolution_68: "bf16[128, 192, 17, 17]", arg344_1: "bf16[192]", arg345_1: "bf16[192]", arg347_1: "bf16[192]", arg348_1: "bf16[192]", convolution_69: "bf16[128, 192, 17, 17]", arg349_1: "bf16[192]", arg350_1: "bf16[192]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[192]" = torch.ops.prims.convert_element_type.default(arg302_1, torch.float32);  arg302_1 = None
        convert_element_type_default_1: "f32[192]" = torch.ops.prims.convert_element_type.default(arg303_1, torch.float32);  arg303_1 = None
        add_tensor: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 0.001);  convert_element_type_default_1 = None
        sqrt_default: "f32[192]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        sub_tensor: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_default_1);  convolution_60 = unsqueeze_default_1 = None
        mul_tensor_1: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
        unsqueeze_default_5: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_default_7: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "bf16[128, 192, 17, 17]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        relu_default: "bf16[128, 192, 17, 17]" = torch.ops.aten.relu.default(convert_element_type_default_2);  convert_element_type_default_2 = None
        convert_element_type_default_3: "f32[192]" = torch.ops.prims.convert_element_type.default(arg317_1, torch.float32);  arg317_1 = None
        convert_element_type_default_4: "f32[192]" = torch.ops.prims.convert_element_type.default(arg318_1, torch.float32);  arg318_1 = None
        add_tensor_2: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_default_4, 0.001);  convert_element_type_default_4 = None
        sqrt_default_1: "f32[192]" = torch.ops.aten.sqrt.default(add_tensor_2);  add_tensor_2 = None
        reciprocal_default_1: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_8: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_3, -1);  convert_element_type_default_3 = None
        unsqueeze_default_9: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        sub_tensor_1: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_default_9);  convolution_63 = unsqueeze_default_9 = None
        mul_tensor_4: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_default_13: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_default_15: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_3: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None
        convert_element_type_default_5: "bf16[128, 192, 17, 17]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        relu_default_1: "bf16[128, 192, 17, 17]" = torch.ops.aten.relu.default(convert_element_type_default_5);  convert_element_type_default_5 = None
        convert_element_type_default_6: "f32[192]" = torch.ops.prims.convert_element_type.default(arg342_1, torch.float32);  arg342_1 = None
        convert_element_type_default_7: "f32[192]" = torch.ops.prims.convert_element_type.default(arg343_1, torch.float32);  arg343_1 = None
        add_tensor_4: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_default_7, 0.001);  convert_element_type_default_7 = None
        sqrt_default_2: "f32[192]" = torch.ops.aten.sqrt.default(add_tensor_4);  add_tensor_4 = None
        reciprocal_default_2: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_default_2);  sqrt_default_2 = None
        mul_tensor_6: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_default_2, 1);  reciprocal_default_2 = None
        unsqueeze_default_16: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_6, -1);  convert_element_type_default_6 = None
        unsqueeze_default_17: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, -1);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, -1);  mul_tensor_6 = None
        unsqueeze_default_19: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, -1);  unsqueeze_default_18 = None
        sub_tensor_2: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_default_17);  convolution_68 = unsqueeze_default_17 = None
        mul_tensor_7: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_19);  sub_tensor_2 = unsqueeze_default_19 = None
        unsqueeze_default_20: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_default_21: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, -1);  unsqueeze_default_20 = None
        mul_tensor_8: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_21);  mul_tensor_7 = unsqueeze_default_21 = None
        unsqueeze_default_22: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg345_1, -1);  arg345_1 = None
        unsqueeze_default_23: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, -1);  unsqueeze_default_22 = None
        add_tensor_5: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_8, unsqueeze_default_23);  mul_tensor_8 = unsqueeze_default_23 = None
        convert_element_type_default_8: "bf16[128, 192, 17, 17]" = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.bfloat16);  add_tensor_5 = None
        relu_default_2: "bf16[128, 192, 17, 17]" = torch.ops.aten.relu.default(convert_element_type_default_8);  convert_element_type_default_8 = None
        convert_element_type_default_9: "f32[192]" = torch.ops.prims.convert_element_type.default(arg347_1, torch.float32);  arg347_1 = None
        convert_element_type_default_10: "f32[192]" = torch.ops.prims.convert_element_type.default(arg348_1, torch.float32);  arg348_1 = None
        add_tensor_6: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_default_10, 0.001);  convert_element_type_default_10 = None
        sqrt_default_3: "f32[192]" = torch.ops.aten.sqrt.default(add_tensor_6);  add_tensor_6 = None
        reciprocal_default_3: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_default_3);  sqrt_default_3 = None
        mul_tensor_9: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_default_3, 1);  reciprocal_default_3 = None
        unsqueeze_default_24: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_9, -1);  convert_element_type_default_9 = None
        unsqueeze_default_25: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, -1);  unsqueeze_default_24 = None
        unsqueeze_default_26: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, -1);  mul_tensor_9 = None
        unsqueeze_default_27: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, -1);  unsqueeze_default_26 = None
        sub_tensor_3: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_default_25);  convolution_69 = unsqueeze_default_25 = None
        mul_tensor_10: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_27);  sub_tensor_3 = unsqueeze_default_27 = None
        unsqueeze_default_28: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_default_29: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, -1);  unsqueeze_default_28 = None
        mul_tensor_11: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_10, unsqueeze_default_29);  mul_tensor_10 = unsqueeze_default_29 = None
        unsqueeze_default_30: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg350_1, -1);  arg350_1 = None
        unsqueeze_default_31: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, -1);  unsqueeze_default_30 = None
        add_tensor_7: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_11, unsqueeze_default_31);  mul_tensor_11 = unsqueeze_default_31 = None
        convert_element_type_default_11: "bf16[128, 192, 17, 17]" = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.bfloat16);  add_tensor_7 = None
        relu_default_3: "bf16[128, 192, 17, 17]" = torch.ops.aten.relu.default(convert_element_type_default_11);  convert_element_type_default_11 = None
        cat_default: "bf16[128, 768, 17, 17]" = torch.ops.aten.cat.default([relu_default, relu_default_1, relu_default_2, relu_default_3], 1);  relu_default = relu_default_1 = relu_default_2 = relu_default_3 = None
        return cat_default


def _default_make_inputs():
    return [
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 192, 17, 17], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 192, 17, 17], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 192, 17, 17], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 192, 17, 17], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
