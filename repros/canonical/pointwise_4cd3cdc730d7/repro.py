"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s1_g10
Pattern hash: 4cd3cdc730d7
Shape hash: 0f29bc98
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
    def forward(self, arg357_1: "bf16[320]", arg358_1: "bf16[320]", convolution_71: "bf16[128, 320, 8, 8]", arg359_1: "bf16[320]", arg360_1: "bf16[320]", arg377_1: "bf16[192]", arg378_1: "bf16[192]", convolution_75: "bf16[128, 192, 8, 8]", arg379_1: "bf16[192]", arg380_1: "bf16[192]", getitem_6: "bf16[128, 768, 8, 8]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[320]" = torch.ops.prims.convert_element_type.default(arg357_1, torch.float32);  arg357_1 = None
        convert_element_type_default_1: "f32[320]" = torch.ops.prims.convert_element_type.default(arg358_1, torch.float32);  arg358_1 = None
        add_tensor: "f32[320]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 0.001);  convert_element_type_default_1 = None
        sqrt_default: "f32[320]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[320]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[320]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        unsqueeze_default_2: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        sub_tensor: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_default_1);  convolution_71 = unsqueeze_default_1 = None
        mul_tensor_1: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "bf16[320, 1]" = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
        unsqueeze_default_5: "bf16[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "bf16[320, 1]" = torch.ops.aten.unsqueeze.default(arg360_1, -1);  arg360_1 = None
        unsqueeze_default_7: "bf16[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "bf16[128, 320, 8, 8]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        relu_default: "bf16[128, 320, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_default_2);  convert_element_type_default_2 = None
        convert_element_type_default_3: "f32[192]" = torch.ops.prims.convert_element_type.default(arg377_1, torch.float32);  arg377_1 = None
        convert_element_type_default_4: "f32[192]" = torch.ops.prims.convert_element_type.default(arg378_1, torch.float32);  arg378_1 = None
        add_tensor_2: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_default_4, 0.001);  convert_element_type_default_4 = None
        sqrt_default_1: "f32[192]" = torch.ops.aten.sqrt.default(add_tensor_2);  add_tensor_2 = None
        reciprocal_default_1: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_8: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_3, -1);  convert_element_type_default_3 = None
        unsqueeze_default_9: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        sub_tensor_1: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_default_9);  convolution_75 = unsqueeze_default_9 = None
        mul_tensor_4: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
        unsqueeze_default_13: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "bf16[192, 1]" = torch.ops.aten.unsqueeze.default(arg380_1, -1);  arg380_1 = None
        unsqueeze_default_15: "bf16[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_3: "f32[128, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None
        convert_element_type_default_5: "bf16[128, 192, 8, 8]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        relu_default_1: "bf16[128, 192, 8, 8]" = torch.ops.aten.relu.default(convert_element_type_default_5);  convert_element_type_default_5 = None
        cat_default: "bf16[128, 1280, 8, 8]" = torch.ops.aten.cat.default([relu_default, relu_default_1, getitem_6], 1);  relu_default = relu_default_1 = getitem_6 = None
        return cat_default


def _default_make_inputs():
    return [
    torch.randn([320], dtype=torch.bfloat16, device='cuda'),
    torch.randn([320], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 320, 8, 8], dtype=torch.bfloat16, device='cuda'),
    torch.randn([320], dtype=torch.bfloat16, device='cuda'),
    torch.randn([320], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 192, 8, 8], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([192], dtype=torch.bfloat16, device='cuda'),
    torch.randn([128, 768, 8, 8], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
