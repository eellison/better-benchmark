"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s7_g20
Pattern hash: d8d9e20492d5
Shape hash: 65f9f5f0
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_36: "f32[1, 384, 1, 1]", convolution_14: "f16[128, 384, 14, 14]", getitem_37: "f32[1, 384, 1, 1]", arg109_1: "f32[384]", arg110_1: "f32[384]", getitem_38: "f32[1, 384, 1, 1]", convolution_15: "f16[128, 384, 14, 14]", getitem_39: "f32[1, 384, 1, 1]", arg115_1: "f32[384]", arg116_1: "f32[384]", arg122_1: "f32[384, 384, 1, 1]", arg128_1: "f32[384, 384, 3, 3]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_default: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_37);  convolution_14 = getitem_37 = None
        mul_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_default_1: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_default_3: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        convert_element_type_default: "f16[128, 384, 14, 14]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        add_tensor_2: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_default_1: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_39);  convolution_15 = getitem_39 = None
        mul_tensor_2: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_default_5: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_default_7: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None
        convert_element_type_default_1: "f16[128, 384, 14, 14]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float16);  add_tensor_3 = None
        add_tensor_4: "f16[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_1);  convert_element_type_default = convert_element_type_default_1 = None
        relu_default: "f16[128, 384, 14, 14]" = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None
        convert_element_type_default_2: "f16[384, 384, 1, 1]" = torch.ops.prims.convert_element_type.default(arg122_1, torch.float16);  arg122_1 = None
        convert_element_type_default_3: "f16[384, 384, 3, 3]" = torch.ops.prims.convert_element_type.default(arg128_1, torch.float16);  arg128_1 = None
        return (relu_default, convert_element_type_default_2, convert_element_type_default_3)


def _default_make_inputs():
    return [
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 384, 14, 14], dtype=torch.float16, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128, 384, 14, 14], dtype=torch.float16, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
