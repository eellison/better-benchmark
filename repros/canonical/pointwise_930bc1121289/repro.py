"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s7_g10
Pattern hash: 930bc1121289
Shape hash: 45a11541
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg87_1: "bf16[64]", arg88_1: "bf16[64]", arg85_1: "bf16[64]", arg86_1: "bf16[64]", convolution_16: "bf16[16, 64, 128, 128]"):
        # No stacktrace found for following nodes
        repeat_default: "bf16[1024]" = torch.ops.aten.repeat.default(arg87_1, [16]);  arg87_1 = None
        repeat_default_1: "bf16[1024]" = torch.ops.aten.repeat.default(arg88_1, [16]);  arg88_1 = None
        repeat_default_2: "bf16[1024]" = torch.ops.aten.repeat.default(arg85_1, [16]);  arg85_1 = None
        repeat_default_3: "bf16[1024]" = torch.ops.aten.repeat.default(arg86_1, [16]);  arg86_1 = None
        reshape_default: "bf16[1, 1024, 128, 128]" = torch.ops.aten.reshape.default(convolution_16, [1, 1024, 128, 128]);  convolution_16 = None
        convert_element_type_default: "f32[1024]" = torch.ops.prims.convert_element_type.default(repeat_default_2, torch.float32);  repeat_default_2 = None
        convert_element_type_default_1: "f32[1024]" = torch.ops.prims.convert_element_type.default(repeat_default_3, torch.float32);  repeat_default_3 = None
        add_tensor: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[1024]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        sub_tensor: "f32[1, 1024, 128, 128]" = torch.ops.aten.sub.Tensor(reshape_default, unsqueeze_default_1);  reshape_default = unsqueeze_default_1 = None
        mul_tensor_1: "f32[1, 1024, 128, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "bf16[1024, 1]" = torch.ops.aten.unsqueeze.default(repeat_default, -1);  repeat_default = None
        unsqueeze_default_5: "bf16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[1, 1024, 128, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "bf16[1024, 1]" = torch.ops.aten.unsqueeze.default(repeat_default_1, -1);  repeat_default_1 = None
        unsqueeze_default_7: "bf16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[1, 1024, 128, 128]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "bf16[1, 1024, 128, 128]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        reshape_default_1: "bf16[16, 64, 128, 128]" = torch.ops.aten.reshape.default(convert_element_type_default_2, [16, 64, 128, 128]);  convert_element_type_default_2 = None
        relu_default: "bf16[16, 64, 128, 128]" = torch.ops.aten.relu.default(reshape_default_1);  reshape_default_1 = None
        return relu_default


def _default_make_inputs():
    return [
    torch.randn([64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([16, 64, 128, 128], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
