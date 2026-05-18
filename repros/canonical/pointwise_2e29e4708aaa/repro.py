"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s5_g10
Pattern hash: 2e29e4708aaa
Shape hash: f49a9ca5
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
    def forward(self, arg255_1: "bf16[160]", arg256_1: "bf16[160]", convolution_60: "bf16[8, 160, 7, 7]", arg257_1: "bf16[160]", arg258_1: "bf16[160]", add_116: "bf16[8, 160, 7, 7]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[160]" = torch.ops.prims.convert_element_type.default(arg255_1, torch.float32);  arg255_1 = None
        convert_element_type_default_1: "f32[160]" = torch.ops.prims.convert_element_type.default(arg256_1, torch.float32);  arg256_1 = None
        add_tensor: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[160]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        unsqueeze_default_2: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        sub_tensor: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_default_1);  convolution_60 = unsqueeze_default_1 = None
        mul_tensor_1: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "bf16[160, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_default_5: "bf16[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "bf16[160, 1]" = torch.ops.aten.unsqueeze.default(arg258_1, -1);  arg258_1 = None
        unsqueeze_default_7: "bf16[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "bf16[8, 160, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        add_tensor_2: "bf16[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_default_2, add_116);  convert_element_type_default_2 = add_116 = None
        return add_tensor_2


def _default_make_inputs():
    return [
    torch.randn([160], dtype=torch.bfloat16, device='cuda'),
    torch.randn([160], dtype=torch.bfloat16, device='cuda'),
    torch.randn([8, 160, 7, 7], dtype=torch.bfloat16, device='cuda'),
    torch.randn([160], dtype=torch.bfloat16, device='cuda'),
    torch.randn([160], dtype=torch.bfloat16, device='cuda'),
    torch.randn([8, 160, 7, 7], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
