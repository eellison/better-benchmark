"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s5_g10
Pattern hash: 5209af342ead
Shape hash: 83ec2c9d
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg91_1: "bf16[120]", arg92_1: "bf16[120]", convolution_20: "bf16[8, 120, 28, 28]", arg93_1: "bf16[120]", arg94_1: "bf16[120]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[120]" = torch.ops.prims.convert_element_type.default(arg91_1, torch.float32);  arg91_1 = None
        convert_element_type_default_1: "f32[120]" = torch.ops.prims.convert_element_type.default(arg92_1, torch.float32);  arg92_1 = None
        add_tensor: "f32[120]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[120]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[120]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[120]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        unsqueeze_default_2: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        sub_tensor: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_default_1);  convolution_20 = unsqueeze_default_1 = None
        mul_tensor_1: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "bf16[120, 1]" = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_default_5: "bf16[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "bf16[120, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_default_7: "bf16[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "bf16[8, 120, 28, 28]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        relu_default: "bf16[8, 120, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_default_2);  convert_element_type_default_2 = None
        mean_dim: "bf16[8, 120, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [2, 3], True);  relu_default = None
        return mean_dim


def _default_make_inputs():
    return [
    torch.randn([120], dtype=torch.bfloat16, device='cuda'),
    torch.randn([120], dtype=torch.bfloat16, device='cuda'),
    torch.randn([8, 120, 28, 28], dtype=torch.bfloat16, device='cuda'),
    torch.randn([120], dtype=torch.bfloat16, device='cuda'),
    torch.randn([120], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
