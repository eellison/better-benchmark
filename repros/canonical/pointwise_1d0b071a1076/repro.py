"""
Standalone repro captured via capture_hook.
Label: mobilenet_v2_inference
Pattern hash: 1d0b071a1076
Shape hash: 854351d6
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg247_1: "f32[960]", convolution_49: "f32[1, 960, 7, 7]", arg248_1: "f32[960]", arg249_1: "f32[960]", arg250_1: "f32[960]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:64 in forward, code: return self.conv(x)
        unsqueeze_default: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_default_1: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[1, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_default_1);  convolution_49 = unsqueeze_default_1 = None
        add_tensor: "f32[960]" = torch.ops.aten.add.Tensor(arg248_1, 1e-05);  arg248_1 = None
        sqrt_default: "f32[960]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[1, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_default_5: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[1, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_default_7: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[1, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        clamp_min_default: "f32[1, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_tensor_1, 0.0);  add_tensor_1 = None
        clamp_max_default: "f32[1, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6.0);  clamp_min_default = None
        return clamp_max_default


def _default_make_inputs():
    return [
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([1, 960, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
