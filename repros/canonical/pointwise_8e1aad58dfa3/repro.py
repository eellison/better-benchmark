"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_infer
Pattern hash: 8e1aad58dfa3
Shape hash: 70a8079a
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([80], f32), T([512, 80, 7, 7], f32, stride=(3920, 1, 560, 80)), T([80], f32), T([80], f32), T([80], f32), T([112], f32), T([512, 112, 7, 7], f32, stride=(5488, 1, 784, 112)), T([112], f32), T([112], f32), T([112], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg317_1: "f32[80]", convolution_69: "f32[512, 80, 7, 7]", arg318_1: "f32[80]", arg319_1: "f32[80]", arg320_1: "f32[80]", arg327_1: "f32[112]", convolution_71: "f32[512, 112, 7, 7]", arg328_1: "f32[112]", arg329_1: "f32[112]", arg330_1: "f32[112]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg317_1, -1);  arg317_1 = None
        unsqueeze_default_1: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[512, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_default_1);  convolution_69 = unsqueeze_default_1 = None
        add_tensor: "f32[80]" = torch.ops.aten.add.Tensor(arg318_1, 1e-05);  arg318_1 = None
        sqrt_default: "f32[80]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_default_5: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[512, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_default_7: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[512, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        unsqueeze_default_8: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg327_1, -1);  arg327_1 = None
        unsqueeze_default_9: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        sub_tensor_1: "f32[512, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_default_9);  convolution_71 = unsqueeze_default_9 = None
        add_tensor_2: "f32[112]" = torch.ops.aten.add.Tensor(arg328_1, 1e-05);  arg328_1 = None
        sqrt_default_1: "f32[112]" = torch.ops.aten.sqrt.default(add_tensor_2);  add_tensor_2 = None
        reciprocal_default_1: "f32[112]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[112]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_10: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        mul_tensor_4: "f32[512, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg329_1, -1);  arg329_1 = None
        unsqueeze_default_13: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[512, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_default_15: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_3: "f32[512, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None
        return (add_tensor_1, add_tensor_3)


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
