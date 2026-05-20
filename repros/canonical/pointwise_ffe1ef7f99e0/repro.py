"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv3_large_100_infer
Pattern hash: ffe1ef7f99e0
Shape hash: c8bce8a6
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([160], f32), T([512, 160, 7, 7], f32, stride=(7840, 1, 1120, 160)), T([160], f32), T([160], f32), T([160], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg216_1: "f32[160]", convolution_50: "f32[512, 160, 7, 7]", arg217_1: "f32[160]", arg218_1: "f32[160]", arg219_1: "f32[160]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_default_1: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[512, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_default_1);  convolution_50 = unsqueeze_default_1 = None
        add_tensor: "f32[160]" = torch.ops.aten.add.Tensor(arg217_1, 1e-05);  arg217_1 = None
        sqrt_default: "f32[160]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_default_5: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[512, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_default_7: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[512, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        return add_tensor_1


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
