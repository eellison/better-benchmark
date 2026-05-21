"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_infer_000
Pattern hash: 8be04df3df80
Shape hash: f7dee424
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([480], f32), T([512, 480, 7, 7], f32), T([480], f32), T([480], f32), T([480], f32), T([512, 480, 7, 7], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg386_1: "f32[480]", convolution_84: "f32[512, 480, 7, 7]", arg387_1: "f32[480]", arg388_1: "f32[480]", arg389_1: "f32[480]", relu_35: "f32[512, 480, 7, 7]"):
        # No stacktrace found for following nodes
        unsqueeze_default: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg386_1, -1);  arg386_1 = None
        unsqueeze_default_1: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[512, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_default_1);  convolution_84 = unsqueeze_default_1 = None
        add_tensor: "f32[480]" = torch.ops.aten.add.Tensor(arg387_1, 1e-05);  arg387_1 = None
        sqrt_default: "f32[480]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg388_1, -1);  arg388_1 = None
        unsqueeze_default_5: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[512, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(arg389_1, -1);  arg389_1 = None
        unsqueeze_default_7: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[512, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        relu_default: "f32[512, 480, 7, 7]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        cat_default: "f32[512, 960, 7, 7]" = torch.ops.aten.cat.default([relu_35, relu_default], 1);  relu_35 = relu_default = None
        return cat_default



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
