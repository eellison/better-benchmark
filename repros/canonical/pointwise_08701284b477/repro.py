"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_infer_000
Pattern hash: 08701284b477
Shape hash: ae16f756
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([80], f32), T([512, 80, 7, 7], f32), T([80], f32), T([80], f32), T([80], f32), T([512, 80, 7, 7], f32), T([512, 160, 7, 7], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg420_1: "f32[80]", convolution_92: "f32[512, 80, 7, 7]", arg421_1: "f32[80]", arg422_1: "f32[80]", arg423_1: "f32[80]", add_177: "f32[512, 80, 7, 7]", add_170: "f32[512, 160, 7, 7]"):
        # No stacktrace found for following nodes
        unsqueeze_default: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg420_1, -1);  arg420_1 = None
        unsqueeze_default_1: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[512, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_default_1);  convolution_92 = unsqueeze_default_1 = None
        add_tensor: "f32[80]" = torch.ops.aten.add.Tensor(arg421_1, 1e-05);  arg421_1 = None
        sqrt_default: "f32[80]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[80]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[80]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[512, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg422_1, -1);  arg422_1 = None
        unsqueeze_default_5: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[512, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(arg423_1, -1);  arg423_1 = None
        unsqueeze_default_7: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[512, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        cat_default: "f32[512, 160, 7, 7]" = torch.ops.aten.cat.default([add_177, add_tensor_1], 1);  add_177 = add_tensor_1 = None
        add_tensor_2: "f32[512, 160, 7, 7]" = torch.ops.aten.add.Tensor(cat_default, add_170);  cat_default = add_170 = None
        return add_tensor_2

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
