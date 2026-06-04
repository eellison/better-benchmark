"""
Standalone repro captured via capture_hook.
Label: torchbench_LearningToPaint_infer_000
Pattern hash: 1545d889b176
Shape hash: 53fb4d3f
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([256], f32), T([1024, 256, 8, 8], f32), T([256], f32), T([256], f32), T([256], f32), T([1024, 256, 8, 8], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg77_1: "f32[256]", convolution_15: "f32[1024, 256, 8, 8]", arg78_1: "f32[256]", arg79_1: "f32[256]", arg80_1: "f32[256]", relu_10: "f32[1024, 256, 8, 8]"):
        # No stacktrace found for following nodes
        unsqueeze_default: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_default_1: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[1024, 256, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_default_1);  convolution_15 = unsqueeze_default_1 = None
        add_tensor: "f32[256]" = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_default: "f32[256]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_default_5: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[1024, 256, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_default_7: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[1024, 256, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        add_tensor_2: "f32[1024, 256, 8, 8]" = torch.ops.aten.add.Tensor(add_tensor_1, relu_10);  add_tensor_1 = relu_10 = None
        relu_default: "f32[1024, 256, 8, 8]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        return relu_default

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
