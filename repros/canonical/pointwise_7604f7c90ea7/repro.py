"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train_000
Pattern hash: 7604f7c90ea7
Shape hash: fe7e1b8d
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 512, 1, 1], f32), T([128, 512, 24, 24], f32), T([], f32), T([128, 512, 24, 24], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_23: "f32[128, 512, 1, 1]", convolution_21: "f32[128, 512, 24, 24]", arg69_1: "f32[]", add_26: "f32[128, 512, 24, 24]"):
        # No stacktrace found for following nodes
        sigmoid_default: "f32[128, 512, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
        mul_tensor: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_21, sigmoid_default);  convolution_21 = sigmoid_default = None
        mul_tensor_1: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor, 2.0);  mul_tensor = None
        mul_tensor_2: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_1, arg69_1);  mul_tensor_1 = arg69_1 = None
        mul_tensor_3: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.2);  mul_tensor_2 = None
        add_tensor: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_tensor_3, add_26);  mul_tensor_3 = add_26 = None
        mul_tensor_4: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5)
        mul_tensor_5: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7071067811865476)
        erf_default: "f32[128, 512, 24, 24]" = torch.ops.aten.erf.default(mul_tensor_5);  mul_tensor_5 = None
        add_tensor_1: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_6: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_4, add_tensor_1);  mul_tensor_4 = None
        mul_tensor_7: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_6, 1.7015043497085571);  mul_tensor_6 = None
        mul_tensor_8: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.9622504486493761);  mul_tensor_7 = None
        avg_pool2d_default: "f32[128, 512, 12, 12]" = torch.ops.aten.avg_pool2d.default(mul_tensor_8, [2, 2], [2, 2], [0, 0], True, False);  mul_tensor_8 = None
        mul_tensor_9: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor_1, 0.5);  add_tensor_1 = None
        mul_tensor_10: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor, add_tensor)
        mul_tensor_11: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_10, -0.5);  mul_tensor_10 = None
        exp_default: "f32[128, 512, 24, 24]" = torch.ops.aten.exp.default(mul_tensor_11);  mul_tensor_11 = None
        mul_tensor_12: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_13: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_12);  add_tensor = mul_tensor_12 = None
        add_tensor_2: "f32[128, 512, 24, 24]" = torch.ops.aten.add.Tensor(mul_tensor_9, mul_tensor_13);  mul_tensor_9 = mul_tensor_13 = None
        return (avg_pool2d_default, add_tensor_2)

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
