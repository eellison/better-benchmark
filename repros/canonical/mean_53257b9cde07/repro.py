"""
Standalone repro captured via capture_hook.
Label: timm_repvgg_a2_infer_000
Pattern hash: 53257b9cde07
Shape hash: 87b46cfc
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1408], f32), T([128, 1408, 7, 7], f32), T([1408], f32), T([1408], f32), T([1408], f32), T([1408], f32), T([128, 1408, 7, 7], f32), T([1408], f32), T([1408], f32), T([1408], f32), S([128, 1408]))"

class Repro(torch.nn.Module):
    def forward(self, arg280_1: "f32[1408]", convolution_42: "f32[128, 1408, 7, 7]", arg281_1: "f32[1408]", arg282_1: "f32[1408]", arg283_1: "f32[1408]", arg285_1: "f32[1408]", convolution_43: "f32[128, 1408, 7, 7]", arg286_1: "f32[1408]", arg287_1: "f32[1408]", arg288_1: "f32[1408]", _shape_param_0):
        # No stacktrace found for following nodes
        unsqueeze_default: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
        unsqueeze_default_1: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_default_1);  convolution_42 = unsqueeze_default_1 = None
        add_tensor: "f32[1408]" = torch.ops.aten.add.Tensor(arg281_1, 1e-05);  arg281_1 = None
        sqrt_default: "f32[1408]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[1408]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[1408]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_default_5: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg283_1, -1);  arg283_1 = None
        unsqueeze_default_7: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        unsqueeze_default_8: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_default_9: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        sub_tensor_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_default_9);  convolution_43 = unsqueeze_default_9 = None
        add_tensor_2: "f32[1408]" = torch.ops.aten.add.Tensor(arg286_1, 1e-05);  arg286_1 = None
        sqrt_default_1: "f32[1408]" = torch.ops.aten.sqrt.default(add_tensor_2);  add_tensor_2 = None
        reciprocal_default_1: "f32[1408]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[1408]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_10: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        mul_tensor_4: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_default_13: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg288_1, -1);  arg288_1 = None
        unsqueeze_default_15: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_3: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None
        add_tensor_4: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, add_tensor_3);  add_tensor_1 = add_tensor_3 = None
        relu_default: "f32[128, 1408, 7, 7]" = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None
        mean_dim: "f32[128, 1408, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True);  relu_default = None
        as_strided_default: "f32[128, 1408, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [128, 1408, 1, 1], [1408, 1, 1408, 1408]);  mean_dim = None
        view_default: "f32[128, 1408]" = torch.ops.aten.view.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None
        return view_default

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
