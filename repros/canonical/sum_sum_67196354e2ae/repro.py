"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train_001
Pattern hash: 67196354e2ae
Shape hash: 685eaed6
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 512, 14, 14], f32), T([128, 512, 28, 28], f32), T([128, 512, 28, 28], f32), T([128, 512, 28, 28], f32), T([128, 512, 28, 28], f32), T([128, 512, 1, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_168: "f32[128, 512, 14, 14]", arg220_1: "f32[128, 512, 28, 28]", getitem_165: "f32[128, 512, 28, 28]", arg219_1: "f32[128, 512, 28, 28]", arg215_1: "f32[128, 512, 28, 28]", arg218_1: "f32[128, 512, 1, 1]"):
        # No stacktrace found for following nodes
        avg_pool2d_backward_default: "f32[128, 512, 28, 28]" = torch.ops.aten.avg_pool2d_backward.default(getitem_168, arg220_1, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_168 = arg220_1 = None
        add_tensor: "f32[128, 512, 28, 28]" = torch.ops.aten.add.Tensor(getitem_165, avg_pool2d_backward_default);  getitem_165 = avg_pool2d_backward_default = None
        mul_tensor: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_tensor, 0.9622504486493761);  add_tensor = None
        neg_default: "f32[128, 512, 28, 28]" = torch.ops.aten.neg.default(arg219_1)
        exp_default: "f32[128, 512, 28, 28]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[128, 512, 28, 28]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[128, 512, 28, 28]" = torch.ops.aten.reciprocal.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_1: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_2: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = None
        sub_tensor: "f32[128, 512, 28, 28]" = torch.ops.aten.sub.Tensor(1, mul_tensor_1);  mul_tensor_1 = None
        mul_tensor_3: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(arg219_1, sub_tensor);  arg219_1 = sub_tensor = None
        add_tensor_2: "f32[128, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_3, 1);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_2);  mul_tensor_2 = add_tensor_2 = None
        mul_tensor_5: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 0.2);  mul_tensor_4 = None
        mul_tensor_6: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_5, 2.0);  mul_tensor_5 = None
        mul_tensor_7: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_6, arg215_1);  mul_tensor_6 = arg215_1 = None
        sigmoid_default: "f32[128, 512, 1, 1]" = torch.ops.aten.sigmoid.default(arg218_1);  arg218_1 = None
        sum_dim_int_list: "f32[128, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [2, 3], True);  mul_tensor_7 = None
        sub_tensor_1: "f32[128, 512, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_8: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor_1);  sigmoid_default = sub_tensor_1 = None
        mul_tensor_9: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_8);  sum_dim_int_list = mul_tensor_8 = None
        sum_dim_int_list_1: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 2, 3]);  mul_tensor_9 = None
        return sum_dim_int_list_1

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
