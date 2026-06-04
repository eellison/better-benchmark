"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train_001
Pattern hash: c6666009132a
Shape hash: 37ea57af
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 512, 28, 28], f32), T([128, 512, 1, 1], f32), T([128, 512, 28, 28], f32), T([128, 512, 28, 28], f32), T([128, 512, 28, 28], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_186: "f32[128, 512, 28, 28]", arg199_1: "f32[128, 512, 1, 1]", arg196_1: "f32[128, 512, 28, 28]", arg181_1: "f32[128, 512, 28, 28]", mul_645: "f32[128, 512, 28, 28]"):
        # No stacktrace found for following nodes
        mul_tensor: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_186, 0.9805806756909201);  getitem_186 = None
        sigmoid_default: "f32[128, 512, 1, 1]" = torch.ops.aten.sigmoid.default(arg199_1);  arg199_1 = None
        mul_tensor_1: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(arg196_1, sigmoid_default)
        mul_tensor_2: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_1, 2.0);  mul_tensor_1 = None
        mul_tensor_3: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.2);  mul_tensor_2 = None
        add_tensor: "f32[128, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_3, arg181_1);  mul_tensor_3 = arg181_1 = None
        neg_default: "f32[128, 512, 28, 28]" = torch.ops.aten.neg.default(add_tensor)
        exp_default: "f32[128, 512, 28, 28]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[128, 512, 28, 28]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[128, 512, 28, 28]" = torch.ops.aten.reciprocal.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_4: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_5: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_4);  mul_tensor = None
        sub_tensor: "f32[128, 512, 28, 28]" = torch.ops.aten.sub.Tensor(1, mul_tensor_4);  mul_tensor_4 = None
        mul_tensor_6: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_tensor, sub_tensor);  add_tensor = sub_tensor = None
        add_tensor_2: "f32[128, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_6, 1);  mul_tensor_6 = None
        mul_tensor_7: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_5, add_tensor_2);  mul_tensor_5 = add_tensor_2 = None
        add_tensor_3: "f32[128, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_645, mul_tensor_7);  mul_645 = mul_tensor_7 = None
        mul_tensor_8: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(add_tensor_3, 0.2)
        mul_tensor_9: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_8, 2.0);  mul_tensor_8 = None
        mul_tensor_10: "f32[128, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_9, arg196_1);  mul_tensor_9 = arg196_1 = None
        sum_dim_int_list: "f32[128, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [2, 3], True);  mul_tensor_10 = None
        sub_tensor_1: "f32[128, 512, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_default)
        mul_tensor_11: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_default, sub_tensor_1);  sigmoid_default = sub_tensor_1 = None
        mul_tensor_12: "f32[128, 512, 1, 1]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, mul_tensor_11);  sum_dim_int_list = mul_tensor_11 = None
        sum_dim_int_list_1: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 2, 3]);  mul_tensor_12 = None
        sum_dim_int_list_2: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_tensor_3, [0, 2, 3]);  add_tensor_3 = None
        return (sum_dim_int_list_1, sum_dim_int_list_2)

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
