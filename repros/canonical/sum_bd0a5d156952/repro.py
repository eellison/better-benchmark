"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train_001
Pattern hash: bd0a5d156952
Shape hash: 18c119c1
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 128, 56, 56], f32), T([128, 128, 56, 56], f32), T([128, 128, 56, 56], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_225: "f32[128, 128, 56, 56]", getitem_228: "f32[128, 128, 56, 56]", arg154_1: "f32[128, 128, 56, 56]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[128, 128, 56, 56]" = torch.ops.aten.add.Tensor(getitem_225, getitem_228);  getitem_225 = getitem_228 = None
        mul_tensor: "f32[128, 128, 56, 56]" = torch.ops.aten.mul.Tensor(add_tensor, 1.0);  add_tensor = None
        neg_default: "f32[128, 128, 56, 56]" = torch.ops.aten.neg.default(arg154_1)
        exp_default: "f32[128, 128, 56, 56]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[128, 128, 56, 56]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[128, 128, 56, 56]" = torch.ops.aten.reciprocal.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_1: "f32[128, 128, 56, 56]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_2: "f32[128, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = None
        sub_tensor: "f32[128, 128, 56, 56]" = torch.ops.aten.sub.Tensor(1, mul_tensor_1);  mul_tensor_1 = None
        mul_tensor_3: "f32[128, 128, 56, 56]" = torch.ops.aten.mul.Tensor(arg154_1, sub_tensor);  arg154_1 = sub_tensor = None
        add_tensor_2: "f32[128, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_3, 1);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_2);  mul_tensor_2 = add_tensor_2 = None
        sum_dim_int_list: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 2, 3]);  mul_tensor_4 = None
        return sum_dim_int_list

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
