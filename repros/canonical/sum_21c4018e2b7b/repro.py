"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train_001
Pattern hash: 21c4018e2b7b
Shape hash: 8799e580
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 16, 112, 112], f32), T([128, 16, 112, 112], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg142_1: "f32[128, 16, 112, 112]", getitem_237: "f32[128, 16, 112, 112]"):
        # No stacktrace found for following nodes
        neg_default: "f32[128, 16, 112, 112]" = torch.ops.aten.neg.default(arg142_1)
        exp_default: "f32[128, 16, 112, 112]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[128, 16, 112, 112]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[128, 16, 112, 112]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[128, 16, 112, 112]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_1: "f32[128, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_237, mul_tensor);  getitem_237 = None
        sub_tensor: "f32[128, 16, 112, 112]" = torch.ops.aten.sub.Tensor(1, mul_tensor);  mul_tensor = None
        mul_tensor_2: "f32[128, 16, 112, 112]" = torch.ops.aten.mul.Tensor(arg142_1, sub_tensor);  arg142_1 = sub_tensor = None
        add_tensor_1: "f32[128, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_2, 1);  mul_tensor_2 = None
        mul_tensor_3: "f32[128, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_1);  mul_tensor_1 = add_tensor_1 = None
        sum_dim_int_list: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 2, 3]);  mul_tensor_3 = None
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
