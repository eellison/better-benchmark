"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train_001
Pattern hash: 97633bc3e047
Shape hash: dd64a33b
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 16, 96, 96], f32), T([128, 16, 96, 96], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_237: "f32[128, 16, 96, 96]", arg154_1: "f32[128, 16, 96, 96]"):
        # No stacktrace found for following nodes
        mul_tensor: "f32[128, 16, 96, 96]" = torch.ops.aten.mul.Tensor(getitem_237, 1.7015043497085571);  getitem_237 = None
        mul_tensor_1: "f32[128, 16, 96, 96]" = torch.ops.aten.mul.Tensor(arg154_1, 0.7071067811865476)
        erf_default: "f32[128, 16, 96, 96]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 16, 96, 96]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 16, 96, 96]" = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_3: "f32[128, 16, 96, 96]" = torch.ops.aten.mul.Tensor(arg154_1, arg154_1)
        mul_tensor_4: "f32[128, 16, 96, 96]" = torch.ops.aten.mul.Tensor(mul_tensor_3, -0.5);  mul_tensor_3 = None
        exp_default: "f32[128, 16, 96, 96]" = torch.ops.aten.exp.default(mul_tensor_4);  mul_tensor_4 = None
        mul_tensor_5: "f32[128, 16, 96, 96]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_6: "f32[128, 16, 96, 96]" = torch.ops.aten.mul.Tensor(arg154_1, mul_tensor_5);  arg154_1 = mul_tensor_5 = None
        add_tensor_1: "f32[128, 16, 96, 96]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_6);  mul_tensor_2 = mul_tensor_6 = None
        mul_tensor_7: "f32[128, 16, 96, 96]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None
        sum_dim_int_list: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 2, 3]);  mul_tensor_7 = None
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
