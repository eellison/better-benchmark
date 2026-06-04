"""
Standalone repro captured via capture_hook.
Label: timm_timm_visformer_small_train_train_001
Pattern hash: 2eb4fbbc04e1
Shape hash: 3f44ae3f
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 192, 28, 28], f32), T([128, 192, 28, 28], f32), T([192], f32), T([192], f32), T([128, 192, 28, 28], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_153: "f32[128, 192, 28, 28]", arg261_1: "f32[128, 192, 28, 28]", arg105_1: "f32[192]", arg11_1: "f32[192]", add_60: "f32[128, 192, 28, 28]"):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_153, [0, 2, 3])
        mul_tensor: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_153, arg261_1)
        sum_dim_int_list_1: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 9.964923469387754e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 9.964923469387754e-06)
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(arg105_1, arg105_1)
        mul_tensor_4: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[192]" = torch.ops.aten.mul.Tensor(arg105_1, arg11_1);  arg11_1 = None
        unsqueeze_default_6: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(arg261_1, unsqueeze_default_5);  arg261_1 = unsqueeze_default_5 = None
        sub_tensor: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_153, mul_tensor_6);  getitem_153 = mul_tensor_6 = None
        sub_tensor_1: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor, unsqueeze_default_2);  sub_tensor = unsqueeze_default_2 = None
        mul_tensor_7: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_8);  sub_tensor_1 = unsqueeze_default_8 = None
        mul_tensor_8: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg105_1);  sum_dim_int_list_1 = arg105_1 = None
        add_tensor: "f32[128, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_60, mul_tensor_7);  add_60 = mul_tensor_7 = None
        return (mul_tensor_8, add_tensor)

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
