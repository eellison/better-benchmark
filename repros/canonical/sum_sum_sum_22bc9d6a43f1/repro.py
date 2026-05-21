"""
Standalone repro captured via capture_hook.
Label: timm_timm_visformer_small_train_train_001
Pattern hash: 22bc9d6a43f1
Shape hash: 5fce0067
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 192, 28, 28], f32), T([128, 192, 28, 28], f32), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([192], f32), T([1, 192, 28, 28], f32), T([1, 192, 1, 1], f32), T([192], f32), T([192], f32), T([128, 192, 28, 28], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_162: "f32[128, 192, 28, 28]", arg96_1: "f32[128, 192, 28, 28]", arg97_1: "f32[1, 192, 1, 1]", arg98_1: "f32[1, 192, 1, 1]", arg4_1: "f32[192]", arg5_1: "f32[192]", arg6_1: "f32[1, 192, 28, 28]", arg262_1: "f32[1, 192, 1, 1]", arg99_1: "f32[192]", arg7_1: "f32[192]", add_65: "f32[128, 192, 28, 28]"):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_162, [0, 2, 3])
        sub_tensor: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(arg96_1, arg97_1)
        mul_tensor: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, arg98_1);  sub_tensor = None
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1)
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        add_tensor_1: "f32[128, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor, arg6_1);  add_tensor = arg6_1 = None
        sub_tensor_1: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_tensor_1, arg262_1);  add_tensor_1 = arg262_1 = None
        mul_tensor_2: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_162, sub_tensor_1)
        sum_dim_int_list_1: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 9.964923469387754e-06);  sum_dim_int_list = None
        unsqueeze_default_4: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_5: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        mul_tensor_4: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 9.964923469387754e-06)
        mul_tensor_5: "f32[192]" = torch.ops.aten.mul.Tensor(arg99_1, arg99_1)
        mul_tensor_6: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_8: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_7: "f32[192]" = torch.ops.aten.mul.Tensor(arg99_1, arg7_1);  arg7_1 = None
        unsqueeze_default_10: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_11: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_8: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_9);  sub_tensor_1 = unsqueeze_default_9 = None
        sub_tensor_2: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_162, mul_tensor_8);  getitem_162 = mul_tensor_8 = None
        sub_tensor_3: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_6);  sub_tensor_2 = unsqueeze_default_6 = None
        mul_tensor_9: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_12);  sub_tensor_3 = unsqueeze_default_12 = None
        mul_tensor_10: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg99_1);  sum_dim_int_list_1 = arg99_1 = None
        add_tensor_2: "f32[128, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_65, mul_tensor_9);  add_65 = mul_tensor_9 = None
        sum_dim_int_list_2: "f32[1, 192, 28, 28]" = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0], True)
        squeeze_dims: "f32[192]" = torch.ops.aten.squeeze.dims(arg97_1, [0, 2, 3]);  arg97_1 = None
        unsqueeze_default_13: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_14: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        sum_dim_int_list_3: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 2, 3])
        sub_tensor_4: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(arg96_1, unsqueeze_default_15);  arg96_1 = unsqueeze_default_15 = None
        mul_tensor_11: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_tensor_2, sub_tensor_4)
        sum_dim_int_list_4: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 2, 3]);  mul_tensor_11 = None
        mul_tensor_12: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 9.964923469387754e-06);  sum_dim_int_list_3 = None
        unsqueeze_default_16: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, 0);  mul_tensor_12 = None
        unsqueeze_default_17: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        mul_tensor_13: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_4, 9.964923469387754e-06)
        squeeze_dims_1: "f32[192]" = torch.ops.aten.squeeze.dims(arg98_1, [0, 2, 3]);  arg98_1 = None
        mul_tensor_14: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_15: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_13, mul_tensor_14);  mul_tensor_13 = mul_tensor_14 = None
        unsqueeze_default_19: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_15, 0);  mul_tensor_15 = None
        unsqueeze_default_20: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        mul_tensor_16: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg4_1);  arg4_1 = None
        unsqueeze_default_22: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_tensor_16, 0);  mul_tensor_16 = None
        unsqueeze_default_23: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        mul_tensor_17: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_21);  sub_tensor_4 = unsqueeze_default_21 = None
        sub_tensor_5: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_tensor_2, mul_tensor_17);  add_tensor_2 = mul_tensor_17 = None
        sub_tensor_6: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_tensor_5, unsqueeze_default_18);  sub_tensor_5 = unsqueeze_default_18 = None
        mul_tensor_18: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_6, unsqueeze_default_24);  sub_tensor_6 = unsqueeze_default_24 = None
        mul_tensor_19: "f32[192]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_4, squeeze_dims_1);  sum_dim_int_list_4 = squeeze_dims_1 = None
        sum_dim_int_list_5: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 2, 3]);  mul_tensor_18 = None
        return (mul_tensor_10, sum_dim_int_list_2, mul_tensor_19, sum_dim_int_list_5)



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
