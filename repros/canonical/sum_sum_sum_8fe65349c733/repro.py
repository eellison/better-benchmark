"""
Standalone repro captured via capture_hook.
Label: torchbench_densenet121_train_001
Pattern hash: 8fe65349c733
Shape hash: 5d33166a
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
_shapes_config = "(T([64, 256, 56, 56], f32), T([64, 224, 56, 56], f32), T([64, 192, 56, 56], f32), T([64, 160, 56, 56], f32), T([64, 128, 56, 56], f32), T([64, 96, 56, 56], f32), T([64, 64, 56, 56], f32), T([], f32), T([64, 64, 56, 56], f32), T([64, 64, 56, 56], f32), T([64], f32), T([64], f32), T([64, 64, 56, 56], i8, gen=Index(9)), T([64, 64, 112, 112], f32), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), S([4096, 3136]), S([4096, 3136]), S([64, 64, 112, 112]))"

class Repro(torch.nn.Module):
    def forward(self, mul_972: "f32[64, 256, 56, 56]", mul_990: "f32[64, 224, 56, 56]", mul_1008: "f32[64, 192, 56, 56]", mul_1026: "f32[64, 160, 56, 56]", mul_1044: "f32[64, 128, 56, 56]", mul_1062: "f32[64, 96, 56, 56]", arg250_1: "f32[64, 64, 56, 56]", full: "f32[]", getitem_354: "f32[64, 64, 56, 56]", arg730_1: "f32[64, 64, 56, 56]", arg249_1: "f32[64]", arg4_1: "f32[64]", arg248_1: "i8[64, 64, 56, 56]", arg245_1: "f32[64, 64, 112, 112]", arg246_1: "f32[1, 64, 1, 1]", arg247_1: "f32[1, 64, 1, 1]", arg2_1: "f32[64]", arg3_1: "f32[64]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        slice_tensor: "f32[64, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_972, 1, 0, 64);  mul_972 = None
        slice_tensor_1: "f32[64, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_990, 1, 0, 64);  mul_990 = None
        add_tensor: "f32[64, 64, 56, 56]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None
        slice_tensor_2: "f32[64, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1008, 1, 0, 64);  mul_1008 = None
        add_tensor_1: "f32[64, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_tensor, slice_tensor_2);  add_tensor = slice_tensor_2 = None
        slice_tensor_3: "f32[64, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1026, 1, 0, 64);  mul_1026 = None
        add_tensor_2: "f32[64, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_tensor_1, slice_tensor_3);  add_tensor_1 = slice_tensor_3 = None
        slice_tensor_4: "f32[64, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1044, 1, 0, 64);  mul_1044 = None
        add_tensor_3: "f32[64, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_tensor_2, slice_tensor_4);  add_tensor_2 = slice_tensor_4 = None
        slice_tensor_5: "f32[64, 64, 56, 56]" = torch.ops.aten.slice.Tensor(mul_1062, 1, 0, 64);  mul_1062 = None
        add_tensor_4: "f32[64, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_tensor_3, slice_tensor_5);  add_tensor_3 = slice_tensor_5 = None
        le_scalar: "b8[64, 64, 56, 56]" = torch.ops.aten.le.Scalar(arg250_1, 0);  arg250_1 = None
        where_self: "f32[64, 64, 56, 56]" = torch.ops.aten.where.self(le_scalar, full, getitem_354);  le_scalar = getitem_354 = None
        sum_dim_int_list: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        mul_tensor: "f32[64, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_self, arg730_1)
        sum_dim_int_list_1: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3]);  mul_tensor = None
        mul_tensor_1: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 4.982461734693877e-06);  sum_dim_int_list = None
        unsqueeze_default: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_2: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 4.982461734693877e-06)
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(arg249_1, arg249_1)
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_3: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_4: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(arg249_1, arg4_1);  arg4_1 = None
        unsqueeze_default_6: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_5, 0);  mul_tensor_5 = None
        unsqueeze_default_7: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_6: "f32[64, 64, 56, 56]" = torch.ops.aten.mul.Tensor(arg730_1, unsqueeze_default_5);  arg730_1 = unsqueeze_default_5 = None
        sub_tensor: "f32[64, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_6);  where_self = mul_tensor_6 = None
        sub_tensor_1: "f32[64, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_tensor, unsqueeze_default_2);  sub_tensor = unsqueeze_default_2 = None
        mul_tensor_7: "f32[64, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_8);  sub_tensor_1 = unsqueeze_default_8 = None
        mul_tensor_8: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, arg249_1);  sum_dim_int_list_1 = arg249_1 = None
        add_tensor_5: "f32[64, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_tensor_4, mul_tensor_7);  add_tensor_4 = mul_tensor_7 = None
        full_default: "f32[4096, 12544]" = torch.ops.aten.full.default([4096, 12544], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_default: "f32[4096, 3136]" = torch.ops.aten.view.default(add_tensor_5, _shape_param_0);  add_tensor_5 = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[64, 64, 56, 56]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(arg248_1, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]);  arg248_1 = None
        view_default_1: "i64[4096, 3136]" = torch.ops.aten.view.default(_low_memory_max_pool_offsets_to_indices_default, _shape_param_1);  _low_memory_max_pool_offsets_to_indices_default = _shape_param_1 = None
        scatter_add_default: "f32[4096, 12544]" = torch.ops.aten.scatter_add.default(full_default, 1, view_default_1, view_default);  full_default = view_default_1 = view_default = None
        view_default_2: "f32[64, 64, 112, 112]" = torch.ops.aten.view.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        sub_tensor_2: "f32[64, 64, 112, 112]" = torch.ops.aten.sub.Tensor(arg245_1, arg246_1)
        mul_tensor_9: "f32[64, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_2, arg247_1);  sub_tensor_2 = None
        unsqueeze_default_9: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1)
        unsqueeze_default_10: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, -1);  unsqueeze_default_9 = None
        mul_tensor_10: "f32[64, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor_9, unsqueeze_default_10);  mul_tensor_9 = unsqueeze_default_10 = None
        unsqueeze_default_11: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        unsqueeze_default_12: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, -1);  unsqueeze_default_11 = None
        add_tensor_6: "f32[64, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_10, unsqueeze_default_12);  mul_tensor_10 = unsqueeze_default_12 = None
        relu_default: "f32[64, 64, 112, 112]" = torch.ops.aten.relu.default(add_tensor_6);  add_tensor_6 = None
        le_scalar_1: "b8[64, 64, 112, 112]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self_1: "f32[64, 64, 112, 112]" = torch.ops.aten.where.self(le_scalar_1, full, view_default_2);  le_scalar_1 = full = view_default_2 = None
        squeeze_dims: "f32[64]" = torch.ops.aten.squeeze.dims(arg246_1, [0, 2, 3]);  arg246_1 = None
        unsqueeze_default_13: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_14: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        sum_dim_int_list_2: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
        sub_tensor_3: "f32[64, 64, 112, 112]" = torch.ops.aten.sub.Tensor(arg245_1, unsqueeze_default_15);  arg245_1 = unsqueeze_default_15 = None
        mul_tensor_11: "f32[64, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_self_1, sub_tensor_3)
        sum_dim_int_list_3: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 2, 3]);  mul_tensor_11 = None
        mul_tensor_12: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 1.2456154336734693e-06);  sum_dim_int_list_2 = None
        unsqueeze_default_16: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_12, 0);  mul_tensor_12 = None
        unsqueeze_default_17: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        mul_tensor_13: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 1.2456154336734693e-06)
        squeeze_dims_1: "f32[64]" = torch.ops.aten.squeeze.dims(arg247_1, [0, 2, 3]);  arg247_1 = None
        mul_tensor_14: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_15: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_13, mul_tensor_14);  mul_tensor_13 = mul_tensor_14 = None
        unsqueeze_default_19: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_15, 0);  mul_tensor_15 = None
        unsqueeze_default_20: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        mul_tensor_16: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg2_1);  arg2_1 = None
        unsqueeze_default_22: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_16, 0);  mul_tensor_16 = None
        unsqueeze_default_23: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        mul_tensor_17: "f32[64, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_21);  sub_tensor_3 = unsqueeze_default_21 = None
        sub_tensor_4: "f32[64, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_self_1, mul_tensor_17);  where_self_1 = mul_tensor_17 = None
        sub_tensor_5: "f32[64, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_18);  sub_tensor_4 = unsqueeze_default_18 = None
        mul_tensor_18: "f32[64, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_24);  sub_tensor_5 = unsqueeze_default_24 = None
        mul_tensor_19: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, squeeze_dims_1);  sum_dim_int_list_3 = squeeze_dims_1 = None
        return (mul_tensor_8, mul_tensor_18, mul_tensor_19)



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
