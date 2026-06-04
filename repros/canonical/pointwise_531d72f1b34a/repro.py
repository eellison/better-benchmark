"""
Standalone repro captured via capture_hook.
Label: torchbench_pyhpc_turbulent_kinetic_energy_infer_000
Pattern hash: 531d72f1b34a
Shape hash: 3b8fd76f
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([204, 204, 26, 3], f64), T([204, 204, 26], f64), T([204, 204], f64), T([26], f64), T([204, 204], i64), T([26], f64), T([204, 204, 26], f64), T([204, 204, 26], f64), T([204, 204, 26], f64), T([204], f64), T([204], f64), T([204, 204, 26], f64), T([204], f64), T([204], f64), T([204, 204, 26], f64), T([204], f64), T([204], f64), T([204, 204, 26, 3], f64), T([204, 204, 26, 3], f64), T([204, 204, 26, 3], f64), T([204, 204, 26, 3], f64))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f64[204, 204, 26, 3]", arg8_1: "f64[204, 204, 26]", arg9_1: "f64[204, 204]", arg6_1: "f64[26]", arg3_1: "i64[204, 204]", arg4_1: "f64[26]", arg5_1: "f64[204, 204, 26]", arg7_1: "f64[204, 204, 26]", arg15_1: "f64[204, 204, 26]", arg10_1: "f64[204]", arg11_1: "f64[204]", arg2_1: "f64[204, 204, 26]", arg16_1: "f64[204]", arg12_1: "f64[204]", arg13_1: "f64[204, 204, 26]", arg14_1: "f64[204]", arg17_1: "f64[204]", arg1_1: "f64[204, 204, 26, 3]", arg18_1: "f64[204, 204, 26, 3]", arg19_1: "f64[204, 204, 26, 3]", arg20_1: "f64[204, 204, 26, 3]"):
        # No stacktrace found for following nodes
        slice_tensor: "f64[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 2, -2)
        slice_tensor_1: "f64[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_tensor, 1, 2, -2);  slice_tensor = None
        select_int: "f64[200, 200, 26]" = torch.ops.aten.select.int(slice_tensor_1, 3, 0);  slice_tensor_1 = None
        slice_tensor_2: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 2, -2);  arg8_1 = None
        slice_tensor_3: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 1, 2, -2);  slice_tensor_2 = None
        mul_tensor: "f64[200, 200, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_3, 1);  slice_tensor_3 = None
        add_tensor: "f64[200, 200, 26]" = torch.ops.aten.add.Tensor(select_int, mul_tensor);  select_int = mul_tensor = None
        select_int_1: "f64[200, 200]" = torch.ops.aten.select.int(add_tensor, 2, -1)
        slice_tensor_4: "f64[200, 204]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2);  arg9_1 = None
        slice_tensor_5: "f64[200, 200]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 1, 2, -2);  slice_tensor_4 = None
        mul_tensor_1: "f64[200, 200]" = torch.ops.aten.mul.Tensor(slice_tensor_5, 1);  slice_tensor_5 = None
        select_int_2: "f64[]" = torch.ops.aten.select.int(arg6_1, 0, -1)
        mul_tensor_2: "f64[]" = torch.ops.aten.mul.Tensor(select_int_2, 0.5);  select_int_2 = None
        div_tensor: "f64[200, 200]" = torch.ops.aten.div.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        add_tensor_1: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_1, div_tensor);  select_int_1 = div_tensor = None
        select_scatter_default: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(add_tensor, add_tensor_1, 2, -1);  add_tensor = add_tensor_1 = None
        select_int_3: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default, 2, -1)
        slice_tensor_6: "i64[200, 204]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 2, -2);  arg3_1 = None
        slice_tensor_7: "i64[200, 200]" = torch.ops.aten.slice.Tensor(slice_tensor_6, 1, 2, -2);  slice_tensor_6 = None
        sub_tensor: "i64[200, 200]" = torch.ops.aten.sub.Tensor(slice_tensor_7, 1);  slice_tensor_7 = None
        ge_scalar: "b8[200, 200]" = torch.ops.aten.ge.Scalar(sub_tensor, 0)
        unsqueeze_default: "b8[200, 200, 1]" = torch.ops.aten.unsqueeze.default(ge_scalar, 2);  ge_scalar = None
        iota_default: "i64[26]" = torch.ops.prims.iota.default(26, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_1: "i64[1, 26]" = torch.ops.aten.unsqueeze.default(iota_default, 0);  iota_default = None
        unsqueeze_default_2: "i64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 1);  unsqueeze_default_1 = None
        unsqueeze_default_3: "i64[200, 200, 1]" = torch.ops.aten.unsqueeze.default(sub_tensor, 2)
        eq_tensor: "b8[200, 200, 26]" = torch.ops.aten.eq.Tensor(unsqueeze_default_2, unsqueeze_default_3);  unsqueeze_default_2 = unsqueeze_default_3 = None
        bitwise_and_tensor: "b8[200, 200, 26]" = torch.ops.aten.bitwise_and.Tensor(unsqueeze_default, eq_tensor);  eq_tensor = None
        full_default: "f64[200, 200, 26]" = torch.ops.aten.full.default([200, 200, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_8: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(full_default, 2, 0, -1)
        unsqueeze_default_4: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg4_1, 0);  arg4_1 = None
        unsqueeze_default_5: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 1);  unsqueeze_default_4 = None
        slice_tensor_9: "f64[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_default_5, 2, 1, 9223372036854775807);  unsqueeze_default_5 = None
        reciprocal_default: "f64[1, 1, 25]" = torch.ops.aten.reciprocal.default(slice_tensor_9);  slice_tensor_9 = None
        mul_tensor_3: "f64[1, 1, 25]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_4: "f64[1, 1, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 1.0);  mul_tensor_3 = None
        mul_tensor_5: "f64[1, 1, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 0.5);  mul_tensor_4 = None
        slice_tensor_10: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_tensor_11: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_10, 1, 2, -2);  slice_tensor_10 = None
        slice_tensor_12: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_11, 2, 0, -1);  slice_tensor_11 = None
        slice_tensor_13: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2);  arg5_1 = None
        slice_tensor_14: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_13, 1, 2, -2);  slice_tensor_13 = None
        slice_tensor_15: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_14, 2, 1, 9223372036854775807);  slice_tensor_14 = None
        add_tensor_2: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_tensor_12, slice_tensor_15);  slice_tensor_12 = slice_tensor_15 = None
        mul_tensor_6: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_5, add_tensor_2);  mul_tensor_5 = add_tensor_2 = None
        copy_default: "f64[200, 200, 25]" = torch.ops.aten.copy.default(slice_tensor_8, mul_tensor_6);  slice_tensor_8 = mul_tensor_6 = None
        slice_scatter_default: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(full_default, copy_default, 2, 0, -1);  full_default = copy_default = None
        unsqueeze_default_6: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_default_7: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 1);  unsqueeze_default_6 = None
        div_tensor_1: "f64[200, 200, 26]" = torch.ops.aten.div.Tensor(slice_scatter_default, unsqueeze_default_7);  unsqueeze_default_7 = None
        add_tensor_3: "f64[200, 200, 26]" = torch.ops.aten.add.Tensor(div_tensor_1, 1);  div_tensor_1 = None
        slice_tensor_16: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 2, -2)
        slice_tensor_17: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_16, 1, 2, -2);  slice_tensor_16 = None
        reciprocal_default_1: "f64[200, 200, 26]" = torch.ops.aten.reciprocal.default(slice_tensor_17);  slice_tensor_17 = None
        mul_tensor_7: "f64[200, 200, 26]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 0.7);  reciprocal_default_1 = None
        full_default_1: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_int_4: "f64[204, 204, 26]" = torch.ops.aten.select.int(arg0_1, 3, 0)
        maximum_default: "f64[204, 204, 26]" = torch.ops.aten.maximum.default(full_default_1, select_int_4);  full_default_1 = select_int_4 = None
        sqrt_default: "f64[204, 204, 26]" = torch.ops.aten.sqrt.default(maximum_default);  maximum_default = None
        slice_tensor_18: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(sqrt_default, 0, 2, -2)
        slice_tensor_19: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_18, 1, 2, -2);  slice_tensor_18 = None
        mul_tensor_8: "f64[200, 200, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_7, slice_tensor_19);  mul_tensor_7 = slice_tensor_19 = None
        add_tensor_4: "f64[200, 200, 26]" = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_8);  add_tensor_3 = mul_tensor_8 = None
        iota_default_1: "i64[26]" = torch.ops.prims.iota.default(26, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_8: "i64[1, 26]" = torch.ops.aten.unsqueeze.default(iota_default_1, 0);  iota_default_1 = None
        unsqueeze_default_9: "i64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 1);  unsqueeze_default_8 = None
        unsqueeze_default_10: "i64[200, 200, 1]" = torch.ops.aten.unsqueeze.default(sub_tensor, 2);  sub_tensor = None
        ge_tensor: "b8[200, 200, 26]" = torch.ops.aten.ge.Tensor(unsqueeze_default_9, unsqueeze_default_10);  unsqueeze_default_9 = unsqueeze_default_10 = None
        bitwise_and_tensor_1: "b8[200, 200, 26]" = torch.ops.aten.bitwise_and.Tensor(unsqueeze_default, ge_tensor);  unsqueeze_default = ge_tensor = None
        full_default_2: "f64[200, 200, 26]" = torch.ops.aten.full.default([200, 200, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_20: "f64[200, 200, 24]" = torch.ops.aten.slice.Tensor(full_default_2, 2, 1, -1)
        slice_tensor_21: "f64[200, 200, 24]" = torch.ops.aten.slice.Tensor(slice_scatter_default, 2, 1, -1)
        slice_tensor_22: "f64[200, 200, 24]" = torch.ops.aten.slice.Tensor(slice_scatter_default, 2, 0, -2)
        add_tensor_5: "f64[200, 200, 24]" = torch.ops.aten.add.Tensor(slice_tensor_21, slice_tensor_22);  slice_tensor_21 = slice_tensor_22 = None
        unsqueeze_default_11: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_default_12: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 1);  unsqueeze_default_11 = None
        slice_tensor_23: "f64[1, 1, 24]" = torch.ops.aten.slice.Tensor(unsqueeze_default_12, 2, 1, -1);  unsqueeze_default_12 = None
        div_tensor_2: "f64[200, 200, 24]" = torch.ops.aten.div.Tensor(add_tensor_5, slice_tensor_23);  add_tensor_5 = slice_tensor_23 = None
        add_tensor_6: "f64[200, 200, 24]" = torch.ops.aten.add.Tensor(div_tensor_2, 1);  div_tensor_2 = None
        slice_tensor_24: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(sqrt_default, 0, 2, -2)
        slice_tensor_25: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_24, 1, 2, -2);  slice_tensor_24 = None
        slice_tensor_26: "f64[200, 200, 24]" = torch.ops.aten.slice.Tensor(slice_tensor_25, 2, 1, -1);  slice_tensor_25 = None
        mul_tensor_9: "f64[200, 200, 24]" = torch.ops.aten.mul.Tensor(slice_tensor_26, 0.7);  slice_tensor_26 = None
        slice_tensor_27: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 2, -2)
        slice_tensor_28: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_27, 1, 2, -2);  slice_tensor_27 = None
        slice_tensor_29: "f64[200, 200, 24]" = torch.ops.aten.slice.Tensor(slice_tensor_28, 2, 1, -1);  slice_tensor_28 = None
        div_tensor_3: "f64[200, 200, 24]" = torch.ops.aten.div.Tensor(mul_tensor_9, slice_tensor_29);  mul_tensor_9 = slice_tensor_29 = None
        add_tensor_7: "f64[200, 200, 24]" = torch.ops.aten.add.Tensor(add_tensor_6, div_tensor_3);  add_tensor_6 = div_tensor_3 = None
        copy_default_1: "f64[200, 200, 24]" = torch.ops.aten.copy.default(slice_tensor_20, add_tensor_7);  slice_tensor_20 = add_tensor_7 = None
        slice_scatter_default_1: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(full_default_2, copy_default_1, 2, 1, -1);  full_default_2 = copy_default_1 = None
        select_int_5: "f64[200, 200]" = torch.ops.aten.select.int(slice_scatter_default_1, 2, -1)
        select_int_6: "f64[200, 200]" = torch.ops.aten.select.int(slice_scatter_default, 2, -2)
        select_int_7: "f64[]" = torch.ops.aten.select.int(arg6_1, 0, -1)
        mul_tensor_10: "f64[]" = torch.ops.aten.mul.Tensor(select_int_7, 0.5);  select_int_7 = None
        div_tensor_4: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_6, mul_tensor_10);  select_int_6 = mul_tensor_10 = None
        add_tensor_8: "f64[200, 200]" = torch.ops.aten.add.Tensor(div_tensor_4, 1);  div_tensor_4 = None
        slice_tensor_30: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 2, -2);  arg7_1 = None
        slice_tensor_31: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_30, 1, 2, -2);  slice_tensor_30 = None
        select_int_8: "f64[200, 200]" = torch.ops.aten.select.int(slice_tensor_31, 2, -1);  slice_tensor_31 = None
        reciprocal_default_2: "f64[200, 200]" = torch.ops.aten.reciprocal.default(select_int_8);  select_int_8 = None
        mul_tensor_11: "f64[200, 200]" = torch.ops.aten.mul.Tensor(reciprocal_default_2, 0.7);  reciprocal_default_2 = None
        slice_tensor_32: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(sqrt_default, 0, 2, -2);  sqrt_default = None
        slice_tensor_33: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_32, 1, 2, -2);  slice_tensor_32 = None
        select_int_9: "f64[200, 200]" = torch.ops.aten.select.int(slice_tensor_33, 2, -1);  slice_tensor_33 = None
        mul_tensor_12: "f64[200, 200]" = torch.ops.aten.mul.Tensor(mul_tensor_11, select_int_9);  mul_tensor_11 = select_int_9 = None
        add_tensor_9: "f64[200, 200]" = torch.ops.aten.add.Tensor(add_tensor_8, mul_tensor_12);  add_tensor_8 = mul_tensor_12 = None
        copy_default_2: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_5, add_tensor_9);  select_int_5 = add_tensor_9 = None
        select_scatter_default_1: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(slice_scatter_default_1, copy_default_2, 2, -1);  slice_scatter_default_1 = copy_default_2 = None
        full_default_3: "f64[]" = torch.ops.aten.full.default([], 1.0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f64[200, 200, 26]" = torch.ops.aten.where.self(bitwise_and_tensor_1, select_scatter_default_1, full_default_3);  select_scatter_default_1 = full_default_3 = None
        where_self_1: "f64[200, 200, 26]" = torch.ops.aten.where.self(bitwise_and_tensor, add_tensor_4, where_self);  add_tensor_4 = where_self = None
        select_int_10: "f64[200, 200]" = torch.ops.aten.select.int(where_self_1, 2, 1)
        full_default_4: "f64[200, 200, 26]" = torch.ops.aten.full.default([200, 200, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_34: "f64[200, 200, 24]" = torch.ops.aten.slice.Tensor(full_default_4, 2, 1, -1)
        slice_tensor_35: "f64[200, 200, 24]" = torch.ops.aten.slice.Tensor(slice_scatter_default, 2, 0, -2)
        neg_default: "f64[200, 200, 24]" = torch.ops.aten.neg.default(slice_tensor_35);  slice_tensor_35 = None
        unsqueeze_default_13: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_default_14: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 1);  unsqueeze_default_13 = None
        slice_tensor_36: "f64[1, 1, 24]" = torch.ops.aten.slice.Tensor(unsqueeze_default_14, 2, 1, -1);  unsqueeze_default_14 = None
        div_tensor_5: "f64[200, 200, 24]" = torch.ops.aten.div.Tensor(neg_default, slice_tensor_36);  neg_default = slice_tensor_36 = None
        copy_default_3: "f64[200, 200, 24]" = torch.ops.aten.copy.default(slice_tensor_34, div_tensor_5);  slice_tensor_34 = div_tensor_5 = None
        slice_scatter_default_2: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(full_default_4, copy_default_3, 2, 1, -1);  full_default_4 = copy_default_3 = None
        select_int_11: "f64[200, 200]" = torch.ops.aten.select.int(slice_scatter_default_2, 2, -1)
        select_int_12: "f64[200, 200]" = torch.ops.aten.select.int(slice_scatter_default, 2, -2)
        neg_default_1: "f64[200, 200]" = torch.ops.aten.neg.default(select_int_12);  select_int_12 = None
        select_int_13: "f64[]" = torch.ops.aten.select.int(arg6_1, 0, -1)
        mul_tensor_13: "f64[]" = torch.ops.aten.mul.Tensor(select_int_13, 0.5);  select_int_13 = None
        div_tensor_6: "f64[200, 200]" = torch.ops.aten.div.Tensor(neg_default_1, mul_tensor_13);  neg_default_1 = mul_tensor_13 = None
        copy_default_4: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_11, div_tensor_6);  select_int_11 = div_tensor_6 = None
        select_scatter_default_2: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(slice_scatter_default_2, copy_default_4, 2, -1);  slice_scatter_default_2 = copy_default_4 = None
        mul_tensor_14: "f64[200, 200, 26]" = torch.ops.aten.mul.Tensor(bitwise_and_tensor_1, select_scatter_default_2);  select_scatter_default_2 = None
        logical_not_default: "b8[200, 200, 26]" = torch.ops.aten.logical_not.default(bitwise_and_tensor);  bitwise_and_tensor = None
        mul_tensor_15: "f64[200, 200, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_14, logical_not_default);  mul_tensor_14 = logical_not_default = None
        select_int_14: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 1)
        select_int_15: "f64[200, 200]" = torch.ops.aten.select.int(where_self_1, 2, 0)
        div_tensor_7: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_14, select_int_15);  select_int_14 = select_int_15 = None
        neg_default_2: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_7)
        full_default_5: "f64[200, 200, 26]" = torch.ops.aten.full.default([200, 200, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_37: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(full_default_5, 2, 0, -1)
        slice_tensor_38: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_scatter_default, 2, 0, -1);  slice_scatter_default = None
        neg_default_3: "f64[200, 200, 25]" = torch.ops.aten.neg.default(slice_tensor_38);  slice_tensor_38 = None
        unsqueeze_default_15: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_default_16: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 1);  unsqueeze_default_15 = None
        slice_tensor_39: "f64[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_default_16, 2, 0, -1);  unsqueeze_default_16 = None
        div_tensor_8: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_default_3, slice_tensor_39);  neg_default_3 = slice_tensor_39 = None
        copy_default_5: "f64[200, 200, 25]" = torch.ops.aten.copy.default(slice_tensor_37, div_tensor_8);  slice_tensor_37 = div_tensor_8 = None
        slice_scatter_default_3: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(full_default_5, copy_default_5, 2, 0, -1);  full_default_5 = copy_default_5 = None
        mul_tensor_16: "f64[200, 200, 26]" = torch.ops.aten.mul.Tensor(bitwise_and_tensor_1, slice_scatter_default_3);  slice_scatter_default_3 = None
        select_int_16: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 0)
        mul_tensor_17: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_2, select_int_16);  neg_default_2 = select_int_16 = None
        add_tensor_10: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_10, mul_tensor_17);  select_int_10 = mul_tensor_17 = None
        select_scatter_default_3: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(where_self_1, add_tensor_10, 2, 1);  where_self_1 = add_tensor_10 = None
        select_int_17: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_3, 2, 1)
        select_int_18: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default, 2, -1)
        select_scatter_default_4: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default, select_int_18, 2, -1);  select_scatter_default = select_int_18 = None
        mul_tensor_18: "f64[200, 200, 26]" = torch.ops.aten.mul.Tensor(bitwise_and_tensor_1, select_scatter_default_4);  select_scatter_default_4 = None
        select_int_19: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_18, 2, 1)
        neg_default_4: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_7);  div_tensor_7 = None
        select_int_20: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_18, 2, 0)
        mul_tensor_19: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_4, select_int_20);  neg_default_4 = select_int_20 = None
        add_tensor_11: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_19, mul_tensor_19);  select_int_19 = mul_tensor_19 = None
        select_scatter_default_5: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(mul_tensor_18, add_tensor_11, 2, 1);  mul_tensor_18 = add_tensor_11 = None
        select_int_21: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_5, 2, 1)
        select_int_22: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_3, 2, 1)
        select_scatter_default_6: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_3, select_int_22, 2, 1);  select_scatter_default_3 = select_int_22 = None
        select_int_23: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_6, 2, 2)
        select_int_24: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 2)
        select_int_25: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_6, 2, 1)
        div_tensor_9: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_24, select_int_25);  select_int_24 = select_int_25 = None
        neg_default_5: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_9)
        select_int_26: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 1)
        mul_tensor_20: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_5, select_int_26);  neg_default_5 = select_int_26 = None
        add_tensor_12: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_23, mul_tensor_20);  select_int_23 = mul_tensor_20 = None
        select_scatter_default_7: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_6, add_tensor_12, 2, 2);  select_scatter_default_6 = add_tensor_12 = None
        select_int_27: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_7, 2, 2)
        select_int_28: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_5, 2, 1)
        select_scatter_default_8: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_5, select_int_28, 2, 1);  select_scatter_default_5 = select_int_28 = None
        select_int_29: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_8, 2, 2)
        neg_default_6: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_9);  div_tensor_9 = None
        select_int_30: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_8, 2, 1)
        mul_tensor_21: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_6, select_int_30);  neg_default_6 = select_int_30 = None
        add_tensor_13: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_29, mul_tensor_21);  select_int_29 = mul_tensor_21 = None
        select_scatter_default_9: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_8, add_tensor_13, 2, 2);  select_scatter_default_8 = add_tensor_13 = None
        select_int_31: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_9, 2, 2)
        select_int_32: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_7, 2, 2)
        select_scatter_default_10: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_7, select_int_32, 2, 2);  select_scatter_default_7 = select_int_32 = None
        select_int_33: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_10, 2, 3)
        select_int_34: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 3)
        select_int_35: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_10, 2, 2)
        div_tensor_10: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_34, select_int_35);  select_int_34 = select_int_35 = None
        neg_default_7: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_10)
        select_int_36: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 2)
        mul_tensor_22: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_7, select_int_36);  neg_default_7 = select_int_36 = None
        add_tensor_14: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_33, mul_tensor_22);  select_int_33 = mul_tensor_22 = None
        select_scatter_default_11: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_10, add_tensor_14, 2, 3);  select_scatter_default_10 = add_tensor_14 = None
        select_int_37: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_11, 2, 3)
        select_int_38: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_9, 2, 2)
        select_scatter_default_12: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_9, select_int_38, 2, 2);  select_scatter_default_9 = select_int_38 = None
        select_int_39: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_12, 2, 3)
        neg_default_8: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_10);  div_tensor_10 = None
        select_int_40: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_12, 2, 2)
        mul_tensor_23: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_8, select_int_40);  neg_default_8 = select_int_40 = None
        add_tensor_15: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_39, mul_tensor_23);  select_int_39 = mul_tensor_23 = None
        select_scatter_default_13: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_12, add_tensor_15, 2, 3);  select_scatter_default_12 = add_tensor_15 = None
        select_int_41: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_13, 2, 3)
        select_int_42: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_11, 2, 3)
        select_scatter_default_14: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_11, select_int_42, 2, 3);  select_scatter_default_11 = select_int_42 = None
        select_int_43: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_14, 2, 4)
        select_int_44: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 4)
        select_int_45: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_14, 2, 3)
        div_tensor_11: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_44, select_int_45);  select_int_44 = select_int_45 = None
        neg_default_9: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_11)
        select_int_46: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 3)
        mul_tensor_24: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_9, select_int_46);  neg_default_9 = select_int_46 = None
        add_tensor_16: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_43, mul_tensor_24);  select_int_43 = mul_tensor_24 = None
        select_scatter_default_15: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_14, add_tensor_16, 2, 4);  select_scatter_default_14 = add_tensor_16 = None
        select_int_47: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_15, 2, 4)
        select_int_48: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_13, 2, 3)
        select_scatter_default_16: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_13, select_int_48, 2, 3);  select_scatter_default_13 = select_int_48 = None
        select_int_49: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_16, 2, 4)
        neg_default_10: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_11);  div_tensor_11 = None
        select_int_50: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_16, 2, 3)
        mul_tensor_25: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_10, select_int_50);  neg_default_10 = select_int_50 = None
        add_tensor_17: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_49, mul_tensor_25);  select_int_49 = mul_tensor_25 = None
        select_scatter_default_17: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_16, add_tensor_17, 2, 4);  select_scatter_default_16 = add_tensor_17 = None
        select_int_51: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_17, 2, 4)
        select_int_52: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_15, 2, 4)
        select_scatter_default_18: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_15, select_int_52, 2, 4);  select_scatter_default_15 = select_int_52 = None
        select_int_53: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_18, 2, 5)
        select_int_54: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 5)
        select_int_55: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_18, 2, 4)
        div_tensor_12: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_54, select_int_55);  select_int_54 = select_int_55 = None
        neg_default_11: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_12)
        select_int_56: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 4)
        mul_tensor_26: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_11, select_int_56);  neg_default_11 = select_int_56 = None
        add_tensor_18: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_53, mul_tensor_26);  select_int_53 = mul_tensor_26 = None
        select_scatter_default_19: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_18, add_tensor_18, 2, 5);  select_scatter_default_18 = add_tensor_18 = None
        select_int_57: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_19, 2, 5)
        select_int_58: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_17, 2, 4)
        select_scatter_default_20: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_17, select_int_58, 2, 4);  select_scatter_default_17 = select_int_58 = None
        select_int_59: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_20, 2, 5)
        neg_default_12: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_12);  div_tensor_12 = None
        select_int_60: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_20, 2, 4)
        mul_tensor_27: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_12, select_int_60);  neg_default_12 = select_int_60 = None
        add_tensor_19: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_59, mul_tensor_27);  select_int_59 = mul_tensor_27 = None
        select_scatter_default_21: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_20, add_tensor_19, 2, 5);  select_scatter_default_20 = add_tensor_19 = None
        select_int_61: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_21, 2, 5)
        select_int_62: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_19, 2, 5)
        select_scatter_default_22: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_19, select_int_62, 2, 5);  select_scatter_default_19 = select_int_62 = None
        select_int_63: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_22, 2, 6)
        select_int_64: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 6)
        select_int_65: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_22, 2, 5)
        div_tensor_13: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_64, select_int_65);  select_int_64 = select_int_65 = None
        neg_default_13: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_13)
        select_int_66: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 5)
        mul_tensor_28: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_13, select_int_66);  neg_default_13 = select_int_66 = None
        add_tensor_20: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_63, mul_tensor_28);  select_int_63 = mul_tensor_28 = None
        select_scatter_default_23: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_22, add_tensor_20, 2, 6);  select_scatter_default_22 = add_tensor_20 = None
        select_int_67: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_23, 2, 6)
        select_int_68: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_21, 2, 5)
        select_scatter_default_24: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_21, select_int_68, 2, 5);  select_scatter_default_21 = select_int_68 = None
        select_int_69: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_24, 2, 6)
        neg_default_14: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_13);  div_tensor_13 = None
        select_int_70: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_24, 2, 5)
        mul_tensor_29: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_14, select_int_70);  neg_default_14 = select_int_70 = None
        add_tensor_21: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_69, mul_tensor_29);  select_int_69 = mul_tensor_29 = None
        select_scatter_default_25: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_24, add_tensor_21, 2, 6);  select_scatter_default_24 = add_tensor_21 = None
        select_int_71: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_25, 2, 6)
        select_int_72: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_23, 2, 6)
        select_scatter_default_26: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_23, select_int_72, 2, 6);  select_scatter_default_23 = select_int_72 = None
        select_int_73: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_26, 2, 7)
        select_int_74: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 7)
        select_int_75: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_26, 2, 6)
        div_tensor_14: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_74, select_int_75);  select_int_74 = select_int_75 = None
        neg_default_15: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_14)
        select_int_76: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 6)
        mul_tensor_30: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_15, select_int_76);  neg_default_15 = select_int_76 = None
        add_tensor_22: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_73, mul_tensor_30);  select_int_73 = mul_tensor_30 = None
        select_scatter_default_27: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_26, add_tensor_22, 2, 7);  select_scatter_default_26 = add_tensor_22 = None
        select_int_77: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_27, 2, 7)
        select_int_78: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_25, 2, 6)
        select_scatter_default_28: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_25, select_int_78, 2, 6);  select_scatter_default_25 = select_int_78 = None
        select_int_79: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_28, 2, 7)
        neg_default_16: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_14);  div_tensor_14 = None
        select_int_80: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_28, 2, 6)
        mul_tensor_31: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_16, select_int_80);  neg_default_16 = select_int_80 = None
        add_tensor_23: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_79, mul_tensor_31);  select_int_79 = mul_tensor_31 = None
        select_scatter_default_29: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_28, add_tensor_23, 2, 7);  select_scatter_default_28 = add_tensor_23 = None
        select_int_81: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_29, 2, 7)
        select_int_82: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_27, 2, 7)
        select_scatter_default_30: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_27, select_int_82, 2, 7);  select_scatter_default_27 = select_int_82 = None
        select_int_83: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_30, 2, 8)
        select_int_84: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 8)
        select_int_85: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_30, 2, 7)
        div_tensor_15: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_84, select_int_85);  select_int_84 = select_int_85 = None
        neg_default_17: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_15)
        select_int_86: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 7)
        mul_tensor_32: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_17, select_int_86);  neg_default_17 = select_int_86 = None
        add_tensor_24: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_83, mul_tensor_32);  select_int_83 = mul_tensor_32 = None
        select_scatter_default_31: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_30, add_tensor_24, 2, 8);  select_scatter_default_30 = add_tensor_24 = None
        select_int_87: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_31, 2, 8)
        select_int_88: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_29, 2, 7)
        select_scatter_default_32: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_29, select_int_88, 2, 7);  select_scatter_default_29 = select_int_88 = None
        select_int_89: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_32, 2, 8)
        neg_default_18: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_15);  div_tensor_15 = None
        select_int_90: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_32, 2, 7)
        mul_tensor_33: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_18, select_int_90);  neg_default_18 = select_int_90 = None
        add_tensor_25: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_89, mul_tensor_33);  select_int_89 = mul_tensor_33 = None
        select_scatter_default_33: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_32, add_tensor_25, 2, 8);  select_scatter_default_32 = add_tensor_25 = None
        select_int_91: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_33, 2, 8)
        select_int_92: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_31, 2, 8)
        select_scatter_default_34: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_31, select_int_92, 2, 8);  select_scatter_default_31 = select_int_92 = None
        select_int_93: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_34, 2, 9)
        select_int_94: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 9)
        select_int_95: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_34, 2, 8)
        div_tensor_16: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_94, select_int_95);  select_int_94 = select_int_95 = None
        neg_default_19: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_16)
        select_int_96: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 8)
        mul_tensor_34: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_19, select_int_96);  neg_default_19 = select_int_96 = None
        add_tensor_26: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_93, mul_tensor_34);  select_int_93 = mul_tensor_34 = None
        select_scatter_default_35: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_34, add_tensor_26, 2, 9);  select_scatter_default_34 = add_tensor_26 = None
        select_int_97: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_35, 2, 9)
        select_int_98: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_33, 2, 8)
        select_scatter_default_36: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_33, select_int_98, 2, 8);  select_scatter_default_33 = select_int_98 = None
        select_int_99: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_36, 2, 9)
        neg_default_20: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_16);  div_tensor_16 = None
        select_int_100: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_36, 2, 8)
        mul_tensor_35: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_20, select_int_100);  neg_default_20 = select_int_100 = None
        add_tensor_27: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_99, mul_tensor_35);  select_int_99 = mul_tensor_35 = None
        select_scatter_default_37: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_36, add_tensor_27, 2, 9);  select_scatter_default_36 = add_tensor_27 = None
        select_int_101: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_37, 2, 9)
        select_int_102: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_35, 2, 9)
        select_scatter_default_38: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_35, select_int_102, 2, 9);  select_scatter_default_35 = select_int_102 = None
        select_int_103: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_38, 2, 10)
        select_int_104: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 10)
        select_int_105: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_38, 2, 9)
        div_tensor_17: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_104, select_int_105);  select_int_104 = select_int_105 = None
        neg_default_21: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_17)
        select_int_106: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 9)
        mul_tensor_36: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_21, select_int_106);  neg_default_21 = select_int_106 = None
        add_tensor_28: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_103, mul_tensor_36);  select_int_103 = mul_tensor_36 = None
        select_scatter_default_39: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_38, add_tensor_28, 2, 10);  select_scatter_default_38 = add_tensor_28 = None
        select_int_107: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_39, 2, 10)
        select_int_108: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_37, 2, 9)
        select_scatter_default_40: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_37, select_int_108, 2, 9);  select_scatter_default_37 = select_int_108 = None
        select_int_109: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_40, 2, 10)
        neg_default_22: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_17);  div_tensor_17 = None
        select_int_110: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_40, 2, 9)
        mul_tensor_37: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_22, select_int_110);  neg_default_22 = select_int_110 = None
        add_tensor_29: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_109, mul_tensor_37);  select_int_109 = mul_tensor_37 = None
        select_scatter_default_41: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_40, add_tensor_29, 2, 10);  select_scatter_default_40 = add_tensor_29 = None
        select_int_111: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_41, 2, 10)
        select_int_112: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_39, 2, 10)
        select_scatter_default_42: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_39, select_int_112, 2, 10);  select_scatter_default_39 = select_int_112 = None
        select_int_113: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_42, 2, 11)
        select_int_114: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 11)
        select_int_115: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_42, 2, 10)
        div_tensor_18: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_114, select_int_115);  select_int_114 = select_int_115 = None
        neg_default_23: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_18)
        select_int_116: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 10)
        mul_tensor_38: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_23, select_int_116);  neg_default_23 = select_int_116 = None
        add_tensor_30: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_113, mul_tensor_38);  select_int_113 = mul_tensor_38 = None
        select_scatter_default_43: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_42, add_tensor_30, 2, 11);  select_scatter_default_42 = add_tensor_30 = None
        select_int_117: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_43, 2, 11)
        select_int_118: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_41, 2, 10)
        select_scatter_default_44: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_41, select_int_118, 2, 10);  select_scatter_default_41 = select_int_118 = None
        select_int_119: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_44, 2, 11)
        neg_default_24: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_18);  div_tensor_18 = None
        select_int_120: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_44, 2, 10)
        mul_tensor_39: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_24, select_int_120);  neg_default_24 = select_int_120 = None
        add_tensor_31: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_119, mul_tensor_39);  select_int_119 = mul_tensor_39 = None
        select_scatter_default_45: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_44, add_tensor_31, 2, 11);  select_scatter_default_44 = add_tensor_31 = None
        select_int_121: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_45, 2, 11)
        select_int_122: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_43, 2, 11)
        select_scatter_default_46: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_43, select_int_122, 2, 11);  select_scatter_default_43 = select_int_122 = None
        select_int_123: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_46, 2, 12)
        select_int_124: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 12)
        select_int_125: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_46, 2, 11)
        div_tensor_19: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_124, select_int_125);  select_int_124 = select_int_125 = None
        neg_default_25: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_19)
        select_int_126: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 11)
        mul_tensor_40: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_25, select_int_126);  neg_default_25 = select_int_126 = None
        add_tensor_32: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_123, mul_tensor_40);  select_int_123 = mul_tensor_40 = None
        select_scatter_default_47: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_46, add_tensor_32, 2, 12);  select_scatter_default_46 = add_tensor_32 = None
        select_int_127: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_47, 2, 12)
        select_int_128: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_45, 2, 11)
        select_scatter_default_48: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_45, select_int_128, 2, 11);  select_scatter_default_45 = select_int_128 = None
        select_int_129: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_48, 2, 12)
        neg_default_26: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_19);  div_tensor_19 = None
        select_int_130: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_48, 2, 11)
        mul_tensor_41: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_26, select_int_130);  neg_default_26 = select_int_130 = None
        add_tensor_33: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_129, mul_tensor_41);  select_int_129 = mul_tensor_41 = None
        select_scatter_default_49: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_48, add_tensor_33, 2, 12);  select_scatter_default_48 = add_tensor_33 = None
        select_int_131: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_49, 2, 12)
        select_int_132: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_47, 2, 12)
        select_scatter_default_50: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_47, select_int_132, 2, 12);  select_scatter_default_47 = select_int_132 = None
        select_int_133: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_50, 2, 13)
        select_int_134: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 13)
        select_int_135: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_50, 2, 12)
        div_tensor_20: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_134, select_int_135);  select_int_134 = select_int_135 = None
        neg_default_27: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_20)
        select_int_136: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 12)
        mul_tensor_42: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_27, select_int_136);  neg_default_27 = select_int_136 = None
        add_tensor_34: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_133, mul_tensor_42);  select_int_133 = mul_tensor_42 = None
        select_scatter_default_51: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_50, add_tensor_34, 2, 13);  select_scatter_default_50 = add_tensor_34 = None
        select_int_137: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_51, 2, 13)
        select_int_138: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_49, 2, 12)
        select_scatter_default_52: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_49, select_int_138, 2, 12);  select_scatter_default_49 = select_int_138 = None
        select_int_139: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_52, 2, 13)
        neg_default_28: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_20);  div_tensor_20 = None
        select_int_140: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_52, 2, 12)
        mul_tensor_43: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_28, select_int_140);  neg_default_28 = select_int_140 = None
        add_tensor_35: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_139, mul_tensor_43);  select_int_139 = mul_tensor_43 = None
        select_scatter_default_53: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_52, add_tensor_35, 2, 13);  select_scatter_default_52 = add_tensor_35 = None
        select_int_141: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_53, 2, 13)
        select_int_142: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_51, 2, 13)
        select_scatter_default_54: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_51, select_int_142, 2, 13);  select_scatter_default_51 = select_int_142 = None
        select_int_143: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_54, 2, 14)
        select_int_144: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 14)
        select_int_145: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_54, 2, 13)
        div_tensor_21: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_144, select_int_145);  select_int_144 = select_int_145 = None
        neg_default_29: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_21)
        select_int_146: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 13)
        mul_tensor_44: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_29, select_int_146);  neg_default_29 = select_int_146 = None
        add_tensor_36: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_143, mul_tensor_44);  select_int_143 = mul_tensor_44 = None
        select_scatter_default_55: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_54, add_tensor_36, 2, 14);  select_scatter_default_54 = add_tensor_36 = None
        select_int_147: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_55, 2, 14)
        select_int_148: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_53, 2, 13)
        select_scatter_default_56: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_53, select_int_148, 2, 13);  select_scatter_default_53 = select_int_148 = None
        select_int_149: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_56, 2, 14)
        neg_default_30: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_21);  div_tensor_21 = None
        select_int_150: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_56, 2, 13)
        mul_tensor_45: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_30, select_int_150);  neg_default_30 = select_int_150 = None
        add_tensor_37: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_149, mul_tensor_45);  select_int_149 = mul_tensor_45 = None
        select_scatter_default_57: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_56, add_tensor_37, 2, 14);  select_scatter_default_56 = add_tensor_37 = None
        select_int_151: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_57, 2, 14)
        select_int_152: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_55, 2, 14)
        select_scatter_default_58: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_55, select_int_152, 2, 14);  select_scatter_default_55 = select_int_152 = None
        select_int_153: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_58, 2, 15)
        select_int_154: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 15)
        select_int_155: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_58, 2, 14)
        div_tensor_22: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_154, select_int_155);  select_int_154 = select_int_155 = None
        neg_default_31: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_22)
        select_int_156: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 14)
        mul_tensor_46: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_31, select_int_156);  neg_default_31 = select_int_156 = None
        add_tensor_38: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_153, mul_tensor_46);  select_int_153 = mul_tensor_46 = None
        select_scatter_default_59: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_58, add_tensor_38, 2, 15);  select_scatter_default_58 = add_tensor_38 = None
        select_int_157: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_59, 2, 15)
        select_int_158: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_57, 2, 14)
        select_scatter_default_60: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_57, select_int_158, 2, 14);  select_scatter_default_57 = select_int_158 = None
        select_int_159: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_60, 2, 15)
        neg_default_32: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_22);  div_tensor_22 = None
        select_int_160: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_60, 2, 14)
        mul_tensor_47: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_32, select_int_160);  neg_default_32 = select_int_160 = None
        add_tensor_39: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_159, mul_tensor_47);  select_int_159 = mul_tensor_47 = None
        select_scatter_default_61: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_60, add_tensor_39, 2, 15);  select_scatter_default_60 = add_tensor_39 = None
        select_int_161: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_61, 2, 15)
        select_int_162: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_59, 2, 15)
        select_scatter_default_62: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_59, select_int_162, 2, 15);  select_scatter_default_59 = select_int_162 = None
        select_int_163: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_62, 2, 16)
        select_int_164: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 16)
        select_int_165: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_62, 2, 15)
        div_tensor_23: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_164, select_int_165);  select_int_164 = select_int_165 = None
        neg_default_33: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_23)
        select_int_166: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 15)
        mul_tensor_48: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_33, select_int_166);  neg_default_33 = select_int_166 = None
        add_tensor_40: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_163, mul_tensor_48);  select_int_163 = mul_tensor_48 = None
        select_scatter_default_63: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_62, add_tensor_40, 2, 16);  select_scatter_default_62 = add_tensor_40 = None
        select_int_167: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_63, 2, 16)
        select_int_168: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_61, 2, 15)
        select_scatter_default_64: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_61, select_int_168, 2, 15);  select_scatter_default_61 = select_int_168 = None
        select_int_169: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_64, 2, 16)
        neg_default_34: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_23);  div_tensor_23 = None
        select_int_170: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_64, 2, 15)
        mul_tensor_49: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_34, select_int_170);  neg_default_34 = select_int_170 = None
        add_tensor_41: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_169, mul_tensor_49);  select_int_169 = mul_tensor_49 = None
        select_scatter_default_65: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_64, add_tensor_41, 2, 16);  select_scatter_default_64 = add_tensor_41 = None
        select_int_171: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_65, 2, 16)
        select_int_172: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_63, 2, 16)
        select_scatter_default_66: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_63, select_int_172, 2, 16);  select_scatter_default_63 = select_int_172 = None
        select_int_173: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_66, 2, 17)
        select_int_174: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 17)
        select_int_175: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_66, 2, 16)
        div_tensor_24: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_174, select_int_175);  select_int_174 = select_int_175 = None
        neg_default_35: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_24)
        select_int_176: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 16)
        mul_tensor_50: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_35, select_int_176);  neg_default_35 = select_int_176 = None
        add_tensor_42: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_173, mul_tensor_50);  select_int_173 = mul_tensor_50 = None
        select_scatter_default_67: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_66, add_tensor_42, 2, 17);  select_scatter_default_66 = add_tensor_42 = None
        select_int_177: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_67, 2, 17)
        select_int_178: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_65, 2, 16)
        select_scatter_default_68: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_65, select_int_178, 2, 16);  select_scatter_default_65 = select_int_178 = None
        select_int_179: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_68, 2, 17)
        neg_default_36: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_24);  div_tensor_24 = None
        select_int_180: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_68, 2, 16)
        mul_tensor_51: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_36, select_int_180);  neg_default_36 = select_int_180 = None
        add_tensor_43: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_179, mul_tensor_51);  select_int_179 = mul_tensor_51 = None
        select_scatter_default_69: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_68, add_tensor_43, 2, 17);  select_scatter_default_68 = add_tensor_43 = None
        select_int_181: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_69, 2, 17)
        select_int_182: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_67, 2, 17)
        select_scatter_default_70: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_67, select_int_182, 2, 17);  select_scatter_default_67 = select_int_182 = None
        select_int_183: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_70, 2, 18)
        select_int_184: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 18)
        select_int_185: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_70, 2, 17)
        div_tensor_25: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_184, select_int_185);  select_int_184 = select_int_185 = None
        neg_default_37: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_25)
        select_int_186: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 17)
        mul_tensor_52: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_37, select_int_186);  neg_default_37 = select_int_186 = None
        add_tensor_44: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_183, mul_tensor_52);  select_int_183 = mul_tensor_52 = None
        select_scatter_default_71: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_70, add_tensor_44, 2, 18);  select_scatter_default_70 = add_tensor_44 = None
        select_int_187: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_71, 2, 18)
        select_int_188: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_69, 2, 17)
        select_scatter_default_72: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_69, select_int_188, 2, 17);  select_scatter_default_69 = select_int_188 = None
        select_int_189: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_72, 2, 18)
        neg_default_38: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_25);  div_tensor_25 = None
        select_int_190: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_72, 2, 17)
        mul_tensor_53: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_38, select_int_190);  neg_default_38 = select_int_190 = None
        add_tensor_45: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_189, mul_tensor_53);  select_int_189 = mul_tensor_53 = None
        select_scatter_default_73: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_72, add_tensor_45, 2, 18);  select_scatter_default_72 = add_tensor_45 = None
        select_int_191: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_73, 2, 18)
        select_int_192: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_71, 2, 18)
        select_scatter_default_74: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_71, select_int_192, 2, 18);  select_scatter_default_71 = select_int_192 = None
        select_int_193: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_74, 2, 19)
        select_int_194: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 19)
        select_int_195: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_74, 2, 18)
        div_tensor_26: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_194, select_int_195);  select_int_194 = select_int_195 = None
        neg_default_39: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_26)
        select_int_196: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 18)
        mul_tensor_54: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_39, select_int_196);  neg_default_39 = select_int_196 = None
        add_tensor_46: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_193, mul_tensor_54);  select_int_193 = mul_tensor_54 = None
        select_scatter_default_75: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_74, add_tensor_46, 2, 19);  select_scatter_default_74 = add_tensor_46 = None
        select_int_197: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_75, 2, 19)
        select_int_198: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_73, 2, 18)
        select_scatter_default_76: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_73, select_int_198, 2, 18);  select_scatter_default_73 = select_int_198 = None
        select_int_199: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_76, 2, 19)
        neg_default_40: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_26);  div_tensor_26 = None
        select_int_200: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_76, 2, 18)
        mul_tensor_55: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_40, select_int_200);  neg_default_40 = select_int_200 = None
        add_tensor_47: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_199, mul_tensor_55);  select_int_199 = mul_tensor_55 = None
        select_scatter_default_77: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_76, add_tensor_47, 2, 19);  select_scatter_default_76 = add_tensor_47 = None
        select_int_201: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_77, 2, 19)
        select_int_202: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_75, 2, 19)
        select_scatter_default_78: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_75, select_int_202, 2, 19);  select_scatter_default_75 = select_int_202 = None
        select_int_203: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_78, 2, 20)
        select_int_204: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 20)
        select_int_205: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_78, 2, 19)
        div_tensor_27: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_204, select_int_205);  select_int_204 = select_int_205 = None
        neg_default_41: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_27)
        select_int_206: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 19)
        mul_tensor_56: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_41, select_int_206);  neg_default_41 = select_int_206 = None
        add_tensor_48: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_203, mul_tensor_56);  select_int_203 = mul_tensor_56 = None
        select_scatter_default_79: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_78, add_tensor_48, 2, 20);  select_scatter_default_78 = add_tensor_48 = None
        select_int_207: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_79, 2, 20)
        select_int_208: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_77, 2, 19)
        select_scatter_default_80: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_77, select_int_208, 2, 19);  select_scatter_default_77 = select_int_208 = None
        select_int_209: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_80, 2, 20)
        neg_default_42: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_27);  div_tensor_27 = None
        select_int_210: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_80, 2, 19)
        mul_tensor_57: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_42, select_int_210);  neg_default_42 = select_int_210 = None
        add_tensor_49: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_209, mul_tensor_57);  select_int_209 = mul_tensor_57 = None
        select_scatter_default_81: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_80, add_tensor_49, 2, 20);  select_scatter_default_80 = add_tensor_49 = None
        select_int_211: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_81, 2, 20)
        select_int_212: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_79, 2, 20)
        select_scatter_default_82: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_79, select_int_212, 2, 20);  select_scatter_default_79 = select_int_212 = None
        select_int_213: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_82, 2, 21)
        select_int_214: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 21)
        select_int_215: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_82, 2, 20)
        div_tensor_28: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_214, select_int_215);  select_int_214 = select_int_215 = None
        neg_default_43: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_28)
        select_int_216: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 20)
        mul_tensor_58: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_43, select_int_216);  neg_default_43 = select_int_216 = None
        add_tensor_50: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_213, mul_tensor_58);  select_int_213 = mul_tensor_58 = None
        select_scatter_default_83: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_82, add_tensor_50, 2, 21);  select_scatter_default_82 = add_tensor_50 = None
        select_int_217: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_83, 2, 21)
        select_int_218: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_81, 2, 20)
        select_scatter_default_84: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_81, select_int_218, 2, 20);  select_scatter_default_81 = select_int_218 = None
        select_int_219: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_84, 2, 21)
        neg_default_44: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_28);  div_tensor_28 = None
        select_int_220: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_84, 2, 20)
        mul_tensor_59: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_44, select_int_220);  neg_default_44 = select_int_220 = None
        add_tensor_51: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_219, mul_tensor_59);  select_int_219 = mul_tensor_59 = None
        select_scatter_default_85: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_84, add_tensor_51, 2, 21);  select_scatter_default_84 = add_tensor_51 = None
        select_int_221: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_85, 2, 21)
        select_int_222: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_83, 2, 21)
        select_scatter_default_86: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_83, select_int_222, 2, 21);  select_scatter_default_83 = select_int_222 = None
        select_int_223: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_86, 2, 22)
        select_int_224: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 22)
        select_int_225: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_86, 2, 21)
        div_tensor_29: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_224, select_int_225);  select_int_224 = select_int_225 = None
        neg_default_45: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_29)
        select_int_226: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 21)
        mul_tensor_60: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_45, select_int_226);  neg_default_45 = select_int_226 = None
        add_tensor_52: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_223, mul_tensor_60);  select_int_223 = mul_tensor_60 = None
        select_scatter_default_87: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_86, add_tensor_52, 2, 22);  select_scatter_default_86 = add_tensor_52 = None
        select_int_227: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_87, 2, 22)
        select_int_228: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_85, 2, 21)
        select_scatter_default_88: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_85, select_int_228, 2, 21);  select_scatter_default_85 = select_int_228 = None
        select_int_229: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_88, 2, 22)
        neg_default_46: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_29);  div_tensor_29 = None
        select_int_230: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_88, 2, 21)
        mul_tensor_61: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_46, select_int_230);  neg_default_46 = select_int_230 = None
        add_tensor_53: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_229, mul_tensor_61);  select_int_229 = mul_tensor_61 = None
        select_scatter_default_89: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_88, add_tensor_53, 2, 22);  select_scatter_default_88 = add_tensor_53 = None
        select_int_231: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_89, 2, 22)
        select_int_232: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_87, 2, 22)
        select_scatter_default_90: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_87, select_int_232, 2, 22);  select_scatter_default_87 = select_int_232 = None
        select_int_233: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_90, 2, 23)
        select_int_234: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 23)
        select_int_235: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_90, 2, 22)
        div_tensor_30: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_234, select_int_235);  select_int_234 = select_int_235 = None
        neg_default_47: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_30)
        select_int_236: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 22)
        mul_tensor_62: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_47, select_int_236);  neg_default_47 = select_int_236 = None
        add_tensor_54: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_233, mul_tensor_62);  select_int_233 = mul_tensor_62 = None
        select_scatter_default_91: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_90, add_tensor_54, 2, 23);  select_scatter_default_90 = add_tensor_54 = None
        select_int_237: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_91, 2, 23)
        select_int_238: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_89, 2, 22)
        select_scatter_default_92: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_89, select_int_238, 2, 22);  select_scatter_default_89 = select_int_238 = None
        select_int_239: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_92, 2, 23)
        neg_default_48: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_30);  div_tensor_30 = None
        select_int_240: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_92, 2, 22)
        mul_tensor_63: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_48, select_int_240);  neg_default_48 = select_int_240 = None
        add_tensor_55: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_239, mul_tensor_63);  select_int_239 = mul_tensor_63 = None
        select_scatter_default_93: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_92, add_tensor_55, 2, 23);  select_scatter_default_92 = add_tensor_55 = None
        select_int_241: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_93, 2, 23)
        select_int_242: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_91, 2, 23)
        select_scatter_default_94: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_91, select_int_242, 2, 23);  select_scatter_default_91 = select_int_242 = None
        select_int_243: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_94, 2, 24)
        select_int_244: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 24)
        select_int_245: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_94, 2, 23)
        div_tensor_31: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_244, select_int_245);  select_int_244 = select_int_245 = None
        neg_default_49: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_31)
        select_int_246: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 23)
        mul_tensor_64: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_49, select_int_246);  neg_default_49 = select_int_246 = None
        add_tensor_56: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_243, mul_tensor_64);  select_int_243 = mul_tensor_64 = None
        select_scatter_default_95: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_94, add_tensor_56, 2, 24);  select_scatter_default_94 = add_tensor_56 = None
        select_int_247: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_95, 2, 24)
        select_int_248: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_93, 2, 23)
        select_scatter_default_96: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_93, select_int_248, 2, 23);  select_scatter_default_93 = select_int_248 = None
        select_int_249: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_96, 2, 24)
        neg_default_50: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_31);  div_tensor_31 = None
        select_int_250: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_96, 2, 23)
        mul_tensor_65: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_50, select_int_250);  neg_default_50 = select_int_250 = None
        add_tensor_57: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_249, mul_tensor_65);  select_int_249 = mul_tensor_65 = None
        select_scatter_default_97: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_96, add_tensor_57, 2, 24);  select_scatter_default_96 = add_tensor_57 = None
        select_int_251: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_97, 2, 24)
        select_int_252: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_95, 2, 24)
        select_scatter_default_98: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_95, select_int_252, 2, 24);  select_scatter_default_95 = select_int_252 = None
        select_int_253: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_98, 2, 25)
        select_int_254: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_15, 2, 25);  mul_tensor_15 = None
        select_int_255: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_98, 2, 24)
        div_tensor_32: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_254, select_int_255);  select_int_254 = select_int_255 = None
        neg_default_51: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_32)
        select_int_256: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 24)
        mul_tensor_66: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_51, select_int_256);  neg_default_51 = select_int_256 = None
        add_tensor_58: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_253, mul_tensor_66);  select_int_253 = mul_tensor_66 = None
        select_scatter_default_99: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_98, add_tensor_58, 2, 25);  select_scatter_default_98 = add_tensor_58 = None
        select_int_257: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_99, 2, 25)
        select_int_258: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_97, 2, 24)
        select_scatter_default_100: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_97, select_int_258, 2, 24);  select_scatter_default_97 = select_int_258 = None
        select_int_259: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_100, 2, 25)
        neg_default_52: "f64[200, 200]" = torch.ops.aten.neg.default(div_tensor_32);  div_tensor_32 = None
        select_int_260: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_100, 2, 24)
        mul_tensor_67: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_52, select_int_260);  neg_default_52 = select_int_260 = None
        add_tensor_59: "f64[200, 200]" = torch.ops.aten.add.Tensor(select_int_259, mul_tensor_67);  select_int_259 = mul_tensor_67 = None
        select_scatter_default_101: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_100, add_tensor_59, 2, 25);  select_scatter_default_100 = add_tensor_59 = None
        select_int_261: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_101, 2, 25)
        full_default_6: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_40: "f64[203, 204, 26]" = torch.ops.aten.slice.Tensor(full_default_6, 0, 0, -1)
        slice_tensor_41: "f64[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 2, -2)
        slice_tensor_42: "f64[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_41, 1, 2, -2)
        slice_tensor_43: "f64[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 2, -2)
        slice_tensor_44: "f64[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_43, 1, 2, -2);  slice_tensor_43 = None
        select_int_262: "f64[200, 200, 26]" = torch.ops.aten.select.int(slice_tensor_44, 3, 1);  slice_tensor_44 = None
        empty_memory_format: "f64[200, 200, 26]" = torch.ops.aten.empty.memory_format([200, 200, 26], dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_int_263: "f64[200, 200]" = torch.ops.aten.select.int(empty_memory_format, 2, -1)
        select_int_264: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_101, 2, 25)
        select_scatter_default_102: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_101, select_int_264, 2, 25);  select_scatter_default_101 = select_int_264 = None
        select_int_265: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, -1)
        select_int_266: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_99, 2, 25)
        select_scatter_default_103: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_99, select_int_266, 2, 25);  select_scatter_default_99 = select_int_266 = None
        select_int_267: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, -1)
        div_tensor_33: "f64[200, 200]" = torch.ops.aten.div.Tensor(select_int_265, select_int_267);  select_int_265 = select_int_267 = None
        copy_default_6: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_263, div_tensor_33);  select_int_263 = div_tensor_33 = None
        select_scatter_default_104: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(empty_memory_format, copy_default_6, 2, -1);  empty_memory_format = copy_default_6 = None
        select_int_268: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_104, 2, 24)
        select_int_269: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 24)
        select_int_270: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 24)
        select_int_271: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_104, 2, 25)
        mul_tensor_68: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_270, select_int_271);  select_int_270 = select_int_271 = None
        sub_tensor_1: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_269, mul_tensor_68);  select_int_269 = mul_tensor_68 = None
        select_int_272: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 24)
        div_tensor_34: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_1, select_int_272);  sub_tensor_1 = select_int_272 = None
        copy_default_7: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_268, div_tensor_34);  select_int_268 = div_tensor_34 = None
        select_scatter_default_105: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_104, copy_default_7, 2, 24);  select_scatter_default_104 = copy_default_7 = None
        select_int_273: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_105, 2, 23)
        select_int_274: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 23)
        select_int_275: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 23)
        select_int_276: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_105, 2, 24)
        mul_tensor_69: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_275, select_int_276);  select_int_275 = select_int_276 = None
        sub_tensor_2: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_274, mul_tensor_69);  select_int_274 = mul_tensor_69 = None
        select_int_277: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 23)
        div_tensor_35: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_2, select_int_277);  sub_tensor_2 = select_int_277 = None
        copy_default_8: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_273, div_tensor_35);  select_int_273 = div_tensor_35 = None
        select_scatter_default_106: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_105, copy_default_8, 2, 23);  select_scatter_default_105 = copy_default_8 = None
        select_int_278: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_106, 2, 22)
        select_int_279: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 22)
        select_int_280: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 22)
        select_int_281: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_106, 2, 23)
        mul_tensor_70: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_280, select_int_281);  select_int_280 = select_int_281 = None
        sub_tensor_3: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_279, mul_tensor_70);  select_int_279 = mul_tensor_70 = None
        select_int_282: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 22)
        div_tensor_36: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_3, select_int_282);  sub_tensor_3 = select_int_282 = None
        copy_default_9: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_278, div_tensor_36);  select_int_278 = div_tensor_36 = None
        select_scatter_default_107: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_106, copy_default_9, 2, 22);  select_scatter_default_106 = copy_default_9 = None
        select_int_283: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_107, 2, 21)
        select_int_284: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 21)
        select_int_285: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 21)
        select_int_286: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_107, 2, 22)
        mul_tensor_71: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_285, select_int_286);  select_int_285 = select_int_286 = None
        sub_tensor_4: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_284, mul_tensor_71);  select_int_284 = mul_tensor_71 = None
        select_int_287: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 21)
        div_tensor_37: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_4, select_int_287);  sub_tensor_4 = select_int_287 = None
        copy_default_10: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_283, div_tensor_37);  select_int_283 = div_tensor_37 = None
        select_scatter_default_108: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_107, copy_default_10, 2, 21);  select_scatter_default_107 = copy_default_10 = None
        select_int_288: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_108, 2, 20)
        select_int_289: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 20)
        select_int_290: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 20)
        select_int_291: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_108, 2, 21)
        mul_tensor_72: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_290, select_int_291);  select_int_290 = select_int_291 = None
        sub_tensor_5: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_289, mul_tensor_72);  select_int_289 = mul_tensor_72 = None
        select_int_292: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 20)
        div_tensor_38: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_5, select_int_292);  sub_tensor_5 = select_int_292 = None
        copy_default_11: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_288, div_tensor_38);  select_int_288 = div_tensor_38 = None
        select_scatter_default_109: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_108, copy_default_11, 2, 20);  select_scatter_default_108 = copy_default_11 = None
        select_int_293: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_109, 2, 19)
        select_int_294: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 19)
        select_int_295: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 19)
        select_int_296: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_109, 2, 20)
        mul_tensor_73: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_295, select_int_296);  select_int_295 = select_int_296 = None
        sub_tensor_6: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_294, mul_tensor_73);  select_int_294 = mul_tensor_73 = None
        select_int_297: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 19)
        div_tensor_39: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_6, select_int_297);  sub_tensor_6 = select_int_297 = None
        copy_default_12: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_293, div_tensor_39);  select_int_293 = div_tensor_39 = None
        select_scatter_default_110: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_109, copy_default_12, 2, 19);  select_scatter_default_109 = copy_default_12 = None
        select_int_298: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_110, 2, 18)
        select_int_299: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 18)
        select_int_300: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 18)
        select_int_301: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_110, 2, 19)
        mul_tensor_74: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_300, select_int_301);  select_int_300 = select_int_301 = None
        sub_tensor_7: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_299, mul_tensor_74);  select_int_299 = mul_tensor_74 = None
        select_int_302: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 18)
        div_tensor_40: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_7, select_int_302);  sub_tensor_7 = select_int_302 = None
        copy_default_13: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_298, div_tensor_40);  select_int_298 = div_tensor_40 = None
        select_scatter_default_111: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_110, copy_default_13, 2, 18);  select_scatter_default_110 = copy_default_13 = None
        select_int_303: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_111, 2, 17)
        select_int_304: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 17)
        select_int_305: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 17)
        select_int_306: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_111, 2, 18)
        mul_tensor_75: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_305, select_int_306);  select_int_305 = select_int_306 = None
        sub_tensor_8: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_304, mul_tensor_75);  select_int_304 = mul_tensor_75 = None
        select_int_307: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 17)
        div_tensor_41: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_8, select_int_307);  sub_tensor_8 = select_int_307 = None
        copy_default_14: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_303, div_tensor_41);  select_int_303 = div_tensor_41 = None
        select_scatter_default_112: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_111, copy_default_14, 2, 17);  select_scatter_default_111 = copy_default_14 = None
        select_int_308: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_112, 2, 16)
        select_int_309: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 16)
        select_int_310: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 16)
        select_int_311: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_112, 2, 17)
        mul_tensor_76: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_310, select_int_311);  select_int_310 = select_int_311 = None
        sub_tensor_9: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_309, mul_tensor_76);  select_int_309 = mul_tensor_76 = None
        select_int_312: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 16)
        div_tensor_42: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_9, select_int_312);  sub_tensor_9 = select_int_312 = None
        copy_default_15: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_308, div_tensor_42);  select_int_308 = div_tensor_42 = None
        select_scatter_default_113: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_112, copy_default_15, 2, 16);  select_scatter_default_112 = copy_default_15 = None
        select_int_313: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_113, 2, 15)
        select_int_314: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 15)
        select_int_315: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 15)
        select_int_316: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_113, 2, 16)
        mul_tensor_77: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_315, select_int_316);  select_int_315 = select_int_316 = None
        sub_tensor_10: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_314, mul_tensor_77);  select_int_314 = mul_tensor_77 = None
        select_int_317: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 15)
        div_tensor_43: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_10, select_int_317);  sub_tensor_10 = select_int_317 = None
        copy_default_16: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_313, div_tensor_43);  select_int_313 = div_tensor_43 = None
        select_scatter_default_114: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_113, copy_default_16, 2, 15);  select_scatter_default_113 = copy_default_16 = None
        select_int_318: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_114, 2, 14)
        select_int_319: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 14)
        select_int_320: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 14)
        select_int_321: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_114, 2, 15)
        mul_tensor_78: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_320, select_int_321);  select_int_320 = select_int_321 = None
        sub_tensor_11: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_319, mul_tensor_78);  select_int_319 = mul_tensor_78 = None
        select_int_322: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 14)
        div_tensor_44: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_11, select_int_322);  sub_tensor_11 = select_int_322 = None
        copy_default_17: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_318, div_tensor_44);  select_int_318 = div_tensor_44 = None
        select_scatter_default_115: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_114, copy_default_17, 2, 14);  select_scatter_default_114 = copy_default_17 = None
        select_int_323: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_115, 2, 13)
        select_int_324: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 13)
        select_int_325: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 13)
        select_int_326: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_115, 2, 14)
        mul_tensor_79: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_325, select_int_326);  select_int_325 = select_int_326 = None
        sub_tensor_12: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_324, mul_tensor_79);  select_int_324 = mul_tensor_79 = None
        select_int_327: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 13)
        div_tensor_45: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_12, select_int_327);  sub_tensor_12 = select_int_327 = None
        copy_default_18: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_323, div_tensor_45);  select_int_323 = div_tensor_45 = None
        select_scatter_default_116: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_115, copy_default_18, 2, 13);  select_scatter_default_115 = copy_default_18 = None
        select_int_328: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_116, 2, 12)
        select_int_329: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 12)
        select_int_330: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 12)
        select_int_331: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_116, 2, 13)
        mul_tensor_80: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_330, select_int_331);  select_int_330 = select_int_331 = None
        sub_tensor_13: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_329, mul_tensor_80);  select_int_329 = mul_tensor_80 = None
        select_int_332: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 12)
        div_tensor_46: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_13, select_int_332);  sub_tensor_13 = select_int_332 = None
        copy_default_19: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_328, div_tensor_46);  select_int_328 = div_tensor_46 = None
        select_scatter_default_117: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_116, copy_default_19, 2, 12);  select_scatter_default_116 = copy_default_19 = None
        select_int_333: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_117, 2, 11)
        select_int_334: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 11)
        select_int_335: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 11)
        select_int_336: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_117, 2, 12)
        mul_tensor_81: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_335, select_int_336);  select_int_335 = select_int_336 = None
        sub_tensor_14: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_334, mul_tensor_81);  select_int_334 = mul_tensor_81 = None
        select_int_337: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 11)
        div_tensor_47: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_14, select_int_337);  sub_tensor_14 = select_int_337 = None
        copy_default_20: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_333, div_tensor_47);  select_int_333 = div_tensor_47 = None
        select_scatter_default_118: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_117, copy_default_20, 2, 11);  select_scatter_default_117 = copy_default_20 = None
        select_int_338: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_118, 2, 10)
        select_int_339: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 10)
        select_int_340: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 10)
        select_int_341: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_118, 2, 11)
        mul_tensor_82: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_340, select_int_341);  select_int_340 = select_int_341 = None
        sub_tensor_15: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_339, mul_tensor_82);  select_int_339 = mul_tensor_82 = None
        select_int_342: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 10)
        div_tensor_48: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_15, select_int_342);  sub_tensor_15 = select_int_342 = None
        copy_default_21: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_338, div_tensor_48);  select_int_338 = div_tensor_48 = None
        select_scatter_default_119: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_118, copy_default_21, 2, 10);  select_scatter_default_118 = copy_default_21 = None
        select_int_343: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_119, 2, 9)
        select_int_344: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 9)
        select_int_345: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 9)
        select_int_346: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_119, 2, 10)
        mul_tensor_83: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_345, select_int_346);  select_int_345 = select_int_346 = None
        sub_tensor_16: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_344, mul_tensor_83);  select_int_344 = mul_tensor_83 = None
        select_int_347: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 9)
        div_tensor_49: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_16, select_int_347);  sub_tensor_16 = select_int_347 = None
        copy_default_22: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_343, div_tensor_49);  select_int_343 = div_tensor_49 = None
        select_scatter_default_120: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_119, copy_default_22, 2, 9);  select_scatter_default_119 = copy_default_22 = None
        select_int_348: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_120, 2, 8)
        select_int_349: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 8)
        select_int_350: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 8)
        select_int_351: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_120, 2, 9)
        mul_tensor_84: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_350, select_int_351);  select_int_350 = select_int_351 = None
        sub_tensor_17: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_349, mul_tensor_84);  select_int_349 = mul_tensor_84 = None
        select_int_352: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 8)
        div_tensor_50: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_17, select_int_352);  sub_tensor_17 = select_int_352 = None
        copy_default_23: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_348, div_tensor_50);  select_int_348 = div_tensor_50 = None
        select_scatter_default_121: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_120, copy_default_23, 2, 8);  select_scatter_default_120 = copy_default_23 = None
        select_int_353: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_121, 2, 7)
        select_int_354: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 7)
        select_int_355: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 7)
        select_int_356: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_121, 2, 8)
        mul_tensor_85: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_355, select_int_356);  select_int_355 = select_int_356 = None
        sub_tensor_18: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_354, mul_tensor_85);  select_int_354 = mul_tensor_85 = None
        select_int_357: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 7)
        div_tensor_51: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_18, select_int_357);  sub_tensor_18 = select_int_357 = None
        copy_default_24: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_353, div_tensor_51);  select_int_353 = div_tensor_51 = None
        select_scatter_default_122: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_121, copy_default_24, 2, 7);  select_scatter_default_121 = copy_default_24 = None
        select_int_358: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_122, 2, 6)
        select_int_359: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 6)
        select_int_360: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 6)
        select_int_361: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_122, 2, 7)
        mul_tensor_86: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_360, select_int_361);  select_int_360 = select_int_361 = None
        sub_tensor_19: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_359, mul_tensor_86);  select_int_359 = mul_tensor_86 = None
        select_int_362: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 6)
        div_tensor_52: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_19, select_int_362);  sub_tensor_19 = select_int_362 = None
        copy_default_25: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_358, div_tensor_52);  select_int_358 = div_tensor_52 = None
        select_scatter_default_123: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_122, copy_default_25, 2, 6);  select_scatter_default_122 = copy_default_25 = None
        select_int_363: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_123, 2, 5)
        select_int_364: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 5)
        select_int_365: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 5)
        select_int_366: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_123, 2, 6)
        mul_tensor_87: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_365, select_int_366);  select_int_365 = select_int_366 = None
        sub_tensor_20: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_364, mul_tensor_87);  select_int_364 = mul_tensor_87 = None
        select_int_367: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 5)
        div_tensor_53: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_20, select_int_367);  sub_tensor_20 = select_int_367 = None
        copy_default_26: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_363, div_tensor_53);  select_int_363 = div_tensor_53 = None
        select_scatter_default_124: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_123, copy_default_26, 2, 5);  select_scatter_default_123 = copy_default_26 = None
        select_int_368: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_124, 2, 4)
        select_int_369: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 4)
        select_int_370: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 4)
        select_int_371: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_124, 2, 5)
        mul_tensor_88: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_370, select_int_371);  select_int_370 = select_int_371 = None
        sub_tensor_21: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_369, mul_tensor_88);  select_int_369 = mul_tensor_88 = None
        select_int_372: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 4)
        div_tensor_54: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_21, select_int_372);  sub_tensor_21 = select_int_372 = None
        copy_default_27: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_368, div_tensor_54);  select_int_368 = div_tensor_54 = None
        select_scatter_default_125: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_124, copy_default_27, 2, 4);  select_scatter_default_124 = copy_default_27 = None
        select_int_373: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_125, 2, 3)
        select_int_374: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 3)
        select_int_375: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 3)
        select_int_376: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_125, 2, 4)
        mul_tensor_89: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_375, select_int_376);  select_int_375 = select_int_376 = None
        sub_tensor_22: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_374, mul_tensor_89);  select_int_374 = mul_tensor_89 = None
        select_int_377: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 3)
        div_tensor_55: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_22, select_int_377);  sub_tensor_22 = select_int_377 = None
        copy_default_28: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_373, div_tensor_55);  select_int_373 = div_tensor_55 = None
        select_scatter_default_126: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_125, copy_default_28, 2, 3);  select_scatter_default_125 = copy_default_28 = None
        select_int_378: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_126, 2, 2)
        select_int_379: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 2)
        select_int_380: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 2)
        select_int_381: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_126, 2, 3)
        mul_tensor_90: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_380, select_int_381);  select_int_380 = select_int_381 = None
        sub_tensor_23: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_379, mul_tensor_90);  select_int_379 = mul_tensor_90 = None
        select_int_382: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 2)
        div_tensor_56: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_23, select_int_382);  sub_tensor_23 = select_int_382 = None
        copy_default_29: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_378, div_tensor_56);  select_int_378 = div_tensor_56 = None
        select_scatter_default_127: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_126, copy_default_29, 2, 2);  select_scatter_default_126 = copy_default_29 = None
        select_int_383: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_127, 2, 1)
        select_int_384: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 1)
        select_int_385: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 1)
        select_int_386: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_127, 2, 2)
        mul_tensor_91: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_385, select_int_386);  select_int_385 = select_int_386 = None
        sub_tensor_24: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_384, mul_tensor_91);  select_int_384 = mul_tensor_91 = None
        select_int_387: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 1)
        div_tensor_57: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_24, select_int_387);  sub_tensor_24 = select_int_387 = None
        copy_default_30: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_383, div_tensor_57);  select_int_383 = div_tensor_57 = None
        select_scatter_default_128: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_127, copy_default_30, 2, 1);  select_scatter_default_127 = copy_default_30 = None
        select_int_388: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_128, 2, 0)
        select_int_389: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_102, 2, 0);  select_scatter_default_102 = None
        select_int_390: "f64[200, 200]" = torch.ops.aten.select.int(mul_tensor_16, 2, 0);  mul_tensor_16 = None
        select_int_391: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_128, 2, 1)
        mul_tensor_92: "f64[200, 200]" = torch.ops.aten.mul.Tensor(select_int_390, select_int_391);  select_int_390 = select_int_391 = None
        sub_tensor_25: "f64[200, 200]" = torch.ops.aten.sub.Tensor(select_int_389, mul_tensor_92);  select_int_389 = mul_tensor_92 = None
        select_int_392: "f64[200, 200]" = torch.ops.aten.select.int(select_scatter_default_103, 2, 0);  select_scatter_default_103 = None
        div_tensor_58: "f64[200, 200]" = torch.ops.aten.div.Tensor(sub_tensor_25, select_int_392);  sub_tensor_25 = select_int_392 = None
        copy_default_31: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_388, div_tensor_58);  select_int_388 = div_tensor_58 = None
        select_scatter_default_129: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_default_128, copy_default_31, 2, 0);  select_scatter_default_128 = copy_default_31 = None
        slice_tensor_45: "f64[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 2, -2)
        slice_tensor_46: "f64[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_45, 1, 2, -2);  slice_tensor_45 = None
        select_int_393: "f64[200, 200, 26]" = torch.ops.aten.select.int(slice_tensor_46, 3, 1);  slice_tensor_46 = None
        where_self_2: "f64[200, 200, 26]" = torch.ops.aten.where.self(bitwise_and_tensor_1, select_scatter_default_129, select_int_393);  bitwise_and_tensor_1 = select_scatter_default_129 = select_int_393 = None
        copy_default_32: "f64[200, 200, 26]" = torch.ops.aten.copy.default(select_int_262, where_self_2);  select_int_262 = where_self_2 = None
        select_scatter_default_130: "f64[200, 200, 26, 3]" = torch.ops.aten.select_scatter.default(slice_tensor_42, copy_default_32, 3, 1);  slice_tensor_42 = copy_default_32 = None
        slice_scatter_default_4: "f64[200, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_tensor_41, select_scatter_default_130, 1, 2, -2);  slice_tensor_41 = select_scatter_default_130 = None
        slice_scatter_default_5: "f64[204, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(arg0_1, slice_scatter_default_4, 0, 2, -2);  arg0_1 = slice_scatter_default_4 = None
        slice_tensor_47: "f64[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_default_5, 0, 2, -2)
        slice_tensor_48: "f64[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_47, 1, 2, -2)
        select_int_394: "f64[200, 200, 3]" = torch.ops.aten.select.int(slice_tensor_48, 2, -1)
        slice_tensor_49: "f64[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_default_5, 0, 2, -2)
        slice_tensor_50: "f64[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_49, 1, 2, -2);  slice_tensor_49 = None
        select_int_395: "f64[200, 200, 3]" = torch.ops.aten.select.int(slice_tensor_50, 2, -1);  slice_tensor_50 = None
        select_int_396: "f64[200, 200]" = torch.ops.aten.select.int(select_int_395, 2, 1);  select_int_395 = None
        full_default_7: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_51: "f64[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_default_5, 0, 2, -2)
        slice_tensor_52: "f64[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_51, 1, 2, -2);  slice_tensor_51 = None
        select_int_397: "f64[200, 200, 3]" = torch.ops.aten.select.int(slice_tensor_52, 2, -1);  slice_tensor_52 = None
        select_int_398: "f64[200, 200]" = torch.ops.aten.select.int(select_int_397, 2, 1);  select_int_397 = None
        maximum_default_1: "f64[200, 200]" = torch.ops.aten.maximum.default(full_default_7, select_int_398);  full_default_7 = select_int_398 = None
        copy_default_33: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_396, maximum_default_1);  select_int_396 = maximum_default_1 = None
        select_scatter_default_131: "f64[200, 200, 3]" = torch.ops.aten.select_scatter.default(select_int_394, copy_default_33, 2, 1);  select_int_394 = copy_default_33 = None
        select_scatter_default_132: "f64[200, 200, 26, 3]" = torch.ops.aten.select_scatter.default(slice_tensor_48, select_scatter_default_131, 2, -1);  slice_tensor_48 = select_scatter_default_131 = None
        slice_scatter_default_6: "f64[200, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_tensor_47, select_scatter_default_132, 1, 2, -2);  slice_tensor_47 = select_scatter_default_132 = None
        slice_scatter_default_7: "f64[204, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_5, slice_scatter_default_6, 0, 2, -2);  slice_scatter_default_6 = None
        slice_tensor_53: "f64[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, 2, -2)
        slice_tensor_54: "f64[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_53, 1, 2, -2)
        slice_tensor_55: "f64[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, 2, -2)
        slice_tensor_56: "f64[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_55, 1, 2, -2);  slice_tensor_55 = None
        select_int_399: "f64[200, 200, 26]" = torch.ops.aten.select.int(slice_tensor_56, 3, 1);  slice_tensor_56 = None
        slice_tensor_57: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg15_1, 0, 2, -2)
        slice_tensor_58: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_57, 1, 2, -2);  slice_tensor_57 = None
        mul_tensor_93: "f64[200, 200, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_58, 1);  slice_tensor_58 = None
        slice_tensor_59: "f64[203, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, 1, 9223372036854775807)
        select_int_400: "f64[203, 204, 26]" = torch.ops.aten.select.int(slice_tensor_59, 3, 0);  slice_tensor_59 = None
        slice_tensor_60: "f64[203, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, 0, -1)
        select_int_401: "f64[203, 204, 26]" = torch.ops.aten.select.int(slice_tensor_60, 3, 0);  slice_tensor_60 = None
        sub_tensor_26: "f64[203, 204, 26]" = torch.ops.aten.sub.Tensor(select_int_400, select_int_401);  select_int_400 = select_int_401 = None
        mul_tensor_94: "f64[203, 204, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_26, 2000.0);  sub_tensor_26 = None
        unsqueeze_default_17: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg10_1, 0)
        unsqueeze_default_18: "f64[1, 204, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 2);  unsqueeze_default_17 = None
        slice_tensor_61: "f64[203]" = torch.ops.aten.slice.Tensor(arg11_1, 0, 0, -1);  arg11_1 = None
        unsqueeze_default_19: "f64[203, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_61, 1);  slice_tensor_61 = None
        unsqueeze_default_20: "f64[203, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        mul_tensor_95: "f64[203, 204, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_default_18, unsqueeze_default_20);  unsqueeze_default_18 = unsqueeze_default_20 = None
        div_tensor_59: "f64[203, 204, 26]" = torch.ops.aten.div.Tensor(mul_tensor_94, mul_tensor_95);  mul_tensor_94 = mul_tensor_95 = None
        slice_tensor_62: "f64[203, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 0, -1);  arg2_1 = None
        mul_tensor_96: "f64[203, 204, 26]" = torch.ops.aten.mul.Tensor(div_tensor_59, slice_tensor_62);  div_tensor_59 = slice_tensor_62 = None
        slice_scatter_default_8: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_default_6, mul_tensor_96, 0, 0, -1);  full_default_6 = mul_tensor_96 = None
        select_int_402: "f64[204, 26]" = torch.ops.aten.select.int(slice_scatter_default_8, 0, -1)
        full_default_8: "f64[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default_34: "f64[204, 26]" = torch.ops.aten.copy.default(select_int_402, full_default_8);  select_int_402 = full_default_8 = None
        select_scatter_default_133: "f64[204, 204, 26]" = torch.ops.aten.select_scatter.default(slice_scatter_default_8, copy_default_34, 0, -1);  slice_scatter_default_8 = copy_default_34 = None
        slice_tensor_63: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_scatter_default_133, 0, 2, -2)
        slice_tensor_64: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_63, 1, 2, -2);  slice_tensor_63 = None
        slice_tensor_65: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_scatter_default_133, 0, 1, -3)
        slice_tensor_66: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_65, 1, 2, -2);  slice_tensor_65 = None
        sub_tensor_27: "f64[200, 200, 26]" = torch.ops.aten.sub.Tensor(slice_tensor_64, slice_tensor_66);  slice_tensor_64 = slice_tensor_66 = None
        unsqueeze_default_21: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg10_1, 0)
        slice_tensor_67: "f64[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_default_21, 1, 2, -2);  unsqueeze_default_21 = None
        unsqueeze_default_22: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_67, 2);  slice_tensor_67 = None
        slice_tensor_68: "f64[200]" = torch.ops.aten.slice.Tensor(arg16_1, 0, 2, -2)
        unsqueeze_default_23: "f64[200, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_68, 1);  slice_tensor_68 = None
        unsqueeze_default_24: "f64[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 2);  unsqueeze_default_23 = None
        mul_tensor_97: "f64[200, 200, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_default_22, unsqueeze_default_24);  unsqueeze_default_22 = unsqueeze_default_24 = None
        div_tensor_60: "f64[200, 200, 26]" = torch.ops.aten.div.Tensor(sub_tensor_27, mul_tensor_97);  sub_tensor_27 = mul_tensor_97 = None
        full_default_9: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_69: "f64[204, 203, 26]" = torch.ops.aten.slice.Tensor(full_default_9, 1, 0, -1)
        slice_tensor_70: "f64[204, 203, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 1, 1, 9223372036854775807)
        select_int_403: "f64[204, 203, 26]" = torch.ops.aten.select.int(slice_tensor_70, 3, 0);  slice_tensor_70 = None
        slice_tensor_71: "f64[204, 203, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 1, 0, -1)
        select_int_404: "f64[204, 203, 26]" = torch.ops.aten.select.int(slice_tensor_71, 3, 0);  slice_tensor_71 = None
        sub_tensor_28: "f64[204, 203, 26]" = torch.ops.aten.sub.Tensor(select_int_403, select_int_404);  select_int_403 = select_int_404 = None
        mul_tensor_98: "f64[204, 203, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_28, 2000.0);  sub_tensor_28 = None
        unsqueeze_default_25: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg12_1, 0);  arg12_1 = None
        slice_tensor_72: "f64[1, 203]" = torch.ops.aten.slice.Tensor(unsqueeze_default_25, 1, 0, -1);  unsqueeze_default_25 = None
        unsqueeze_default_26: "f64[1, 203, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_72, 2);  slice_tensor_72 = None
        div_tensor_61: "f64[204, 203, 26]" = torch.ops.aten.div.Tensor(mul_tensor_98, unsqueeze_default_26);  mul_tensor_98 = unsqueeze_default_26 = None
        slice_tensor_73: "f64[204, 203, 26]" = torch.ops.aten.slice.Tensor(arg13_1, 1, 0, -1);  arg13_1 = None
        mul_tensor_99: "f64[204, 203, 26]" = torch.ops.aten.mul.Tensor(div_tensor_61, slice_tensor_73);  div_tensor_61 = slice_tensor_73 = None
        unsqueeze_default_27: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg14_1, 0)
        slice_tensor_74: "f64[1, 203]" = torch.ops.aten.slice.Tensor(unsqueeze_default_27, 1, 0, -1);  unsqueeze_default_27 = None
        unsqueeze_default_28: "f64[1, 203, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_74, 2);  slice_tensor_74 = None
        mul_tensor_100: "f64[204, 203, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_99, unsqueeze_default_28);  mul_tensor_99 = unsqueeze_default_28 = None
        copy_default_35: "f64[204, 203, 26]" = torch.ops.aten.copy.default(slice_tensor_69, mul_tensor_100);  slice_tensor_69 = mul_tensor_100 = None
        slice_scatter_default_9: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_default_9, copy_default_35, 1, 0, -1);  full_default_9 = copy_default_35 = None
        select_int_405: "f64[204, 26]" = torch.ops.aten.select.int(slice_scatter_default_9, 1, -1)
        full_default_10: "f64[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default_36: "f64[204, 26]" = torch.ops.aten.copy.default(select_int_405, full_default_10);  select_int_405 = full_default_10 = None
        select_scatter_default_134: "f64[204, 204, 26]" = torch.ops.aten.select_scatter.default(slice_scatter_default_9, copy_default_36, 1, -1);  slice_scatter_default_9 = copy_default_36 = None
        slice_tensor_75: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_scatter_default_134, 0, 2, -2)
        slice_tensor_76: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_75, 1, 2, -2);  slice_tensor_75 = None
        slice_tensor_77: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_scatter_default_134, 0, 2, -2)
        slice_tensor_78: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_77, 1, 1, -3);  slice_tensor_77 = None
        sub_tensor_29: "f64[200, 200, 26]" = torch.ops.aten.sub.Tensor(slice_tensor_76, slice_tensor_78);  slice_tensor_76 = slice_tensor_78 = None
        unsqueeze_default_29: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg10_1, 0)
        slice_tensor_79: "f64[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_default_29, 1, 2, -2);  unsqueeze_default_29 = None
        unsqueeze_default_30: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_79, 2);  slice_tensor_79 = None
        unsqueeze_default_31: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg17_1, 0)
        slice_tensor_80: "f64[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_default_31, 1, 2, -2);  unsqueeze_default_31 = None
        unsqueeze_default_32: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_80, 2);  slice_tensor_80 = None
        mul_tensor_101: "f64[1, 200, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_default_30, unsqueeze_default_32);  unsqueeze_default_30 = unsqueeze_default_32 = None
        div_tensor_62: "f64[200, 200, 26]" = torch.ops.aten.div.Tensor(sub_tensor_29, mul_tensor_101);  sub_tensor_29 = mul_tensor_101 = None
        add_tensor_60: "f64[200, 200, 26]" = torch.ops.aten.add.Tensor(div_tensor_60, div_tensor_62);  div_tensor_60 = div_tensor_62 = None
        mul_tensor_102: "f64[200, 200, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_93, add_tensor_60);  mul_tensor_93 = add_tensor_60 = None
        add_tensor_61: "f64[200, 200, 26]" = torch.ops.aten.add.Tensor(select_int_399, mul_tensor_102);  select_int_399 = mul_tensor_102 = None
        select_scatter_default_135: "f64[200, 200, 26, 3]" = torch.ops.aten.select_scatter.default(slice_tensor_54, add_tensor_61, 3, 1);  slice_tensor_54 = add_tensor_61 = None
        slice_scatter_default_10: "f64[200, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_tensor_53, select_scatter_default_135, 1, 2, -2);  slice_tensor_53 = select_scatter_default_135 = None
        slice_scatter_default_11: "f64[204, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_7, slice_scatter_default_10, 0, 2, -2);  slice_scatter_default_7 = slice_scatter_default_10 = None
        slice_tensor_81: "f64[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_default_11, 0, 2, -2)
        slice_tensor_82: "f64[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_81, 1, 2, -2);  slice_tensor_81 = None
        select_int_406: "f64[200, 200, 26]" = torch.ops.aten.select.int(slice_tensor_82, 3, 1);  slice_tensor_82 = None
        full_default_11: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_83: "f64[203, 204, 26]" = torch.ops.aten.slice.Tensor(full_default_11, 0, 0, -1)
        slice_tensor_84: "f64[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 2, -2)
        slice_tensor_85: "f64[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_84, 1, 2, -2)
        slice_tensor_86: "f64[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 2, -2)
        slice_tensor_87: "f64[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_86, 1, 2, -2);  slice_tensor_86 = None
        select_int_407: "f64[200, 200, 26]" = torch.ops.aten.select.int(slice_tensor_87, 3, 0);  slice_tensor_87 = None
        slice_tensor_88: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg15_1, 0, 2, -2)
        slice_tensor_89: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_88, 1, 2, -2);  slice_tensor_88 = None
        full_default_12: "f64[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default_37: "f64[204, 204, 26]" = torch.ops.aten.copy.default(select_scatter_default_133, full_default_12);  select_scatter_default_133 = full_default_12 = None
        slice_tensor_90: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(copy_default_37, 0, 1, -2)
        slice_tensor_91: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(copy_default_37, 0, 1, -2)
        slice_tensor_92: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_91, 1, 2, -2);  slice_tensor_91 = None
        select_int_408: "f64[204, 204, 26]" = torch.ops.aten.select.int(arg18_1, 3, 0);  arg18_1 = None
        slice_tensor_93: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_408, 0, 1, -2)
        slice_tensor_94: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_93, 1, 2, -2);  slice_tensor_93 = None
        slice_tensor_95: "f64[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_default_11, 0, 2, -2)
        slice_tensor_96: "f64[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_95, 1, 2, -2)
        slice_tensor_97: "f64[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_default_11, 0, 2, -2)
        slice_tensor_98: "f64[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_97, 1, 2, -2);  slice_tensor_97 = None
        select_int_409: "f64[200, 200, 26]" = torch.ops.aten.select.int(slice_tensor_98, 3, 1);  slice_tensor_98 = None
        select_scatter_default_136: "f64[200, 200, 26, 3]" = torch.ops.aten.select_scatter.default(slice_tensor_96, select_int_409, 3, 1);  slice_tensor_96 = select_int_409 = None
        slice_scatter_default_12: "f64[200, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_tensor_95, select_scatter_default_136, 1, 2, -2);  slice_tensor_95 = select_scatter_default_136 = None
        slice_scatter_default_13: "f64[204, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_11, slice_scatter_default_12, 0, 2, -2);  slice_scatter_default_11 = slice_scatter_default_12 = None
        select_int_410: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 0)
        slice_tensor_99: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_410, 0, 2, -1);  select_int_410 = None
        slice_tensor_100: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_99, 1, 2, -2);  slice_tensor_99 = None
        select_int_411: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 0)
        slice_tensor_101: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_411, 0, 1, -2);  select_int_411 = None
        slice_tensor_102: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_101, 1, 2, -2);  slice_tensor_101 = None
        add_tensor_62: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(slice_tensor_100, slice_tensor_102);  slice_tensor_100 = slice_tensor_102 = None
        mul_tensor_103: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_94, add_tensor_62);  slice_tensor_94 = add_tensor_62 = None
        mul_tensor_104: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_103, 0.5);  mul_tensor_103 = None
        slice_tensor_103: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_408, 0, 1, -2)
        slice_tensor_104: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_103, 1, 2, -2);  slice_tensor_103 = None
        abs_default: "f64[201, 200, 26]" = torch.ops.aten.abs.default(slice_tensor_104);  slice_tensor_104 = None
        full_default_13: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "f32[1]" = torch.ops.aten.full.default([1], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_105: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_408, 0, 1, -2)
        slice_tensor_106: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_105, 1, 2, -2);  slice_tensor_105 = None
        gt_scalar: "b8[201, 200, 26]" = torch.ops.aten.gt.Scalar(slice_tensor_106, 0.0);  slice_tensor_106 = None
        select_int_412: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 0)
        slice_tensor_107: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_412, 0, 1, -2);  select_int_412 = None
        slice_tensor_108: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_107, 1, 2, -2);  slice_tensor_107 = None
        select_int_413: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 0)
        slice_tensor_109: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_413, 0, 0, -3);  select_int_413 = None
        slice_tensor_110: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_109, 1, 2, -2);  slice_tensor_109 = None
        sub_tensor_30: "f64[201, 200, 26]" = torch.ops.aten.sub.Tensor(slice_tensor_108, slice_tensor_110);  slice_tensor_108 = slice_tensor_110 = None
        slice_tensor_111: "f64[203, 204, 26]" = torch.ops.aten.slice.Tensor(arg15_1, 0, 1, 9223372036854775807)
        slice_tensor_112: "f64[203, 204, 26]" = torch.ops.aten.slice.Tensor(arg15_1, 0, 0, -1)
        mul_tensor_105: "f64[203, 204, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_111, slice_tensor_112);  slice_tensor_111 = slice_tensor_112 = None
        slice_scatter_default_14: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_default_11, mul_tensor_105, 0, 0, -1);  full_default_11 = mul_tensor_105 = None
        slice_tensor_113: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_14, 0, 0, -3)
        slice_tensor_114: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_113, 1, 2, -2);  slice_tensor_113 = None
        mul_tensor_106: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_30, slice_tensor_114);  sub_tensor_30 = slice_tensor_114 = None
        select_int_414: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 0)
        slice_tensor_115: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_414, 0, 3, 9223372036854775807);  select_int_414 = None
        slice_tensor_116: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_115, 1, 2, -2);  slice_tensor_115 = None
        select_int_415: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 0)
        slice_tensor_117: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_415, 0, 2, -1);  select_int_415 = None
        slice_tensor_118: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_117, 1, 2, -2);  slice_tensor_117 = None
        sub_tensor_31: "f64[201, 200, 26]" = torch.ops.aten.sub.Tensor(slice_tensor_116, slice_tensor_118);  slice_tensor_116 = slice_tensor_118 = None
        slice_tensor_119: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_14, 0, 2, -1)
        slice_tensor_120: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_119, 1, 2, -2);  slice_tensor_119 = None
        mul_tensor_107: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_31, slice_tensor_120);  sub_tensor_31 = slice_tensor_120 = None
        where_self_3: "f64[201, 200, 26]" = torch.ops.aten.where.self(gt_scalar, mul_tensor_106, mul_tensor_107);  gt_scalar = mul_tensor_106 = mul_tensor_107 = None
        select_int_416: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 0)
        slice_tensor_121: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_416, 0, 2, -1);  select_int_416 = None
        slice_tensor_122: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_121, 1, 2, -2);  slice_tensor_121 = None
        select_int_417: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 0)
        slice_tensor_123: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_417, 0, 1, -2);  select_int_417 = None
        slice_tensor_124: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_123, 1, 2, -2);  slice_tensor_123 = None
        sub_tensor_32: "f64[201, 200, 26]" = torch.ops.aten.sub.Tensor(slice_tensor_122, slice_tensor_124);  slice_tensor_122 = slice_tensor_124 = None
        slice_tensor_125: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_14, 0, 1, -2);  slice_scatter_default_14 = None
        slice_tensor_126: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_125, 1, 2, -2);  slice_tensor_125 = None
        mul_tensor_108: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_32, slice_tensor_126);  sub_tensor_32 = slice_tensor_126 = None
        abs_default_1: "f64[201, 200, 26]" = torch.ops.aten.abs.default(mul_tensor_108)
        lt_scalar: "b8[201, 200, 26]" = torch.ops.aten.lt.Scalar(abs_default_1, 1e-20);  abs_default_1 = None
        full_default_15: "f64[]" = torch.ops.aten.full.default([], 1e-20, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_4: "f64[201, 200, 26]" = torch.ops.aten.where.self(lt_scalar, full_default_15, mul_tensor_108);  lt_scalar = full_default_15 = None
        div_tensor_63: "f64[201, 200, 26]" = torch.ops.aten.div.Tensor(where_self_3, where_self_4);  where_self_3 = where_self_4 = None
        mul_tensor_109: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(div_tensor_63, 2)
        minimum_default: "f64[201, 200, 26]" = torch.ops.aten.minimum.default(full_default_14, mul_tensor_109);  full_default_14 = mul_tensor_109 = None
        full_default_16: "f32[1]" = torch.ops.aten.full.default([1], 2.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default_1: "f64[201, 200, 26]" = torch.ops.aten.minimum.default(full_default_16, div_tensor_63);  full_default_16 = div_tensor_63 = None
        maximum_default_2: "f64[201, 200, 26]" = torch.ops.aten.maximum.default(minimum_default, minimum_default_1);  minimum_default = minimum_default_1 = None
        maximum_default_3: "f64[201, 200, 26]" = torch.ops.aten.maximum.default(full_default_13, maximum_default_2);  full_default_13 = maximum_default_2 = None
        sub_tensor_33: "f64[201, 200, 26]" = torch.ops.aten.sub.Tensor(1.0, maximum_default_3)
        slice_tensor_127: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_408, 0, 1, -2);  select_int_408 = None
        slice_tensor_128: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_127, 1, 2, -2);  slice_tensor_127 = None
        mul_tensor_110: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_128, 1.0);  slice_tensor_128 = None
        unsqueeze_default_33: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg10_1, 0)
        slice_tensor_129: "f64[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_default_33, 1, 2, -2);  unsqueeze_default_33 = None
        unsqueeze_default_34: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_129, 2);  slice_tensor_129 = None
        slice_tensor_130: "f64[201]" = torch.ops.aten.slice.Tensor(arg16_1, 0, 1, -2)
        unsqueeze_default_35: "f64[201, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_130, 1);  slice_tensor_130 = None
        unsqueeze_default_36: "f64[201, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_35, 2);  unsqueeze_default_35 = None
        mul_tensor_111: "f64[201, 200, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_default_34, unsqueeze_default_36);  unsqueeze_default_34 = unsqueeze_default_36 = None
        div_tensor_64: "f64[201, 200, 26]" = torch.ops.aten.div.Tensor(mul_tensor_110, mul_tensor_111);  mul_tensor_110 = mul_tensor_111 = None
        abs_default_2: "f64[201, 200, 26]" = torch.ops.aten.abs.default(div_tensor_64);  div_tensor_64 = None
        mul_tensor_112: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(abs_default_2, maximum_default_3);  abs_default_2 = maximum_default_3 = None
        add_tensor_63: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(sub_tensor_33, mul_tensor_112);  sub_tensor_33 = mul_tensor_112 = None
        mul_tensor_113: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(abs_default, add_tensor_63);  abs_default = add_tensor_63 = None
        mul_tensor_114: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_113, mul_tensor_108);  mul_tensor_113 = mul_tensor_108 = None
        mul_tensor_115: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_114, 0.5);  mul_tensor_114 = None
        sub_tensor_34: "f64[201, 200, 26]" = torch.ops.aten.sub.Tensor(mul_tensor_104, mul_tensor_115);  mul_tensor_104 = mul_tensor_115 = None
        copy_default_38: "f64[201, 200, 26]" = torch.ops.aten.copy.default(slice_tensor_92, sub_tensor_34);  slice_tensor_92 = sub_tensor_34 = None
        slice_scatter_default_15: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_90, copy_default_38, 1, 2, -2);  slice_tensor_90 = copy_default_38 = None
        slice_scatter_default_16: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(copy_default_37, slice_scatter_default_15, 0, 1, -2);  copy_default_37 = slice_scatter_default_15 = None
        slice_tensor_131: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_16, 0, 2, -2)
        slice_tensor_132: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_131, 1, 2, -2);  slice_tensor_131 = None
        slice_tensor_133: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_16, 0, 1, -3);  slice_scatter_default_16 = None
        slice_tensor_134: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_133, 1, 2, -2);  slice_tensor_133 = None
        sub_tensor_35: "f64[200, 200, 26]" = torch.ops.aten.sub.Tensor(slice_tensor_132, slice_tensor_134);  slice_tensor_132 = slice_tensor_134 = None
        neg_default_53: "f64[200, 200, 26]" = torch.ops.aten.neg.default(sub_tensor_35);  sub_tensor_35 = None
        unsqueeze_default_37: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg10_1, 0)
        slice_tensor_135: "f64[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_default_37, 1, 2, -2);  unsqueeze_default_37 = None
        unsqueeze_default_38: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_135, 2);  slice_tensor_135 = None
        slice_tensor_136: "f64[200]" = torch.ops.aten.slice.Tensor(arg16_1, 0, 2, -2);  arg16_1 = None
        unsqueeze_default_39: "f64[200, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_136, 1);  slice_tensor_136 = None
        unsqueeze_default_40: "f64[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_39, 2);  unsqueeze_default_39 = None
        mul_tensor_116: "f64[200, 200, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_default_38, unsqueeze_default_40);  unsqueeze_default_38 = unsqueeze_default_40 = None
        div_tensor_65: "f64[200, 200, 26]" = torch.ops.aten.div.Tensor(neg_default_53, mul_tensor_116);  neg_default_53 = mul_tensor_116 = None
        full_default_17: "f64[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default_39: "f64[204, 204, 26]" = torch.ops.aten.copy.default(select_scatter_default_134, full_default_17);  select_scatter_default_134 = full_default_17 = None
        slice_tensor_137: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(copy_default_39, 0, 2, -2)
        slice_tensor_138: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(copy_default_39, 0, 2, -2)
        slice_tensor_139: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_138, 1, 1, -2);  slice_tensor_138 = None
        unsqueeze_default_41: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg14_1, 0);  arg14_1 = None
        slice_tensor_140: "f64[1, 201]" = torch.ops.aten.slice.Tensor(unsqueeze_default_41, 1, 1, -2);  unsqueeze_default_41 = None
        unsqueeze_default_42: "f64[1, 201, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_140, 2);  slice_tensor_140 = None
        select_int_418: "f64[204, 204, 26]" = torch.ops.aten.select.int(arg19_1, 3, 0);  arg19_1 = None
        slice_tensor_141: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_418, 0, 2, -2)
        slice_tensor_142: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_141, 1, 1, -2);  slice_tensor_141 = None
        mul_tensor_117: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_default_42, slice_tensor_142);  slice_tensor_142 = None
        select_int_419: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 0)
        slice_tensor_143: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_419, 0, 2, -2);  select_int_419 = None
        slice_tensor_144: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_143, 1, 2, -1);  slice_tensor_143 = None
        select_int_420: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 0)
        slice_tensor_145: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_420, 0, 2, -2);  select_int_420 = None
        slice_tensor_146: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_145, 1, 1, -2);  slice_tensor_145 = None
        add_tensor_64: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(slice_tensor_144, slice_tensor_146);  slice_tensor_144 = slice_tensor_146 = None
        mul_tensor_118: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_117, add_tensor_64);  mul_tensor_117 = add_tensor_64 = None
        mul_tensor_119: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_118, 0.5);  mul_tensor_118 = None
        slice_tensor_147: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_418, 0, 2, -2)
        slice_tensor_148: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_147, 1, 1, -2);  slice_tensor_147 = None
        mul_tensor_120: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_default_42, slice_tensor_148);  slice_tensor_148 = None
        abs_default_3: "f64[200, 201, 26]" = torch.ops.aten.abs.default(mul_tensor_120);  mul_tensor_120 = None
        full_default_18: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "f32[1]" = torch.ops.aten.full.default([1], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_149: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_418, 0, 2, -2)
        slice_tensor_150: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_149, 1, 1, -2);  slice_tensor_149 = None
        gt_scalar_1: "b8[200, 201, 26]" = torch.ops.aten.gt.Scalar(slice_tensor_150, 0.0);  slice_tensor_150 = None
        select_int_421: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 0)
        slice_tensor_151: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_421, 0, 2, -2);  select_int_421 = None
        slice_tensor_152: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_151, 1, 1, -2);  slice_tensor_151 = None
        select_int_422: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 0)
        slice_tensor_153: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_422, 0, 2, -2);  select_int_422 = None
        slice_tensor_154: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_153, 1, 0, -3);  slice_tensor_153 = None
        sub_tensor_36: "f64[200, 201, 26]" = torch.ops.aten.sub.Tensor(slice_tensor_152, slice_tensor_154);  slice_tensor_152 = slice_tensor_154 = None
        full_default_20: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_155: "f64[204, 203, 26]" = torch.ops.aten.slice.Tensor(full_default_20, 1, 0, -1)
        slice_tensor_156: "f64[204, 203, 26]" = torch.ops.aten.slice.Tensor(arg15_1, 1, 1, 9223372036854775807)
        slice_tensor_157: "f64[204, 203, 26]" = torch.ops.aten.slice.Tensor(arg15_1, 1, 0, -1)
        mul_tensor_121: "f64[204, 203, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_156, slice_tensor_157);  slice_tensor_156 = slice_tensor_157 = None
        copy_default_40: "f64[204, 203, 26]" = torch.ops.aten.copy.default(slice_tensor_155, mul_tensor_121);  slice_tensor_155 = mul_tensor_121 = None
        slice_scatter_default_17: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_default_20, copy_default_40, 1, 0, -1);  full_default_20 = copy_default_40 = None
        slice_tensor_158: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_17, 0, 2, -2)
        slice_tensor_159: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_158, 1, 0, -3);  slice_tensor_158 = None
        mul_tensor_122: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_36, slice_tensor_159);  sub_tensor_36 = slice_tensor_159 = None
        select_int_423: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 0)
        slice_tensor_160: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_423, 0, 2, -2);  select_int_423 = None
        slice_tensor_161: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_160, 1, 3, 9223372036854775807);  slice_tensor_160 = None
        select_int_424: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 0)
        slice_tensor_162: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_424, 0, 2, -2);  select_int_424 = None
        slice_tensor_163: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_162, 1, 2, -1);  slice_tensor_162 = None
        sub_tensor_37: "f64[200, 201, 26]" = torch.ops.aten.sub.Tensor(slice_tensor_161, slice_tensor_163);  slice_tensor_161 = slice_tensor_163 = None
        slice_tensor_164: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_17, 0, 2, -2)
        slice_tensor_165: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_164, 1, 2, -1);  slice_tensor_164 = None
        mul_tensor_123: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_37, slice_tensor_165);  sub_tensor_37 = slice_tensor_165 = None
        where_self_5: "f64[200, 201, 26]" = torch.ops.aten.where.self(gt_scalar_1, mul_tensor_122, mul_tensor_123);  gt_scalar_1 = mul_tensor_122 = mul_tensor_123 = None
        select_int_425: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 0)
        slice_tensor_166: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_425, 0, 2, -2);  select_int_425 = None
        slice_tensor_167: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_166, 1, 2, -1);  slice_tensor_166 = None
        select_int_426: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 0)
        slice_tensor_168: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_426, 0, 2, -2);  select_int_426 = None
        slice_tensor_169: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_168, 1, 1, -2);  slice_tensor_168 = None
        sub_tensor_38: "f64[200, 201, 26]" = torch.ops.aten.sub.Tensor(slice_tensor_167, slice_tensor_169);  slice_tensor_167 = slice_tensor_169 = None
        slice_tensor_170: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_17, 0, 2, -2);  slice_scatter_default_17 = None
        slice_tensor_171: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_170, 1, 1, -2);  slice_tensor_170 = None
        mul_tensor_124: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_38, slice_tensor_171);  sub_tensor_38 = slice_tensor_171 = None
        abs_default_4: "f64[200, 201, 26]" = torch.ops.aten.abs.default(mul_tensor_124)
        lt_scalar_1: "b8[200, 201, 26]" = torch.ops.aten.lt.Scalar(abs_default_4, 1e-20);  abs_default_4 = None
        full_default_21: "f64[]" = torch.ops.aten.full.default([], 1e-20, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_6: "f64[200, 201, 26]" = torch.ops.aten.where.self(lt_scalar_1, full_default_21, mul_tensor_124);  lt_scalar_1 = full_default_21 = None
        div_tensor_66: "f64[200, 201, 26]" = torch.ops.aten.div.Tensor(where_self_5, where_self_6);  where_self_5 = where_self_6 = None
        mul_tensor_125: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(div_tensor_66, 2)
        minimum_default_2: "f64[200, 201, 26]" = torch.ops.aten.minimum.default(full_default_19, mul_tensor_125);  full_default_19 = mul_tensor_125 = None
        full_default_22: "f32[1]" = torch.ops.aten.full.default([1], 2.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default_3: "f64[200, 201, 26]" = torch.ops.aten.minimum.default(full_default_22, div_tensor_66);  full_default_22 = div_tensor_66 = None
        maximum_default_4: "f64[200, 201, 26]" = torch.ops.aten.maximum.default(minimum_default_2, minimum_default_3);  minimum_default_2 = minimum_default_3 = None
        maximum_default_5: "f64[200, 201, 26]" = torch.ops.aten.maximum.default(full_default_18, maximum_default_4);  full_default_18 = maximum_default_4 = None
        sub_tensor_39: "f64[200, 201, 26]" = torch.ops.aten.sub.Tensor(1.0, maximum_default_5)
        slice_tensor_172: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_int_418, 0, 2, -2);  select_int_418 = None
        slice_tensor_173: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_172, 1, 1, -2);  slice_tensor_172 = None
        mul_tensor_126: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_default_42, slice_tensor_173);  unsqueeze_default_42 = slice_tensor_173 = None
        mul_tensor_127: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_126, 1.0);  mul_tensor_126 = None
        mul_tensor_128: "f64[204]" = torch.ops.aten.mul.Tensor(arg10_1, arg17_1)
        unsqueeze_default_43: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(mul_tensor_128, 0);  mul_tensor_128 = None
        slice_tensor_174: "f64[1, 201]" = torch.ops.aten.slice.Tensor(unsqueeze_default_43, 1, 1, -2);  unsqueeze_default_43 = None
        unsqueeze_default_44: "f64[1, 201, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_174, 2);  slice_tensor_174 = None
        div_tensor_67: "f64[200, 201, 26]" = torch.ops.aten.div.Tensor(mul_tensor_127, unsqueeze_default_44);  mul_tensor_127 = unsqueeze_default_44 = None
        abs_default_5: "f64[200, 201, 26]" = torch.ops.aten.abs.default(div_tensor_67);  div_tensor_67 = None
        mul_tensor_129: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(abs_default_5, maximum_default_5);  abs_default_5 = maximum_default_5 = None
        add_tensor_65: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(sub_tensor_39, mul_tensor_129);  sub_tensor_39 = mul_tensor_129 = None
        mul_tensor_130: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(abs_default_3, add_tensor_65);  abs_default_3 = add_tensor_65 = None
        mul_tensor_131: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_130, mul_tensor_124);  mul_tensor_130 = mul_tensor_124 = None
        mul_tensor_132: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_131, 0.5);  mul_tensor_131 = None
        sub_tensor_40: "f64[200, 201, 26]" = torch.ops.aten.sub.Tensor(mul_tensor_119, mul_tensor_132);  mul_tensor_119 = mul_tensor_132 = None
        copy_default_41: "f64[200, 201, 26]" = torch.ops.aten.copy.default(slice_tensor_139, sub_tensor_40);  slice_tensor_139 = sub_tensor_40 = None
        slice_scatter_default_18: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_137, copy_default_41, 1, 1, -2);  slice_tensor_137 = copy_default_41 = None
        slice_scatter_default_19: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(copy_default_39, slice_scatter_default_18, 0, 2, -2);  copy_default_39 = slice_scatter_default_18 = None
        slice_tensor_175: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_19, 0, 2, -2)
        slice_tensor_176: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_175, 1, 2, -2);  slice_tensor_175 = None
        slice_tensor_177: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_19, 0, 2, -2);  slice_scatter_default_19 = None
        slice_tensor_178: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_177, 1, 1, -3);  slice_tensor_177 = None
        sub_tensor_41: "f64[200, 200, 26]" = torch.ops.aten.sub.Tensor(slice_tensor_176, slice_tensor_178);  slice_tensor_176 = slice_tensor_178 = None
        unsqueeze_default_45: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg10_1, 0);  arg10_1 = None
        slice_tensor_179: "f64[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_default_45, 1, 2, -2);  unsqueeze_default_45 = None
        unsqueeze_default_46: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_179, 2);  slice_tensor_179 = None
        unsqueeze_default_47: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg17_1, 0);  arg17_1 = None
        slice_tensor_180: "f64[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_default_47, 1, 2, -2);  unsqueeze_default_47 = None
        unsqueeze_default_48: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_180, 2);  slice_tensor_180 = None
        mul_tensor_133: "f64[1, 200, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_default_46, unsqueeze_default_48);  unsqueeze_default_46 = unsqueeze_default_48 = None
        div_tensor_68: "f64[200, 200, 26]" = torch.ops.aten.div.Tensor(sub_tensor_41, mul_tensor_133);  sub_tensor_41 = mul_tensor_133 = None
        sub_tensor_42: "f64[200, 200, 26]" = torch.ops.aten.sub.Tensor(div_tensor_65, div_tensor_68);  div_tensor_65 = div_tensor_68 = None
        mul_tensor_134: "f64[200, 200, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_89, sub_tensor_42);  slice_tensor_89 = sub_tensor_42 = None
        copy_default_42: "f64[200, 200, 26]" = torch.ops.aten.copy.default(select_int_407, mul_tensor_134);  select_int_407 = mul_tensor_134 = None
        select_scatter_default_137: "f64[200, 200, 26, 3]" = torch.ops.aten.select_scatter.default(slice_tensor_85, copy_default_42, 3, 0);  slice_tensor_85 = copy_default_42 = None
        slice_scatter_default_20: "f64[200, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_tensor_84, select_scatter_default_137, 1, 2, -2);  slice_tensor_84 = select_scatter_default_137 = None
        slice_scatter_default_21: "f64[204, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(arg1_1, slice_scatter_default_20, 0, 2, -2);  arg1_1 = slice_scatter_default_20 = None
        select_int_427: "f64[204, 204, 3]" = torch.ops.aten.select.int(slice_scatter_default_21, 2, 0)
        select_int_428: "f64[204, 204, 3]" = torch.ops.aten.select.int(slice_scatter_default_21, 2, 0)
        select_int_429: "f64[204, 204]" = torch.ops.aten.select.int(select_int_428, 2, 0);  select_int_428 = None
        full_default_23: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "f64[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default_43: "f64[204, 204, 26]" = torch.ops.aten.copy.default(full_default_23, full_default_24);  full_default_23 = full_default_24 = None
        slice_tensor_181: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(copy_default_43, 0, 2, -2)
        slice_tensor_182: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_181, 1, 2, -2)
        slice_tensor_183: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(copy_default_43, 0, 2, -2)
        slice_tensor_184: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_183, 1, 2, -2);  slice_tensor_183 = None
        slice_tensor_185: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_184, 2, 0, -1);  slice_tensor_184 = None
        full_default_25: "f64[204, 204, 28]" = torch.ops.aten.full.default([204, 204, 28], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_186: "f64[204, 204, 26]" = torch.ops.aten.slice.Tensor(full_default_25, 2, 1, -1)
        select_int_430: "f64[204, 204, 26]" = torch.ops.aten.select.int(arg20_1, 3, 0);  arg20_1 = None
        copy_default_44: "f64[204, 204, 26]" = torch.ops.aten.copy.default(slice_tensor_186, select_int_430);  slice_tensor_186 = select_int_430 = None
        slice_scatter_default_22: "f64[204, 204, 28]" = torch.ops.aten.slice_scatter.default(full_default_25, copy_default_44, 2, 1, -1);  full_default_25 = copy_default_44 = None
        slice_tensor_187: "f64[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_default_22, 0, 2, -2)
        slice_tensor_188: "f64[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_tensor_187, 1, 2, -2);  slice_tensor_187 = None
        slice_tensor_189: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_188, 2, 1, -2);  slice_tensor_188 = None
        full_default_26: "f64[204, 204, 28]" = torch.ops.aten.full.default([204, 204, 28], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_190: "f64[204, 204, 26]" = torch.ops.aten.slice.Tensor(full_default_26, 2, 1, -1)
        select_int_431: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 0)
        copy_default_45: "f64[204, 204, 26]" = torch.ops.aten.copy.default(slice_tensor_190, select_int_431);  slice_tensor_190 = select_int_431 = None
        slice_scatter_default_23: "f64[204, 204, 28]" = torch.ops.aten.slice_scatter.default(full_default_26, copy_default_45, 2, 1, -1);  full_default_26 = copy_default_45 = None
        slice_tensor_191: "f64[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_default_23, 0, 2, -2)
        slice_tensor_192: "f64[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_tensor_191, 1, 2, -2);  slice_tensor_191 = None
        slice_tensor_193: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_192, 2, 2, -1);  slice_tensor_192 = None
        slice_tensor_194: "f64[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_default_23, 0, 2, -2)
        slice_tensor_195: "f64[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_tensor_194, 1, 2, -2);  slice_tensor_194 = None
        slice_tensor_196: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_195, 2, 1, -2);  slice_tensor_195 = None
        add_tensor_66: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_tensor_193, slice_tensor_196);  slice_tensor_193 = slice_tensor_196 = None
        mul_tensor_135: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_189, add_tensor_66);  slice_tensor_189 = add_tensor_66 = None
        mul_tensor_136: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_135, 0.5);  mul_tensor_135 = None
        slice_tensor_197: "f64[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_default_22, 0, 2, -2)
        slice_tensor_198: "f64[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_tensor_197, 1, 2, -2);  slice_tensor_197 = None
        slice_tensor_199: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_198, 2, 1, -2);  slice_tensor_198 = None
        abs_default_6: "f64[200, 200, 25]" = torch.ops.aten.abs.default(slice_tensor_199);  slice_tensor_199 = None
        full_default_27: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "f32[1]" = torch.ops.aten.full.default([1], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_200: "f64[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_default_22, 0, 2, -2)
        slice_tensor_201: "f64[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_tensor_200, 1, 2, -2);  slice_tensor_200 = None
        slice_tensor_202: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_201, 2, 1, -2);  slice_tensor_201 = None
        gt_scalar_2: "b8[200, 200, 25]" = torch.ops.aten.gt.Scalar(slice_tensor_202, 0.0);  slice_tensor_202 = None
        slice_tensor_203: "f64[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_default_23, 0, 2, -2)
        slice_tensor_204: "f64[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_tensor_203, 1, 2, -2);  slice_tensor_203 = None
        slice_tensor_205: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_204, 2, 1, -2);  slice_tensor_204 = None
        slice_tensor_206: "f64[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_default_23, 0, 2, -2)
        slice_tensor_207: "f64[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_tensor_206, 1, 2, -2);  slice_tensor_206 = None
        slice_tensor_208: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_207, 2, 0, -3);  slice_tensor_207 = None
        sub_tensor_43: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(slice_tensor_205, slice_tensor_208);  slice_tensor_205 = slice_tensor_208 = None
        full_default_29: "f64[204, 204, 28]" = torch.ops.aten.full.default([204, 204, 28], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_209: "f64[204, 204, 26]" = torch.ops.aten.slice.Tensor(full_default_29, 2, 1, -1)
        full_default_30: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_210: "f64[204, 204, 25]" = torch.ops.aten.slice.Tensor(full_default_30, 2, 0, -1)
        slice_tensor_211: "f64[204, 204, 25]" = torch.ops.aten.slice.Tensor(arg15_1, 2, 1, 9223372036854775807)
        slice_tensor_212: "f64[204, 204, 25]" = torch.ops.aten.slice.Tensor(arg15_1, 2, 0, -1);  arg15_1 = None
        mul_tensor_137: "f64[204, 204, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_211, slice_tensor_212);  slice_tensor_211 = slice_tensor_212 = None
        copy_default_46: "f64[204, 204, 25]" = torch.ops.aten.copy.default(slice_tensor_210, mul_tensor_137);  slice_tensor_210 = mul_tensor_137 = None
        slice_scatter_default_24: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_default_30, copy_default_46, 2, 0, -1);  full_default_30 = copy_default_46 = None
        copy_default_47: "f64[204, 204, 26]" = torch.ops.aten.copy.default(slice_tensor_209, slice_scatter_default_24);  slice_tensor_209 = slice_scatter_default_24 = None
        slice_scatter_default_25: "f64[204, 204, 28]" = torch.ops.aten.slice_scatter.default(full_default_29, copy_default_47, 2, 1, -1);  full_default_29 = copy_default_47 = None
        slice_tensor_213: "f64[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_default_25, 0, 2, -2)
        slice_tensor_214: "f64[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_tensor_213, 1, 2, -2);  slice_tensor_213 = None
        slice_tensor_215: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_214, 2, 0, -3);  slice_tensor_214 = None
        mul_tensor_138: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(sub_tensor_43, slice_tensor_215);  sub_tensor_43 = slice_tensor_215 = None
        slice_tensor_216: "f64[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_default_23, 0, 2, -2)
        slice_tensor_217: "f64[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_tensor_216, 1, 2, -2);  slice_tensor_216 = None
        slice_tensor_218: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_217, 2, 3, 9223372036854775807);  slice_tensor_217 = None
        slice_tensor_219: "f64[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_default_23, 0, 2, -2)
        slice_tensor_220: "f64[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_tensor_219, 1, 2, -2);  slice_tensor_219 = None
        slice_tensor_221: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_220, 2, 2, -1);  slice_tensor_220 = None
        sub_tensor_44: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(slice_tensor_218, slice_tensor_221);  slice_tensor_218 = slice_tensor_221 = None
        slice_tensor_222: "f64[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_default_25, 0, 2, -2)
        slice_tensor_223: "f64[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_tensor_222, 1, 2, -2);  slice_tensor_222 = None
        slice_tensor_224: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_223, 2, 2, -1);  slice_tensor_223 = None
        mul_tensor_139: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(sub_tensor_44, slice_tensor_224);  sub_tensor_44 = slice_tensor_224 = None
        where_self_7: "f64[200, 200, 25]" = torch.ops.aten.where.self(gt_scalar_2, mul_tensor_138, mul_tensor_139);  gt_scalar_2 = mul_tensor_138 = mul_tensor_139 = None
        slice_tensor_225: "f64[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_default_23, 0, 2, -2)
        slice_tensor_226: "f64[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_tensor_225, 1, 2, -2);  slice_tensor_225 = None
        slice_tensor_227: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_226, 2, 2, -1);  slice_tensor_226 = None
        slice_tensor_228: "f64[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_default_23, 0, 2, -2);  slice_scatter_default_23 = None
        slice_tensor_229: "f64[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_tensor_228, 1, 2, -2);  slice_tensor_228 = None
        slice_tensor_230: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_229, 2, 1, -2);  slice_tensor_229 = None
        sub_tensor_45: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(slice_tensor_227, slice_tensor_230);  slice_tensor_227 = slice_tensor_230 = None
        slice_tensor_231: "f64[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_default_25, 0, 2, -2);  slice_scatter_default_25 = None
        slice_tensor_232: "f64[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_tensor_231, 1, 2, -2);  slice_tensor_231 = None
        slice_tensor_233: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_232, 2, 1, -2);  slice_tensor_232 = None
        mul_tensor_140: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(sub_tensor_45, slice_tensor_233);  sub_tensor_45 = slice_tensor_233 = None
        abs_default_7: "f64[200, 200, 25]" = torch.ops.aten.abs.default(mul_tensor_140)
        lt_scalar_2: "b8[200, 200, 25]" = torch.ops.aten.lt.Scalar(abs_default_7, 1e-20);  abs_default_7 = None
        full_default_31: "f64[]" = torch.ops.aten.full.default([], 1e-20, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_8: "f64[200, 200, 25]" = torch.ops.aten.where.self(lt_scalar_2, full_default_31, mul_tensor_140);  lt_scalar_2 = full_default_31 = None
        div_tensor_69: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(where_self_7, where_self_8);  where_self_7 = where_self_8 = None
        mul_tensor_141: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(div_tensor_69, 2)
        minimum_default_4: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_28, mul_tensor_141);  full_default_28 = mul_tensor_141 = None
        full_default_32: "f32[1]" = torch.ops.aten.full.default([1], 2.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default_5: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_32, div_tensor_69);  full_default_32 = div_tensor_69 = None
        maximum_default_6: "f64[200, 200, 25]" = torch.ops.aten.maximum.default(minimum_default_4, minimum_default_5);  minimum_default_4 = minimum_default_5 = None
        maximum_default_7: "f64[200, 200, 25]" = torch.ops.aten.maximum.default(full_default_27, maximum_default_6);  full_default_27 = maximum_default_6 = None
        sub_tensor_46: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(1.0, maximum_default_7)
        slice_tensor_234: "f64[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_default_22, 0, 2, -2);  slice_scatter_default_22 = None
        slice_tensor_235: "f64[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_tensor_234, 1, 2, -2);  slice_tensor_234 = None
        slice_tensor_236: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_235, 2, 1, -2);  slice_tensor_235 = None
        mul_tensor_142: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_236, 1.0);  slice_tensor_236 = None
        unsqueeze_default_49: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_default_50: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_49, 1);  unsqueeze_default_49 = None
        slice_tensor_237: "f64[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_default_50, 2, 0, -1);  unsqueeze_default_50 = None
        div_tensor_70: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(mul_tensor_142, slice_tensor_237);  mul_tensor_142 = slice_tensor_237 = None
        abs_default_8: "f64[200, 200, 25]" = torch.ops.aten.abs.default(div_tensor_70);  div_tensor_70 = None
        mul_tensor_143: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(abs_default_8, maximum_default_7);  abs_default_8 = maximum_default_7 = None
        add_tensor_67: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(sub_tensor_46, mul_tensor_143);  sub_tensor_46 = mul_tensor_143 = None
        mul_tensor_144: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(abs_default_6, add_tensor_67);  abs_default_6 = add_tensor_67 = None
        mul_tensor_145: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_144, mul_tensor_140);  mul_tensor_144 = mul_tensor_140 = None
        mul_tensor_146: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_145, 0.5);  mul_tensor_145 = None
        sub_tensor_47: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(mul_tensor_136, mul_tensor_146);  mul_tensor_136 = mul_tensor_146 = None
        copy_default_48: "f64[200, 200, 25]" = torch.ops.aten.copy.default(slice_tensor_185, sub_tensor_47);  slice_tensor_185 = sub_tensor_47 = None
        slice_scatter_default_26: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_182, copy_default_48, 2, 0, -1);  slice_tensor_182 = copy_default_48 = None
        slice_scatter_default_27: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_181, slice_scatter_default_26, 1, 2, -2);  slice_tensor_181 = slice_scatter_default_26 = None
        slice_scatter_default_28: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(copy_default_43, slice_scatter_default_27, 0, 2, -2);  copy_default_43 = slice_scatter_default_27 = None
        select_int_432: "f64[204, 204]" = torch.ops.aten.select.int(slice_scatter_default_28, 2, 0)
        neg_default_54: "f64[204, 204]" = torch.ops.aten.neg.default(select_int_432);  select_int_432 = None
        select_int_433: "f64[]" = torch.ops.aten.select.int(arg6_1, 0, 0)
        div_tensor_71: "f64[204, 204]" = torch.ops.aten.div.Tensor(neg_default_54, select_int_433);  neg_default_54 = select_int_433 = None
        add_tensor_68: "f64[204, 204]" = torch.ops.aten.add.Tensor(select_int_429, div_tensor_71);  select_int_429 = div_tensor_71 = None
        select_scatter_default_138: "f64[204, 204, 3]" = torch.ops.aten.select_scatter.default(select_int_427, add_tensor_68, 2, 0);  select_int_427 = add_tensor_68 = None
        select_scatter_default_139: "f64[204, 204, 26, 3]" = torch.ops.aten.select_scatter.default(slice_scatter_default_21, select_scatter_default_138, 2, 0);  slice_scatter_default_21 = select_scatter_default_138 = None
        select_int_434: "f64[204, 204, 3]" = torch.ops.aten.select.int(select_scatter_default_139, 2, 0)
        select_int_435: "f64[204, 204]" = torch.ops.aten.select.int(select_int_434, 2, 0);  select_int_434 = None
        select_int_436: "f64[204, 204, 3]" = torch.ops.aten.select.int(select_scatter_default_139, 2, 0)
        select_int_437: "f64[204, 204, 3]" = torch.ops.aten.select.int(select_scatter_default_139, 2, 0)
        select_int_438: "f64[204, 204]" = torch.ops.aten.select.int(select_int_437, 2, 0);  select_int_437 = None
        select_scatter_default_140: "f64[204, 204, 3]" = torch.ops.aten.select_scatter.default(select_int_436, select_int_438, 2, 0);  select_int_436 = select_int_438 = None
        select_scatter_default_141: "f64[204, 204, 26, 3]" = torch.ops.aten.select_scatter.default(select_scatter_default_139, select_scatter_default_140, 2, 0);  select_scatter_default_139 = select_scatter_default_140 = None
        slice_tensor_238: "f64[204, 204, 24, 3]" = torch.ops.aten.slice.Tensor(select_scatter_default_141, 2, 1, -1)
        slice_tensor_239: "f64[204, 204, 24, 3]" = torch.ops.aten.slice.Tensor(select_scatter_default_141, 2, 1, -1)
        select_int_439: "f64[204, 204, 24]" = torch.ops.aten.select.int(slice_tensor_239, 3, 0);  slice_tensor_239 = None
        slice_tensor_240: "f64[204, 204, 24]" = torch.ops.aten.slice.Tensor(slice_scatter_default_28, 2, 1, -1)
        slice_tensor_241: "f64[204, 204, 24]" = torch.ops.aten.slice.Tensor(slice_scatter_default_28, 2, 0, -2)
        sub_tensor_48: "f64[204, 204, 24]" = torch.ops.aten.sub.Tensor(slice_tensor_240, slice_tensor_241);  slice_tensor_240 = slice_tensor_241 = None
        neg_default_55: "f64[204, 204, 24]" = torch.ops.aten.neg.default(sub_tensor_48);  sub_tensor_48 = None
        slice_tensor_242: "f64[24]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 1, -1)
        div_tensor_72: "f64[204, 204, 24]" = torch.ops.aten.div.Tensor(neg_default_55, slice_tensor_242);  neg_default_55 = slice_tensor_242 = None
        add_tensor_69: "f64[204, 204, 24]" = torch.ops.aten.add.Tensor(select_int_439, div_tensor_72);  select_int_439 = div_tensor_72 = None
        select_scatter_default_142: "f64[204, 204, 24, 3]" = torch.ops.aten.select_scatter.default(slice_tensor_238, add_tensor_69, 3, 0);  slice_tensor_238 = add_tensor_69 = None
        slice_scatter_default_29: "f64[204, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(select_scatter_default_141, select_scatter_default_142, 2, 1, -1);  select_scatter_default_141 = select_scatter_default_142 = None
        slice_tensor_243: "f64[204, 204, 24, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_default_29, 2, 1, -1)
        select_int_440: "f64[204, 204, 24]" = torch.ops.aten.select.int(slice_tensor_243, 3, 0);  slice_tensor_243 = None
        slice_tensor_244: "f64[204, 204, 24, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_default_29, 2, 1, -1)
        slice_tensor_245: "f64[204, 204, 24, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_default_29, 2, 1, -1)
        select_int_441: "f64[204, 204, 24]" = torch.ops.aten.select.int(slice_tensor_245, 3, 0);  slice_tensor_245 = None
        select_scatter_default_143: "f64[204, 204, 24, 3]" = torch.ops.aten.select_scatter.default(slice_tensor_244, select_int_441, 3, 0);  slice_tensor_244 = select_int_441 = None
        slice_scatter_default_30: "f64[204, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_29, select_scatter_default_143, 2, 1, -1);  slice_scatter_default_29 = select_scatter_default_143 = None
        select_int_442: "f64[204, 204, 3]" = torch.ops.aten.select.int(slice_scatter_default_30, 2, -1)
        select_int_443: "f64[204, 204, 3]" = torch.ops.aten.select.int(slice_scatter_default_30, 2, -1)
        select_int_444: "f64[204, 204]" = torch.ops.aten.select.int(select_int_443, 2, 0);  select_int_443 = None
        select_int_445: "f64[204, 204]" = torch.ops.aten.select.int(slice_scatter_default_28, 2, -1)
        select_int_446: "f64[204, 204]" = torch.ops.aten.select.int(slice_scatter_default_28, 2, -2);  slice_scatter_default_28 = None
        sub_tensor_49: "f64[204, 204]" = torch.ops.aten.sub.Tensor(select_int_445, select_int_446);  select_int_445 = select_int_446 = None
        neg_default_56: "f64[204, 204]" = torch.ops.aten.neg.default(sub_tensor_49);  sub_tensor_49 = None
        select_int_447: "f64[]" = torch.ops.aten.select.int(arg6_1, 0, -1)
        mul_tensor_147: "f64[]" = torch.ops.aten.mul.Tensor(select_int_447, 0.5);  select_int_447 = None
        div_tensor_73: "f64[204, 204]" = torch.ops.aten.div.Tensor(neg_default_56, mul_tensor_147);  neg_default_56 = mul_tensor_147 = None
        add_tensor_70: "f64[204, 204]" = torch.ops.aten.add.Tensor(select_int_444, div_tensor_73);  select_int_444 = div_tensor_73 = None
        select_scatter_default_144: "f64[204, 204, 3]" = torch.ops.aten.select_scatter.default(select_int_442, add_tensor_70, 2, 0);  select_int_442 = add_tensor_70 = None
        select_scatter_default_145: "f64[204, 204, 26, 3]" = torch.ops.aten.select_scatter.default(slice_scatter_default_30, select_scatter_default_144, 2, -1);  slice_scatter_default_30 = select_scatter_default_144 = None
        select_int_448: "f64[204, 204, 3]" = torch.ops.aten.select.int(select_scatter_default_145, 2, -1)
        select_int_449: "f64[204, 204]" = torch.ops.aten.select.int(select_int_448, 2, 0);  select_int_448 = None
        select_int_450: "f64[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_default_13, 3, 1)
        select_int_451: "f64[204, 204, 3]" = torch.ops.aten.select.int(select_scatter_default_145, 2, -1)
        select_int_452: "f64[204, 204, 3]" = torch.ops.aten.select.int(select_scatter_default_145, 2, -1)
        select_int_453: "f64[204, 204]" = torch.ops.aten.select.int(select_int_452, 2, 0);  select_int_452 = None
        select_scatter_default_146: "f64[204, 204, 3]" = torch.ops.aten.select_scatter.default(select_int_451, select_int_453, 2, 0);  select_int_451 = select_int_453 = None
        select_scatter_default_147: "f64[204, 204, 26, 3]" = torch.ops.aten.select_scatter.default(select_scatter_default_145, select_scatter_default_146, 2, -1);  select_scatter_default_145 = select_scatter_default_146 = None
        select_int_454: "f64[204, 204, 26]" = torch.ops.aten.select.int(select_scatter_default_147, 3, 0)
        mul_tensor_148: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(select_int_454, 1.6);  select_int_454 = None
        select_int_455: "f64[204, 204, 26]" = torch.ops.aten.select.int(select_scatter_default_147, 3, 2);  select_scatter_default_147 = None
        mul_tensor_149: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(select_int_455, 0.6);  select_int_455 = None
        sub_tensor_50: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_tensor_148, mul_tensor_149);  mul_tensor_148 = mul_tensor_149 = None
        mul_tensor_150: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_tensor_50, 1.0);  sub_tensor_50 = None
        add_tensor_71: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(select_int_450, mul_tensor_150);  select_int_450 = mul_tensor_150 = None
        select_scatter_default_148: "f64[204, 204, 26, 3]" = torch.ops.aten.select_scatter.default(slice_scatter_default_13, add_tensor_71, 3, 1);  slice_scatter_default_13 = add_tensor_71 = None
        select_int_456: "f64[204, 204, 26]" = torch.ops.aten.select.int(select_scatter_default_148, 3, 1)
        select_int_457: "f64[204, 204, 26]" = torch.ops.aten.select.int(select_scatter_default_148, 3, 1)
        select_scatter_default_149: "f64[204, 204, 26, 3]" = torch.ops.aten.select_scatter.default(select_scatter_default_148, select_int_457, 3, 1);  select_scatter_default_148 = select_int_457 = None
        full_default_33: "f32[204, 204]" = torch.ops.aten.full.default([204, 204], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_246: "f32[200, 204]" = torch.ops.aten.slice.Tensor(full_default_33, 0, 2, -2)
        slice_tensor_247: "f32[200, 204]" = torch.ops.aten.slice.Tensor(full_default_33, 0, 2, -2)
        slice_tensor_248: "f32[200, 200]" = torch.ops.aten.slice.Tensor(slice_tensor_247, 1, 2, -2);  slice_tensor_247 = None
        slice_tensor_249: "f64[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_default_5, 0, 2, -2)
        slice_tensor_250: "f64[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_249, 1, 2, -2);  slice_tensor_249 = None
        select_int_458: "f64[200, 200, 3]" = torch.ops.aten.select.int(slice_tensor_250, 2, -1);  slice_tensor_250 = None
        select_int_459: "f64[200, 200]" = torch.ops.aten.select.int(select_int_458, 2, 1);  select_int_458 = None
        lt_scalar_3: "b8[200, 200]" = torch.ops.aten.lt.Scalar(select_int_459, 0.0);  select_int_459 = None
        slice_tensor_251: "f64[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_default_5, 0, 2, -2);  slice_scatter_default_5 = None
        slice_tensor_252: "f64[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_251, 1, 2, -2);  slice_tensor_251 = None
        select_int_460: "f64[200, 200, 3]" = torch.ops.aten.select.int(slice_tensor_252, 2, -1);  slice_tensor_252 = None
        select_int_461: "f64[200, 200]" = torch.ops.aten.select.int(select_int_460, 2, 1);  select_int_460 = None
        neg_default_57: "f64[200, 200]" = torch.ops.aten.neg.default(select_int_461);  select_int_461 = None
        mul_tensor_151: "f64[200, 200]" = torch.ops.aten.mul.Tensor(neg_default_57, 0.5);  neg_default_57 = None
        select_int_462: "f64[]" = torch.ops.aten.select.int(arg6_1, 0, -1);  arg6_1 = None
        mul_tensor_152: "f64[200, 200]" = torch.ops.aten.mul.Tensor(mul_tensor_151, select_int_462);  mul_tensor_151 = select_int_462 = None
        div_tensor_74: "f64[200, 200]" = torch.ops.aten.div.Tensor(mul_tensor_152, 1);  mul_tensor_152 = None
        full_default_34: "f64[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_9: "f64[200, 200]" = torch.ops.aten.where.self(lt_scalar_3, div_tensor_74, full_default_34);  lt_scalar_3 = div_tensor_74 = full_default_34 = None
        copy_default_49: "f32[200, 200]" = torch.ops.aten.copy.default(slice_tensor_248, where_self_9);  slice_tensor_248 = where_self_9 = None
        slice_scatter_default_31: "f32[200, 204]" = torch.ops.aten.slice_scatter.default(slice_tensor_246, copy_default_49, 1, 2, -2);  slice_tensor_246 = copy_default_49 = None
        slice_scatter_default_32: "f32[204, 204]" = torch.ops.aten.slice_scatter.default(full_default_33, slice_scatter_default_31, 0, 2, -2);  full_default_33 = slice_scatter_default_31 = None
        return (select_int_3, select_int_17, select_int_21, select_int_27, select_int_31, select_int_37, select_int_41, select_int_47, select_int_51, select_int_57, select_int_61, select_int_67, select_int_71, select_int_77, select_int_81, select_int_87, select_int_91, select_int_97, select_int_101, select_int_107, select_int_111, select_int_117, select_int_121, select_int_127, select_int_131, select_int_137, select_int_141, select_int_147, select_int_151, select_int_157, select_int_161, select_int_167, select_int_171, select_int_177, select_int_181, select_int_187, select_int_191, select_int_197, select_int_201, select_int_207, select_int_211, select_int_217, select_int_221, select_int_227, select_int_231, select_int_237, select_int_241, select_int_247, select_int_251, select_int_257, select_int_261, slice_tensor_40, select_int_406, slice_tensor_83, select_int_435, select_int_440, select_int_449, select_int_456, select_scatter_default_149, slice_scatter_default_32)

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
