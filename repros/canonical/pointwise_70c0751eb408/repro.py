"""
Standalone repro captured via capture_hook.
Label: torchbench_pyhpc_isoneutral_mixing_infer_000
Pattern hash: 70c0751eb408
Shape hash: d6b8a8e0
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([26], f64), T([204, 204, 26], f64), T([204, 204, 26], f64), T([204, 204, 26, 3], f64), T([26], f64), T([204, 204, 26], f64), T([204], f64), T([204], f64), T([204, 204, 26, 3], f64), T([204, 204, 26], f64), T([204, 204, 26, 2, 2], f64), T([204, 204, 26], f64), T([26], f64), T([204, 204, 26], f64), T([204], f64), T([204, 204, 26, 2, 2], f64), T([204, 204, 26], f64), T([204, 204, 26, 2, 2], f64), T([204, 204, 26, 2, 2], f64), T([204, 204, 26], f64), T([204], f64), T([204], f64), T([204], f64))"

class Repro(torch.nn.Module):
    def forward(self, arg6_1: "f64[26]", arg7_1: "f64[204, 204, 26]", arg12_1: "f64[204, 204, 26]", arg2_1: "f64[204, 204, 26, 3]", arg3_1: "f64[26]", arg4_1: "f64[204, 204, 26]", arg8_1: "f64[204]", arg9_1: "f64[204]", arg1_1: "f64[204, 204, 26, 3]", arg5_1: "f64[204, 204, 26]", arg13_1: "f64[204, 204, 26, 2, 2]", arg0_1: "f64[204, 204, 26]", arg14_1: "f64[26]", arg10_1: "f64[204, 204, 26]", arg11_1: "f64[204]", arg15_1: "f64[204, 204, 26, 2, 2]", arg16_1: "f64[204, 204, 26]", arg17_1: "f64[204, 204, 26, 2, 2]", arg19_1: "f64[204, 204, 26, 2, 2]", arg22_1: "f64[204, 204, 26]", arg20_1: "f64[204]", arg18_1: "f64[204]", arg21_1: "f64[204]"):
        # No stacktrace found for following nodes
        full_default: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor: "f64[203, 204, 26]" = torch.ops.aten.slice.Tensor(full_default, 0, 0, -1)
        full_default_1: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_1: "f64[203, 204, 26]" = torch.ops.aten.slice.Tensor(full_default_1, 0, 0, -1)
        full_default_2: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_2: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(full_default_2, 0, 1, -2)
        slice_tensor_3: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 1, 2, -2)
        slice_tensor_4: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(full_default_2, 0, 1, -2)
        slice_tensor_5: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 1, 2, -2);  slice_tensor_4 = None
        slice_tensor_6: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_5, 2, 1, 9223372036854775807);  slice_tensor_5 = None
        unsqueeze_default: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_default_1: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 1);  unsqueeze_default = None
        slice_tensor_7: "f64[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_default_1, 2, 0, 25);  unsqueeze_default_1 = None
        slice_tensor_8: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 1, -2)
        slice_tensor_9: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_8, 1, 2, -2);  slice_tensor_8 = None
        slice_tensor_10: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_9, 2, 1, 9223372036854775807);  slice_tensor_9 = None
        mul_tensor: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_7, slice_tensor_10);  slice_tensor_7 = slice_tensor_10 = None
        full_default_3: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_11: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(full_default_4, 0, 1, -2)
        slice_tensor_12: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_11, 1, 2, -2)
        slice_tensor_13: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(full_default_4, 0, 1, -2)
        slice_tensor_14: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_13, 1, 2, -2);  slice_tensor_13 = None
        slice_tensor_15: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_14, 2, 1, 9223372036854775807);  slice_tensor_14 = None
        slice_tensor_16: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 1, -2)
        slice_tensor_17: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_16, 1, 2, -2);  slice_tensor_16 = None
        slice_tensor_18: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_17, 2, 1, 9223372036854775807);  slice_tensor_17 = None
        slice_tensor_19: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 1, -2)
        slice_tensor_20: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_19, 1, 2, -2);  slice_tensor_19 = None
        slice_tensor_21: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_20, 2, 0, -1);  slice_tensor_20 = None
        add_tensor: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(slice_tensor_18, slice_tensor_21);  slice_tensor_18 = slice_tensor_21 = None
        slice_tensor_22: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -1)
        slice_tensor_23: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_22, 1, 2, -2);  slice_tensor_22 = None
        slice_tensor_24: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_23, 2, 1, 9223372036854775807);  slice_tensor_23 = None
        add_tensor_1: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(add_tensor, slice_tensor_24);  add_tensor = slice_tensor_24 = None
        slice_tensor_25: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -1)
        slice_tensor_26: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_25, 1, 2, -2);  slice_tensor_25 = None
        slice_tensor_27: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_26, 2, 0, -1);  slice_tensor_26 = None
        add_tensor_2: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(add_tensor_1, slice_tensor_27);  add_tensor_1 = slice_tensor_27 = None
        mul_tensor_1: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(add_tensor_2, 0.25);  add_tensor_2 = None
        copy_default: "f64[201, 200, 25]" = torch.ops.aten.copy.default(slice_tensor_15, mul_tensor_1);  slice_tensor_15 = mul_tensor_1 = None
        slice_scatter_default: "f64[201, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_12, copy_default, 2, 1, 9223372036854775807);  slice_tensor_12 = copy_default = None
        slice_scatter_default_1: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_11, slice_scatter_default, 1, 2, -2);  slice_tensor_11 = slice_scatter_default = None
        slice_scatter_default_2: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_default_4, slice_scatter_default_1, 0, 1, -2);  full_default_4 = slice_scatter_default_1 = None
        slice_tensor_28: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_2, 0, 1, -2)
        slice_tensor_29: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_28, 1, 2, -2)
        slice_tensor_30: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_2, 0, 1, -2)
        slice_tensor_31: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_30, 1, 2, -2);  slice_tensor_30 = None
        select_int: "f64[201, 200]" = torch.ops.aten.select.int(slice_tensor_31, 2, 0);  slice_tensor_31 = None
        slice_tensor_32: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 1, -2)
        slice_tensor_33: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_32, 1, 2, -2);  slice_tensor_32 = None
        select_int_1: "f64[201, 200]" = torch.ops.aten.select.int(slice_tensor_33, 2, 0);  slice_tensor_33 = None
        slice_tensor_34: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -1)
        slice_tensor_35: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_34, 1, 2, -2);  slice_tensor_34 = None
        select_int_2: "f64[201, 200]" = torch.ops.aten.select.int(slice_tensor_35, 2, 0);  slice_tensor_35 = None
        add_tensor_3: "f64[201, 200]" = torch.ops.aten.add.Tensor(select_int_1, select_int_2);  select_int_1 = select_int_2 = None
        mul_tensor_2: "f64[201, 200]" = torch.ops.aten.mul.Tensor(add_tensor_3, 0.5);  add_tensor_3 = None
        copy_default_1: "f64[201, 200]" = torch.ops.aten.copy.default(select_int, mul_tensor_2);  select_int = mul_tensor_2 = None
        select_scatter_default: "f64[201, 200, 26]" = torch.ops.aten.select_scatter.default(slice_tensor_29, copy_default_1, 2, 0);  slice_tensor_29 = copy_default_1 = None
        slice_scatter_default_3: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_28, select_scatter_default, 1, 2, -2);  slice_tensor_28 = select_scatter_default = None
        slice_scatter_default_4: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_2, slice_scatter_default_3, 0, 1, -2);  slice_scatter_default_2 = slice_scatter_default_3 = None
        slice_tensor_36: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_4, 0, 1, -2)
        slice_tensor_37: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_36, 1, 2, -2);  slice_tensor_36 = None
        slice_tensor_38: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_37, 2, 1, 9223372036854775807);  slice_tensor_37 = None
        select_int_3: "f64[204, 204, 26]" = torch.ops.aten.select.int(arg2_1, 3, 0)
        sub_tensor: "f64[204, 204, 26]" = torch.ops.aten.sub.Tensor(select_int_3, 9.850000000000023);  select_int_3 = None
        mul_tensor_3: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_tensor, 1e-05);  sub_tensor = None
        abs_default: "f64[26]" = torch.ops.aten.abs.default(arg3_1);  arg3_1 = None
        neg_default: "f64[26]" = torch.ops.aten.neg.default(abs_default);  abs_default = None
        sub_tensor_1: "f64[26]" = torch.ops.aten.sub.Tensor(neg_default, 0.0);  neg_default = None
        mul_tensor_4: "f64[26]" = torch.ops.aten.mul.Tensor(sub_tensor_1, 1.0790999999999999e-07);  sub_tensor_1 = None
        mul_tensor_5: "f64[26]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 1024.0);  mul_tensor_4 = None
        sub_tensor_2: "f64[26]" = torch.ops.aten.sub.Tensor(1, mul_tensor_5);  mul_tensor_5 = None
        mul_tensor_6: "f64[26]" = torch.ops.aten.mul.Tensor(sub_tensor_2, 0.000167);  sub_tensor_2 = None
        add_tensor_4: "f64[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_6);  mul_tensor_3 = mul_tensor_6 = None
        neg_default_1: "f64[204, 204, 26]" = torch.ops.aten.neg.default(add_tensor_4);  add_tensor_4 = None
        mul_tensor_7: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(neg_default_1, 1024.0);  neg_default_1 = None
        mul_tensor_8: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg4_1, mul_tensor_7);  mul_tensor_7 = None
        slice_tensor_39: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 1, -2)
        slice_tensor_40: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_39, 1, 2, -2);  slice_tensor_39 = None
        slice_tensor_41: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_40, 2, 1, 9223372036854775807);  slice_tensor_40 = None
        slice_tensor_42: "f64[203, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 0, -1)
        slice_tensor_43: "f64[203, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 1, 9223372036854775807)
        select_int_4: "f64[203, 204, 26]" = torch.ops.aten.select.int(slice_tensor_43, 3, 0);  slice_tensor_43 = None
        slice_tensor_44: "f64[203, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 0, -1)
        select_int_5: "f64[203, 204, 26]" = torch.ops.aten.select.int(slice_tensor_44, 3, 0);  slice_tensor_44 = None
        sub_tensor_3: "f64[203, 204, 26]" = torch.ops.aten.sub.Tensor(select_int_4, select_int_5);  select_int_4 = select_int_5 = None
        mul_tensor_9: "f64[203, 204, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_42, sub_tensor_3);  slice_tensor_42 = sub_tensor_3 = None
        slice_tensor_45: "f64[203]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 0, -1)
        unsqueeze_default_2: "f64[203, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_45, 1);  slice_tensor_45 = None
        unsqueeze_default_3: "f64[203, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 2);  unsqueeze_default_2 = None
        unsqueeze_default_4: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg9_1, 0)
        unsqueeze_default_5: "f64[1, 204, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        mul_tensor_10: "f64[203, 204, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_default_3, unsqueeze_default_5);  unsqueeze_default_3 = unsqueeze_default_5 = None
        div_tensor: "f64[203, 204, 26]" = torch.ops.aten.div.Tensor(mul_tensor_9, mul_tensor_10);  mul_tensor_9 = mul_tensor_10 = None
        slice_scatter_default_5: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_default, div_tensor, 0, 0, -1);  full_default = div_tensor = None
        slice_tensor_46: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_5, 0, 1, -2)
        slice_tensor_47: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_46, 1, 2, -2);  slice_tensor_46 = None
        slice_tensor_48: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_47, 2, 1, 9223372036854775807);  slice_tensor_47 = None
        mul_tensor_11: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_41, slice_tensor_48);  slice_tensor_41 = slice_tensor_48 = None
        full_default_5: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0.79872, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        mul_tensor_12: "f64[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg4_1, full_default_5);  arg4_1 = full_default_5 = None
        slice_tensor_49: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 1, -2)
        slice_tensor_50: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_49, 1, 2, -2);  slice_tensor_49 = None
        slice_tensor_51: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_50, 2, 1, 9223372036854775807);  slice_tensor_50 = None
        slice_tensor_52: "f64[203, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 0, -1)
        slice_tensor_53: "f64[203, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 1, 9223372036854775807)
        select_int_6: "f64[203, 204, 26]" = torch.ops.aten.select.int(slice_tensor_53, 3, 0);  slice_tensor_53 = None
        slice_tensor_54: "f64[203, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 0, -1)
        select_int_7: "f64[203, 204, 26]" = torch.ops.aten.select.int(slice_tensor_54, 3, 0);  slice_tensor_54 = None
        sub_tensor_4: "f64[203, 204, 26]" = torch.ops.aten.sub.Tensor(select_int_6, select_int_7);  select_int_6 = select_int_7 = None
        mul_tensor_13: "f64[203, 204, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_52, sub_tensor_4);  slice_tensor_52 = sub_tensor_4 = None
        slice_tensor_55: "f64[203]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 0, -1)
        unsqueeze_default_6: "f64[203, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_55, 1);  slice_tensor_55 = None
        unsqueeze_default_7: "f64[203, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg9_1, 0)
        unsqueeze_default_9: "f64[1, 204, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 2);  unsqueeze_default_8 = None
        mul_tensor_14: "f64[203, 204, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_default_7, unsqueeze_default_9);  unsqueeze_default_7 = unsqueeze_default_9 = None
        div_tensor_1: "f64[203, 204, 26]" = torch.ops.aten.div.Tensor(mul_tensor_13, mul_tensor_14);  mul_tensor_13 = mul_tensor_14 = None
        slice_scatter_default_6: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_default_1, div_tensor_1, 0, 0, -1);  full_default_1 = div_tensor_1 = None
        slice_tensor_56: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_6, 0, 1, -2)
        slice_tensor_57: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_56, 1, 2, -2);  slice_tensor_56 = None
        slice_tensor_58: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_57, 2, 1, 9223372036854775807);  slice_tensor_57 = None
        mul_tensor_15: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_51, slice_tensor_58);  slice_tensor_51 = slice_tensor_58 = None
        add_tensor_5: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(mul_tensor_11, mul_tensor_15);  mul_tensor_11 = mul_tensor_15 = None
        neg_default_2: "f64[201, 200, 25]" = torch.ops.aten.neg.default(add_tensor_5);  add_tensor_5 = None
        slice_tensor_59: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 1, -2)
        slice_tensor_60: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_59, 1, 2, -2);  slice_tensor_59 = None
        slice_tensor_61: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_60, 2, 1, 9223372036854775807);  slice_tensor_60 = None
        full_default_6: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_62: "f64[204, 204, 25]" = torch.ops.aten.slice.Tensor(full_default_6, 2, 0, -1)
        slice_tensor_63: "f64[204, 204, 25]" = torch.ops.aten.slice.Tensor(arg5_1, 2, 0, -1)
        slice_tensor_64: "f64[204, 204, 25, 3]" = torch.ops.aten.slice.Tensor(arg2_1, 2, 1, 9223372036854775807)
        select_int_8: "f64[204, 204, 25]" = torch.ops.aten.select.int(slice_tensor_64, 3, 0);  slice_tensor_64 = None
        slice_tensor_65: "f64[204, 204, 25, 3]" = torch.ops.aten.slice.Tensor(arg2_1, 2, 0, -1)
        select_int_9: "f64[204, 204, 25]" = torch.ops.aten.select.int(slice_tensor_65, 3, 0);  slice_tensor_65 = None
        sub_tensor_5: "f64[204, 204, 25]" = torch.ops.aten.sub.Tensor(select_int_8, select_int_9);  select_int_8 = select_int_9 = None
        mul_tensor_16: "f64[204, 204, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_63, sub_tensor_5);  slice_tensor_63 = sub_tensor_5 = None
        unsqueeze_default_10: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_default_11: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 1);  unsqueeze_default_10 = None
        slice_tensor_66: "f64[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_default_11, 2, 0, -1);  unsqueeze_default_11 = None
        div_tensor_2: "f64[204, 204, 25]" = torch.ops.aten.div.Tensor(mul_tensor_16, slice_tensor_66);  mul_tensor_16 = slice_tensor_66 = None
        copy_default_2: "f64[204, 204, 25]" = torch.ops.aten.copy.default(slice_tensor_62, div_tensor_2);  slice_tensor_62 = div_tensor_2 = None
        slice_scatter_default_7: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_default_6, copy_default_2, 2, 0, -1);  full_default_6 = copy_default_2 = None
        slice_tensor_67: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, 1, -2)
        slice_tensor_68: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_67, 1, 2, -2);  slice_tensor_67 = None
        slice_tensor_69: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_68, 2, 0, 25);  slice_tensor_68 = None
        mul_tensor_17: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_61, slice_tensor_69);  slice_tensor_61 = slice_tensor_69 = None
        slice_tensor_70: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 1, -2)
        slice_tensor_71: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_70, 1, 2, -2);  slice_tensor_70 = None
        slice_tensor_72: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_71, 2, 1, 9223372036854775807);  slice_tensor_71 = None
        full_default_7: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_73: "f64[204, 204, 25]" = torch.ops.aten.slice.Tensor(full_default_7, 2, 0, -1)
        slice_tensor_74: "f64[204, 204, 25]" = torch.ops.aten.slice.Tensor(arg5_1, 2, 0, -1)
        slice_tensor_75: "f64[204, 204, 25, 3]" = torch.ops.aten.slice.Tensor(arg1_1, 2, 1, 9223372036854775807)
        select_int_10: "f64[204, 204, 25]" = torch.ops.aten.select.int(slice_tensor_75, 3, 0);  slice_tensor_75 = None
        slice_tensor_76: "f64[204, 204, 25, 3]" = torch.ops.aten.slice.Tensor(arg1_1, 2, 0, -1)
        select_int_11: "f64[204, 204, 25]" = torch.ops.aten.select.int(slice_tensor_76, 3, 0);  slice_tensor_76 = None
        sub_tensor_6: "f64[204, 204, 25]" = torch.ops.aten.sub.Tensor(select_int_10, select_int_11);  select_int_10 = select_int_11 = None
        mul_tensor_18: "f64[204, 204, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_74, sub_tensor_6);  slice_tensor_74 = sub_tensor_6 = None
        unsqueeze_default_12: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_default_13: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 1);  unsqueeze_default_12 = None
        slice_tensor_77: "f64[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_default_13, 2, 0, -1);  unsqueeze_default_13 = None
        div_tensor_3: "f64[204, 204, 25]" = torch.ops.aten.div.Tensor(mul_tensor_18, slice_tensor_77);  mul_tensor_18 = slice_tensor_77 = None
        copy_default_3: "f64[204, 204, 25]" = torch.ops.aten.copy.default(slice_tensor_73, div_tensor_3);  slice_tensor_73 = div_tensor_3 = None
        slice_scatter_default_8: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_default_7, copy_default_3, 2, 0, -1);  full_default_7 = copy_default_3 = None
        slice_tensor_78: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_8, 0, 1, -2)
        slice_tensor_79: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_78, 1, 2, -2);  slice_tensor_78 = None
        slice_tensor_80: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_79, 2, 0, 25);  slice_tensor_79 = None
        mul_tensor_19: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_72, slice_tensor_80);  slice_tensor_72 = slice_tensor_80 = None
        add_tensor_6: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(mul_tensor_17, mul_tensor_19);  mul_tensor_17 = mul_tensor_19 = None
        full_default_8: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default: "f64[201, 200, 25]" = torch.ops.aten.minimum.default(add_tensor_6, full_default_8);  add_tensor_6 = full_default_8 = None
        sub_tensor_7: "f64[201, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_default, 1e-20);  minimum_default = None
        div_tensor_4: "f64[201, 200, 25]" = torch.ops.aten.div.Tensor(neg_default_2, sub_tensor_7);  neg_default_2 = sub_tensor_7 = None
        abs_default_1: "f64[201, 200, 25]" = torch.ops.aten.abs.default(div_tensor_4)
        neg_default_3: "f64[201, 200, 25]" = torch.ops.aten.neg.default(abs_default_1);  abs_default_1 = None
        add_tensor_7: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(neg_default_3, 0.001);  neg_default_3 = None
        div_tensor_5: "f64[201, 200, 25]" = torch.ops.aten.div.Tensor(add_tensor_7, 0.001);  add_tensor_7 = None
        tanh_default: "f64[201, 200, 25]" = torch.ops.aten.tanh.default(div_tensor_5);  div_tensor_5 = None
        add_tensor_8: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(tanh_default, 1.0);  tanh_default = None
        mul_tensor_20: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(add_tensor_8, 0.5);  add_tensor_8 = None
        mul_tensor_21: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_38, mul_tensor_20);  slice_tensor_38 = None
        maximum_default: "f64[201, 200, 25]" = torch.ops.aten.maximum.default(full_default_3, mul_tensor_21);  full_default_3 = mul_tensor_21 = None
        mul_tensor_22: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor, maximum_default);  mul_tensor = maximum_default = None
        add_tensor_9: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(slice_tensor_6, mul_tensor_22);  slice_tensor_6 = mul_tensor_22 = None
        slice_scatter_default_9: "f64[201, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_3, add_tensor_9, 2, 1, 9223372036854775807);  slice_tensor_3 = add_tensor_9 = None
        slice_scatter_default_10: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_2, slice_scatter_default_9, 1, 2, -2);  slice_tensor_2 = slice_scatter_default_9 = None
        slice_scatter_default_11: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_default_10, 0, 1, -2);  full_default_2 = slice_scatter_default_10 = None
        slice_tensor_81: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_11, 0, 1, -2)
        slice_tensor_82: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_81, 1, 2, -2);  slice_tensor_81 = None
        slice_tensor_83: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_82, 2, 1, 9223372036854775807);  slice_tensor_82 = None
        slice_tensor_84: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_11, 0, 1, -2)
        slice_tensor_85: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_84, 1, 2, -2)
        slice_tensor_86: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_11, 0, 1, -2)
        slice_tensor_87: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_86, 1, 2, -2);  slice_tensor_86 = None
        slice_tensor_88: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_87, 2, 1, 9223372036854775807);  slice_tensor_87 = None
        slice_scatter_default_12: "f64[201, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_85, slice_tensor_88, 2, 1, 9223372036854775807);  slice_tensor_85 = slice_tensor_88 = None
        slice_scatter_default_13: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_84, slice_scatter_default_12, 1, 2, -2);  slice_tensor_84 = slice_scatter_default_12 = None
        slice_scatter_default_14: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_11, slice_scatter_default_13, 0, 1, -2);  slice_scatter_default_11 = slice_scatter_default_13 = None
        slice_tensor_89: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_14, 0, 1, -2)
        slice_tensor_90: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_89, 1, 2, -2)
        slice_tensor_91: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_14, 0, 1, -2)
        slice_tensor_92: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_91, 1, 2, -2);  slice_tensor_91 = None
        slice_tensor_93: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_92, 2, 1, 9223372036854775807);  slice_tensor_92 = None
        unsqueeze_default_14: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_default_15: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 1);  unsqueeze_default_14 = None
        slice_tensor_94: "f64[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_default_15, 2, 0, 25);  unsqueeze_default_15 = None
        slice_tensor_95: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 1, -2)
        slice_tensor_96: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_95, 1, 2, -2);  slice_tensor_95 = None
        slice_tensor_97: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_96, 2, 1, 9223372036854775807);  slice_tensor_96 = None
        mul_tensor_23: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_94, slice_tensor_97);  slice_tensor_94 = slice_tensor_97 = None
        full_default_9: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_98: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_4, 0, 1, -2)
        slice_tensor_99: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_98, 1, 2, -2);  slice_tensor_98 = None
        slice_tensor_100: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_99, 2, 1, 9223372036854775807);  slice_tensor_99 = None
        slice_tensor_101: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -1)
        slice_tensor_102: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_101, 1, 2, -2);  slice_tensor_101 = None
        slice_tensor_103: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_102, 2, 1, 9223372036854775807);  slice_tensor_102 = None
        slice_tensor_104: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_5, 0, 1, -2)
        slice_tensor_105: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_104, 1, 2, -2);  slice_tensor_104 = None
        slice_tensor_106: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_105, 2, 1, 9223372036854775807);  slice_tensor_105 = None
        mul_tensor_24: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_103, slice_tensor_106);  slice_tensor_103 = slice_tensor_106 = None
        slice_tensor_107: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -1)
        slice_tensor_108: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_107, 1, 2, -2);  slice_tensor_107 = None
        slice_tensor_109: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_108, 2, 1, 9223372036854775807);  slice_tensor_108 = None
        slice_tensor_110: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_6, 0, 1, -2)
        slice_tensor_111: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_110, 1, 2, -2);  slice_tensor_110 = None
        slice_tensor_112: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_111, 2, 1, 9223372036854775807);  slice_tensor_111 = None
        mul_tensor_25: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_109, slice_tensor_112);  slice_tensor_109 = slice_tensor_112 = None
        add_tensor_10: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(mul_tensor_24, mul_tensor_25);  mul_tensor_24 = mul_tensor_25 = None
        neg_default_4: "f64[201, 200, 25]" = torch.ops.aten.neg.default(add_tensor_10);  add_tensor_10 = None
        slice_tensor_113: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -1)
        slice_tensor_114: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_113, 1, 2, -2);  slice_tensor_113 = None
        slice_tensor_115: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_114, 2, 1, 9223372036854775807);  slice_tensor_114 = None
        slice_tensor_116: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, 2, -1)
        slice_tensor_117: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_116, 1, 2, -2);  slice_tensor_116 = None
        slice_tensor_118: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_117, 2, 0, 25);  slice_tensor_117 = None
        mul_tensor_26: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_115, slice_tensor_118);  slice_tensor_115 = slice_tensor_118 = None
        slice_tensor_119: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -1)
        slice_tensor_120: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_119, 1, 2, -2);  slice_tensor_119 = None
        slice_tensor_121: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_120, 2, 1, 9223372036854775807);  slice_tensor_120 = None
        slice_tensor_122: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_8, 0, 2, -1)
        slice_tensor_123: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_122, 1, 2, -2);  slice_tensor_122 = None
        slice_tensor_124: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_123, 2, 0, 25);  slice_tensor_123 = None
        mul_tensor_27: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_121, slice_tensor_124);  slice_tensor_121 = slice_tensor_124 = None
        add_tensor_11: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(mul_tensor_26, mul_tensor_27);  mul_tensor_26 = mul_tensor_27 = None
        full_default_10: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default_1: "f64[201, 200, 25]" = torch.ops.aten.minimum.default(add_tensor_11, full_default_10);  add_tensor_11 = full_default_10 = None
        sub_tensor_8: "f64[201, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_default_1, 1e-20);  minimum_default_1 = None
        div_tensor_6: "f64[201, 200, 25]" = torch.ops.aten.div.Tensor(neg_default_4, sub_tensor_8);  neg_default_4 = sub_tensor_8 = None
        abs_default_2: "f64[201, 200, 25]" = torch.ops.aten.abs.default(div_tensor_6)
        neg_default_5: "f64[201, 200, 25]" = torch.ops.aten.neg.default(abs_default_2);  abs_default_2 = None
        add_tensor_12: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(neg_default_5, 0.001);  neg_default_5 = None
        div_tensor_7: "f64[201, 200, 25]" = torch.ops.aten.div.Tensor(add_tensor_12, 0.001);  add_tensor_12 = None
        tanh_default_1: "f64[201, 200, 25]" = torch.ops.aten.tanh.default(div_tensor_7);  div_tensor_7 = None
        add_tensor_13: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(tanh_default_1, 1.0);  tanh_default_1 = None
        mul_tensor_28: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(add_tensor_13, 0.5);  add_tensor_13 = None
        mul_tensor_29: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_100, mul_tensor_28);  slice_tensor_100 = None
        maximum_default_1: "f64[201, 200, 25]" = torch.ops.aten.maximum.default(full_default_9, mul_tensor_29);  full_default_9 = mul_tensor_29 = None
        mul_tensor_30: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_23, maximum_default_1);  mul_tensor_23 = maximum_default_1 = None
        add_tensor_14: "f64[201, 200, 25]" = torch.ops.aten.add.Tensor(slice_tensor_93, mul_tensor_30);  slice_tensor_93 = mul_tensor_30 = None
        slice_scatter_default_15: "f64[201, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_90, add_tensor_14, 2, 1, 9223372036854775807);  slice_tensor_90 = add_tensor_14 = None
        slice_scatter_default_16: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_89, slice_scatter_default_15, 1, 2, -2);  slice_tensor_89 = slice_scatter_default_15 = None
        slice_scatter_default_17: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_14, slice_scatter_default_16, 0, 1, -2);  slice_scatter_default_14 = slice_scatter_default_16 = None
        slice_tensor_125: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_17, 0, 1, -2)
        slice_tensor_126: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_125, 1, 2, -2);  slice_tensor_125 = None
        slice_tensor_127: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_126, 2, 1, 9223372036854775807);  slice_tensor_126 = None
        slice_tensor_128: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_17, 0, 1, -2)
        slice_tensor_129: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_128, 1, 2, -2)
        slice_tensor_130: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_17, 0, 1, -2)
        slice_tensor_131: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_130, 1, 2, -2);  slice_tensor_130 = None
        slice_tensor_132: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_131, 2, 1, 9223372036854775807);  slice_tensor_131 = None
        slice_scatter_default_18: "f64[201, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_129, slice_tensor_132, 2, 1, 9223372036854775807);  slice_tensor_129 = slice_tensor_132 = None
        slice_scatter_default_19: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_128, slice_scatter_default_18, 1, 2, -2);  slice_tensor_128 = slice_scatter_default_18 = None
        slice_scatter_default_20: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_17, slice_scatter_default_19, 0, 1, -2);  slice_scatter_default_17 = slice_scatter_default_19 = None
        slice_tensor_133: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_20, 0, 1, -2)
        slice_tensor_134: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_20, 0, 1, -2)
        slice_tensor_135: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_134, 1, 2, -2);  slice_tensor_134 = None
        unsqueeze_default_16: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_default_17: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 1);  unsqueeze_default_16 = None
        slice_tensor_136: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 1, -2)
        slice_tensor_137: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_136, 1, 2, -2);  slice_tensor_136 = None
        mul_tensor_31: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_default_17, slice_tensor_137);  unsqueeze_default_17 = slice_tensor_137 = None
        full_default_11: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_138: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_4, 0, 1, -2)
        slice_tensor_139: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_138, 1, 2, -2);  slice_tensor_138 = None
        slice_tensor_140: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 1, -2)
        slice_tensor_141: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_140, 1, 2, -2);  slice_tensor_140 = None
        slice_tensor_142: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_5, 0, 1, -2)
        slice_tensor_143: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_142, 1, 2, -2);  slice_tensor_142 = None
        mul_tensor_32: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_141, slice_tensor_143);  slice_tensor_141 = slice_tensor_143 = None
        slice_tensor_144: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 1, -2)
        slice_tensor_145: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_144, 1, 2, -2);  slice_tensor_144 = None
        slice_tensor_146: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_6, 0, 1, -2)
        slice_tensor_147: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_146, 1, 2, -2);  slice_tensor_146 = None
        mul_tensor_33: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_145, slice_tensor_147);  slice_tensor_145 = slice_tensor_147 = None
        add_tensor_15: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(mul_tensor_32, mul_tensor_33);  mul_tensor_32 = mul_tensor_33 = None
        neg_default_6: "f64[201, 200, 26]" = torch.ops.aten.neg.default(add_tensor_15);  add_tensor_15 = None
        slice_tensor_148: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 1, -2)
        slice_tensor_149: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_148, 1, 2, -2);  slice_tensor_148 = None
        slice_tensor_150: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, 1, -2)
        slice_tensor_151: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_150, 1, 2, -2);  slice_tensor_150 = None
        mul_tensor_34: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_149, slice_tensor_151);  slice_tensor_149 = slice_tensor_151 = None
        slice_tensor_152: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 1, -2)
        slice_tensor_153: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_152, 1, 2, -2);  slice_tensor_152 = None
        slice_tensor_154: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_8, 0, 1, -2)
        slice_tensor_155: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_154, 1, 2, -2);  slice_tensor_154 = None
        mul_tensor_35: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_153, slice_tensor_155);  slice_tensor_153 = slice_tensor_155 = None
        add_tensor_16: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(mul_tensor_34, mul_tensor_35);  mul_tensor_34 = mul_tensor_35 = None
        full_default_12: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default_2: "f64[201, 200, 26]" = torch.ops.aten.minimum.default(add_tensor_16, full_default_12);  add_tensor_16 = full_default_12 = None
        sub_tensor_9: "f64[201, 200, 26]" = torch.ops.aten.sub.Tensor(minimum_default_2, 1e-20);  minimum_default_2 = None
        div_tensor_8: "f64[201, 200, 26]" = torch.ops.aten.div.Tensor(neg_default_6, sub_tensor_9);  neg_default_6 = sub_tensor_9 = None
        abs_default_3: "f64[201, 200, 26]" = torch.ops.aten.abs.default(div_tensor_8)
        neg_default_7: "f64[201, 200, 26]" = torch.ops.aten.neg.default(abs_default_3);  abs_default_3 = None
        add_tensor_17: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(neg_default_7, 0.001);  neg_default_7 = None
        div_tensor_9: "f64[201, 200, 26]" = torch.ops.aten.div.Tensor(add_tensor_17, 0.001);  add_tensor_17 = None
        tanh_default_2: "f64[201, 200, 26]" = torch.ops.aten.tanh.default(div_tensor_9);  div_tensor_9 = None
        add_tensor_18: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(tanh_default_2, 1.0);  tanh_default_2 = None
        mul_tensor_36: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(add_tensor_18, 0.5);  add_tensor_18 = None
        mul_tensor_37: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_139, mul_tensor_36);  slice_tensor_139 = None
        maximum_default_2: "f64[201, 200, 26]" = torch.ops.aten.maximum.default(full_default_11, mul_tensor_37);  full_default_11 = mul_tensor_37 = None
        mul_tensor_38: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_31, maximum_default_2);  mul_tensor_31 = maximum_default_2 = None
        add_tensor_19: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(slice_tensor_135, mul_tensor_38);  slice_tensor_135 = mul_tensor_38 = None
        slice_scatter_default_21: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_133, add_tensor_19, 1, 2, -2);  slice_tensor_133 = add_tensor_19 = None
        slice_scatter_default_22: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_20, slice_scatter_default_21, 0, 1, -2);  slice_scatter_default_20 = slice_scatter_default_21 = None
        slice_tensor_156: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_22, 0, 1, -2)
        slice_tensor_157: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_156, 1, 2, -2);  slice_tensor_156 = None
        slice_tensor_158: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_22, 0, 1, -2)
        slice_tensor_159: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_22, 0, 1, -2)
        slice_tensor_160: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_159, 1, 2, -2);  slice_tensor_159 = None
        slice_scatter_default_23: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_158, slice_tensor_160, 1, 2, -2);  slice_tensor_158 = slice_tensor_160 = None
        slice_scatter_default_24: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_22, slice_scatter_default_23, 0, 1, -2);  slice_scatter_default_22 = slice_scatter_default_23 = None
        slice_tensor_161: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_24, 0, 1, -2)
        slice_tensor_162: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_24, 0, 1, -2)
        slice_tensor_163: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_162, 1, 2, -2);  slice_tensor_162 = None
        unsqueeze_default_18: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_default_19: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, 1);  unsqueeze_default_18 = None
        slice_tensor_164: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 1, -2)
        slice_tensor_165: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_164, 1, 2, -2);  slice_tensor_164 = None
        mul_tensor_39: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_default_19, slice_tensor_165);  unsqueeze_default_19 = slice_tensor_165 = None
        full_default_13: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_166: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_4, 0, 1, -2)
        slice_tensor_167: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_166, 1, 2, -2);  slice_tensor_166 = None
        slice_tensor_168: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -1)
        slice_tensor_169: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_168, 1, 2, -2);  slice_tensor_168 = None
        slice_tensor_170: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_5, 0, 1, -2)
        slice_tensor_171: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_170, 1, 2, -2);  slice_tensor_170 = None
        mul_tensor_40: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_169, slice_tensor_171);  slice_tensor_169 = slice_tensor_171 = None
        slice_tensor_172: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -1)
        slice_tensor_173: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_172, 1, 2, -2);  slice_tensor_172 = None
        slice_tensor_174: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_6, 0, 1, -2)
        slice_tensor_175: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_174, 1, 2, -2);  slice_tensor_174 = None
        mul_tensor_41: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_173, slice_tensor_175);  slice_tensor_173 = slice_tensor_175 = None
        add_tensor_20: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(mul_tensor_40, mul_tensor_41);  mul_tensor_40 = mul_tensor_41 = None
        neg_default_8: "f64[201, 200, 26]" = torch.ops.aten.neg.default(add_tensor_20);  add_tensor_20 = None
        slice_tensor_176: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -1)
        slice_tensor_177: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_176, 1, 2, -2);  slice_tensor_176 = None
        slice_tensor_178: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, 2, -1)
        slice_tensor_179: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_178, 1, 2, -2);  slice_tensor_178 = None
        mul_tensor_42: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_177, slice_tensor_179);  slice_tensor_177 = slice_tensor_179 = None
        slice_tensor_180: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -1)
        slice_tensor_181: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_180, 1, 2, -2);  slice_tensor_180 = None
        slice_tensor_182: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_8, 0, 2, -1)
        slice_tensor_183: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_182, 1, 2, -2);  slice_tensor_182 = None
        mul_tensor_43: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_181, slice_tensor_183);  slice_tensor_181 = slice_tensor_183 = None
        add_tensor_21: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(mul_tensor_42, mul_tensor_43);  mul_tensor_42 = mul_tensor_43 = None
        full_default_14: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default_3: "f64[201, 200, 26]" = torch.ops.aten.minimum.default(add_tensor_21, full_default_14);  add_tensor_21 = full_default_14 = None
        sub_tensor_10: "f64[201, 200, 26]" = torch.ops.aten.sub.Tensor(minimum_default_3, 1e-20);  minimum_default_3 = None
        div_tensor_10: "f64[201, 200, 26]" = torch.ops.aten.div.Tensor(neg_default_8, sub_tensor_10);  neg_default_8 = sub_tensor_10 = None
        abs_default_4: "f64[201, 200, 26]" = torch.ops.aten.abs.default(div_tensor_10)
        neg_default_9: "f64[201, 200, 26]" = torch.ops.aten.neg.default(abs_default_4);  abs_default_4 = None
        add_tensor_22: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(neg_default_9, 0.001);  neg_default_9 = None
        div_tensor_11: "f64[201, 200, 26]" = torch.ops.aten.div.Tensor(add_tensor_22, 0.001);  add_tensor_22 = None
        tanh_default_3: "f64[201, 200, 26]" = torch.ops.aten.tanh.default(div_tensor_11);  div_tensor_11 = None
        add_tensor_23: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(tanh_default_3, 1.0);  tanh_default_3 = None
        mul_tensor_44: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(add_tensor_23, 0.5);  add_tensor_23 = None
        mul_tensor_45: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_167, mul_tensor_44);  slice_tensor_167 = None
        maximum_default_3: "f64[201, 200, 26]" = torch.ops.aten.maximum.default(full_default_13, mul_tensor_45);  full_default_13 = mul_tensor_45 = None
        mul_tensor_46: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_39, maximum_default_3);  mul_tensor_39 = maximum_default_3 = None
        add_tensor_24: "f64[201, 200, 26]" = torch.ops.aten.add.Tensor(slice_tensor_163, mul_tensor_46);  slice_tensor_163 = mul_tensor_46 = None
        slice_scatter_default_25: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_161, add_tensor_24, 1, 2, -2);  slice_tensor_161 = add_tensor_24 = None
        slice_scatter_default_26: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_24, slice_scatter_default_25, 0, 1, -2);  slice_scatter_default_24 = slice_scatter_default_25 = None
        slice_tensor_184: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_26, 0, 1, -2)
        slice_tensor_185: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_184, 1, 2, -2);  slice_tensor_184 = None
        slice_tensor_186: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg13_1, 0, 1, -2)
        slice_tensor_187: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_186, 1, 2, -2)
        slice_tensor_188: "f64[201, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_187, 2, 1, 9223372036854775807)
        select_int_12: "f64[201, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_188, 3, 0)
        slice_tensor_189: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg13_1, 0, 1, -2)
        slice_tensor_190: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_189, 1, 2, -2);  slice_tensor_189 = None
        slice_tensor_191: "f64[201, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_190, 2, 1, 9223372036854775807);  slice_tensor_190 = None
        select_int_13: "f64[201, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_191, 3, 0);  slice_tensor_191 = None
        select_int_14: "f64[201, 200, 25]" = torch.ops.aten.select.int(select_int_13, 3, 0);  select_int_13 = None
        mul_tensor_47: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_20, div_tensor_4);  mul_tensor_20 = div_tensor_4 = None
        slice_tensor_192: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 1, -2)
        slice_tensor_193: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_192, 1, 2, -2);  slice_tensor_192 = None
        slice_tensor_194: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_193, 2, 1, 9223372036854775807);  slice_tensor_193 = None
        mul_tensor_48: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_47, slice_tensor_194);  mul_tensor_47 = slice_tensor_194 = None
        copy_default_4: "f64[201, 200, 25]" = torch.ops.aten.copy.default(select_int_14, mul_tensor_48);  select_int_14 = mul_tensor_48 = None
        select_scatter_default_1: "f64[201, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_int_12, copy_default_4, 3, 0);  select_int_12 = copy_default_4 = None
        select_scatter_default_2: "f64[201, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_tensor_188, select_scatter_default_1, 3, 0);  slice_tensor_188 = select_scatter_default_1 = None
        slice_scatter_default_27: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_187, select_scatter_default_2, 2, 1, 9223372036854775807);  slice_tensor_187 = select_scatter_default_2 = None
        slice_scatter_default_28: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_186, slice_scatter_default_27, 1, 2, -2);  slice_tensor_186 = slice_scatter_default_27 = None
        slice_scatter_default_29: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(arg13_1, slice_scatter_default_28, 0, 1, -2);  slice_scatter_default_28 = None
        slice_tensor_195: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_29, 0, 1, -2)
        slice_tensor_196: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_195, 1, 2, -2)
        slice_tensor_197: "f64[201, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_196, 2, 1, 9223372036854775807)
        select_int_15: "f64[201, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_197, 3, 1)
        slice_tensor_198: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_29, 0, 1, -2)
        slice_tensor_199: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_198, 1, 2, -2);  slice_tensor_198 = None
        slice_tensor_200: "f64[201, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_199, 2, 1, 9223372036854775807);  slice_tensor_199 = None
        select_int_16: "f64[201, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_200, 3, 1);  slice_tensor_200 = None
        select_int_17: "f64[201, 200, 25]" = torch.ops.aten.select.int(select_int_16, 3, 0);  select_int_16 = None
        mul_tensor_49: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_28, div_tensor_6);  mul_tensor_28 = div_tensor_6 = None
        slice_tensor_201: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 1, -2)
        slice_tensor_202: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_201, 1, 2, -2);  slice_tensor_201 = None
        slice_tensor_203: "f64[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_202, 2, 1, 9223372036854775807);  slice_tensor_202 = None
        mul_tensor_50: "f64[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_49, slice_tensor_203);  mul_tensor_49 = slice_tensor_203 = None
        copy_default_5: "f64[201, 200, 25]" = torch.ops.aten.copy.default(select_int_17, mul_tensor_50);  select_int_17 = mul_tensor_50 = None
        select_scatter_default_3: "f64[201, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_int_15, copy_default_5, 3, 0);  select_int_15 = copy_default_5 = None
        select_scatter_default_4: "f64[201, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_tensor_197, select_scatter_default_3, 3, 1);  slice_tensor_197 = select_scatter_default_3 = None
        slice_scatter_default_30: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_196, select_scatter_default_4, 2, 1, 9223372036854775807);  slice_tensor_196 = select_scatter_default_4 = None
        slice_scatter_default_31: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_195, slice_scatter_default_30, 1, 2, -2);  slice_tensor_195 = slice_scatter_default_30 = None
        slice_scatter_default_32: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_29, slice_scatter_default_31, 0, 1, -2);  slice_scatter_default_29 = slice_scatter_default_31 = None
        slice_tensor_204: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_32, 0, 1, -2)
        slice_tensor_205: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_204, 1, 2, -2)
        select_int_18: "f64[201, 200, 26, 2]" = torch.ops.aten.select.int(slice_tensor_205, 3, 0)
        slice_tensor_206: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_32, 0, 1, -2)
        slice_tensor_207: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_206, 1, 2, -2);  slice_tensor_206 = None
        select_int_19: "f64[201, 200, 26, 2]" = torch.ops.aten.select.int(slice_tensor_207, 3, 0);  slice_tensor_207 = None
        select_int_20: "f64[201, 200, 26]" = torch.ops.aten.select.int(select_int_19, 3, 1);  select_int_19 = None
        mul_tensor_51: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_36, div_tensor_8);  mul_tensor_36 = div_tensor_8 = None
        slice_tensor_208: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 1, -2)
        slice_tensor_209: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_208, 1, 2, -2);  slice_tensor_208 = None
        mul_tensor_52: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_51, slice_tensor_209);  mul_tensor_51 = slice_tensor_209 = None
        copy_default_6: "f64[201, 200, 26]" = torch.ops.aten.copy.default(select_int_20, mul_tensor_52);  select_int_20 = mul_tensor_52 = None
        select_scatter_default_5: "f64[201, 200, 26, 2]" = torch.ops.aten.select_scatter.default(select_int_18, copy_default_6, 3, 1);  select_int_18 = copy_default_6 = None
        select_scatter_default_6: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.select_scatter.default(slice_tensor_205, select_scatter_default_5, 3, 0);  slice_tensor_205 = select_scatter_default_5 = None
        slice_scatter_default_33: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_204, select_scatter_default_6, 1, 2, -2);  slice_tensor_204 = select_scatter_default_6 = None
        slice_scatter_default_34: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_32, slice_scatter_default_33, 0, 1, -2);  slice_scatter_default_32 = slice_scatter_default_33 = None
        slice_tensor_210: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_34, 0, 1, -2)
        slice_tensor_211: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_210, 1, 2, -2)
        select_int_21: "f64[201, 200, 26, 2]" = torch.ops.aten.select.int(slice_tensor_211, 3, 1)
        slice_tensor_212: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_34, 0, 1, -2)
        slice_tensor_213: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_212, 1, 2, -2);  slice_tensor_212 = None
        select_int_22: "f64[201, 200, 26, 2]" = torch.ops.aten.select.int(slice_tensor_213, 3, 1);  slice_tensor_213 = None
        select_int_23: "f64[201, 200, 26]" = torch.ops.aten.select.int(select_int_22, 3, 1);  select_int_22 = None
        mul_tensor_53: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_44, div_tensor_10);  mul_tensor_44 = div_tensor_10 = None
        slice_tensor_214: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 1, -2);  arg7_1 = None
        slice_tensor_215: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_214, 1, 2, -2);  slice_tensor_214 = None
        mul_tensor_54: "f64[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_53, slice_tensor_215);  mul_tensor_53 = slice_tensor_215 = None
        copy_default_7: "f64[201, 200, 26]" = torch.ops.aten.copy.default(select_int_23, mul_tensor_54);  select_int_23 = mul_tensor_54 = None
        select_scatter_default_7: "f64[201, 200, 26, 2]" = torch.ops.aten.select_scatter.default(select_int_21, copy_default_7, 3, 1);  select_int_21 = copy_default_7 = None
        select_scatter_default_8: "f64[201, 200, 26, 2, 2]" = torch.ops.aten.select_scatter.default(slice_tensor_211, select_scatter_default_7, 3, 1);  slice_tensor_211 = select_scatter_default_7 = None
        slice_scatter_default_35: "f64[201, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_210, select_scatter_default_8, 1, 2, -2);  slice_tensor_210 = select_scatter_default_8 = None
        slice_scatter_default_36: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_34, slice_scatter_default_35, 0, 1, -2);  slice_scatter_default_34 = slice_scatter_default_35 = None
        slice_tensor_216: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 1, -2)
        slice_tensor_217: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 1, -2)
        slice_tensor_218: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_217, 1, 2, -2);  slice_tensor_217 = None
        slice_tensor_219: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_26, 0, 1, -2)
        slice_tensor_220: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_26, 0, 1, -2)
        slice_tensor_221: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_220, 1, 2, -2);  slice_tensor_220 = None
        slice_scatter_default_37: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_219, slice_tensor_221, 1, 2, -2);  slice_tensor_219 = slice_tensor_221 = None
        slice_scatter_default_38: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_26, slice_scatter_default_37, 0, 1, -2);  slice_scatter_default_26 = slice_scatter_default_37 = None
        slice_tensor_222: "f64[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_38, 0, 1, -2);  slice_scatter_default_38 = None
        slice_tensor_223: "f64[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_222, 1, 2, -2);  slice_tensor_222 = None
        unsqueeze_default_20: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg14_1, 0)
        unsqueeze_default_21: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 1);  unsqueeze_default_20 = None
        mul_tensor_55: "f64[1, 1, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_default_21, 4.0);  unsqueeze_default_21 = None
        div_tensor_12: "f64[201, 200, 26]" = torch.ops.aten.div.Tensor(slice_tensor_223, mul_tensor_55);  slice_tensor_223 = mul_tensor_55 = None
        copy_default_8: "f64[201, 200, 26]" = torch.ops.aten.copy.default(slice_tensor_218, div_tensor_12);  slice_tensor_218 = div_tensor_12 = None
        slice_scatter_default_39: "f64[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_216, copy_default_8, 1, 2, -2);  slice_tensor_216 = copy_default_8 = None
        slice_scatter_default_40: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(arg0_1, slice_scatter_default_39, 0, 1, -2);  slice_scatter_default_39 = None
        full_default_15: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_224: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_default_15, 0, 2, -2)
        slice_tensor_225: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_224, 1, 1, -2)
        slice_tensor_226: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_default_15, 0, 2, -2)
        slice_tensor_227: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_226, 1, 1, -2);  slice_tensor_226 = None
        slice_tensor_228: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_227, 2, 1, 9223372036854775807);  slice_tensor_227 = None
        unsqueeze_default_22: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_default_23: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 1);  unsqueeze_default_22 = None
        slice_tensor_229: "f64[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_default_23, 2, 0, 25);  unsqueeze_default_23 = None
        slice_tensor_230: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 2, -2)
        slice_tensor_231: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_230, 1, 1, -2);  slice_tensor_230 = None
        slice_tensor_232: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_231, 2, 1, 9223372036854775807);  slice_tensor_231 = None
        mul_tensor_56: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_229, slice_tensor_232);  slice_tensor_229 = slice_tensor_232 = None
        full_default_16: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f64[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default_9: "f64[204, 204, 26]" = torch.ops.aten.copy.default(slice_scatter_default_4, full_default_17);  slice_scatter_default_4 = full_default_17 = None
        slice_tensor_233: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(copy_default_9, 0, 2, -2)
        slice_tensor_234: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_233, 1, 1, -2)
        slice_tensor_235: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(copy_default_9, 0, 2, -2)
        slice_tensor_236: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_235, 1, 1, -2);  slice_tensor_235 = None
        slice_tensor_237: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_236, 2, 1, 9223372036854775807);  slice_tensor_236 = None
        slice_tensor_238: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_tensor_239: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_238, 1, 1, -2);  slice_tensor_238 = None
        slice_tensor_240: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_239, 2, 1, 9223372036854775807);  slice_tensor_239 = None
        slice_tensor_241: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_tensor_242: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_241, 1, 1, -2);  slice_tensor_241 = None
        slice_tensor_243: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_242, 2, 0, -1);  slice_tensor_242 = None
        add_tensor_25: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(slice_tensor_240, slice_tensor_243);  slice_tensor_240 = slice_tensor_243 = None
        slice_tensor_244: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_tensor_245: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_244, 1, 2, -1);  slice_tensor_244 = None
        slice_tensor_246: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_245, 2, 1, 9223372036854775807);  slice_tensor_245 = None
        add_tensor_26: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(add_tensor_25, slice_tensor_246);  add_tensor_25 = slice_tensor_246 = None
        slice_tensor_247: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_tensor_248: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_247, 1, 2, -1);  slice_tensor_247 = None
        slice_tensor_249: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_248, 2, 0, -1);  slice_tensor_248 = None
        add_tensor_27: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(add_tensor_26, slice_tensor_249);  add_tensor_26 = slice_tensor_249 = None
        mul_tensor_57: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(add_tensor_27, 0.25);  add_tensor_27 = None
        copy_default_10: "f64[200, 201, 25]" = torch.ops.aten.copy.default(slice_tensor_237, mul_tensor_57);  slice_tensor_237 = mul_tensor_57 = None
        slice_scatter_default_41: "f64[200, 201, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_234, copy_default_10, 2, 1, 9223372036854775807);  slice_tensor_234 = copy_default_10 = None
        slice_scatter_default_42: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_233, slice_scatter_default_41, 1, 1, -2);  slice_tensor_233 = slice_scatter_default_41 = None
        slice_scatter_default_43: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(copy_default_9, slice_scatter_default_42, 0, 2, -2);  copy_default_9 = slice_scatter_default_42 = None
        slice_tensor_250: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_43, 0, 2, -2)
        slice_tensor_251: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_250, 1, 1, -2)
        slice_tensor_252: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_43, 0, 2, -2)
        slice_tensor_253: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_252, 1, 1, -2);  slice_tensor_252 = None
        select_int_24: "f64[200, 201]" = torch.ops.aten.select.int(slice_tensor_253, 2, 0);  slice_tensor_253 = None
        slice_tensor_254: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_tensor_255: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_254, 1, 1, -2);  slice_tensor_254 = None
        select_int_25: "f64[200, 201]" = torch.ops.aten.select.int(slice_tensor_255, 2, 0);  slice_tensor_255 = None
        slice_tensor_256: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_tensor_257: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_256, 1, 2, -1);  slice_tensor_256 = None
        select_int_26: "f64[200, 201]" = torch.ops.aten.select.int(slice_tensor_257, 2, 0);  slice_tensor_257 = None
        add_tensor_28: "f64[200, 201]" = torch.ops.aten.add.Tensor(select_int_25, select_int_26);  select_int_25 = select_int_26 = None
        mul_tensor_58: "f64[200, 201]" = torch.ops.aten.mul.Tensor(add_tensor_28, 0.5);  add_tensor_28 = None
        copy_default_11: "f64[200, 201]" = torch.ops.aten.copy.default(select_int_24, mul_tensor_58);  select_int_24 = mul_tensor_58 = None
        select_scatter_default_9: "f64[200, 201, 26]" = torch.ops.aten.select_scatter.default(slice_tensor_251, copy_default_11, 2, 0);  slice_tensor_251 = copy_default_11 = None
        slice_scatter_default_44: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_250, select_scatter_default_9, 1, 1, -2);  slice_tensor_250 = select_scatter_default_9 = None
        slice_scatter_default_45: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_43, slice_scatter_default_44, 0, 2, -2);  slice_scatter_default_43 = slice_scatter_default_44 = None
        slice_tensor_258: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_45, 0, 2, -2)
        slice_tensor_259: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_258, 1, 1, -2);  slice_tensor_258 = None
        slice_tensor_260: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_259, 2, 1, 9223372036854775807);  slice_tensor_259 = None
        slice_tensor_261: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2)
        slice_tensor_262: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_261, 1, 1, -2);  slice_tensor_261 = None
        slice_tensor_263: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_262, 2, 1, 9223372036854775807);  slice_tensor_262 = None
        full_default_18: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_264: "f64[204, 203, 26]" = torch.ops.aten.slice.Tensor(full_default_18, 1, 0, -1)
        slice_tensor_265: "f64[204, 203, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 1, 0, -1)
        slice_tensor_266: "f64[204, 203, 26, 3]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 1, 9223372036854775807)
        select_int_27: "f64[204, 203, 26]" = torch.ops.aten.select.int(slice_tensor_266, 3, 0);  slice_tensor_266 = None
        slice_tensor_267: "f64[204, 203, 26, 3]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, -1);  arg2_1 = None
        select_int_28: "f64[204, 203, 26]" = torch.ops.aten.select.int(slice_tensor_267, 3, 0);  slice_tensor_267 = None
        sub_tensor_11: "f64[204, 203, 26]" = torch.ops.aten.sub.Tensor(select_int_27, select_int_28);  select_int_27 = select_int_28 = None
        mul_tensor_59: "f64[204, 203, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_265, sub_tensor_11);  slice_tensor_265 = sub_tensor_11 = None
        unsqueeze_default_24: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg11_1, 0)
        slice_tensor_268: "f64[1, 203]" = torch.ops.aten.slice.Tensor(unsqueeze_default_24, 1, 0, -1);  unsqueeze_default_24 = None
        unsqueeze_default_25: "f64[1, 203, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_268, 2);  slice_tensor_268 = None
        div_tensor_13: "f64[204, 203, 26]" = torch.ops.aten.div.Tensor(mul_tensor_59, unsqueeze_default_25);  mul_tensor_59 = unsqueeze_default_25 = None
        copy_default_12: "f64[204, 203, 26]" = torch.ops.aten.copy.default(slice_tensor_264, div_tensor_13);  slice_tensor_264 = div_tensor_13 = None
        slice_scatter_default_46: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_default_18, copy_default_12, 1, 0, -1);  full_default_18 = copy_default_12 = None
        slice_tensor_269: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_46, 0, 2, -2)
        slice_tensor_270: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_269, 1, 1, -2);  slice_tensor_269 = None
        slice_tensor_271: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_270, 2, 1, 9223372036854775807);  slice_tensor_270 = None
        mul_tensor_60: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_263, slice_tensor_271);  slice_tensor_263 = slice_tensor_271 = None
        slice_tensor_272: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2)
        slice_tensor_273: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_272, 1, 1, -2);  slice_tensor_272 = None
        slice_tensor_274: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_273, 2, 1, 9223372036854775807);  slice_tensor_273 = None
        full_default_19: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_275: "f64[204, 203, 26]" = torch.ops.aten.slice.Tensor(full_default_19, 1, 0, -1)
        slice_tensor_276: "f64[204, 203, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 1, 0, -1)
        slice_tensor_277: "f64[204, 203, 26, 3]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 1, 9223372036854775807)
        select_int_29: "f64[204, 203, 26]" = torch.ops.aten.select.int(slice_tensor_277, 3, 0);  slice_tensor_277 = None
        slice_tensor_278: "f64[204, 203, 26, 3]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 0, -1);  arg1_1 = None
        select_int_30: "f64[204, 203, 26]" = torch.ops.aten.select.int(slice_tensor_278, 3, 0);  slice_tensor_278 = None
        sub_tensor_12: "f64[204, 203, 26]" = torch.ops.aten.sub.Tensor(select_int_29, select_int_30);  select_int_29 = select_int_30 = None
        mul_tensor_61: "f64[204, 203, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_276, sub_tensor_12);  slice_tensor_276 = sub_tensor_12 = None
        unsqueeze_default_26: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg11_1, 0)
        slice_tensor_279: "f64[1, 203]" = torch.ops.aten.slice.Tensor(unsqueeze_default_26, 1, 0, -1);  unsqueeze_default_26 = None
        unsqueeze_default_27: "f64[1, 203, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_279, 2);  slice_tensor_279 = None
        div_tensor_14: "f64[204, 203, 26]" = torch.ops.aten.div.Tensor(mul_tensor_61, unsqueeze_default_27);  mul_tensor_61 = unsqueeze_default_27 = None
        copy_default_13: "f64[204, 203, 26]" = torch.ops.aten.copy.default(slice_tensor_275, div_tensor_14);  slice_tensor_275 = div_tensor_14 = None
        slice_scatter_default_47: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_default_19, copy_default_13, 1, 0, -1);  full_default_19 = copy_default_13 = None
        slice_tensor_280: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_47, 0, 2, -2)
        slice_tensor_281: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_280, 1, 1, -2);  slice_tensor_280 = None
        slice_tensor_282: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_281, 2, 1, 9223372036854775807);  slice_tensor_281 = None
        mul_tensor_62: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_274, slice_tensor_282);  slice_tensor_274 = slice_tensor_282 = None
        add_tensor_29: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(mul_tensor_60, mul_tensor_62);  mul_tensor_60 = mul_tensor_62 = None
        neg_default_10: "f64[200, 201, 25]" = torch.ops.aten.neg.default(add_tensor_29);  add_tensor_29 = None
        full_default_20: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_283: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2)
        slice_tensor_284: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_283, 1, 1, -2);  slice_tensor_283 = None
        slice_tensor_285: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_284, 2, 1, 9223372036854775807);  slice_tensor_284 = None
        slice_tensor_286: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, 2, -2)
        slice_tensor_287: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_286, 1, 1, -2);  slice_tensor_286 = None
        slice_tensor_288: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_287, 2, 0, 25);  slice_tensor_287 = None
        mul_tensor_63: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_285, slice_tensor_288);  slice_tensor_285 = slice_tensor_288 = None
        slice_tensor_289: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2)
        slice_tensor_290: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_289, 1, 1, -2);  slice_tensor_289 = None
        slice_tensor_291: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_290, 2, 1, 9223372036854775807);  slice_tensor_290 = None
        slice_tensor_292: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_8, 0, 2, -2)
        slice_tensor_293: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_292, 1, 1, -2);  slice_tensor_292 = None
        slice_tensor_294: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_293, 2, 0, 25);  slice_tensor_293 = None
        mul_tensor_64: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_291, slice_tensor_294);  slice_tensor_291 = slice_tensor_294 = None
        add_tensor_30: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(mul_tensor_63, mul_tensor_64);  mul_tensor_63 = mul_tensor_64 = None
        minimum_default_4: "f64[200, 201, 25]" = torch.ops.aten.minimum.default(full_default_20, add_tensor_30);  full_default_20 = add_tensor_30 = None
        sub_tensor_13: "f64[200, 201, 25]" = torch.ops.aten.sub.Tensor(minimum_default_4, 1e-20);  minimum_default_4 = None
        div_tensor_15: "f64[200, 201, 25]" = torch.ops.aten.div.Tensor(neg_default_10, sub_tensor_13);  neg_default_10 = sub_tensor_13 = None
        abs_default_5: "f64[200, 201, 25]" = torch.ops.aten.abs.default(div_tensor_15)
        neg_default_11: "f64[200, 201, 25]" = torch.ops.aten.neg.default(abs_default_5);  abs_default_5 = None
        add_tensor_31: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(neg_default_11, 0.001);  neg_default_11 = None
        div_tensor_16: "f64[200, 201, 25]" = torch.ops.aten.div.Tensor(add_tensor_31, 0.001);  add_tensor_31 = None
        tanh_default_4: "f64[200, 201, 25]" = torch.ops.aten.tanh.default(div_tensor_16);  div_tensor_16 = None
        add_tensor_32: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(tanh_default_4, 1.0);  tanh_default_4 = None
        mul_tensor_65: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(add_tensor_32, 0.5);  add_tensor_32 = None
        mul_tensor_66: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_260, mul_tensor_65);  slice_tensor_260 = None
        maximum_default_4: "f64[200, 201, 25]" = torch.ops.aten.maximum.default(full_default_16, mul_tensor_66);  full_default_16 = mul_tensor_66 = None
        mul_tensor_67: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_56, maximum_default_4);  mul_tensor_56 = maximum_default_4 = None
        add_tensor_33: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(slice_tensor_228, mul_tensor_67);  slice_tensor_228 = mul_tensor_67 = None
        slice_scatter_default_48: "f64[200, 201, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_225, add_tensor_33, 2, 1, 9223372036854775807);  slice_tensor_225 = add_tensor_33 = None
        slice_scatter_default_49: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_224, slice_scatter_default_48, 1, 1, -2);  slice_tensor_224 = slice_scatter_default_48 = None
        slice_scatter_default_50: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_default_15, slice_scatter_default_49, 0, 2, -2);  full_default_15 = slice_scatter_default_49 = None
        slice_tensor_295: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_50, 0, 2, -2)
        slice_tensor_296: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_295, 1, 1, -2);  slice_tensor_295 = None
        slice_tensor_297: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_296, 2, 1, 9223372036854775807);  slice_tensor_296 = None
        slice_tensor_298: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_50, 0, 2, -2)
        slice_tensor_299: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_298, 1, 1, -2)
        slice_tensor_300: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_50, 0, 2, -2)
        slice_tensor_301: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_300, 1, 1, -2);  slice_tensor_300 = None
        slice_tensor_302: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_301, 2, 1, 9223372036854775807);  slice_tensor_301 = None
        slice_scatter_default_51: "f64[200, 201, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_299, slice_tensor_302, 2, 1, 9223372036854775807);  slice_tensor_299 = slice_tensor_302 = None
        slice_scatter_default_52: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_298, slice_scatter_default_51, 1, 1, -2);  slice_tensor_298 = slice_scatter_default_51 = None
        slice_scatter_default_53: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_50, slice_scatter_default_52, 0, 2, -2);  slice_scatter_default_50 = slice_scatter_default_52 = None
        slice_tensor_303: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_53, 0, 2, -2)
        slice_tensor_304: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_303, 1, 1, -2)
        slice_tensor_305: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_53, 0, 2, -2)
        slice_tensor_306: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_305, 1, 1, -2);  slice_tensor_305 = None
        slice_tensor_307: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_306, 2, 1, 9223372036854775807);  slice_tensor_306 = None
        unsqueeze_default_28: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_default_29: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, 1);  unsqueeze_default_28 = None
        slice_tensor_308: "f64[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_default_29, 2, 0, 25);  unsqueeze_default_29 = None
        slice_tensor_309: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 2, -2)
        slice_tensor_310: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_309, 1, 1, -2);  slice_tensor_309 = None
        slice_tensor_311: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_310, 2, 1, 9223372036854775807);  slice_tensor_310 = None
        mul_tensor_68: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_308, slice_tensor_311);  slice_tensor_308 = slice_tensor_311 = None
        full_default_21: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_312: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_45, 0, 2, -2)
        slice_tensor_313: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_312, 1, 1, -2);  slice_tensor_312 = None
        slice_tensor_314: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_313, 2, 1, 9223372036854775807);  slice_tensor_313 = None
        slice_tensor_315: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2)
        slice_tensor_316: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_315, 1, 2, -1);  slice_tensor_315 = None
        slice_tensor_317: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_316, 2, 1, 9223372036854775807);  slice_tensor_316 = None
        slice_tensor_318: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_46, 0, 2, -2)
        slice_tensor_319: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_318, 1, 1, -2);  slice_tensor_318 = None
        slice_tensor_320: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_319, 2, 1, 9223372036854775807);  slice_tensor_319 = None
        mul_tensor_69: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_317, slice_tensor_320);  slice_tensor_317 = slice_tensor_320 = None
        slice_tensor_321: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2)
        slice_tensor_322: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_321, 1, 2, -1);  slice_tensor_321 = None
        slice_tensor_323: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_322, 2, 1, 9223372036854775807);  slice_tensor_322 = None
        slice_tensor_324: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_47, 0, 2, -2)
        slice_tensor_325: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_324, 1, 1, -2);  slice_tensor_324 = None
        slice_tensor_326: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_325, 2, 1, 9223372036854775807);  slice_tensor_325 = None
        mul_tensor_70: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_323, slice_tensor_326);  slice_tensor_323 = slice_tensor_326 = None
        add_tensor_34: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(mul_tensor_69, mul_tensor_70);  mul_tensor_69 = mul_tensor_70 = None
        neg_default_12: "f64[200, 201, 25]" = torch.ops.aten.neg.default(add_tensor_34);  add_tensor_34 = None
        full_default_22: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_327: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2)
        slice_tensor_328: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_327, 1, 2, -1);  slice_tensor_327 = None
        slice_tensor_329: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_328, 2, 1, 9223372036854775807);  slice_tensor_328 = None
        slice_tensor_330: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, 2, -2)
        slice_tensor_331: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_330, 1, 2, -1);  slice_tensor_330 = None
        slice_tensor_332: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_331, 2, 0, 25);  slice_tensor_331 = None
        mul_tensor_71: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_329, slice_tensor_332);  slice_tensor_329 = slice_tensor_332 = None
        slice_tensor_333: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2)
        slice_tensor_334: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_333, 1, 2, -1);  slice_tensor_333 = None
        slice_tensor_335: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_334, 2, 1, 9223372036854775807);  slice_tensor_334 = None
        slice_tensor_336: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_8, 0, 2, -2)
        slice_tensor_337: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_336, 1, 2, -1);  slice_tensor_336 = None
        slice_tensor_338: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_337, 2, 0, 25);  slice_tensor_337 = None
        mul_tensor_72: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_335, slice_tensor_338);  slice_tensor_335 = slice_tensor_338 = None
        add_tensor_35: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(mul_tensor_71, mul_tensor_72);  mul_tensor_71 = mul_tensor_72 = None
        minimum_default_5: "f64[200, 201, 25]" = torch.ops.aten.minimum.default(full_default_22, add_tensor_35);  full_default_22 = add_tensor_35 = None
        sub_tensor_14: "f64[200, 201, 25]" = torch.ops.aten.sub.Tensor(minimum_default_5, 1e-20);  minimum_default_5 = None
        div_tensor_17: "f64[200, 201, 25]" = torch.ops.aten.div.Tensor(neg_default_12, sub_tensor_14);  neg_default_12 = sub_tensor_14 = None
        abs_default_6: "f64[200, 201, 25]" = torch.ops.aten.abs.default(div_tensor_17)
        neg_default_13: "f64[200, 201, 25]" = torch.ops.aten.neg.default(abs_default_6);  abs_default_6 = None
        add_tensor_36: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(neg_default_13, 0.001);  neg_default_13 = None
        div_tensor_18: "f64[200, 201, 25]" = torch.ops.aten.div.Tensor(add_tensor_36, 0.001);  add_tensor_36 = None
        tanh_default_5: "f64[200, 201, 25]" = torch.ops.aten.tanh.default(div_tensor_18);  div_tensor_18 = None
        add_tensor_37: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(tanh_default_5, 1.0);  tanh_default_5 = None
        mul_tensor_73: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(add_tensor_37, 0.5);  add_tensor_37 = None
        mul_tensor_74: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_314, mul_tensor_73);  slice_tensor_314 = None
        maximum_default_5: "f64[200, 201, 25]" = torch.ops.aten.maximum.default(full_default_21, mul_tensor_74);  full_default_21 = mul_tensor_74 = None
        mul_tensor_75: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_68, maximum_default_5);  mul_tensor_68 = maximum_default_5 = None
        add_tensor_38: "f64[200, 201, 25]" = torch.ops.aten.add.Tensor(slice_tensor_307, mul_tensor_75);  slice_tensor_307 = mul_tensor_75 = None
        slice_scatter_default_54: "f64[200, 201, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_304, add_tensor_38, 2, 1, 9223372036854775807);  slice_tensor_304 = add_tensor_38 = None
        slice_scatter_default_55: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_303, slice_scatter_default_54, 1, 1, -2);  slice_tensor_303 = slice_scatter_default_54 = None
        slice_scatter_default_56: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_53, slice_scatter_default_55, 0, 2, -2);  slice_scatter_default_53 = slice_scatter_default_55 = None
        slice_tensor_339: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_56, 0, 2, -2)
        slice_tensor_340: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_339, 1, 1, -2);  slice_tensor_339 = None
        slice_tensor_341: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_340, 2, 1, 9223372036854775807);  slice_tensor_340 = None
        slice_tensor_342: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_56, 0, 2, -2)
        slice_tensor_343: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_342, 1, 1, -2)
        slice_tensor_344: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_56, 0, 2, -2)
        slice_tensor_345: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_344, 1, 1, -2);  slice_tensor_344 = None
        slice_tensor_346: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_345, 2, 1, 9223372036854775807);  slice_tensor_345 = None
        slice_scatter_default_57: "f64[200, 201, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_343, slice_tensor_346, 2, 1, 9223372036854775807);  slice_tensor_343 = slice_tensor_346 = None
        slice_scatter_default_58: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_342, slice_scatter_default_57, 1, 1, -2);  slice_tensor_342 = slice_scatter_default_57 = None
        slice_scatter_default_59: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_56, slice_scatter_default_58, 0, 2, -2);  slice_scatter_default_56 = slice_scatter_default_58 = None
        slice_tensor_347: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_59, 0, 2, -2)
        slice_tensor_348: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_59, 0, 2, -2)
        slice_tensor_349: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_348, 1, 1, -2);  slice_tensor_348 = None
        unsqueeze_default_30: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0)
        unsqueeze_default_31: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 1);  unsqueeze_default_30 = None
        slice_tensor_350: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 2, -2)
        slice_tensor_351: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_350, 1, 1, -2);  slice_tensor_350 = None
        mul_tensor_76: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_default_31, slice_tensor_351);  unsqueeze_default_31 = slice_tensor_351 = None
        full_default_23: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_352: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_45, 0, 2, -2)
        slice_tensor_353: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_352, 1, 1, -2);  slice_tensor_352 = None
        slice_tensor_354: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2)
        slice_tensor_355: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_354, 1, 1, -2);  slice_tensor_354 = None
        slice_tensor_356: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_46, 0, 2, -2)
        slice_tensor_357: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_356, 1, 1, -2);  slice_tensor_356 = None
        mul_tensor_77: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_355, slice_tensor_357);  slice_tensor_355 = slice_tensor_357 = None
        slice_tensor_358: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2)
        slice_tensor_359: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_358, 1, 1, -2);  slice_tensor_358 = None
        slice_tensor_360: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_47, 0, 2, -2)
        slice_tensor_361: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_360, 1, 1, -2);  slice_tensor_360 = None
        mul_tensor_78: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_359, slice_tensor_361);  slice_tensor_359 = slice_tensor_361 = None
        add_tensor_39: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(mul_tensor_77, mul_tensor_78);  mul_tensor_77 = mul_tensor_78 = None
        neg_default_14: "f64[200, 201, 26]" = torch.ops.aten.neg.default(add_tensor_39);  add_tensor_39 = None
        full_default_24: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_362: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2)
        slice_tensor_363: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_362, 1, 1, -2);  slice_tensor_362 = None
        slice_tensor_364: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, 2, -2)
        slice_tensor_365: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_364, 1, 1, -2);  slice_tensor_364 = None
        mul_tensor_79: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_363, slice_tensor_365);  slice_tensor_363 = slice_tensor_365 = None
        slice_tensor_366: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2)
        slice_tensor_367: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_366, 1, 1, -2);  slice_tensor_366 = None
        slice_tensor_368: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_8, 0, 2, -2)
        slice_tensor_369: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_368, 1, 1, -2);  slice_tensor_368 = None
        mul_tensor_80: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_367, slice_tensor_369);  slice_tensor_367 = slice_tensor_369 = None
        add_tensor_40: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(mul_tensor_79, mul_tensor_80);  mul_tensor_79 = mul_tensor_80 = None
        minimum_default_6: "f64[200, 201, 26]" = torch.ops.aten.minimum.default(full_default_24, add_tensor_40);  full_default_24 = add_tensor_40 = None
        sub_tensor_15: "f64[200, 201, 26]" = torch.ops.aten.sub.Tensor(minimum_default_6, 1e-20);  minimum_default_6 = None
        div_tensor_19: "f64[200, 201, 26]" = torch.ops.aten.div.Tensor(neg_default_14, sub_tensor_15);  neg_default_14 = sub_tensor_15 = None
        abs_default_7: "f64[200, 201, 26]" = torch.ops.aten.abs.default(div_tensor_19)
        neg_default_15: "f64[200, 201, 26]" = torch.ops.aten.neg.default(abs_default_7);  abs_default_7 = None
        add_tensor_41: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(neg_default_15, 0.001);  neg_default_15 = None
        div_tensor_20: "f64[200, 201, 26]" = torch.ops.aten.div.Tensor(add_tensor_41, 0.001);  add_tensor_41 = None
        tanh_default_6: "f64[200, 201, 26]" = torch.ops.aten.tanh.default(div_tensor_20);  div_tensor_20 = None
        add_tensor_42: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(tanh_default_6, 1.0);  tanh_default_6 = None
        mul_tensor_81: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(add_tensor_42, 0.5);  add_tensor_42 = None
        mul_tensor_82: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_353, mul_tensor_81);  slice_tensor_353 = None
        maximum_default_6: "f64[200, 201, 26]" = torch.ops.aten.maximum.default(full_default_23, mul_tensor_82);  full_default_23 = mul_tensor_82 = None
        mul_tensor_83: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_76, maximum_default_6);  mul_tensor_76 = maximum_default_6 = None
        add_tensor_43: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(slice_tensor_349, mul_tensor_83);  slice_tensor_349 = mul_tensor_83 = None
        slice_scatter_default_60: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_347, add_tensor_43, 1, 1, -2);  slice_tensor_347 = add_tensor_43 = None
        slice_scatter_default_61: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_59, slice_scatter_default_60, 0, 2, -2);  slice_scatter_default_59 = slice_scatter_default_60 = None
        slice_tensor_370: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_61, 0, 2, -2)
        slice_tensor_371: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_370, 1, 1, -2);  slice_tensor_370 = None
        slice_tensor_372: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_61, 0, 2, -2)
        slice_tensor_373: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_61, 0, 2, -2)
        slice_tensor_374: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_373, 1, 1, -2);  slice_tensor_373 = None
        slice_scatter_default_62: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_372, slice_tensor_374, 1, 1, -2);  slice_tensor_372 = slice_tensor_374 = None
        slice_scatter_default_63: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_61, slice_scatter_default_62, 0, 2, -2);  slice_scatter_default_61 = slice_scatter_default_62 = None
        slice_tensor_375: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_63, 0, 2, -2)
        slice_tensor_376: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_63, 0, 2, -2)
        slice_tensor_377: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_376, 1, 1, -2);  slice_tensor_376 = None
        unsqueeze_default_32: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg6_1, 0);  arg6_1 = None
        unsqueeze_default_33: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, 1);  unsqueeze_default_32 = None
        slice_tensor_378: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 2, -2)
        slice_tensor_379: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_378, 1, 1, -2);  slice_tensor_378 = None
        mul_tensor_84: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_default_33, slice_tensor_379);  unsqueeze_default_33 = slice_tensor_379 = None
        full_default_25: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_380: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_45, 0, 2, -2);  slice_scatter_default_45 = None
        slice_tensor_381: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_380, 1, 1, -2);  slice_tensor_380 = None
        slice_tensor_382: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2)
        slice_tensor_383: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_382, 1, 2, -1);  slice_tensor_382 = None
        slice_tensor_384: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_46, 0, 2, -2)
        slice_tensor_385: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_384, 1, 1, -2);  slice_tensor_384 = None
        mul_tensor_85: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_383, slice_tensor_385);  slice_tensor_383 = slice_tensor_385 = None
        slice_tensor_386: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2)
        slice_tensor_387: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_386, 1, 2, -1);  slice_tensor_386 = None
        slice_tensor_388: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_47, 0, 2, -2)
        slice_tensor_389: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_388, 1, 1, -2);  slice_tensor_388 = None
        mul_tensor_86: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_387, slice_tensor_389);  slice_tensor_387 = slice_tensor_389 = None
        add_tensor_44: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(mul_tensor_85, mul_tensor_86);  mul_tensor_85 = mul_tensor_86 = None
        neg_default_16: "f64[200, 201, 26]" = torch.ops.aten.neg.default(add_tensor_44);  add_tensor_44 = None
        full_default_26: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_390: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2)
        slice_tensor_391: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_390, 1, 2, -1);  slice_tensor_390 = None
        slice_tensor_392: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, 2, -2)
        slice_tensor_393: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_392, 1, 2, -1);  slice_tensor_392 = None
        mul_tensor_87: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_391, slice_tensor_393);  slice_tensor_391 = slice_tensor_393 = None
        slice_tensor_394: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2)
        slice_tensor_395: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_394, 1, 2, -1);  slice_tensor_394 = None
        slice_tensor_396: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_8, 0, 2, -2)
        slice_tensor_397: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_396, 1, 2, -1);  slice_tensor_396 = None
        mul_tensor_88: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_395, slice_tensor_397);  slice_tensor_395 = slice_tensor_397 = None
        add_tensor_45: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(mul_tensor_87, mul_tensor_88);  mul_tensor_87 = mul_tensor_88 = None
        minimum_default_7: "f64[200, 201, 26]" = torch.ops.aten.minimum.default(full_default_26, add_tensor_45);  full_default_26 = add_tensor_45 = None
        sub_tensor_16: "f64[200, 201, 26]" = torch.ops.aten.sub.Tensor(minimum_default_7, 1e-20);  minimum_default_7 = None
        div_tensor_21: "f64[200, 201, 26]" = torch.ops.aten.div.Tensor(neg_default_16, sub_tensor_16);  neg_default_16 = sub_tensor_16 = None
        abs_default_8: "f64[200, 201, 26]" = torch.ops.aten.abs.default(div_tensor_21)
        neg_default_17: "f64[200, 201, 26]" = torch.ops.aten.neg.default(abs_default_8);  abs_default_8 = None
        add_tensor_46: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(neg_default_17, 0.001);  neg_default_17 = None
        div_tensor_22: "f64[200, 201, 26]" = torch.ops.aten.div.Tensor(add_tensor_46, 0.001);  add_tensor_46 = None
        tanh_default_7: "f64[200, 201, 26]" = torch.ops.aten.tanh.default(div_tensor_22);  div_tensor_22 = None
        add_tensor_47: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(tanh_default_7, 1.0);  tanh_default_7 = None
        mul_tensor_89: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(add_tensor_47, 0.5);  add_tensor_47 = None
        mul_tensor_90: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_tensor_381, mul_tensor_89);  slice_tensor_381 = None
        maximum_default_7: "f64[200, 201, 26]" = torch.ops.aten.maximum.default(full_default_25, mul_tensor_90);  full_default_25 = mul_tensor_90 = None
        mul_tensor_91: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_84, maximum_default_7);  mul_tensor_84 = maximum_default_7 = None
        add_tensor_48: "f64[200, 201, 26]" = torch.ops.aten.add.Tensor(slice_tensor_377, mul_tensor_91);  slice_tensor_377 = mul_tensor_91 = None
        slice_scatter_default_64: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_375, add_tensor_48, 1, 1, -2);  slice_tensor_375 = add_tensor_48 = None
        slice_scatter_default_65: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_63, slice_scatter_default_64, 0, 2, -2);  slice_scatter_default_63 = slice_scatter_default_64 = None
        slice_tensor_398: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_65, 0, 2, -2)
        slice_tensor_399: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_398, 1, 1, -2);  slice_tensor_398 = None
        slice_tensor_400: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg15_1, 0, 2, -2)
        slice_tensor_401: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_400, 1, 1, -2)
        slice_tensor_402: "f64[200, 201, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_401, 2, 1, 9223372036854775807)
        select_int_31: "f64[200, 201, 25, 2]" = torch.ops.aten.select.int(slice_tensor_402, 3, 0)
        slice_tensor_403: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg15_1, 0, 2, -2)
        slice_tensor_404: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_403, 1, 1, -2);  slice_tensor_403 = None
        slice_tensor_405: "f64[200, 201, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_404, 2, 1, 9223372036854775807);  slice_tensor_404 = None
        select_int_32: "f64[200, 201, 25, 2]" = torch.ops.aten.select.int(slice_tensor_405, 3, 0);  slice_tensor_405 = None
        select_int_33: "f64[200, 201, 25]" = torch.ops.aten.select.int(select_int_32, 3, 0);  select_int_32 = None
        mul_tensor_92: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_65, div_tensor_15);  mul_tensor_65 = div_tensor_15 = None
        slice_tensor_406: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 2, -2)
        slice_tensor_407: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_406, 1, 1, -2);  slice_tensor_406 = None
        slice_tensor_408: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_407, 2, 1, 9223372036854775807);  slice_tensor_407 = None
        mul_tensor_93: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_92, slice_tensor_408);  mul_tensor_92 = slice_tensor_408 = None
        copy_default_14: "f64[200, 201, 25]" = torch.ops.aten.copy.default(select_int_33, mul_tensor_93);  select_int_33 = mul_tensor_93 = None
        select_scatter_default_10: "f64[200, 201, 25, 2]" = torch.ops.aten.select_scatter.default(select_int_31, copy_default_14, 3, 0);  select_int_31 = copy_default_14 = None
        select_scatter_default_11: "f64[200, 201, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_tensor_402, select_scatter_default_10, 3, 0);  slice_tensor_402 = select_scatter_default_10 = None
        slice_scatter_default_66: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_401, select_scatter_default_11, 2, 1, 9223372036854775807);  slice_tensor_401 = select_scatter_default_11 = None
        slice_scatter_default_67: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_400, slice_scatter_default_66, 1, 1, -2);  slice_tensor_400 = slice_scatter_default_66 = None
        slice_scatter_default_68: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(arg15_1, slice_scatter_default_67, 0, 2, -2);  slice_scatter_default_67 = None
        slice_tensor_409: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_68, 0, 2, -2)
        slice_tensor_410: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_409, 1, 1, -2)
        slice_tensor_411: "f64[200, 201, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_410, 2, 1, 9223372036854775807)
        select_int_34: "f64[200, 201, 25, 2]" = torch.ops.aten.select.int(slice_tensor_411, 3, 1)
        slice_tensor_412: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_68, 0, 2, -2)
        slice_tensor_413: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_412, 1, 1, -2);  slice_tensor_412 = None
        slice_tensor_414: "f64[200, 201, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_413, 2, 1, 9223372036854775807);  slice_tensor_413 = None
        select_int_35: "f64[200, 201, 25, 2]" = torch.ops.aten.select.int(slice_tensor_414, 3, 1);  slice_tensor_414 = None
        select_int_36: "f64[200, 201, 25]" = torch.ops.aten.select.int(select_int_35, 3, 0);  select_int_35 = None
        mul_tensor_94: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_73, div_tensor_17);  mul_tensor_73 = div_tensor_17 = None
        slice_tensor_415: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 2, -2)
        slice_tensor_416: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_415, 1, 1, -2);  slice_tensor_415 = None
        slice_tensor_417: "f64[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_416, 2, 1, 9223372036854775807);  slice_tensor_416 = None
        mul_tensor_95: "f64[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_94, slice_tensor_417);  mul_tensor_94 = slice_tensor_417 = None
        copy_default_15: "f64[200, 201, 25]" = torch.ops.aten.copy.default(select_int_36, mul_tensor_95);  select_int_36 = mul_tensor_95 = None
        select_scatter_default_12: "f64[200, 201, 25, 2]" = torch.ops.aten.select_scatter.default(select_int_34, copy_default_15, 3, 0);  select_int_34 = copy_default_15 = None
        select_scatter_default_13: "f64[200, 201, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_tensor_411, select_scatter_default_12, 3, 1);  slice_tensor_411 = select_scatter_default_12 = None
        slice_scatter_default_69: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_410, select_scatter_default_13, 2, 1, 9223372036854775807);  slice_tensor_410 = select_scatter_default_13 = None
        slice_scatter_default_70: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_409, slice_scatter_default_69, 1, 1, -2);  slice_tensor_409 = slice_scatter_default_69 = None
        slice_scatter_default_71: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_68, slice_scatter_default_70, 0, 2, -2);  slice_scatter_default_68 = slice_scatter_default_70 = None
        slice_tensor_418: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_71, 0, 2, -2)
        slice_tensor_419: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_418, 1, 1, -2)
        select_int_37: "f64[200, 201, 26, 2]" = torch.ops.aten.select.int(slice_tensor_419, 3, 0)
        slice_tensor_420: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_71, 0, 2, -2)
        slice_tensor_421: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_420, 1, 1, -2);  slice_tensor_420 = None
        select_int_38: "f64[200, 201, 26, 2]" = torch.ops.aten.select.int(slice_tensor_421, 3, 0);  slice_tensor_421 = None
        select_int_39: "f64[200, 201, 26]" = torch.ops.aten.select.int(select_int_38, 3, 1);  select_int_38 = None
        mul_tensor_96: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_81, div_tensor_19);  mul_tensor_81 = div_tensor_19 = None
        slice_tensor_422: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 2, -2)
        slice_tensor_423: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_422, 1, 1, -2);  slice_tensor_422 = None
        mul_tensor_97: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_96, slice_tensor_423);  mul_tensor_96 = slice_tensor_423 = None
        copy_default_16: "f64[200, 201, 26]" = torch.ops.aten.copy.default(select_int_39, mul_tensor_97);  select_int_39 = mul_tensor_97 = None
        select_scatter_default_14: "f64[200, 201, 26, 2]" = torch.ops.aten.select_scatter.default(select_int_37, copy_default_16, 3, 1);  select_int_37 = copy_default_16 = None
        select_scatter_default_15: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.select_scatter.default(slice_tensor_419, select_scatter_default_14, 3, 0);  slice_tensor_419 = select_scatter_default_14 = None
        slice_scatter_default_72: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_418, select_scatter_default_15, 1, 1, -2);  slice_tensor_418 = select_scatter_default_15 = None
        slice_scatter_default_73: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_71, slice_scatter_default_72, 0, 2, -2);  slice_scatter_default_71 = slice_scatter_default_72 = None
        slice_tensor_424: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_73, 0, 2, -2)
        slice_tensor_425: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_424, 1, 1, -2)
        select_int_40: "f64[200, 201, 26, 2]" = torch.ops.aten.select.int(slice_tensor_425, 3, 1)
        slice_tensor_426: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_73, 0, 2, -2)
        slice_tensor_427: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_426, 1, 1, -2);  slice_tensor_426 = None
        select_int_41: "f64[200, 201, 26, 2]" = torch.ops.aten.select.int(slice_tensor_427, 3, 1);  slice_tensor_427 = None
        select_int_42: "f64[200, 201, 26]" = torch.ops.aten.select.int(select_int_41, 3, 1);  select_int_41 = None
        mul_tensor_98: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_89, div_tensor_21);  mul_tensor_89 = div_tensor_21 = None
        slice_tensor_428: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 2, -2);  arg10_1 = None
        slice_tensor_429: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_428, 1, 1, -2);  slice_tensor_428 = None
        mul_tensor_99: "f64[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_tensor_98, slice_tensor_429);  mul_tensor_98 = slice_tensor_429 = None
        copy_default_17: "f64[200, 201, 26]" = torch.ops.aten.copy.default(select_int_42, mul_tensor_99);  select_int_42 = mul_tensor_99 = None
        select_scatter_default_16: "f64[200, 201, 26, 2]" = torch.ops.aten.select_scatter.default(select_int_40, copy_default_17, 3, 1);  select_int_40 = copy_default_17 = None
        select_scatter_default_17: "f64[200, 201, 26, 2, 2]" = torch.ops.aten.select_scatter.default(slice_tensor_425, select_scatter_default_16, 3, 1);  slice_tensor_425 = select_scatter_default_16 = None
        slice_scatter_default_74: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_424, select_scatter_default_17, 1, 1, -2);  slice_tensor_424 = select_scatter_default_17 = None
        slice_scatter_default_75: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_73, slice_scatter_default_74, 0, 2, -2);  slice_scatter_default_73 = slice_scatter_default_74 = None
        slice_tensor_430: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg16_1, 0, 2, -2)
        slice_tensor_431: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg16_1, 0, 2, -2)
        slice_tensor_432: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_431, 1, 1, -2);  slice_tensor_431 = None
        slice_tensor_433: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_65, 0, 2, -2)
        slice_tensor_434: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_65, 0, 2, -2)
        slice_tensor_435: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_434, 1, 1, -2);  slice_tensor_434 = None
        slice_scatter_default_76: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_433, slice_tensor_435, 1, 1, -2);  slice_tensor_433 = slice_tensor_435 = None
        slice_scatter_default_77: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_65, slice_scatter_default_76, 0, 2, -2);  slice_scatter_default_65 = slice_scatter_default_76 = None
        slice_tensor_436: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_77, 0, 2, -2);  slice_scatter_default_77 = None
        slice_tensor_437: "f64[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_436, 1, 1, -2);  slice_tensor_436 = None
        unsqueeze_default_34: "f64[1, 26]" = torch.ops.aten.unsqueeze.default(arg14_1, 0);  arg14_1 = None
        unsqueeze_default_35: "f64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, 1);  unsqueeze_default_34 = None
        mul_tensor_100: "f64[1, 1, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_default_35, 4.0);  unsqueeze_default_35 = None
        div_tensor_23: "f64[200, 201, 26]" = torch.ops.aten.div.Tensor(slice_tensor_437, mul_tensor_100);  slice_tensor_437 = mul_tensor_100 = None
        copy_default_18: "f64[200, 201, 26]" = torch.ops.aten.copy.default(slice_tensor_432, div_tensor_23);  slice_tensor_432 = div_tensor_23 = None
        slice_scatter_default_78: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_430, copy_default_18, 1, 1, -2);  slice_tensor_430 = copy_default_18 = None
        slice_scatter_default_79: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(arg16_1, slice_scatter_default_78, 0, 2, -2);  slice_scatter_default_78 = None
        slice_tensor_438: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg17_1, 0, 2, -2)
        slice_tensor_439: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_438, 1, 2, -2)
        slice_tensor_440: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_439, 2, 0, -1)
        select_int_43: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_440, 3, 0)
        slice_tensor_441: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg17_1, 0, 2, -2)
        slice_tensor_442: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_441, 1, 2, -2);  slice_tensor_441 = None
        slice_tensor_443: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_442, 2, 0, -1);  slice_tensor_442 = None
        select_int_44: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_443, 3, 0);  slice_tensor_443 = None
        select_int_45: "f64[200, 200, 25]" = torch.ops.aten.select.int(select_int_44, 3, 0);  select_int_44 = None
        slice_tensor_444: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2)
        slice_tensor_445: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_444, 1, 2, -2);  slice_tensor_444 = None
        slice_tensor_446: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_445, 2, 0, 25);  slice_tensor_445 = None
        slice_tensor_447: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_5, 0, 1, -3)
        slice_tensor_448: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_447, 1, 2, -2);  slice_tensor_447 = None
        slice_tensor_449: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_448, 2, 0, 25);  slice_tensor_448 = None
        mul_tensor_101: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_446, slice_tensor_449);  slice_tensor_446 = slice_tensor_449 = None
        slice_tensor_450: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2)
        slice_tensor_451: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_450, 1, 2, -2);  slice_tensor_450 = None
        slice_tensor_452: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_451, 2, 0, 25);  slice_tensor_451 = None
        slice_tensor_453: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_6, 0, 1, -3)
        slice_tensor_454: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_453, 1, 2, -2);  slice_tensor_453 = None
        slice_tensor_455: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_454, 2, 0, 25);  slice_tensor_454 = None
        mul_tensor_102: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_452, slice_tensor_455);  slice_tensor_452 = slice_tensor_455 = None
        add_tensor_49: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_tensor_101, mul_tensor_102);  mul_tensor_101 = mul_tensor_102 = None
        neg_default_18: "f64[200, 200, 25]" = torch.ops.aten.neg.default(add_tensor_49);  add_tensor_49 = None
        full_default_27: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_456: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2)
        slice_tensor_457: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_456, 1, 2, -2);  slice_tensor_456 = None
        slice_tensor_458: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_457, 2, 0, 25);  slice_tensor_457 = None
        slice_tensor_459: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, 2, -2)
        slice_tensor_460: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_459, 1, 2, -2);  slice_tensor_459 = None
        slice_tensor_461: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_460, 2, 0, -1);  slice_tensor_460 = None
        mul_tensor_103: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_458, slice_tensor_461);  slice_tensor_458 = slice_tensor_461 = None
        slice_tensor_462: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2)
        slice_tensor_463: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_462, 1, 2, -2);  slice_tensor_462 = None
        slice_tensor_464: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_463, 2, 0, 25);  slice_tensor_463 = None
        slice_tensor_465: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_8, 0, 2, -2)
        slice_tensor_466: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_465, 1, 2, -2);  slice_tensor_465 = None
        slice_tensor_467: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_466, 2, 0, -1);  slice_tensor_466 = None
        mul_tensor_104: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_464, slice_tensor_467);  slice_tensor_464 = slice_tensor_467 = None
        add_tensor_50: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_tensor_103, mul_tensor_104);  mul_tensor_103 = mul_tensor_104 = None
        minimum_default_8: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_27, add_tensor_50);  full_default_27 = None
        sub_tensor_17: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_default_8, 1e-20);  minimum_default_8 = None
        div_tensor_24: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_default_18, sub_tensor_17);  neg_default_18 = sub_tensor_17 = None
        abs_default_9: "f64[200, 200, 25]" = torch.ops.aten.abs.default(div_tensor_24)
        neg_default_19: "f64[200, 200, 25]" = torch.ops.aten.neg.default(abs_default_9);  abs_default_9 = None
        add_tensor_51: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_default_19, 0.001);  neg_default_19 = None
        div_tensor_25: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(add_tensor_51, 0.001);  add_tensor_51 = None
        tanh_default_8: "f64[200, 200, 25]" = torch.ops.aten.tanh.default(div_tensor_25);  div_tensor_25 = None
        add_tensor_52: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_default_8, 1.0);  tanh_default_8 = None
        mul_tensor_105: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_tensor_52, 0.5);  add_tensor_52 = None
        mul_tensor_106: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_105, div_tensor_24)
        slice_tensor_468: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_tensor_469: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_468, 1, 2, -2);  slice_tensor_468 = None
        slice_tensor_470: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_469, 2, 0, -1);  slice_tensor_469 = None
        mul_tensor_107: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_106, slice_tensor_470);  mul_tensor_106 = slice_tensor_470 = None
        copy_default_19: "f64[200, 200, 25]" = torch.ops.aten.copy.default(select_int_45, mul_tensor_107);  select_int_45 = mul_tensor_107 = None
        select_scatter_default_18: "f64[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_int_43, copy_default_19, 3, 0);  select_int_43 = copy_default_19 = None
        select_scatter_default_19: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_tensor_440, select_scatter_default_18, 3, 0);  slice_tensor_440 = select_scatter_default_18 = None
        slice_scatter_default_80: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_439, select_scatter_default_19, 2, 0, -1);  slice_tensor_439 = select_scatter_default_19 = None
        slice_scatter_default_81: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_438, slice_scatter_default_80, 1, 2, -2);  slice_tensor_438 = slice_scatter_default_80 = None
        slice_scatter_default_82: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(arg17_1, slice_scatter_default_81, 0, 2, -2);  slice_scatter_default_81 = None
        slice_tensor_471: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_82, 0, 2, -2)
        slice_tensor_472: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_471, 1, 2, -2)
        slice_tensor_473: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_472, 2, 0, -1)
        select_int_46: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_473, 3, 1)
        slice_tensor_474: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_82, 0, 2, -2)
        slice_tensor_475: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_474, 1, 2, -2);  slice_tensor_474 = None
        slice_tensor_476: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_475, 2, 0, -1);  slice_tensor_475 = None
        select_int_47: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_476, 3, 1);  slice_tensor_476 = None
        select_int_48: "f64[200, 200, 25]" = torch.ops.aten.select.int(select_int_47, 3, 0);  select_int_47 = None
        slice_tensor_477: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2)
        slice_tensor_478: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_477, 1, 2, -2);  slice_tensor_477 = None
        slice_tensor_479: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_478, 2, 0, 25);  slice_tensor_478 = None
        slice_tensor_480: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_5, 0, 2, -2)
        slice_tensor_481: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_480, 1, 2, -2);  slice_tensor_480 = None
        slice_tensor_482: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_481, 2, 0, 25);  slice_tensor_481 = None
        mul_tensor_108: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_479, slice_tensor_482);  slice_tensor_479 = slice_tensor_482 = None
        slice_tensor_483: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2)
        slice_tensor_484: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_483, 1, 2, -2);  slice_tensor_483 = None
        slice_tensor_485: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_484, 2, 0, 25);  slice_tensor_484 = None
        slice_tensor_486: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_6, 0, 2, -2)
        slice_tensor_487: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_486, 1, 2, -2);  slice_tensor_486 = None
        slice_tensor_488: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_487, 2, 0, 25);  slice_tensor_487 = None
        mul_tensor_109: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_485, slice_tensor_488);  slice_tensor_485 = slice_tensor_488 = None
        add_tensor_53: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_tensor_108, mul_tensor_109);  mul_tensor_108 = mul_tensor_109 = None
        neg_default_20: "f64[200, 200, 25]" = torch.ops.aten.neg.default(add_tensor_53);  add_tensor_53 = None
        full_default_28: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default_9: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_28, add_tensor_50);  full_default_28 = None
        sub_tensor_18: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_default_9, 1e-20);  minimum_default_9 = None
        div_tensor_26: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_default_20, sub_tensor_18);  neg_default_20 = sub_tensor_18 = None
        abs_default_10: "f64[200, 200, 25]" = torch.ops.aten.abs.default(div_tensor_26)
        neg_default_21: "f64[200, 200, 25]" = torch.ops.aten.neg.default(abs_default_10);  abs_default_10 = None
        add_tensor_54: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_default_21, 0.001);  neg_default_21 = None
        div_tensor_27: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(add_tensor_54, 0.001);  add_tensor_54 = None
        tanh_default_9: "f64[200, 200, 25]" = torch.ops.aten.tanh.default(div_tensor_27);  div_tensor_27 = None
        add_tensor_55: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_default_9, 1.0);  tanh_default_9 = None
        mul_tensor_110: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_tensor_55, 0.5);  add_tensor_55 = None
        mul_tensor_111: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_110, div_tensor_26)
        slice_tensor_489: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_tensor_490: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_489, 1, 2, -2);  slice_tensor_489 = None
        slice_tensor_491: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_490, 2, 0, -1);  slice_tensor_490 = None
        mul_tensor_112: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_111, slice_tensor_491);  mul_tensor_111 = slice_tensor_491 = None
        copy_default_20: "f64[200, 200, 25]" = torch.ops.aten.copy.default(select_int_48, mul_tensor_112);  select_int_48 = mul_tensor_112 = None
        select_scatter_default_20: "f64[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_int_46, copy_default_20, 3, 0);  select_int_46 = copy_default_20 = None
        select_scatter_default_21: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_tensor_473, select_scatter_default_20, 3, 1);  slice_tensor_473 = select_scatter_default_20 = None
        slice_scatter_default_83: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_472, select_scatter_default_21, 2, 0, -1);  slice_tensor_472 = select_scatter_default_21 = None
        slice_scatter_default_84: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_471, slice_scatter_default_83, 1, 2, -2);  slice_tensor_471 = slice_scatter_default_83 = None
        slice_scatter_default_85: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_82, slice_scatter_default_84, 0, 2, -2);  slice_scatter_default_82 = slice_scatter_default_84 = None
        slice_tensor_492: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_85, 0, 2, -2)
        slice_tensor_493: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_492, 1, 2, -2)
        slice_tensor_494: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_493, 2, 0, -1)
        select_int_49: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_494, 3, 0)
        slice_tensor_495: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_85, 0, 2, -2)
        slice_tensor_496: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_495, 1, 2, -2);  slice_tensor_495 = None
        slice_tensor_497: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_496, 2, 0, -1);  slice_tensor_496 = None
        select_int_50: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_497, 3, 0);  slice_tensor_497 = None
        select_int_51: "f64[200, 200, 25]" = torch.ops.aten.select.int(select_int_50, 3, 1);  select_int_50 = None
        slice_tensor_498: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2)
        slice_tensor_499: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_498, 1, 2, -2);  slice_tensor_498 = None
        slice_tensor_500: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_499, 2, 1, 26);  slice_tensor_499 = None
        slice_tensor_501: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_5, 0, 1, -3)
        slice_tensor_502: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_501, 1, 2, -2);  slice_tensor_501 = None
        slice_tensor_503: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_502, 2, 1, 26);  slice_tensor_502 = None
        mul_tensor_113: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_500, slice_tensor_503);  slice_tensor_500 = slice_tensor_503 = None
        slice_tensor_504: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2)
        slice_tensor_505: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_504, 1, 2, -2);  slice_tensor_504 = None
        slice_tensor_506: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_505, 2, 1, 26);  slice_tensor_505 = None
        slice_tensor_507: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_6, 0, 1, -3)
        slice_tensor_508: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_507, 1, 2, -2);  slice_tensor_507 = None
        slice_tensor_509: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_508, 2, 1, 26);  slice_tensor_508 = None
        mul_tensor_114: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_506, slice_tensor_509);  slice_tensor_506 = slice_tensor_509 = None
        add_tensor_56: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_tensor_113, mul_tensor_114);  mul_tensor_113 = mul_tensor_114 = None
        neg_default_22: "f64[200, 200, 25]" = torch.ops.aten.neg.default(add_tensor_56);  add_tensor_56 = None
        full_default_29: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_510: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2)
        slice_tensor_511: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_510, 1, 2, -2);  slice_tensor_510 = None
        slice_tensor_512: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_511, 2, 1, 26);  slice_tensor_511 = None
        slice_tensor_513: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, 2, -2);  slice_scatter_default_7 = None
        slice_tensor_514: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_513, 1, 2, -2);  slice_tensor_513 = None
        slice_tensor_515: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_514, 2, 0, -1);  slice_tensor_514 = None
        mul_tensor_115: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_512, slice_tensor_515);  slice_tensor_512 = slice_tensor_515 = None
        slice_tensor_516: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2)
        slice_tensor_517: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_516, 1, 2, -2);  slice_tensor_516 = None
        slice_tensor_518: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_517, 2, 1, 26);  slice_tensor_517 = None
        slice_tensor_519: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_8, 0, 2, -2);  slice_scatter_default_8 = None
        slice_tensor_520: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_519, 1, 2, -2);  slice_tensor_519 = None
        slice_tensor_521: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_520, 2, 0, -1);  slice_tensor_520 = None
        mul_tensor_116: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_518, slice_tensor_521);  slice_tensor_518 = slice_tensor_521 = None
        add_tensor_57: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_tensor_115, mul_tensor_116);  mul_tensor_115 = mul_tensor_116 = None
        minimum_default_10: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_29, add_tensor_57);  full_default_29 = None
        sub_tensor_19: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_default_10, 1e-20);  minimum_default_10 = None
        div_tensor_28: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_default_22, sub_tensor_19);  neg_default_22 = sub_tensor_19 = None
        abs_default_11: "f64[200, 200, 25]" = torch.ops.aten.abs.default(div_tensor_28)
        neg_default_23: "f64[200, 200, 25]" = torch.ops.aten.neg.default(abs_default_11);  abs_default_11 = None
        add_tensor_58: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_default_23, 0.001);  neg_default_23 = None
        div_tensor_29: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(add_tensor_58, 0.001);  add_tensor_58 = None
        tanh_default_10: "f64[200, 200, 25]" = torch.ops.aten.tanh.default(div_tensor_29);  div_tensor_29 = None
        add_tensor_59: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_default_10, 1.0);  tanh_default_10 = None
        mul_tensor_117: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_tensor_59, 0.5);  add_tensor_59 = None
        mul_tensor_118: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_117, div_tensor_28)
        slice_tensor_522: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_tensor_523: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_522, 1, 2, -2);  slice_tensor_522 = None
        slice_tensor_524: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_523, 2, 0, -1);  slice_tensor_523 = None
        mul_tensor_119: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_118, slice_tensor_524);  mul_tensor_118 = slice_tensor_524 = None
        copy_default_21: "f64[200, 200, 25]" = torch.ops.aten.copy.default(select_int_51, mul_tensor_119);  select_int_51 = mul_tensor_119 = None
        select_scatter_default_22: "f64[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_int_49, copy_default_21, 3, 1);  select_int_49 = copy_default_21 = None
        select_scatter_default_23: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_tensor_494, select_scatter_default_22, 3, 0);  slice_tensor_494 = select_scatter_default_22 = None
        slice_scatter_default_86: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_493, select_scatter_default_23, 2, 0, -1);  slice_tensor_493 = select_scatter_default_23 = None
        slice_scatter_default_87: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_492, slice_scatter_default_86, 1, 2, -2);  slice_tensor_492 = slice_scatter_default_86 = None
        slice_scatter_default_88: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_85, slice_scatter_default_87, 0, 2, -2);  slice_scatter_default_85 = slice_scatter_default_87 = None
        slice_tensor_525: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_88, 0, 2, -2)
        slice_tensor_526: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_525, 1, 2, -2)
        slice_tensor_527: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_526, 2, 0, -1)
        select_int_52: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_527, 3, 1)
        slice_tensor_528: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_88, 0, 2, -2)
        slice_tensor_529: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_528, 1, 2, -2);  slice_tensor_528 = None
        slice_tensor_530: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_529, 2, 0, -1);  slice_tensor_529 = None
        select_int_53: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_530, 3, 1);  slice_tensor_530 = None
        select_int_54: "f64[200, 200, 25]" = torch.ops.aten.select.int(select_int_53, 3, 1);  select_int_53 = None
        slice_tensor_531: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2)
        slice_tensor_532: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_531, 1, 2, -2);  slice_tensor_531 = None
        slice_tensor_533: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_532, 2, 1, 26);  slice_tensor_532 = None
        slice_tensor_534: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_5, 0, 2, -2);  slice_scatter_default_5 = None
        slice_tensor_535: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_534, 1, 2, -2);  slice_tensor_534 = None
        slice_tensor_536: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_535, 2, 1, 26);  slice_tensor_535 = None
        mul_tensor_120: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_533, slice_tensor_536);  slice_tensor_533 = slice_tensor_536 = None
        slice_tensor_537: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2)
        slice_tensor_538: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_537, 1, 2, -2);  slice_tensor_537 = None
        slice_tensor_539: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_538, 2, 1, 26);  slice_tensor_538 = None
        slice_tensor_540: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_6, 0, 2, -2);  slice_scatter_default_6 = None
        slice_tensor_541: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_540, 1, 2, -2);  slice_tensor_540 = None
        slice_tensor_542: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_541, 2, 1, 26);  slice_tensor_541 = None
        mul_tensor_121: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_539, slice_tensor_542);  slice_tensor_539 = slice_tensor_542 = None
        add_tensor_60: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_tensor_120, mul_tensor_121);  mul_tensor_120 = mul_tensor_121 = None
        neg_default_24: "f64[200, 200, 25]" = torch.ops.aten.neg.default(add_tensor_60);  add_tensor_60 = None
        full_default_30: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default_11: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_30, add_tensor_57);  full_default_30 = None
        sub_tensor_20: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_default_11, 1e-20);  minimum_default_11 = None
        div_tensor_30: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_default_24, sub_tensor_20);  neg_default_24 = sub_tensor_20 = None
        abs_default_12: "f64[200, 200, 25]" = torch.ops.aten.abs.default(div_tensor_30)
        neg_default_25: "f64[200, 200, 25]" = torch.ops.aten.neg.default(abs_default_12);  abs_default_12 = None
        add_tensor_61: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_default_25, 0.001);  neg_default_25 = None
        div_tensor_31: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(add_tensor_61, 0.001);  add_tensor_61 = None
        tanh_default_11: "f64[200, 200, 25]" = torch.ops.aten.tanh.default(div_tensor_31);  div_tensor_31 = None
        add_tensor_62: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_default_11, 1.0);  tanh_default_11 = None
        mul_tensor_122: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_tensor_62, 0.5);  add_tensor_62 = None
        mul_tensor_123: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_122, div_tensor_30)
        slice_tensor_543: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_tensor_544: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_543, 1, 2, -2);  slice_tensor_543 = None
        slice_tensor_545: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_544, 2, 0, -1);  slice_tensor_544 = None
        mul_tensor_124: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_123, slice_tensor_545);  mul_tensor_123 = slice_tensor_545 = None
        copy_default_22: "f64[200, 200, 25]" = torch.ops.aten.copy.default(select_int_54, mul_tensor_124);  select_int_54 = mul_tensor_124 = None
        select_scatter_default_24: "f64[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_int_52, copy_default_22, 3, 1);  select_int_52 = copy_default_22 = None
        select_scatter_default_25: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_tensor_527, select_scatter_default_24, 3, 1);  slice_tensor_527 = select_scatter_default_24 = None
        slice_scatter_default_89: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_526, select_scatter_default_25, 2, 0, -1);  slice_tensor_526 = select_scatter_default_25 = None
        slice_scatter_default_90: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_525, slice_scatter_default_89, 1, 2, -2);  slice_tensor_525 = slice_scatter_default_89 = None
        slice_scatter_default_91: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_88, slice_scatter_default_90, 0, 2, -2);  slice_scatter_default_88 = slice_scatter_default_90 = None
        slice_tensor_546: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg19_1, 0, 2, -2)
        slice_tensor_547: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_546, 1, 2, -2)
        slice_tensor_548: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_547, 2, 0, -1)
        select_int_55: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_548, 3, 0)
        slice_tensor_549: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg19_1, 0, 2, -2)
        slice_tensor_550: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_549, 1, 2, -2);  slice_tensor_549 = None
        slice_tensor_551: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_550, 2, 0, -1);  slice_tensor_550 = None
        select_int_56: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_551, 3, 0);  slice_tensor_551 = None
        select_int_57: "f64[200, 200, 25]" = torch.ops.aten.select.int(select_int_56, 3, 0);  select_int_56 = None
        slice_tensor_552: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2)
        slice_tensor_553: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_552, 1, 2, -2);  slice_tensor_552 = None
        slice_tensor_554: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_553, 2, 0, 25);  slice_tensor_553 = None
        slice_tensor_555: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_46, 0, 2, -2)
        slice_tensor_556: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_555, 1, 1, -3);  slice_tensor_555 = None
        slice_tensor_557: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_556, 2, 0, 25);  slice_tensor_556 = None
        mul_tensor_125: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_554, slice_tensor_557);  slice_tensor_554 = slice_tensor_557 = None
        slice_tensor_558: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2)
        slice_tensor_559: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_558, 1, 2, -2);  slice_tensor_558 = None
        slice_tensor_560: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_559, 2, 0, 25);  slice_tensor_559 = None
        slice_tensor_561: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_47, 0, 2, -2)
        slice_tensor_562: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_561, 1, 1, -3);  slice_tensor_561 = None
        slice_tensor_563: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_562, 2, 0, 25);  slice_tensor_562 = None
        mul_tensor_126: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_560, slice_tensor_563);  slice_tensor_560 = slice_tensor_563 = None
        add_tensor_63: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_tensor_125, mul_tensor_126);  mul_tensor_125 = mul_tensor_126 = None
        neg_default_26: "f64[200, 200, 25]" = torch.ops.aten.neg.default(add_tensor_63);  add_tensor_63 = None
        full_default_31: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default_12: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_31, add_tensor_50);  full_default_31 = None
        sub_tensor_21: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_default_12, 1e-20);  minimum_default_12 = None
        div_tensor_32: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_default_26, sub_tensor_21);  neg_default_26 = sub_tensor_21 = None
        abs_default_13: "f64[200, 200, 25]" = torch.ops.aten.abs.default(div_tensor_32)
        neg_default_27: "f64[200, 200, 25]" = torch.ops.aten.neg.default(abs_default_13);  abs_default_13 = None
        add_tensor_64: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_default_27, 0.001);  neg_default_27 = None
        div_tensor_33: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(add_tensor_64, 0.001);  add_tensor_64 = None
        tanh_default_12: "f64[200, 200, 25]" = torch.ops.aten.tanh.default(div_tensor_33);  div_tensor_33 = None
        add_tensor_65: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_default_12, 1.0);  tanh_default_12 = None
        mul_tensor_127: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_tensor_65, 0.5);  add_tensor_65 = None
        mul_tensor_128: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_127, div_tensor_32)
        slice_tensor_564: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_tensor_565: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_564, 1, 2, -2);  slice_tensor_564 = None
        slice_tensor_566: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_565, 2, 0, -1);  slice_tensor_565 = None
        mul_tensor_129: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_128, slice_tensor_566);  mul_tensor_128 = slice_tensor_566 = None
        copy_default_23: "f64[200, 200, 25]" = torch.ops.aten.copy.default(select_int_57, mul_tensor_129);  select_int_57 = mul_tensor_129 = None
        select_scatter_default_26: "f64[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_int_55, copy_default_23, 3, 0);  select_int_55 = copy_default_23 = None
        select_scatter_default_27: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_tensor_548, select_scatter_default_26, 3, 0);  slice_tensor_548 = select_scatter_default_26 = None
        slice_scatter_default_92: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_547, select_scatter_default_27, 2, 0, -1);  slice_tensor_547 = select_scatter_default_27 = None
        slice_scatter_default_93: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_546, slice_scatter_default_92, 1, 2, -2);  slice_tensor_546 = slice_scatter_default_92 = None
        slice_scatter_default_94: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(arg19_1, slice_scatter_default_93, 0, 2, -2);  slice_scatter_default_93 = None
        slice_tensor_567: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_94, 0, 2, -2)
        slice_tensor_568: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_567, 1, 2, -2)
        slice_tensor_569: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_568, 2, 0, -1)
        select_int_58: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_569, 3, 1)
        slice_tensor_570: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_94, 0, 2, -2)
        slice_tensor_571: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_570, 1, 2, -2);  slice_tensor_570 = None
        slice_tensor_572: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_571, 2, 0, -1);  slice_tensor_571 = None
        select_int_59: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_572, 3, 1);  slice_tensor_572 = None
        select_int_60: "f64[200, 200, 25]" = torch.ops.aten.select.int(select_int_59, 3, 0);  select_int_59 = None
        slice_tensor_573: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2)
        slice_tensor_574: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_573, 1, 2, -2);  slice_tensor_573 = None
        slice_tensor_575: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_574, 2, 0, 25);  slice_tensor_574 = None
        slice_tensor_576: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_46, 0, 2, -2)
        slice_tensor_577: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_576, 1, 2, -2);  slice_tensor_576 = None
        slice_tensor_578: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_577, 2, 0, 25);  slice_tensor_577 = None
        mul_tensor_130: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_575, slice_tensor_578);  slice_tensor_575 = slice_tensor_578 = None
        slice_tensor_579: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2)
        slice_tensor_580: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_579, 1, 2, -2);  slice_tensor_579 = None
        slice_tensor_581: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_580, 2, 0, 25);  slice_tensor_580 = None
        slice_tensor_582: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_47, 0, 2, -2)
        slice_tensor_583: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_582, 1, 2, -2);  slice_tensor_582 = None
        slice_tensor_584: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_583, 2, 0, 25);  slice_tensor_583 = None
        mul_tensor_131: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_581, slice_tensor_584);  slice_tensor_581 = slice_tensor_584 = None
        add_tensor_66: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_tensor_130, mul_tensor_131);  mul_tensor_130 = mul_tensor_131 = None
        neg_default_28: "f64[200, 200, 25]" = torch.ops.aten.neg.default(add_tensor_66);  add_tensor_66 = None
        full_default_32: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default_13: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_32, add_tensor_50);  full_default_32 = add_tensor_50 = None
        sub_tensor_22: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_default_13, 1e-20);  minimum_default_13 = None
        div_tensor_34: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_default_28, sub_tensor_22);  neg_default_28 = sub_tensor_22 = None
        abs_default_14: "f64[200, 200, 25]" = torch.ops.aten.abs.default(div_tensor_34)
        neg_default_29: "f64[200, 200, 25]" = torch.ops.aten.neg.default(abs_default_14);  abs_default_14 = None
        add_tensor_67: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_default_29, 0.001);  neg_default_29 = None
        div_tensor_35: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(add_tensor_67, 0.001);  add_tensor_67 = None
        tanh_default_13: "f64[200, 200, 25]" = torch.ops.aten.tanh.default(div_tensor_35);  div_tensor_35 = None
        add_tensor_68: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_default_13, 1.0);  tanh_default_13 = None
        mul_tensor_132: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_tensor_68, 0.5);  add_tensor_68 = None
        mul_tensor_133: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_132, div_tensor_34)
        slice_tensor_585: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_tensor_586: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_585, 1, 2, -2);  slice_tensor_585 = None
        slice_tensor_587: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_586, 2, 0, -1);  slice_tensor_586 = None
        mul_tensor_134: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_133, slice_tensor_587);  mul_tensor_133 = slice_tensor_587 = None
        copy_default_24: "f64[200, 200, 25]" = torch.ops.aten.copy.default(select_int_60, mul_tensor_134);  select_int_60 = mul_tensor_134 = None
        select_scatter_default_28: "f64[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_int_58, copy_default_24, 3, 0);  select_int_58 = copy_default_24 = None
        select_scatter_default_29: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_tensor_569, select_scatter_default_28, 3, 1);  slice_tensor_569 = select_scatter_default_28 = None
        slice_scatter_default_95: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_568, select_scatter_default_29, 2, 0, -1);  slice_tensor_568 = select_scatter_default_29 = None
        slice_scatter_default_96: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_567, slice_scatter_default_95, 1, 2, -2);  slice_tensor_567 = slice_scatter_default_95 = None
        slice_scatter_default_97: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_94, slice_scatter_default_96, 0, 2, -2);  slice_scatter_default_94 = slice_scatter_default_96 = None
        slice_tensor_588: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_97, 0, 2, -2)
        slice_tensor_589: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_588, 1, 2, -2)
        slice_tensor_590: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_589, 2, 0, -1)
        select_int_61: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_590, 3, 0)
        slice_tensor_591: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_97, 0, 2, -2)
        slice_tensor_592: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_591, 1, 2, -2);  slice_tensor_591 = None
        slice_tensor_593: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_592, 2, 0, -1);  slice_tensor_592 = None
        select_int_62: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_593, 3, 0);  slice_tensor_593 = None
        select_int_63: "f64[200, 200, 25]" = torch.ops.aten.select.int(select_int_62, 3, 1);  select_int_62 = None
        slice_tensor_594: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2)
        slice_tensor_595: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_594, 1, 2, -2);  slice_tensor_594 = None
        slice_tensor_596: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_595, 2, 1, 26);  slice_tensor_595 = None
        slice_tensor_597: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_46, 0, 2, -2)
        slice_tensor_598: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_597, 1, 1, -3);  slice_tensor_597 = None
        slice_tensor_599: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_598, 2, 1, 26);  slice_tensor_598 = None
        mul_tensor_135: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_596, slice_tensor_599);  slice_tensor_596 = slice_tensor_599 = None
        slice_tensor_600: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2)
        slice_tensor_601: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_600, 1, 2, -2);  slice_tensor_600 = None
        slice_tensor_602: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_601, 2, 1, 26);  slice_tensor_601 = None
        slice_tensor_603: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_47, 0, 2, -2)
        slice_tensor_604: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_603, 1, 1, -3);  slice_tensor_603 = None
        slice_tensor_605: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_604, 2, 1, 26);  slice_tensor_604 = None
        mul_tensor_136: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_602, slice_tensor_605);  slice_tensor_602 = slice_tensor_605 = None
        add_tensor_69: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_tensor_135, mul_tensor_136);  mul_tensor_135 = mul_tensor_136 = None
        neg_default_30: "f64[200, 200, 25]" = torch.ops.aten.neg.default(add_tensor_69);  add_tensor_69 = None
        full_default_33: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default_14: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_33, add_tensor_57);  full_default_33 = None
        sub_tensor_23: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_default_14, 1e-20);  minimum_default_14 = None
        div_tensor_36: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_default_30, sub_tensor_23);  neg_default_30 = sub_tensor_23 = None
        abs_default_15: "f64[200, 200, 25]" = torch.ops.aten.abs.default(div_tensor_36)
        neg_default_31: "f64[200, 200, 25]" = torch.ops.aten.neg.default(abs_default_15);  abs_default_15 = None
        add_tensor_70: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_default_31, 0.001);  neg_default_31 = None
        div_tensor_37: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(add_tensor_70, 0.001);  add_tensor_70 = None
        tanh_default_14: "f64[200, 200, 25]" = torch.ops.aten.tanh.default(div_tensor_37);  div_tensor_37 = None
        add_tensor_71: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_default_14, 1.0);  tanh_default_14 = None
        mul_tensor_137: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_tensor_71, 0.5);  add_tensor_71 = None
        mul_tensor_138: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_137, div_tensor_36)
        slice_tensor_606: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_tensor_607: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_606, 1, 2, -2);  slice_tensor_606 = None
        slice_tensor_608: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_607, 2, 0, -1);  slice_tensor_607 = None
        mul_tensor_139: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_138, slice_tensor_608);  mul_tensor_138 = slice_tensor_608 = None
        copy_default_25: "f64[200, 200, 25]" = torch.ops.aten.copy.default(select_int_63, mul_tensor_139);  select_int_63 = mul_tensor_139 = None
        select_scatter_default_30: "f64[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_int_61, copy_default_25, 3, 1);  select_int_61 = copy_default_25 = None
        select_scatter_default_31: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_tensor_590, select_scatter_default_30, 3, 0);  slice_tensor_590 = select_scatter_default_30 = None
        slice_scatter_default_98: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_589, select_scatter_default_31, 2, 0, -1);  slice_tensor_589 = select_scatter_default_31 = None
        slice_scatter_default_99: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_588, slice_scatter_default_98, 1, 2, -2);  slice_tensor_588 = slice_scatter_default_98 = None
        slice_scatter_default_100: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_97, slice_scatter_default_99, 0, 2, -2);  slice_scatter_default_97 = slice_scatter_default_99 = None
        slice_tensor_609: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_100, 0, 2, -2)
        slice_tensor_610: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_609, 1, 2, -2)
        slice_tensor_611: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_610, 2, 0, -1)
        select_int_64: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_611, 3, 1)
        slice_tensor_612: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_default_100, 0, 2, -2)
        slice_tensor_613: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_612, 1, 2, -2);  slice_tensor_612 = None
        slice_tensor_614: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_tensor_613, 2, 0, -1);  slice_tensor_613 = None
        select_int_65: "f64[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_tensor_614, 3, 1);  slice_tensor_614 = None
        select_int_66: "f64[200, 200, 25]" = torch.ops.aten.select.int(select_int_65, 3, 1);  select_int_65 = None
        slice_tensor_615: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_8, 0, 2, -2);  mul_tensor_8 = None
        slice_tensor_616: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_615, 1, 2, -2);  slice_tensor_615 = None
        slice_tensor_617: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_616, 2, 1, 26);  slice_tensor_616 = None
        slice_tensor_618: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_46, 0, 2, -2);  slice_scatter_default_46 = None
        slice_tensor_619: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_618, 1, 2, -2);  slice_tensor_618 = None
        slice_tensor_620: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_619, 2, 1, 26);  slice_tensor_619 = None
        mul_tensor_140: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_617, slice_tensor_620);  slice_tensor_617 = slice_tensor_620 = None
        slice_tensor_621: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_tensor_12, 0, 2, -2);  mul_tensor_12 = None
        slice_tensor_622: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_621, 1, 2, -2);  slice_tensor_621 = None
        slice_tensor_623: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_622, 2, 1, 26);  slice_tensor_622 = None
        slice_tensor_624: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_47, 0, 2, -2);  slice_scatter_default_47 = None
        slice_tensor_625: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_624, 1, 2, -2);  slice_tensor_624 = None
        slice_tensor_626: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_625, 2, 1, 26);  slice_tensor_625 = None
        mul_tensor_141: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_tensor_623, slice_tensor_626);  slice_tensor_623 = slice_tensor_626 = None
        add_tensor_72: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_tensor_140, mul_tensor_141);  mul_tensor_140 = mul_tensor_141 = None
        neg_default_32: "f64[200, 200, 25]" = torch.ops.aten.neg.default(add_tensor_72);  add_tensor_72 = None
        full_default_34: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_default_15: "f64[200, 200, 25]" = torch.ops.aten.minimum.default(full_default_34, add_tensor_57);  full_default_34 = add_tensor_57 = None
        sub_tensor_24: "f64[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_default_15, 1e-20);  minimum_default_15 = None
        div_tensor_38: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_default_32, sub_tensor_24);  neg_default_32 = sub_tensor_24 = None
        abs_default_16: "f64[200, 200, 25]" = torch.ops.aten.abs.default(div_tensor_38)
        neg_default_33: "f64[200, 200, 25]" = torch.ops.aten.neg.default(abs_default_16);  abs_default_16 = None
        add_tensor_73: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_default_33, 0.001);  neg_default_33 = None
        div_tensor_39: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(add_tensor_73, 0.001);  add_tensor_73 = None
        tanh_default_15: "f64[200, 200, 25]" = torch.ops.aten.tanh.default(div_tensor_39);  div_tensor_39 = None
        add_tensor_74: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_default_15, 1.0);  tanh_default_15 = None
        mul_tensor_142: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_tensor_74, 0.5);  add_tensor_74 = None
        mul_tensor_143: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_142, div_tensor_38)
        slice_tensor_627: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_tensor_628: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_627, 1, 2, -2);  slice_tensor_627 = None
        slice_tensor_629: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_628, 2, 0, -1);  slice_tensor_628 = None
        mul_tensor_144: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_143, slice_tensor_629);  mul_tensor_143 = slice_tensor_629 = None
        copy_default_26: "f64[200, 200, 25]" = torch.ops.aten.copy.default(select_int_66, mul_tensor_144);  select_int_66 = mul_tensor_144 = None
        select_scatter_default_32: "f64[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_int_64, copy_default_26, 3, 1);  select_int_64 = copy_default_26 = None
        select_scatter_default_33: "f64[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_tensor_611, select_scatter_default_32, 3, 1);  slice_tensor_611 = select_scatter_default_32 = None
        slice_scatter_default_101: "f64[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_610, select_scatter_default_33, 2, 0, -1);  slice_tensor_610 = select_scatter_default_33 = None
        slice_scatter_default_102: "f64[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_tensor_609, slice_scatter_default_101, 1, 2, -2);  slice_tensor_609 = slice_scatter_default_101 = None
        slice_scatter_default_103: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_100, slice_scatter_default_102, 0, 2, -2);  slice_scatter_default_100 = slice_scatter_default_102 = None
        slice_tensor_630: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg22_1, 0, 2, -2)
        slice_tensor_631: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_630, 1, 2, -2)
        slice_tensor_632: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg22_1, 0, 2, -2)
        slice_tensor_633: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_632, 1, 2, -2);  slice_tensor_632 = None
        slice_tensor_634: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_633, 2, 0, -1);  slice_tensor_633 = None
        full_default_35: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_635: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_default_35, 0, 2, -2)
        slice_tensor_636: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_635, 1, 2, -2)
        slice_tensor_637: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_default_35, 0, 2, -2)
        slice_tensor_638: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_637, 1, 2, -2);  slice_tensor_637 = None
        slice_tensor_639: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_638, 2, 0, -1);  slice_tensor_638 = None
        slice_tensor_640: "f64[200]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 1, -3)
        unsqueeze_default_36: "f64[200, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_640, 1);  slice_tensor_640 = None
        unsqueeze_default_37: "f64[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_36, 2);  unsqueeze_default_36 = None
        slice_tensor_641: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_tensor_642: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_641, 1, 2, -2);  slice_tensor_641 = None
        slice_tensor_643: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_642, 2, 0, -1);  slice_tensor_642 = None
        mul_tensor_145: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_default_37, slice_tensor_643);  unsqueeze_default_37 = slice_tensor_643 = None
        mul_tensor_146: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_145, mul_tensor_105);  mul_tensor_145 = mul_tensor_105 = None
        pow_tensor_scalar: "f64[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_tensor_24, 2);  div_tensor_24 = None
        mul_tensor_147: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_146, pow_tensor_scalar);  mul_tensor_146 = pow_tensor_scalar = None
        slice_tensor_644: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_tensor_645: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_644, 1, 2, -2);  slice_tensor_644 = None
        slice_tensor_646: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_645, 2, 0, -1);  slice_tensor_645 = None
        mul_tensor_148: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_147, slice_tensor_646);  mul_tensor_147 = slice_tensor_646 = None
        add_tensor_75: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_tensor_639, mul_tensor_148);  slice_tensor_639 = mul_tensor_148 = None
        slice_scatter_default_104: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_636, add_tensor_75, 2, 0, -1);  slice_tensor_636 = add_tensor_75 = None
        slice_scatter_default_105: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_635, slice_scatter_default_104, 1, 2, -2);  slice_tensor_635 = slice_scatter_default_104 = None
        slice_scatter_default_106: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_default_35, slice_scatter_default_105, 0, 2, -2);  full_default_35 = slice_scatter_default_105 = None
        slice_tensor_647: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_106, 0, 2, -2)
        slice_tensor_648: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_647, 1, 2, -2)
        slice_tensor_649: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_106, 0, 2, -2)
        slice_tensor_650: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_649, 1, 2, -2);  slice_tensor_649 = None
        slice_tensor_651: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_650, 2, 0, -1);  slice_tensor_650 = None
        slice_tensor_652: "f64[200]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 2, -2)
        unsqueeze_default_38: "f64[200, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_652, 1);  slice_tensor_652 = None
        unsqueeze_default_39: "f64[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_38, 2);  unsqueeze_default_38 = None
        slice_tensor_653: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_tensor_654: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_653, 1, 2, -2);  slice_tensor_653 = None
        slice_tensor_655: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_654, 2, 0, -1);  slice_tensor_654 = None
        mul_tensor_149: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_default_39, slice_tensor_655);  unsqueeze_default_39 = slice_tensor_655 = None
        mul_tensor_150: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_149, mul_tensor_110);  mul_tensor_149 = mul_tensor_110 = None
        pow_tensor_scalar_1: "f64[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_tensor_26, 2);  div_tensor_26 = None
        mul_tensor_151: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_150, pow_tensor_scalar_1);  mul_tensor_150 = pow_tensor_scalar_1 = None
        slice_tensor_656: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_tensor_657: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_656, 1, 2, -2);  slice_tensor_656 = None
        slice_tensor_658: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_657, 2, 0, -1);  slice_tensor_657 = None
        mul_tensor_152: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_151, slice_tensor_658);  mul_tensor_151 = slice_tensor_658 = None
        add_tensor_76: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_tensor_651, mul_tensor_152);  slice_tensor_651 = mul_tensor_152 = None
        slice_scatter_default_107: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_648, add_tensor_76, 2, 0, -1);  slice_tensor_648 = add_tensor_76 = None
        slice_scatter_default_108: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_647, slice_scatter_default_107, 1, 2, -2);  slice_tensor_647 = slice_scatter_default_107 = None
        slice_scatter_default_109: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_106, slice_scatter_default_108, 0, 2, -2);  slice_scatter_default_106 = slice_scatter_default_108 = None
        slice_tensor_659: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_109, 0, 2, -2)
        slice_tensor_660: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_659, 1, 2, -2)
        slice_tensor_661: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_109, 0, 2, -2)
        slice_tensor_662: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_661, 1, 2, -2);  slice_tensor_661 = None
        slice_tensor_663: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_662, 2, 0, -1);  slice_tensor_662 = None
        slice_tensor_664: "f64[200]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 1, -3)
        unsqueeze_default_40: "f64[200, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_664, 1);  slice_tensor_664 = None
        unsqueeze_default_41: "f64[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_40, 2);  unsqueeze_default_40 = None
        slice_tensor_665: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_tensor_666: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_665, 1, 2, -2);  slice_tensor_665 = None
        slice_tensor_667: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_666, 2, 0, -1);  slice_tensor_666 = None
        mul_tensor_153: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_default_41, slice_tensor_667);  unsqueeze_default_41 = slice_tensor_667 = None
        mul_tensor_154: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_153, mul_tensor_117);  mul_tensor_153 = mul_tensor_117 = None
        pow_tensor_scalar_2: "f64[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_tensor_28, 2);  div_tensor_28 = None
        mul_tensor_155: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_154, pow_tensor_scalar_2);  mul_tensor_154 = pow_tensor_scalar_2 = None
        slice_tensor_668: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_tensor_669: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_668, 1, 2, -2);  slice_tensor_668 = None
        slice_tensor_670: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_669, 2, 0, -1);  slice_tensor_669 = None
        mul_tensor_156: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_155, slice_tensor_670);  mul_tensor_155 = slice_tensor_670 = None
        add_tensor_77: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_tensor_663, mul_tensor_156);  slice_tensor_663 = mul_tensor_156 = None
        slice_scatter_default_110: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_660, add_tensor_77, 2, 0, -1);  slice_tensor_660 = add_tensor_77 = None
        slice_scatter_default_111: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_659, slice_scatter_default_110, 1, 2, -2);  slice_tensor_659 = slice_scatter_default_110 = None
        slice_scatter_default_112: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_109, slice_scatter_default_111, 0, 2, -2);  slice_scatter_default_109 = slice_scatter_default_111 = None
        slice_tensor_671: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_112, 0, 2, -2)
        slice_tensor_672: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_671, 1, 2, -2)
        slice_tensor_673: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_112, 0, 2, -2)
        slice_tensor_674: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_673, 1, 2, -2);  slice_tensor_673 = None
        slice_tensor_675: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_674, 2, 0, -1);  slice_tensor_674 = None
        slice_tensor_676: "f64[200]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 2, -2);  arg8_1 = None
        unsqueeze_default_42: "f64[200, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_676, 1);  slice_tensor_676 = None
        unsqueeze_default_43: "f64[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_42, 2);  unsqueeze_default_42 = None
        slice_tensor_677: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_tensor_678: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_677, 1, 2, -2);  slice_tensor_677 = None
        slice_tensor_679: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_678, 2, 0, -1);  slice_tensor_678 = None
        mul_tensor_157: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_default_43, slice_tensor_679);  unsqueeze_default_43 = slice_tensor_679 = None
        mul_tensor_158: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_157, mul_tensor_122);  mul_tensor_157 = mul_tensor_122 = None
        pow_tensor_scalar_3: "f64[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_tensor_30, 2);  div_tensor_30 = None
        mul_tensor_159: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_158, pow_tensor_scalar_3);  mul_tensor_158 = pow_tensor_scalar_3 = None
        slice_tensor_680: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_tensor_681: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_680, 1, 2, -2);  slice_tensor_680 = None
        slice_tensor_682: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_681, 2, 0, -1);  slice_tensor_681 = None
        mul_tensor_160: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_159, slice_tensor_682);  mul_tensor_159 = slice_tensor_682 = None
        add_tensor_78: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_tensor_675, mul_tensor_160);  slice_tensor_675 = mul_tensor_160 = None
        slice_scatter_default_113: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_672, add_tensor_78, 2, 0, -1);  slice_tensor_672 = add_tensor_78 = None
        slice_scatter_default_114: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_671, slice_scatter_default_113, 1, 2, -2);  slice_tensor_671 = slice_scatter_default_113 = None
        slice_scatter_default_115: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_112, slice_scatter_default_114, 0, 2, -2);  slice_scatter_default_112 = slice_scatter_default_114 = None
        slice_tensor_683: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_115, 0, 2, -2);  slice_scatter_default_115 = None
        slice_tensor_684: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_683, 1, 2, -2);  slice_tensor_683 = None
        slice_tensor_685: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_684, 2, 0, -1);  slice_tensor_684 = None
        slice_tensor_686: "f64[200]" = torch.ops.aten.slice.Tensor(arg20_1, 0, 2, -2);  arg20_1 = None
        unsqueeze_default_44: "f64[200, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_686, 1);  slice_tensor_686 = None
        unsqueeze_default_45: "f64[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_44, 2);  unsqueeze_default_44 = None
        mul_tensor_161: "f64[200, 1, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_default_45, 4);  unsqueeze_default_45 = None
        div_tensor_40: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(slice_tensor_685, mul_tensor_161);  slice_tensor_685 = mul_tensor_161 = None
        full_default_36: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor_687: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_default_36, 0, 2, -2)
        slice_tensor_688: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_687, 1, 2, -2)
        slice_tensor_689: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_default_36, 0, 2, -2)
        slice_tensor_690: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_689, 1, 2, -2);  slice_tensor_689 = None
        slice_tensor_691: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_690, 2, 0, -1);  slice_tensor_690 = None
        slice_tensor_692: "f64[200]" = torch.ops.aten.slice.Tensor(arg18_1, 0, 1, -3)
        slice_tensor_693: "f64[200]" = torch.ops.aten.slice.Tensor(arg11_1, 0, 1, -3)
        mul_tensor_162: "f64[200]" = torch.ops.aten.mul.Tensor(slice_tensor_692, slice_tensor_693);  slice_tensor_692 = slice_tensor_693 = None
        unsqueeze_default_46: "f64[1, 200]" = torch.ops.aten.unsqueeze.default(mul_tensor_162, 0);  mul_tensor_162 = None
        unsqueeze_default_47: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_46, 2);  unsqueeze_default_46 = None
        slice_tensor_694: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_tensor_695: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_694, 1, 2, -2);  slice_tensor_694 = None
        slice_tensor_696: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_695, 2, 0, -1);  slice_tensor_695 = None
        mul_tensor_163: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_default_47, slice_tensor_696);  unsqueeze_default_47 = slice_tensor_696 = None
        mul_tensor_164: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_163, mul_tensor_127);  mul_tensor_163 = mul_tensor_127 = None
        pow_tensor_scalar_4: "f64[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_tensor_32, 2);  div_tensor_32 = None
        mul_tensor_165: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_164, pow_tensor_scalar_4);  mul_tensor_164 = pow_tensor_scalar_4 = None
        slice_tensor_697: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_tensor_698: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_697, 1, 2, -2);  slice_tensor_697 = None
        slice_tensor_699: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_698, 2, 0, -1);  slice_tensor_698 = None
        mul_tensor_166: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_165, slice_tensor_699);  mul_tensor_165 = slice_tensor_699 = None
        add_tensor_79: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_tensor_691, mul_tensor_166);  slice_tensor_691 = mul_tensor_166 = None
        slice_scatter_default_116: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_688, add_tensor_79, 2, 0, -1);  slice_tensor_688 = add_tensor_79 = None
        slice_scatter_default_117: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_687, slice_scatter_default_116, 1, 2, -2);  slice_tensor_687 = slice_scatter_default_116 = None
        slice_scatter_default_118: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_default_36, slice_scatter_default_117, 0, 2, -2);  full_default_36 = slice_scatter_default_117 = None
        slice_tensor_700: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_118, 0, 2, -2)
        slice_tensor_701: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_700, 1, 2, -2)
        slice_tensor_702: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_118, 0, 2, -2)
        slice_tensor_703: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_702, 1, 2, -2);  slice_tensor_702 = None
        slice_tensor_704: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_703, 2, 0, -1);  slice_tensor_703 = None
        slice_tensor_705: "f64[200]" = torch.ops.aten.slice.Tensor(arg18_1, 0, 2, -2)
        slice_tensor_706: "f64[200]" = torch.ops.aten.slice.Tensor(arg11_1, 0, 2, -2)
        mul_tensor_167: "f64[200]" = torch.ops.aten.mul.Tensor(slice_tensor_705, slice_tensor_706);  slice_tensor_705 = slice_tensor_706 = None
        unsqueeze_default_48: "f64[1, 200]" = torch.ops.aten.unsqueeze.default(mul_tensor_167, 0);  mul_tensor_167 = None
        unsqueeze_default_49: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_48, 2);  unsqueeze_default_48 = None
        slice_tensor_707: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_tensor_708: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_707, 1, 2, -2);  slice_tensor_707 = None
        slice_tensor_709: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_708, 2, 0, -1);  slice_tensor_708 = None
        mul_tensor_168: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_default_49, slice_tensor_709);  unsqueeze_default_49 = slice_tensor_709 = None
        mul_tensor_169: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_168, mul_tensor_132);  mul_tensor_168 = mul_tensor_132 = None
        pow_tensor_scalar_5: "f64[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_tensor_34, 2);  div_tensor_34 = None
        mul_tensor_170: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_169, pow_tensor_scalar_5);  mul_tensor_169 = pow_tensor_scalar_5 = None
        slice_tensor_710: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_tensor_711: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_710, 1, 2, -2);  slice_tensor_710 = None
        slice_tensor_712: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_711, 2, 0, -1);  slice_tensor_711 = None
        mul_tensor_171: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_170, slice_tensor_712);  mul_tensor_170 = slice_tensor_712 = None
        add_tensor_80: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_tensor_704, mul_tensor_171);  slice_tensor_704 = mul_tensor_171 = None
        slice_scatter_default_119: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_701, add_tensor_80, 2, 0, -1);  slice_tensor_701 = add_tensor_80 = None
        slice_scatter_default_120: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_700, slice_scatter_default_119, 1, 2, -2);  slice_tensor_700 = slice_scatter_default_119 = None
        slice_scatter_default_121: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_118, slice_scatter_default_120, 0, 2, -2);  slice_scatter_default_118 = slice_scatter_default_120 = None
        slice_tensor_713: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_121, 0, 2, -2)
        slice_tensor_714: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_713, 1, 2, -2)
        slice_tensor_715: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_121, 0, 2, -2)
        slice_tensor_716: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_715, 1, 2, -2);  slice_tensor_715 = None
        slice_tensor_717: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_716, 2, 0, -1);  slice_tensor_716 = None
        slice_tensor_718: "f64[200]" = torch.ops.aten.slice.Tensor(arg18_1, 0, 1, -3)
        slice_tensor_719: "f64[200]" = torch.ops.aten.slice.Tensor(arg11_1, 0, 1, -3)
        mul_tensor_172: "f64[200]" = torch.ops.aten.mul.Tensor(slice_tensor_718, slice_tensor_719);  slice_tensor_718 = slice_tensor_719 = None
        unsqueeze_default_50: "f64[1, 200]" = torch.ops.aten.unsqueeze.default(mul_tensor_172, 0);  mul_tensor_172 = None
        unsqueeze_default_51: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_50, 2);  unsqueeze_default_50 = None
        slice_tensor_720: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        slice_tensor_721: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_720, 1, 2, -2);  slice_tensor_720 = None
        slice_tensor_722: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_721, 2, 0, -1);  slice_tensor_721 = None
        mul_tensor_173: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_default_51, slice_tensor_722);  unsqueeze_default_51 = slice_tensor_722 = None
        mul_tensor_174: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_173, mul_tensor_137);  mul_tensor_173 = mul_tensor_137 = None
        pow_tensor_scalar_6: "f64[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_tensor_36, 2);  div_tensor_36 = None
        mul_tensor_175: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_174, pow_tensor_scalar_6);  mul_tensor_174 = pow_tensor_scalar_6 = None
        slice_tensor_723: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2)
        slice_tensor_724: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_723, 1, 2, -2);  slice_tensor_723 = None
        slice_tensor_725: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_724, 2, 0, -1);  slice_tensor_724 = None
        mul_tensor_176: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_175, slice_tensor_725);  mul_tensor_175 = slice_tensor_725 = None
        add_tensor_81: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_tensor_717, mul_tensor_176);  slice_tensor_717 = mul_tensor_176 = None
        slice_scatter_default_122: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_714, add_tensor_81, 2, 0, -1);  slice_tensor_714 = add_tensor_81 = None
        slice_scatter_default_123: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_713, slice_scatter_default_122, 1, 2, -2);  slice_tensor_713 = slice_scatter_default_122 = None
        slice_scatter_default_124: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_121, slice_scatter_default_123, 0, 2, -2);  slice_scatter_default_121 = slice_scatter_default_123 = None
        slice_tensor_726: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_124, 0, 2, -2)
        slice_tensor_727: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_726, 1, 2, -2)
        slice_tensor_728: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_124, 0, 2, -2)
        slice_tensor_729: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_728, 1, 2, -2);  slice_tensor_728 = None
        slice_tensor_730: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_729, 2, 0, -1);  slice_tensor_729 = None
        slice_tensor_731: "f64[200]" = torch.ops.aten.slice.Tensor(arg18_1, 0, 2, -2);  arg18_1 = None
        slice_tensor_732: "f64[200]" = torch.ops.aten.slice.Tensor(arg11_1, 0, 2, -2);  arg11_1 = None
        mul_tensor_177: "f64[200]" = torch.ops.aten.mul.Tensor(slice_tensor_731, slice_tensor_732);  slice_tensor_731 = slice_tensor_732 = None
        unsqueeze_default_52: "f64[1, 200]" = torch.ops.aten.unsqueeze.default(mul_tensor_177, 0);  mul_tensor_177 = None
        unsqueeze_default_53: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_52, 2);  unsqueeze_default_52 = None
        slice_tensor_733: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2);  arg12_1 = None
        slice_tensor_734: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_733, 1, 2, -2);  slice_tensor_733 = None
        slice_tensor_735: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_734, 2, 0, -1);  slice_tensor_734 = None
        mul_tensor_178: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_default_53, slice_tensor_735);  unsqueeze_default_53 = slice_tensor_735 = None
        mul_tensor_179: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_178, mul_tensor_142);  mul_tensor_178 = mul_tensor_142 = None
        pow_tensor_scalar_7: "f64[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_tensor_38, 2);  div_tensor_38 = None
        mul_tensor_180: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_179, pow_tensor_scalar_7);  mul_tensor_179 = pow_tensor_scalar_7 = None
        slice_tensor_736: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg5_1, 0, 2, -2);  arg5_1 = None
        slice_tensor_737: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_736, 1, 2, -2);  slice_tensor_736 = None
        slice_tensor_738: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_737, 2, 0, -1);  slice_tensor_737 = None
        mul_tensor_181: "f64[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_tensor_180, slice_tensor_738);  mul_tensor_180 = slice_tensor_738 = None
        add_tensor_82: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_tensor_730, mul_tensor_181);  slice_tensor_730 = mul_tensor_181 = None
        slice_scatter_default_125: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_727, add_tensor_82, 2, 0, -1);  slice_tensor_727 = add_tensor_82 = None
        slice_scatter_default_126: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_726, slice_scatter_default_125, 1, 2, -2);  slice_tensor_726 = slice_scatter_default_125 = None
        slice_scatter_default_127: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_124, slice_scatter_default_126, 0, 2, -2);  slice_scatter_default_124 = slice_scatter_default_126 = None
        slice_tensor_739: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_127, 0, 2, -2);  slice_scatter_default_127 = None
        slice_tensor_740: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_739, 1, 2, -2);  slice_tensor_739 = None
        slice_tensor_741: "f64[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_tensor_740, 2, 0, -1);  slice_tensor_740 = None
        unsqueeze_default_54: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg21_1, 0);  arg21_1 = None
        slice_tensor_742: "f64[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_default_54, 1, 2, -2);  unsqueeze_default_54 = None
        unsqueeze_default_55: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_742, 2);  slice_tensor_742 = None
        mul_tensor_182: "f64[1, 200, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_default_55, 4);  unsqueeze_default_55 = None
        unsqueeze_default_56: "f64[1, 204]" = torch.ops.aten.unsqueeze.default(arg9_1, 0);  arg9_1 = None
        slice_tensor_743: "f64[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_default_56, 1, 2, -2);  unsqueeze_default_56 = None
        unsqueeze_default_57: "f64[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_tensor_743, 2);  slice_tensor_743 = None
        mul_tensor_183: "f64[1, 200, 1]" = torch.ops.aten.mul.Tensor(mul_tensor_182, unsqueeze_default_57);  mul_tensor_182 = unsqueeze_default_57 = None
        div_tensor_41: "f64[200, 200, 25]" = torch.ops.aten.div.Tensor(slice_tensor_741, mul_tensor_183);  slice_tensor_741 = mul_tensor_183 = None
        add_tensor_83: "f64[200, 200, 25]" = torch.ops.aten.add.Tensor(div_tensor_40, div_tensor_41);  div_tensor_40 = div_tensor_41 = None
        copy_default_27: "f64[200, 200, 25]" = torch.ops.aten.copy.default(slice_tensor_634, add_tensor_83);  slice_tensor_634 = add_tensor_83 = None
        slice_scatter_default_128: "f64[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_631, copy_default_27, 2, 0, -1);  slice_tensor_631 = copy_default_27 = None
        slice_scatter_default_129: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_630, slice_scatter_default_128, 1, 2, -2);  slice_tensor_630 = slice_scatter_default_128 = None
        slice_scatter_default_130: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(arg22_1, slice_scatter_default_129, 0, 2, -2);  slice_scatter_default_129 = None
        slice_tensor_744: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_130, 0, 2, -2)
        slice_tensor_745: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_744, 1, 2, -2)
        slice_tensor_746: "f64[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_default_130, 0, 2, -2)
        slice_tensor_747: "f64[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_tensor_746, 1, 2, -2);  slice_tensor_746 = None
        select_int_67: "f64[200, 200]" = torch.ops.aten.select.int(slice_tensor_747, 2, -1);  slice_tensor_747 = None
        full_default_37: "f64[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default_28: "f64[200, 200]" = torch.ops.aten.copy.default(select_int_67, full_default_37);  select_int_67 = full_default_37 = None
        select_scatter_default_34: "f64[200, 200, 26]" = torch.ops.aten.select_scatter.default(slice_tensor_745, copy_default_28, 2, -1);  slice_tensor_745 = copy_default_28 = None
        slice_scatter_default_131: "f64[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_tensor_744, select_scatter_default_34, 1, 2, -2);  slice_tensor_744 = select_scatter_default_34 = None
        slice_scatter_default_132: "f64[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_130, slice_scatter_default_131, 0, 2, -2);  slice_scatter_default_130 = slice_scatter_default_131 = None
        copy__default: "f64[204, 204, 26]" = torch.ops.aten.copy_.default(arg0_1, slice_scatter_default_40);  arg0_1 = slice_scatter_default_40 = None
        copy__default_1: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.copy_.default(arg13_1, slice_scatter_default_36);  arg13_1 = slice_scatter_default_36 = None
        copy__default_2: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.copy_.default(arg15_1, slice_scatter_default_75);  arg15_1 = slice_scatter_default_75 = None
        copy__default_3: "f64[204, 204, 26]" = torch.ops.aten.copy_.default(arg16_1, slice_scatter_default_79);  arg16_1 = slice_scatter_default_79 = None
        copy__default_4: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.copy_.default(arg17_1, slice_scatter_default_91);  arg17_1 = slice_scatter_default_91 = None
        copy__default_5: "f64[204, 204, 26, 2, 2]" = torch.ops.aten.copy_.default(arg19_1, slice_scatter_default_103);  arg19_1 = slice_scatter_default_103 = None
        copy__default_6: "f64[204, 204, 26]" = torch.ops.aten.copy_.default(arg22_1, slice_scatter_default_132);  arg22_1 = slice_scatter_default_132 = None
        return (slice_tensor, slice_tensor_1, slice_tensor_83, slice_tensor_127, slice_tensor_157, slice_tensor_185, slice_tensor_297, slice_tensor_341, slice_tensor_371, slice_tensor_399, copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6)

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
