"""
Standalone repro captured via capture_hook.
Label: torchbench_pyhpc_isoneutral_mixing_infer
Pattern hash: 656833e29d89
Shape hash: b115bb1f
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[26]", arg1_1: "bf16[204, 204, 26]", arg2_1: "bf16[204, 204, 26]", arg3_1: "bf16[204, 204, 26, 3]", arg4_1: "bf16[26]", arg5_1: "bf16[204, 204, 26]", arg6_1: "bf16[204]", arg7_1: "bf16[204]", arg8_1: "bf16[204, 204, 26, 3]", arg9_1: "bf16[204, 204, 26]", arg10_1: "bf16[204, 204, 26, 2, 2]", arg11_1: "bf16[204, 204, 26]", arg12_1: "bf16[26]", arg13_1: "bf16[204, 204, 26]", arg14_1: "bf16[204]", arg15_1: "bf16[204, 204, 26, 2, 2]", arg16_1: "bf16[204, 204, 26]", arg17_1: "bf16[204, 204, 26, 2, 2]", arg18_1: "bf16[204, 204, 26, 2, 2]", arg19_1: "bf16[204, 204, 26]", arg20_1: "bf16[204]", arg21_1: "bf16[204]", arg22_1: "bf16[204]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # No stacktrace found for following nodes
        full: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        slice_1: "bf16[203, 204, 26]" = torch.ops.aten.slice.Tensor(full, 0, 0, -1);  slice_1 = None
        full_1: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        slice_2: "bf16[203, 204, 26]" = torch.ops.aten.slice.Tensor(full_1, 0, 0, -1);  slice_2 = None
        full_2: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_2, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_2 = None
        slice_3: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(full_2, 0, 1, -2)
        slice_4: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_3, 1, 2, -2)
        slice_5: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(full_2, 0, 1, -2)
        slice_6: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_5, 1, 2, -2);  slice_5 = None
        slice_7: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_6, 2, 1, 9223372036854775807);  slice_6 = None
        unsqueeze: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg0_1, 0)
        unsqueeze_1: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        slice_8: "bf16[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_1, 2, 0, 25);  unsqueeze_1 = None
        slice_9: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 1, -2)
        slice_10: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_9, 1, 2, -2);  slice_9 = None
        slice_11: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_10, 2, 1, 9223372036854775807);  slice_10 = None
        mul: "bf16[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_8, slice_11);  slice_8 = slice_11 = None
        full_3: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_4: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        slice_12: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(full_4, 0, 1, -2)
        slice_13: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_12, 1, 2, -2)
        slice_14: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(full_4, 0, 1, -2)
        slice_15: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_14, 1, 2, -2);  slice_14 = None
        slice_16: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_15, 2, 1, 9223372036854775807);  slice_15 = None
        slice_17: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 1, -2)
        slice_18: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_17, 1, 2, -2);  slice_17 = None
        slice_19: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_18, 2, 1, 9223372036854775807);  slice_18 = None
        slice_20: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 1, -2)
        slice_21: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_20, 1, 2, -2);  slice_20 = None
        slice_22: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_21, 2, 0, -1);  slice_21 = None
        add: "bf16[201, 200, 25]" = torch.ops.aten.add.Tensor(slice_19, slice_22);  slice_19 = slice_22 = None
        slice_23: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -1)
        slice_24: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_23, 1, 2, -2);  slice_23 = None
        slice_25: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_24, 2, 1, 9223372036854775807);  slice_24 = None
        add_1: "bf16[201, 200, 25]" = torch.ops.aten.add.Tensor(add, slice_25);  add = slice_25 = None
        slice_26: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -1)
        slice_27: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_26, 1, 2, -2);  slice_26 = None
        slice_28: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_27, 2, 0, -1);  slice_27 = None
        add_2: "bf16[201, 200, 25]" = torch.ops.aten.add.Tensor(add_1, slice_28);  add_1 = slice_28 = None
        mul_1: "bf16[201, 200, 25]" = torch.ops.aten.mul.Tensor(add_2, 0.25);  add_2 = None
        copy: "bf16[201, 200, 25]" = torch.ops.aten.copy.default(slice_16, mul_1);  slice_16 = mul_1 = None
        slice_scatter: "bf16[201, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_13, copy, 2, 1, 9223372036854775807);  slice_13 = copy = None
        slice_scatter_1: "bf16[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_12, slice_scatter, 1, 2, -2);  slice_12 = slice_scatter = None
        slice_scatter_2: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_4, slice_scatter_1, 0, 1, -2);  full_4 = slice_scatter_1 = None
        slice_29: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 1, -2)
        slice_30: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_29, 1, 2, -2)
        slice_31: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 1, -2)
        slice_32: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_31, 1, 2, -2);  slice_31 = None
        select: "bf16[201, 200]" = torch.ops.aten.select.int(slice_32, 2, 0);  slice_32 = None
        slice_33: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 1, -2)
        slice_34: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_33, 1, 2, -2);  slice_33 = None
        select_1: "bf16[201, 200]" = torch.ops.aten.select.int(slice_34, 2, 0);  slice_34 = None
        slice_35: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -1)
        slice_36: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_35, 1, 2, -2);  slice_35 = None
        select_2: "bf16[201, 200]" = torch.ops.aten.select.int(slice_36, 2, 0);  slice_36 = None
        add_3: "bf16[201, 200]" = torch.ops.aten.add.Tensor(select_1, select_2);  select_1 = select_2 = None
        mul_2: "bf16[201, 200]" = torch.ops.aten.mul.Tensor(add_3, 0.5);  add_3 = None
        copy_1: "bf16[201, 200]" = torch.ops.aten.copy.default(select, mul_2);  select = mul_2 = None
        select_scatter: "bf16[201, 200, 26]" = torch.ops.aten.select_scatter.default(slice_30, copy_1, 2, 0);  slice_30 = copy_1 = None
        slice_scatter_3: "bf16[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_29, select_scatter, 1, 2, -2);  slice_29 = select_scatter = None
        slice_scatter_4: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_2, slice_scatter_3, 0, 1, -2);  slice_scatter_2 = slice_scatter_3 = None
        slice_37: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_4, 0, 1, -2)
        slice_38: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_37, 1, 2, -2);  slice_37 = None
        slice_39: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_38, 2, 1, 9223372036854775807);  slice_38 = None
        select_3: "bf16[204, 204, 26]" = torch.ops.aten.select.int(arg3_1, 3, 0)
        sub: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(select_3, 9.850000000000023);  select_3 = None
        mul_3: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub, 1e-05);  sub = None
        abs_1: "bf16[26]" = torch.ops.aten.abs.default(arg4_1);  arg4_1 = None
        neg: "bf16[26]" = torch.ops.aten.neg.default(abs_1);  abs_1 = None
        sub_1: "bf16[26]" = torch.ops.aten.sub.Tensor(neg, 0.0);  neg = None
        mul_4: "bf16[26]" = torch.ops.aten.mul.Tensor(sub_1, 1.0790999999999999e-07);  sub_1 = None
        mul_5: "bf16[26]" = torch.ops.aten.mul.Tensor(mul_4, 1024.0);  mul_4 = None
        sub_2: "bf16[26]" = torch.ops.aten.sub.Tensor(1, mul_5);  mul_5 = None
        mul_6: "bf16[26]" = torch.ops.aten.mul.Tensor(sub_2, 0.000167);  sub_2 = None
        add_4: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(mul_3, mul_6);  mul_3 = mul_6 = None
        neg_1: "bf16[204, 204, 26]" = torch.ops.aten.neg.default(add_4);  add_4 = None
        mul_7: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(neg_1, 1024.0);  neg_1 = None
        mul_8: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg5_1, mul_7);  mul_7 = None
        slice_40: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 1, -2)
        slice_41: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_40, 1, 2, -2);  slice_40 = None
        slice_42: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_41, 2, 1, 9223372036854775807);  slice_41 = None
        slice_43: "bf16[203, 204, 26]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 0, -1)
        slice_44: "bf16[203, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 1, 9223372036854775807)
        select_4: "bf16[203, 204, 26]" = torch.ops.aten.select.int(slice_44, 3, 0);  slice_44 = None
        slice_45: "bf16[203, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 0, -1)
        select_5: "bf16[203, 204, 26]" = torch.ops.aten.select.int(slice_45, 3, 0);  slice_45 = None
        sub_3: "bf16[203, 204, 26]" = torch.ops.aten.sub.Tensor(select_4, select_5);  select_4 = select_5 = None
        mul_9: "bf16[203, 204, 26]" = torch.ops.aten.mul.Tensor(slice_43, sub_3);  slice_43 = sub_3 = None
        slice_46: "bf16[203]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 0, -1)
        unsqueeze_2: "bf16[203, 1]" = torch.ops.aten.unsqueeze.default(slice_46, 1);  slice_46 = None
        unsqueeze_3: "bf16[203, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 2);  unsqueeze_2 = None
        unsqueeze_4: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(arg7_1, 0)
        unsqueeze_5: "bf16[1, 204, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        mul_10: "bf16[203, 204, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_3, unsqueeze_5);  unsqueeze_3 = unsqueeze_5 = None
        div: "bf16[203, 204, 26]" = torch.ops.aten.div.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
        slice_scatter_5: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full, div, 0, 0, -1);  full = div = None
        slice_47: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 1, -2)
        slice_48: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_47, 1, 2, -2);  slice_47 = None
        slice_49: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_48, 2, 1, 9223372036854775807);  slice_48 = None
        mul_11: "bf16[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_42, slice_49);  slice_42 = slice_49 = None
        full_5: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_4, 0.796875, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        mul_12: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(arg5_1, full_5);  arg5_1 = full_5 = None
        slice_50: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 1, -2)
        slice_51: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_50, 1, 2, -2);  slice_50 = None
        slice_52: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_51, 2, 1, 9223372036854775807);  slice_51 = None
        slice_53: "bf16[203, 204, 26]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 0, -1)
        slice_54: "bf16[203, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 1, 9223372036854775807)
        select_6: "bf16[203, 204, 26]" = torch.ops.aten.select.int(slice_54, 3, 0);  slice_54 = None
        slice_55: "bf16[203, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 0, -1)
        select_7: "bf16[203, 204, 26]" = torch.ops.aten.select.int(slice_55, 3, 0);  slice_55 = None
        sub_4: "bf16[203, 204, 26]" = torch.ops.aten.sub.Tensor(select_6, select_7);  select_6 = select_7 = None
        mul_13: "bf16[203, 204, 26]" = torch.ops.aten.mul.Tensor(slice_53, sub_4);  slice_53 = sub_4 = None
        slice_56: "bf16[203]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 0, -1)
        unsqueeze_6: "bf16[203, 1]" = torch.ops.aten.unsqueeze.default(slice_56, 1);  slice_56 = None
        unsqueeze_7: "bf16[203, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(arg7_1, 0)
        unsqueeze_9: "bf16[1, 204, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 2);  unsqueeze_8 = None
        mul_14: "bf16[203, 204, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_7, unsqueeze_9);  unsqueeze_7 = unsqueeze_9 = None
        div_1: "bf16[203, 204, 26]" = torch.ops.aten.div.Tensor(mul_13, mul_14);  mul_13 = mul_14 = None
        slice_scatter_6: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_1, div_1, 0, 0, -1);  full_1 = div_1 = None
        slice_57: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_6, 0, 1, -2)
        slice_58: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_57, 1, 2, -2);  slice_57 = None
        slice_59: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_58, 2, 1, 9223372036854775807);  slice_58 = None
        mul_15: "bf16[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_52, slice_59);  slice_52 = slice_59 = None
        add_5: "bf16[201, 200, 25]" = torch.ops.aten.add.Tensor(mul_11, mul_15);  mul_11 = mul_15 = None
        neg_2: "bf16[201, 200, 25]" = torch.ops.aten.neg.default(add_5);  add_5 = None
        slice_60: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 1, -2)
        slice_61: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_60, 1, 2, -2);  slice_60 = None
        slice_62: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_61, 2, 1, 9223372036854775807);  slice_61 = None
        full_6: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_5, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_5 = None
        slice_63: "bf16[204, 204, 25]" = torch.ops.aten.slice.Tensor(full_6, 2, 0, -1)
        slice_64: "bf16[204, 204, 25]" = torch.ops.aten.slice.Tensor(arg9_1, 2, 0, -1)
        slice_65: "bf16[204, 204, 25, 3]" = torch.ops.aten.slice.Tensor(arg3_1, 2, 1, 9223372036854775807)
        select_8: "bf16[204, 204, 25]" = torch.ops.aten.select.int(slice_65, 3, 0);  slice_65 = None
        slice_66: "bf16[204, 204, 25, 3]" = torch.ops.aten.slice.Tensor(arg3_1, 2, 0, -1)
        select_9: "bf16[204, 204, 25]" = torch.ops.aten.select.int(slice_66, 3, 0);  slice_66 = None
        sub_5: "bf16[204, 204, 25]" = torch.ops.aten.sub.Tensor(select_8, select_9);  select_8 = select_9 = None
        mul_16: "bf16[204, 204, 25]" = torch.ops.aten.mul.Tensor(slice_64, sub_5);  slice_64 = sub_5 = None
        unsqueeze_10: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg0_1, 0)
        unsqueeze_11: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 1);  unsqueeze_10 = None
        slice_67: "bf16[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_11, 2, 0, -1);  unsqueeze_11 = None
        div_2: "bf16[204, 204, 25]" = torch.ops.aten.div.Tensor(mul_16, slice_67);  mul_16 = slice_67 = None
        copy_2: "bf16[204, 204, 25]" = torch.ops.aten.copy.default(slice_63, div_2);  slice_63 = div_2 = None
        slice_scatter_7: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_6, copy_2, 2, 0, -1);  full_6 = copy_2 = None
        slice_68: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 1, -2)
        slice_69: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_68, 1, 2, -2);  slice_68 = None
        slice_70: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_69, 2, 0, 25);  slice_69 = None
        mul_17: "bf16[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_62, slice_70);  slice_62 = slice_70 = None
        slice_71: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 1, -2)
        slice_72: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_71, 1, 2, -2);  slice_71 = None
        slice_73: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_72, 2, 1, 9223372036854775807);  slice_72 = None
        full_7: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_6, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_6 = None
        slice_74: "bf16[204, 204, 25]" = torch.ops.aten.slice.Tensor(full_7, 2, 0, -1)
        slice_75: "bf16[204, 204, 25]" = torch.ops.aten.slice.Tensor(arg9_1, 2, 0, -1)
        slice_76: "bf16[204, 204, 25, 3]" = torch.ops.aten.slice.Tensor(arg8_1, 2, 1, 9223372036854775807)
        select_10: "bf16[204, 204, 25]" = torch.ops.aten.select.int(slice_76, 3, 0);  slice_76 = None
        slice_77: "bf16[204, 204, 25, 3]" = torch.ops.aten.slice.Tensor(arg8_1, 2, 0, -1)
        select_11: "bf16[204, 204, 25]" = torch.ops.aten.select.int(slice_77, 3, 0);  slice_77 = None
        sub_6: "bf16[204, 204, 25]" = torch.ops.aten.sub.Tensor(select_10, select_11);  select_10 = select_11 = None
        mul_18: "bf16[204, 204, 25]" = torch.ops.aten.mul.Tensor(slice_75, sub_6);  slice_75 = sub_6 = None
        unsqueeze_12: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg0_1, 0)
        unsqueeze_13: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, 1);  unsqueeze_12 = None
        slice_78: "bf16[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_13, 2, 0, -1);  unsqueeze_13 = None
        div_3: "bf16[204, 204, 25]" = torch.ops.aten.div.Tensor(mul_18, slice_78);  mul_18 = slice_78 = None
        copy_3: "bf16[204, 204, 25]" = torch.ops.aten.copy.default(slice_74, div_3);  slice_74 = div_3 = None
        slice_scatter_8: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_7, copy_3, 2, 0, -1);  full_7 = copy_3 = None
        slice_79: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_8, 0, 1, -2)
        slice_80: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_79, 1, 2, -2);  slice_79 = None
        slice_81: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_80, 2, 0, 25);  slice_80 = None
        mul_19: "bf16[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_73, slice_81);  slice_73 = slice_81 = None
        add_6: "bf16[201, 200, 25]" = torch.ops.aten.add.Tensor(mul_17, mul_19);  mul_17 = mul_19 = None
        full_8: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum: "f32[201, 200, 25]" = torch.ops.aten.minimum.default(add_6, full_8);  add_6 = full_8 = None
        sub_7: "f32[201, 200, 25]" = torch.ops.aten.sub.Tensor(minimum, 1e-20);  minimum = None
        div_4: "f32[201, 200, 25]" = torch.ops.aten.div.Tensor(neg_2, sub_7);  neg_2 = sub_7 = None
        abs_2: "f32[201, 200, 25]" = torch.ops.aten.abs.default(div_4)
        neg_3: "f32[201, 200, 25]" = torch.ops.aten.neg.default(abs_2);  abs_2 = None
        add_7: "f32[201, 200, 25]" = torch.ops.aten.add.Tensor(neg_3, 0.001);  neg_3 = None
        div_5: "f32[201, 200, 25]" = torch.ops.aten.div.Tensor(add_7, 0.001);  add_7 = None
        tanh: "f32[201, 200, 25]" = torch.ops.aten.tanh.default(div_5);  div_5 = None
        add_8: "f32[201, 200, 25]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_20: "f32[201, 200, 25]" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
        mul_21: "f32[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_39, mul_20);  slice_39 = None
        maximum: "f32[201, 200, 25]" = torch.ops.aten.maximum.default(full_3, mul_21);  full_3 = mul_21 = None
        mul_22: "f32[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul, maximum);  mul = maximum = None
        add_9: "f32[201, 200, 25]" = torch.ops.aten.add.Tensor(slice_7, mul_22);  slice_7 = mul_22 = None
        convert_element_type: "bf16[201, 200, 25]" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        slice_scatter_9: "bf16[201, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_4, convert_element_type, 2, 1, 9223372036854775807);  slice_4 = convert_element_type = None
        slice_scatter_10: "bf16[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_3, slice_scatter_9, 1, 2, -2);  slice_3 = slice_scatter_9 = None
        slice_scatter_11: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_2, slice_scatter_10, 0, 1, -2);  full_2 = slice_scatter_10 = None
        slice_82: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_11, 0, 1, -2)
        slice_83: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_82, 1, 2, -2);  slice_82 = None
        slice_84: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_83, 2, 1, 9223372036854775807);  slice_83 = slice_84 = None
        slice_85: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_11, 0, 1, -2)
        slice_86: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_85, 1, 2, -2)
        slice_87: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_11, 0, 1, -2)
        slice_88: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_87, 1, 2, -2);  slice_87 = None
        slice_89: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_88, 2, 1, 9223372036854775807);  slice_88 = None
        slice_scatter_12: "bf16[201, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_86, slice_89, 2, 1, 9223372036854775807);  slice_86 = slice_89 = None
        slice_scatter_13: "bf16[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_85, slice_scatter_12, 1, 2, -2);  slice_85 = slice_scatter_12 = None
        slice_scatter_14: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_11, slice_scatter_13, 0, 1, -2);  slice_scatter_11 = slice_scatter_13 = None
        slice_90: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_14, 0, 1, -2)
        slice_91: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_90, 1, 2, -2)
        slice_92: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_14, 0, 1, -2)
        slice_93: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_92, 1, 2, -2);  slice_92 = None
        slice_94: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_93, 2, 1, 9223372036854775807);  slice_93 = None
        unsqueeze_14: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg0_1, 0)
        unsqueeze_15: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 1);  unsqueeze_14 = None
        slice_95: "bf16[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_15, 2, 0, 25);  unsqueeze_15 = None
        slice_96: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 1, -2)
        slice_97: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_96, 1, 2, -2);  slice_96 = None
        slice_98: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_97, 2, 1, 9223372036854775807);  slice_97 = None
        mul_23: "bf16[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_95, slice_98);  slice_95 = slice_98 = None
        full_9: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_99: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_4, 0, 1, -2)
        slice_100: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_99, 1, 2, -2);  slice_99 = None
        slice_101: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_100, 2, 1, 9223372036854775807);  slice_100 = None
        slice_102: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -1)
        slice_103: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_102, 1, 2, -2);  slice_102 = None
        slice_104: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_103, 2, 1, 9223372036854775807);  slice_103 = None
        slice_105: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 1, -2)
        slice_106: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_105, 1, 2, -2);  slice_105 = None
        slice_107: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_106, 2, 1, 9223372036854775807);  slice_106 = None
        mul_24: "bf16[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_104, slice_107);  slice_104 = slice_107 = None
        slice_108: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -1)
        slice_109: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_108, 1, 2, -2);  slice_108 = None
        slice_110: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_109, 2, 1, 9223372036854775807);  slice_109 = None
        slice_111: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_6, 0, 1, -2)
        slice_112: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_111, 1, 2, -2);  slice_111 = None
        slice_113: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_112, 2, 1, 9223372036854775807);  slice_112 = None
        mul_25: "bf16[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_110, slice_113);  slice_110 = slice_113 = None
        add_10: "bf16[201, 200, 25]" = torch.ops.aten.add.Tensor(mul_24, mul_25);  mul_24 = mul_25 = None
        neg_4: "bf16[201, 200, 25]" = torch.ops.aten.neg.default(add_10);  add_10 = None
        slice_114: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -1)
        slice_115: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_114, 1, 2, -2);  slice_114 = None
        slice_116: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_115, 2, 1, 9223372036854775807);  slice_115 = None
        slice_117: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 2, -1)
        slice_118: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_117, 1, 2, -2);  slice_117 = None
        slice_119: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_118, 2, 0, 25);  slice_118 = None
        mul_26: "bf16[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_116, slice_119);  slice_116 = slice_119 = None
        slice_120: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -1)
        slice_121: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_120, 1, 2, -2);  slice_120 = None
        slice_122: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_121, 2, 1, 9223372036854775807);  slice_121 = None
        slice_123: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_8, 0, 2, -1)
        slice_124: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_123, 1, 2, -2);  slice_123 = None
        slice_125: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_124, 2, 0, 25);  slice_124 = None
        mul_27: "bf16[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_122, slice_125);  slice_122 = slice_125 = None
        add_11: "bf16[201, 200, 25]" = torch.ops.aten.add.Tensor(mul_26, mul_27);  mul_26 = mul_27 = None
        full_10: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_1: "f32[201, 200, 25]" = torch.ops.aten.minimum.default(add_11, full_10);  add_11 = full_10 = None
        sub_8: "f32[201, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_1, 1e-20);  minimum_1 = None
        div_6: "f32[201, 200, 25]" = torch.ops.aten.div.Tensor(neg_4, sub_8);  neg_4 = sub_8 = None
        abs_3: "f32[201, 200, 25]" = torch.ops.aten.abs.default(div_6)
        neg_5: "f32[201, 200, 25]" = torch.ops.aten.neg.default(abs_3);  abs_3 = None
        add_12: "f32[201, 200, 25]" = torch.ops.aten.add.Tensor(neg_5, 0.001);  neg_5 = None
        div_7: "f32[201, 200, 25]" = torch.ops.aten.div.Tensor(add_12, 0.001);  add_12 = None
        tanh_1: "f32[201, 200, 25]" = torch.ops.aten.tanh.default(div_7);  div_7 = None
        add_13: "f32[201, 200, 25]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_28: "f32[201, 200, 25]" = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None
        mul_29: "f32[201, 200, 25]" = torch.ops.aten.mul.Tensor(slice_101, mul_28);  slice_101 = None
        maximum_1: "f32[201, 200, 25]" = torch.ops.aten.maximum.default(full_9, mul_29);  full_9 = mul_29 = None
        mul_30: "f32[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul_23, maximum_1);  mul_23 = maximum_1 = None
        add_14: "f32[201, 200, 25]" = torch.ops.aten.add.Tensor(slice_94, mul_30);  slice_94 = mul_30 = None
        convert_element_type_1: "bf16[201, 200, 25]" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None
        slice_scatter_15: "bf16[201, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_91, convert_element_type_1, 2, 1, 9223372036854775807);  slice_91 = convert_element_type_1 = None
        slice_scatter_16: "bf16[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_90, slice_scatter_15, 1, 2, -2);  slice_90 = slice_scatter_15 = None
        slice_scatter_17: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_14, slice_scatter_16, 0, 1, -2);  slice_scatter_14 = slice_scatter_16 = None
        slice_126: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_17, 0, 1, -2)
        slice_127: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_126, 1, 2, -2);  slice_126 = None
        slice_128: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_127, 2, 1, 9223372036854775807);  slice_127 = slice_128 = None
        slice_129: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_17, 0, 1, -2)
        slice_130: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_129, 1, 2, -2)
        slice_131: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_17, 0, 1, -2)
        slice_132: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_131, 1, 2, -2);  slice_131 = None
        slice_133: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_132, 2, 1, 9223372036854775807);  slice_132 = None
        slice_scatter_18: "bf16[201, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_130, slice_133, 2, 1, 9223372036854775807);  slice_130 = slice_133 = None
        slice_scatter_19: "bf16[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_129, slice_scatter_18, 1, 2, -2);  slice_129 = slice_scatter_18 = None
        slice_scatter_20: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_17, slice_scatter_19, 0, 1, -2);  slice_scatter_17 = slice_scatter_19 = None
        slice_134: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_20, 0, 1, -2)
        slice_135: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_20, 0, 1, -2)
        slice_136: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_135, 1, 2, -2);  slice_135 = None
        unsqueeze_16: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg0_1, 0)
        unsqueeze_17: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 1);  unsqueeze_16 = None
        slice_137: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 1, -2)
        slice_138: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_137, 1, 2, -2);  slice_137 = None
        mul_31: "bf16[201, 200, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_17, slice_138);  unsqueeze_17 = slice_138 = None
        full_11: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_139: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_4, 0, 1, -2)
        slice_140: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_139, 1, 2, -2);  slice_139 = None
        slice_141: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 1, -2)
        slice_142: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_141, 1, 2, -2);  slice_141 = None
        slice_143: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 1, -2)
        slice_144: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_143, 1, 2, -2);  slice_143 = None
        mul_32: "bf16[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_142, slice_144);  slice_142 = slice_144 = None
        slice_145: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 1, -2)
        slice_146: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_145, 1, 2, -2);  slice_145 = None
        slice_147: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_6, 0, 1, -2)
        slice_148: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_147, 1, 2, -2);  slice_147 = None
        mul_33: "bf16[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_146, slice_148);  slice_146 = slice_148 = None
        add_15: "bf16[201, 200, 26]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        neg_6: "bf16[201, 200, 26]" = torch.ops.aten.neg.default(add_15);  add_15 = None
        slice_149: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 1, -2)
        slice_150: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_149, 1, 2, -2);  slice_149 = None
        slice_151: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 1, -2)
        slice_152: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_151, 1, 2, -2);  slice_151 = None
        mul_34: "bf16[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_150, slice_152);  slice_150 = slice_152 = None
        slice_153: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 1, -2)
        slice_154: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_153, 1, 2, -2);  slice_153 = None
        slice_155: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_8, 0, 1, -2)
        slice_156: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_155, 1, 2, -2);  slice_155 = None
        mul_35: "bf16[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_154, slice_156);  slice_154 = slice_156 = None
        add_16: "bf16[201, 200, 26]" = torch.ops.aten.add.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None
        full_12: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_2: "f32[201, 200, 26]" = torch.ops.aten.minimum.default(add_16, full_12);  add_16 = full_12 = None
        sub_9: "f32[201, 200, 26]" = torch.ops.aten.sub.Tensor(minimum_2, 1e-20);  minimum_2 = None
        div_8: "f32[201, 200, 26]" = torch.ops.aten.div.Tensor(neg_6, sub_9);  neg_6 = sub_9 = None
        abs_4: "f32[201, 200, 26]" = torch.ops.aten.abs.default(div_8)
        neg_7: "f32[201, 200, 26]" = torch.ops.aten.neg.default(abs_4);  abs_4 = None
        add_17: "f32[201, 200, 26]" = torch.ops.aten.add.Tensor(neg_7, 0.001);  neg_7 = None
        div_9: "f32[201, 200, 26]" = torch.ops.aten.div.Tensor(add_17, 0.001);  add_17 = None
        tanh_2: "f32[201, 200, 26]" = torch.ops.aten.tanh.default(div_9);  div_9 = None
        add_18: "f32[201, 200, 26]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_36: "f32[201, 200, 26]" = torch.ops.aten.mul.Tensor(add_18, 0.5);  add_18 = None
        mul_37: "f32[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_140, mul_36);  slice_140 = None
        maximum_2: "f32[201, 200, 26]" = torch.ops.aten.maximum.default(full_11, mul_37);  full_11 = mul_37 = None
        mul_38: "f32[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_31, maximum_2);  mul_31 = maximum_2 = None
        add_19: "f32[201, 200, 26]" = torch.ops.aten.add.Tensor(slice_136, mul_38);  slice_136 = mul_38 = None
        convert_element_type_2: "bf16[201, 200, 26]" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None
        slice_scatter_21: "bf16[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_134, convert_element_type_2, 1, 2, -2);  slice_134 = convert_element_type_2 = None
        slice_scatter_22: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_20, slice_scatter_21, 0, 1, -2);  slice_scatter_20 = slice_scatter_21 = None
        slice_157: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_22, 0, 1, -2)
        slice_158: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_157, 1, 2, -2);  slice_157 = slice_158 = None
        slice_159: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_22, 0, 1, -2)
        slice_160: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_22, 0, 1, -2)
        slice_161: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_160, 1, 2, -2);  slice_160 = None
        slice_scatter_23: "bf16[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_159, slice_161, 1, 2, -2);  slice_159 = slice_161 = None
        slice_scatter_24: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_22, slice_scatter_23, 0, 1, -2);  slice_scatter_22 = slice_scatter_23 = None
        slice_162: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_24, 0, 1, -2)
        slice_163: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_24, 0, 1, -2)
        slice_164: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_163, 1, 2, -2);  slice_163 = None
        unsqueeze_18: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg0_1, 0)
        unsqueeze_19: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 1);  unsqueeze_18 = None
        slice_165: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 1, -2)
        slice_166: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_165, 1, 2, -2);  slice_165 = None
        mul_39: "bf16[201, 200, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_19, slice_166);  unsqueeze_19 = slice_166 = None
        full_13: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_167: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_4, 0, 1, -2)
        slice_168: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_167, 1, 2, -2);  slice_167 = None
        slice_169: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -1)
        slice_170: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_169, 1, 2, -2);  slice_169 = None
        slice_171: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 1, -2)
        slice_172: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_171, 1, 2, -2);  slice_171 = None
        mul_40: "bf16[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_170, slice_172);  slice_170 = slice_172 = None
        slice_173: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -1)
        slice_174: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_173, 1, 2, -2);  slice_173 = None
        slice_175: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_6, 0, 1, -2)
        slice_176: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_175, 1, 2, -2);  slice_175 = None
        mul_41: "bf16[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_174, slice_176);  slice_174 = slice_176 = None
        add_20: "bf16[201, 200, 26]" = torch.ops.aten.add.Tensor(mul_40, mul_41);  mul_40 = mul_41 = None
        neg_8: "bf16[201, 200, 26]" = torch.ops.aten.neg.default(add_20);  add_20 = None
        slice_177: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -1)
        slice_178: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_177, 1, 2, -2);  slice_177 = None
        slice_179: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 2, -1)
        slice_180: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_179, 1, 2, -2);  slice_179 = None
        mul_42: "bf16[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_178, slice_180);  slice_178 = slice_180 = None
        slice_181: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -1)
        slice_182: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_181, 1, 2, -2);  slice_181 = None
        slice_183: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_8, 0, 2, -1)
        slice_184: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_183, 1, 2, -2);  slice_183 = None
        mul_43: "bf16[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_182, slice_184);  slice_182 = slice_184 = None
        add_21: "bf16[201, 200, 26]" = torch.ops.aten.add.Tensor(mul_42, mul_43);  mul_42 = mul_43 = None
        full_14: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_3: "f32[201, 200, 26]" = torch.ops.aten.minimum.default(add_21, full_14);  add_21 = full_14 = None
        sub_10: "f32[201, 200, 26]" = torch.ops.aten.sub.Tensor(minimum_3, 1e-20);  minimum_3 = None
        div_10: "f32[201, 200, 26]" = torch.ops.aten.div.Tensor(neg_8, sub_10);  neg_8 = sub_10 = None
        abs_5: "f32[201, 200, 26]" = torch.ops.aten.abs.default(div_10)
        neg_9: "f32[201, 200, 26]" = torch.ops.aten.neg.default(abs_5);  abs_5 = None
        add_22: "f32[201, 200, 26]" = torch.ops.aten.add.Tensor(neg_9, 0.001);  neg_9 = None
        div_11: "f32[201, 200, 26]" = torch.ops.aten.div.Tensor(add_22, 0.001);  add_22 = None
        tanh_3: "f32[201, 200, 26]" = torch.ops.aten.tanh.default(div_11);  div_11 = None
        add_23: "f32[201, 200, 26]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_44: "f32[201, 200, 26]" = torch.ops.aten.mul.Tensor(add_23, 0.5);  add_23 = None
        mul_45: "f32[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_168, mul_44);  slice_168 = None
        maximum_3: "f32[201, 200, 26]" = torch.ops.aten.maximum.default(full_13, mul_45);  full_13 = mul_45 = None
        mul_46: "f32[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_39, maximum_3);  mul_39 = maximum_3 = None
        add_24: "f32[201, 200, 26]" = torch.ops.aten.add.Tensor(slice_164, mul_46);  slice_164 = mul_46 = None
        convert_element_type_3: "bf16[201, 200, 26]" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None
        slice_scatter_25: "bf16[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_162, convert_element_type_3, 1, 2, -2);  slice_162 = convert_element_type_3 = None
        slice_scatter_26: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_24, slice_scatter_25, 0, 1, -2);  slice_scatter_24 = slice_scatter_25 = None
        slice_185: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_26, 0, 1, -2)
        slice_186: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_185, 1, 2, -2);  slice_185 = slice_186 = None
        slice_187: "bf16[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 1, -2)
        slice_188: "bf16[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_187, 1, 2, -2)
        slice_189: "bf16[201, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_188, 2, 1, 9223372036854775807)
        select_12: "bf16[201, 200, 25, 2]" = torch.ops.aten.select.int(slice_189, 3, 0)
        slice_190: "bf16[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 1, -2)
        slice_191: "bf16[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_190, 1, 2, -2);  slice_190 = None
        slice_192: "bf16[201, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_191, 2, 1, 9223372036854775807);  slice_191 = None
        select_13: "bf16[201, 200, 25, 2]" = torch.ops.aten.select.int(slice_192, 3, 0);  slice_192 = None
        select_14: "bf16[201, 200, 25]" = torch.ops.aten.select.int(select_13, 3, 0);  select_13 = None
        mul_47: "f32[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul_20, div_4);  mul_20 = div_4 = None
        slice_193: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 1, -2)
        slice_194: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_193, 1, 2, -2);  slice_193 = None
        slice_195: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_194, 2, 1, 9223372036854775807);  slice_194 = None
        mul_48: "f32[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul_47, slice_195);  mul_47 = slice_195 = None
        copy_4: "bf16[201, 200, 25]" = torch.ops.aten.copy.default(select_14, mul_48);  select_14 = mul_48 = None
        select_scatter_1: "bf16[201, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_12, copy_4, 3, 0);  select_12 = copy_4 = None
        select_scatter_2: "bf16[201, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_189, select_scatter_1, 3, 0);  slice_189 = select_scatter_1 = None
        slice_scatter_27: "bf16[201, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_188, select_scatter_2, 2, 1, 9223372036854775807);  slice_188 = select_scatter_2 = None
        slice_scatter_28: "bf16[201, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_187, slice_scatter_27, 1, 2, -2);  slice_187 = slice_scatter_27 = None
        slice_scatter_29: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(arg10_1, slice_scatter_28, 0, 1, -2);  slice_scatter_28 = None
        slice_196: "bf16[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_29, 0, 1, -2)
        slice_197: "bf16[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_196, 1, 2, -2)
        slice_198: "bf16[201, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_197, 2, 1, 9223372036854775807)
        select_15: "bf16[201, 200, 25, 2]" = torch.ops.aten.select.int(slice_198, 3, 1)
        slice_199: "bf16[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_29, 0, 1, -2)
        slice_200: "bf16[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_199, 1, 2, -2);  slice_199 = None
        slice_201: "bf16[201, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_200, 2, 1, 9223372036854775807);  slice_200 = None
        select_16: "bf16[201, 200, 25, 2]" = torch.ops.aten.select.int(slice_201, 3, 1);  slice_201 = None
        select_17: "bf16[201, 200, 25]" = torch.ops.aten.select.int(select_16, 3, 0);  select_16 = None
        mul_49: "f32[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul_28, div_6);  mul_28 = div_6 = None
        slice_202: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 1, -2)
        slice_203: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_202, 1, 2, -2);  slice_202 = None
        slice_204: "bf16[201, 200, 25]" = torch.ops.aten.slice.Tensor(slice_203, 2, 1, 9223372036854775807);  slice_203 = None
        mul_50: "f32[201, 200, 25]" = torch.ops.aten.mul.Tensor(mul_49, slice_204);  mul_49 = slice_204 = None
        copy_5: "bf16[201, 200, 25]" = torch.ops.aten.copy.default(select_17, mul_50);  select_17 = mul_50 = None
        select_scatter_3: "bf16[201, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_15, copy_5, 3, 0);  select_15 = copy_5 = None
        select_scatter_4: "bf16[201, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_198, select_scatter_3, 3, 1);  slice_198 = select_scatter_3 = None
        slice_scatter_30: "bf16[201, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_197, select_scatter_4, 2, 1, 9223372036854775807);  slice_197 = select_scatter_4 = None
        slice_scatter_31: "bf16[201, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_196, slice_scatter_30, 1, 2, -2);  slice_196 = slice_scatter_30 = None
        slice_scatter_32: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_29, slice_scatter_31, 0, 1, -2);  slice_scatter_29 = slice_scatter_31 = None
        slice_205: "bf16[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_32, 0, 1, -2)
        slice_206: "bf16[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_205, 1, 2, -2)
        select_18: "bf16[201, 200, 26, 2]" = torch.ops.aten.select.int(slice_206, 3, 0)
        slice_207: "bf16[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_32, 0, 1, -2)
        slice_208: "bf16[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_207, 1, 2, -2);  slice_207 = None
        select_19: "bf16[201, 200, 26, 2]" = torch.ops.aten.select.int(slice_208, 3, 0);  slice_208 = None
        select_20: "bf16[201, 200, 26]" = torch.ops.aten.select.int(select_19, 3, 1);  select_19 = None
        mul_51: "f32[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_36, div_8);  mul_36 = div_8 = None
        slice_209: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 1, -2)
        slice_210: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_209, 1, 2, -2);  slice_209 = None
        mul_52: "f32[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_51, slice_210);  mul_51 = slice_210 = None
        copy_6: "bf16[201, 200, 26]" = torch.ops.aten.copy.default(select_20, mul_52);  select_20 = mul_52 = None
        select_scatter_5: "bf16[201, 200, 26, 2]" = torch.ops.aten.select_scatter.default(select_18, copy_6, 3, 1);  select_18 = copy_6 = None
        select_scatter_6: "bf16[201, 200, 26, 2, 2]" = torch.ops.aten.select_scatter.default(slice_206, select_scatter_5, 3, 0);  slice_206 = select_scatter_5 = None
        slice_scatter_33: "bf16[201, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_205, select_scatter_6, 1, 2, -2);  slice_205 = select_scatter_6 = None
        slice_scatter_34: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_32, slice_scatter_33, 0, 1, -2);  slice_scatter_32 = slice_scatter_33 = None
        slice_211: "bf16[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_34, 0, 1, -2)
        slice_212: "bf16[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_211, 1, 2, -2)
        select_21: "bf16[201, 200, 26, 2]" = torch.ops.aten.select.int(slice_212, 3, 1)
        slice_213: "bf16[201, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_34, 0, 1, -2)
        slice_214: "bf16[201, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_213, 1, 2, -2);  slice_213 = None
        select_22: "bf16[201, 200, 26, 2]" = torch.ops.aten.select.int(slice_214, 3, 1);  slice_214 = None
        select_23: "bf16[201, 200, 26]" = torch.ops.aten.select.int(select_22, 3, 1);  select_22 = None
        mul_53: "f32[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_44, div_10);  mul_44 = div_10 = None
        slice_215: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 1, -2);  arg1_1 = None
        slice_216: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_215, 1, 2, -2);  slice_215 = None
        mul_54: "f32[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_53, slice_216);  mul_53 = slice_216 = None
        copy_7: "bf16[201, 200, 26]" = torch.ops.aten.copy.default(select_23, mul_54);  select_23 = mul_54 = None
        select_scatter_7: "bf16[201, 200, 26, 2]" = torch.ops.aten.select_scatter.default(select_21, copy_7, 3, 1);  select_21 = copy_7 = None
        select_scatter_8: "bf16[201, 200, 26, 2, 2]" = torch.ops.aten.select_scatter.default(slice_212, select_scatter_7, 3, 1);  slice_212 = select_scatter_7 = None
        slice_scatter_35: "bf16[201, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_211, select_scatter_8, 1, 2, -2);  slice_211 = select_scatter_8 = None
        slice_scatter_36: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_34, slice_scatter_35, 0, 1, -2);  slice_scatter_34 = slice_scatter_35 = None
        slice_217: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg11_1, 0, 1, -2)
        slice_218: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(arg11_1, 0, 1, -2)
        slice_219: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_218, 1, 2, -2);  slice_218 = None
        slice_220: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_26, 0, 1, -2)
        slice_221: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_26, 0, 1, -2)
        slice_222: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_221, 1, 2, -2);  slice_221 = None
        slice_scatter_37: "bf16[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_220, slice_222, 1, 2, -2);  slice_220 = slice_222 = None
        slice_scatter_38: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_26, slice_scatter_37, 0, 1, -2);  slice_scatter_26 = slice_scatter_37 = None
        slice_223: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_38, 0, 1, -2);  slice_scatter_38 = None
        slice_224: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_223, 1, 2, -2);  slice_223 = None
        unsqueeze_20: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg12_1, 0)
        unsqueeze_21: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, 1);  unsqueeze_20 = None
        mul_55: "bf16[1, 1, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_21, 4.0);  unsqueeze_21 = None
        div_12: "bf16[201, 200, 26]" = torch.ops.aten.div.Tensor(slice_224, mul_55);  slice_224 = mul_55 = None
        copy_8: "bf16[201, 200, 26]" = torch.ops.aten.copy.default(slice_219, div_12);  slice_219 = div_12 = None
        slice_scatter_39: "bf16[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_217, copy_8, 1, 2, -2);  slice_217 = copy_8 = None
        slice_scatter_40: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(arg11_1, slice_scatter_39, 0, 1, -2);  slice_scatter_39 = None
        full_15: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_7, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_7 = None
        slice_225: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_15, 0, 2, -2)
        slice_226: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_225, 1, 1, -2)
        slice_227: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_15, 0, 2, -2)
        slice_228: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_227, 1, 1, -2);  slice_227 = None
        slice_229: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_228, 2, 1, 9223372036854775807);  slice_228 = None
        unsqueeze_22: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg0_1, 0)
        unsqueeze_23: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 1);  unsqueeze_22 = None
        slice_230: "bf16[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_23, 2, 0, 25);  unsqueeze_23 = None
        slice_231: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg13_1, 0, 2, -2)
        slice_232: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_231, 1, 1, -2);  slice_231 = None
        slice_233: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_232, 2, 1, 9223372036854775807);  slice_232 = None
        mul_56: "bf16[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_230, slice_233);  slice_230 = slice_233 = None
        full_16: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_17: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_9: "bf16[204, 204, 26]" = torch.ops.aten.copy.default(slice_scatter_4, full_17);  slice_scatter_4 = full_17 = None
        slice_234: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(copy_9, 0, 2, -2)
        slice_235: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_234, 1, 1, -2)
        slice_236: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(copy_9, 0, 2, -2)
        slice_237: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_236, 1, 1, -2);  slice_236 = None
        slice_238: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_237, 2, 1, 9223372036854775807);  slice_237 = None
        slice_239: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -2)
        slice_240: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_239, 1, 1, -2);  slice_239 = None
        slice_241: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_240, 2, 1, 9223372036854775807);  slice_240 = None
        slice_242: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -2)
        slice_243: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_242, 1, 1, -2);  slice_242 = None
        slice_244: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_243, 2, 0, -1);  slice_243 = None
        add_25: "bf16[200, 201, 25]" = torch.ops.aten.add.Tensor(slice_241, slice_244);  slice_241 = slice_244 = None
        slice_245: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -2)
        slice_246: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_245, 1, 2, -1);  slice_245 = None
        slice_247: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_246, 2, 1, 9223372036854775807);  slice_246 = None
        add_26: "bf16[200, 201, 25]" = torch.ops.aten.add.Tensor(add_25, slice_247);  add_25 = slice_247 = None
        slice_248: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -2)
        slice_249: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_248, 1, 2, -1);  slice_248 = None
        slice_250: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_249, 2, 0, -1);  slice_249 = None
        add_27: "bf16[200, 201, 25]" = torch.ops.aten.add.Tensor(add_26, slice_250);  add_26 = slice_250 = None
        mul_57: "bf16[200, 201, 25]" = torch.ops.aten.mul.Tensor(add_27, 0.25);  add_27 = None
        copy_10: "bf16[200, 201, 25]" = torch.ops.aten.copy.default(slice_238, mul_57);  slice_238 = mul_57 = None
        slice_scatter_41: "bf16[200, 201, 26]" = torch.ops.aten.slice_scatter.default(slice_235, copy_10, 2, 1, 9223372036854775807);  slice_235 = copy_10 = None
        slice_scatter_42: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_234, slice_scatter_41, 1, 1, -2);  slice_234 = slice_scatter_41 = None
        slice_scatter_43: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(copy_9, slice_scatter_42, 0, 2, -2);  copy_9 = slice_scatter_42 = None
        slice_251: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_43, 0, 2, -2)
        slice_252: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_251, 1, 1, -2)
        slice_253: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_43, 0, 2, -2)
        slice_254: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_253, 1, 1, -2);  slice_253 = None
        select_24: "bf16[200, 201]" = torch.ops.aten.select.int(slice_254, 2, 0);  slice_254 = None
        slice_255: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -2)
        slice_256: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_255, 1, 1, -2);  slice_255 = None
        select_25: "bf16[200, 201]" = torch.ops.aten.select.int(slice_256, 2, 0);  slice_256 = None
        slice_257: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -2)
        slice_258: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_257, 1, 2, -1);  slice_257 = None
        select_26: "bf16[200, 201]" = torch.ops.aten.select.int(slice_258, 2, 0);  slice_258 = None
        add_28: "bf16[200, 201]" = torch.ops.aten.add.Tensor(select_25, select_26);  select_25 = select_26 = None
        mul_58: "bf16[200, 201]" = torch.ops.aten.mul.Tensor(add_28, 0.5);  add_28 = None
        copy_11: "bf16[200, 201]" = torch.ops.aten.copy.default(select_24, mul_58);  select_24 = mul_58 = None
        select_scatter_9: "bf16[200, 201, 26]" = torch.ops.aten.select_scatter.default(slice_252, copy_11, 2, 0);  slice_252 = copy_11 = None
        slice_scatter_44: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_251, select_scatter_9, 1, 1, -2);  slice_251 = select_scatter_9 = None
        slice_scatter_45: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_43, slice_scatter_44, 0, 2, -2);  slice_scatter_43 = slice_scatter_44 = None
        slice_259: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_45, 0, 2, -2)
        slice_260: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_259, 1, 1, -2);  slice_259 = None
        slice_261: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_260, 2, 1, 9223372036854775807);  slice_260 = None
        slice_262: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2)
        slice_263: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_262, 1, 1, -2);  slice_262 = None
        slice_264: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_263, 2, 1, 9223372036854775807);  slice_263 = None
        full_18: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_8, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_8 = None
        slice_265: "bf16[204, 203, 26]" = torch.ops.aten.slice.Tensor(full_18, 1, 0, -1)
        slice_266: "bf16[204, 203, 26]" = torch.ops.aten.slice.Tensor(arg13_1, 1, 0, -1)
        slice_267: "bf16[204, 203, 26, 3]" = torch.ops.aten.slice.Tensor(arg3_1, 1, 1, 9223372036854775807)
        select_27: "bf16[204, 203, 26]" = torch.ops.aten.select.int(slice_267, 3, 0);  slice_267 = None
        slice_268: "bf16[204, 203, 26, 3]" = torch.ops.aten.slice.Tensor(arg3_1, 1, 0, -1);  arg3_1 = None
        select_28: "bf16[204, 203, 26]" = torch.ops.aten.select.int(slice_268, 3, 0);  slice_268 = None
        sub_11: "bf16[204, 203, 26]" = torch.ops.aten.sub.Tensor(select_27, select_28);  select_27 = select_28 = None
        mul_59: "bf16[204, 203, 26]" = torch.ops.aten.mul.Tensor(slice_266, sub_11);  slice_266 = sub_11 = None
        unsqueeze_24: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(arg14_1, 0)
        slice_269: "bf16[1, 203]" = torch.ops.aten.slice.Tensor(unsqueeze_24, 1, 0, -1);  unsqueeze_24 = None
        unsqueeze_25: "bf16[1, 203, 1]" = torch.ops.aten.unsqueeze.default(slice_269, 2);  slice_269 = None
        div_13: "bf16[204, 203, 26]" = torch.ops.aten.div.Tensor(mul_59, unsqueeze_25);  mul_59 = unsqueeze_25 = None
        copy_12: "bf16[204, 203, 26]" = torch.ops.aten.copy.default(slice_265, div_13);  slice_265 = div_13 = None
        slice_scatter_46: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_18, copy_12, 1, 0, -1);  full_18 = copy_12 = None
        slice_270: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_46, 0, 2, -2)
        slice_271: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_270, 1, 1, -2);  slice_270 = None
        slice_272: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_271, 2, 1, 9223372036854775807);  slice_271 = None
        mul_60: "bf16[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_264, slice_272);  slice_264 = slice_272 = None
        slice_273: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2)
        slice_274: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_273, 1, 1, -2);  slice_273 = None
        slice_275: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_274, 2, 1, 9223372036854775807);  slice_274 = None
        full_19: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_9, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_9 = None
        slice_276: "bf16[204, 203, 26]" = torch.ops.aten.slice.Tensor(full_19, 1, 0, -1)
        slice_277: "bf16[204, 203, 26]" = torch.ops.aten.slice.Tensor(arg13_1, 1, 0, -1)
        slice_278: "bf16[204, 203, 26, 3]" = torch.ops.aten.slice.Tensor(arg8_1, 1, 1, 9223372036854775807)
        select_29: "bf16[204, 203, 26]" = torch.ops.aten.select.int(slice_278, 3, 0);  slice_278 = None
        slice_279: "bf16[204, 203, 26, 3]" = torch.ops.aten.slice.Tensor(arg8_1, 1, 0, -1);  arg8_1 = None
        select_30: "bf16[204, 203, 26]" = torch.ops.aten.select.int(slice_279, 3, 0);  slice_279 = None
        sub_12: "bf16[204, 203, 26]" = torch.ops.aten.sub.Tensor(select_29, select_30);  select_29 = select_30 = None
        mul_61: "bf16[204, 203, 26]" = torch.ops.aten.mul.Tensor(slice_277, sub_12);  slice_277 = sub_12 = None
        unsqueeze_26: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(arg14_1, 0)
        slice_280: "bf16[1, 203]" = torch.ops.aten.slice.Tensor(unsqueeze_26, 1, 0, -1);  unsqueeze_26 = None
        unsqueeze_27: "bf16[1, 203, 1]" = torch.ops.aten.unsqueeze.default(slice_280, 2);  slice_280 = None
        div_14: "bf16[204, 203, 26]" = torch.ops.aten.div.Tensor(mul_61, unsqueeze_27);  mul_61 = unsqueeze_27 = None
        copy_13: "bf16[204, 203, 26]" = torch.ops.aten.copy.default(slice_276, div_14);  slice_276 = div_14 = None
        slice_scatter_47: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_19, copy_13, 1, 0, -1);  full_19 = copy_13 = None
        slice_281: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 2, -2)
        slice_282: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_281, 1, 1, -2);  slice_281 = None
        slice_283: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_282, 2, 1, 9223372036854775807);  slice_282 = None
        mul_62: "bf16[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_275, slice_283);  slice_275 = slice_283 = None
        add_29: "bf16[200, 201, 25]" = torch.ops.aten.add.Tensor(mul_60, mul_62);  mul_60 = mul_62 = None
        neg_10: "bf16[200, 201, 25]" = torch.ops.aten.neg.default(add_29);  add_29 = None
        full_20: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_284: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2)
        slice_285: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_284, 1, 1, -2);  slice_284 = None
        slice_286: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_285, 2, 1, 9223372036854775807);  slice_285 = None
        slice_287: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 2, -2)
        slice_288: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_287, 1, 1, -2);  slice_287 = None
        slice_289: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_288, 2, 0, 25);  slice_288 = None
        mul_63: "bf16[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_286, slice_289);  slice_286 = slice_289 = None
        slice_290: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2)
        slice_291: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_290, 1, 1, -2);  slice_290 = None
        slice_292: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_291, 2, 1, 9223372036854775807);  slice_291 = None
        slice_293: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_8, 0, 2, -2)
        slice_294: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_293, 1, 1, -2);  slice_293 = None
        slice_295: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_294, 2, 0, 25);  slice_294 = None
        mul_64: "bf16[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_292, slice_295);  slice_292 = slice_295 = None
        add_30: "bf16[200, 201, 25]" = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None
        minimum_4: "f32[200, 201, 25]" = torch.ops.aten.minimum.default(full_20, add_30);  full_20 = add_30 = None
        sub_13: "f32[200, 201, 25]" = torch.ops.aten.sub.Tensor(minimum_4, 1e-20);  minimum_4 = None
        div_15: "f32[200, 201, 25]" = torch.ops.aten.div.Tensor(neg_10, sub_13);  neg_10 = sub_13 = None
        abs_6: "f32[200, 201, 25]" = torch.ops.aten.abs.default(div_15)
        neg_11: "f32[200, 201, 25]" = torch.ops.aten.neg.default(abs_6);  abs_6 = None
        add_31: "f32[200, 201, 25]" = torch.ops.aten.add.Tensor(neg_11, 0.001);  neg_11 = None
        div_16: "f32[200, 201, 25]" = torch.ops.aten.div.Tensor(add_31, 0.001);  add_31 = None
        tanh_4: "f32[200, 201, 25]" = torch.ops.aten.tanh.default(div_16);  div_16 = None
        add_32: "f32[200, 201, 25]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_65: "f32[200, 201, 25]" = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None
        mul_66: "f32[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_261, mul_65);  slice_261 = None
        maximum_4: "f32[200, 201, 25]" = torch.ops.aten.maximum.default(full_16, mul_66);  full_16 = mul_66 = None
        mul_67: "f32[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_56, maximum_4);  mul_56 = maximum_4 = None
        add_33: "f32[200, 201, 25]" = torch.ops.aten.add.Tensor(slice_229, mul_67);  slice_229 = mul_67 = None
        convert_element_type_4: "bf16[200, 201, 25]" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16);  add_33 = None
        slice_scatter_48: "bf16[200, 201, 26]" = torch.ops.aten.slice_scatter.default(slice_226, convert_element_type_4, 2, 1, 9223372036854775807);  slice_226 = convert_element_type_4 = None
        slice_scatter_49: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_225, slice_scatter_48, 1, 1, -2);  slice_225 = slice_scatter_48 = None
        slice_scatter_50: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_15, slice_scatter_49, 0, 2, -2);  full_15 = slice_scatter_49 = None
        slice_296: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_50, 0, 2, -2)
        slice_297: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_296, 1, 1, -2);  slice_296 = None
        slice_298: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_297, 2, 1, 9223372036854775807);  slice_297 = slice_298 = None
        slice_299: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_50, 0, 2, -2)
        slice_300: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_299, 1, 1, -2)
        slice_301: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_50, 0, 2, -2)
        slice_302: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_301, 1, 1, -2);  slice_301 = None
        slice_303: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_302, 2, 1, 9223372036854775807);  slice_302 = None
        slice_scatter_51: "bf16[200, 201, 26]" = torch.ops.aten.slice_scatter.default(slice_300, slice_303, 2, 1, 9223372036854775807);  slice_300 = slice_303 = None
        slice_scatter_52: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_299, slice_scatter_51, 1, 1, -2);  slice_299 = slice_scatter_51 = None
        slice_scatter_53: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_50, slice_scatter_52, 0, 2, -2);  slice_scatter_50 = slice_scatter_52 = None
        slice_304: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_53, 0, 2, -2)
        slice_305: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_304, 1, 1, -2)
        slice_306: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_53, 0, 2, -2)
        slice_307: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_306, 1, 1, -2);  slice_306 = None
        slice_308: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_307, 2, 1, 9223372036854775807);  slice_307 = None
        unsqueeze_28: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg0_1, 0)
        unsqueeze_29: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 1);  unsqueeze_28 = None
        slice_309: "bf16[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_29, 2, 0, 25);  unsqueeze_29 = None
        slice_310: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg13_1, 0, 2, -2)
        slice_311: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_310, 1, 1, -2);  slice_310 = None
        slice_312: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_311, 2, 1, 9223372036854775807);  slice_311 = None
        mul_68: "bf16[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_309, slice_312);  slice_309 = slice_312 = None
        full_21: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_313: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_45, 0, 2, -2)
        slice_314: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_313, 1, 1, -2);  slice_313 = None
        slice_315: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_314, 2, 1, 9223372036854775807);  slice_314 = None
        slice_316: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2)
        slice_317: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_316, 1, 2, -1);  slice_316 = None
        slice_318: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_317, 2, 1, 9223372036854775807);  slice_317 = None
        slice_319: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_46, 0, 2, -2)
        slice_320: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_319, 1, 1, -2);  slice_319 = None
        slice_321: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_320, 2, 1, 9223372036854775807);  slice_320 = None
        mul_69: "bf16[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_318, slice_321);  slice_318 = slice_321 = None
        slice_322: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2)
        slice_323: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_322, 1, 2, -1);  slice_322 = None
        slice_324: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_323, 2, 1, 9223372036854775807);  slice_323 = None
        slice_325: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 2, -2)
        slice_326: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_325, 1, 1, -2);  slice_325 = None
        slice_327: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_326, 2, 1, 9223372036854775807);  slice_326 = None
        mul_70: "bf16[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_324, slice_327);  slice_324 = slice_327 = None
        add_34: "bf16[200, 201, 25]" = torch.ops.aten.add.Tensor(mul_69, mul_70);  mul_69 = mul_70 = None
        neg_12: "bf16[200, 201, 25]" = torch.ops.aten.neg.default(add_34);  add_34 = None
        full_22: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_328: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2)
        slice_329: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_328, 1, 2, -1);  slice_328 = None
        slice_330: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_329, 2, 1, 9223372036854775807);  slice_329 = None
        slice_331: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 2, -2)
        slice_332: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_331, 1, 2, -1);  slice_331 = None
        slice_333: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_332, 2, 0, 25);  slice_332 = None
        mul_71: "bf16[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_330, slice_333);  slice_330 = slice_333 = None
        slice_334: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2)
        slice_335: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_334, 1, 2, -1);  slice_334 = None
        slice_336: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_335, 2, 1, 9223372036854775807);  slice_335 = None
        slice_337: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_8, 0, 2, -2)
        slice_338: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_337, 1, 2, -1);  slice_337 = None
        slice_339: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_338, 2, 0, 25);  slice_338 = None
        mul_72: "bf16[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_336, slice_339);  slice_336 = slice_339 = None
        add_35: "bf16[200, 201, 25]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        minimum_5: "f32[200, 201, 25]" = torch.ops.aten.minimum.default(full_22, add_35);  full_22 = add_35 = None
        sub_14: "f32[200, 201, 25]" = torch.ops.aten.sub.Tensor(minimum_5, 1e-20);  minimum_5 = None
        div_17: "f32[200, 201, 25]" = torch.ops.aten.div.Tensor(neg_12, sub_14);  neg_12 = sub_14 = None
        abs_7: "f32[200, 201, 25]" = torch.ops.aten.abs.default(div_17)
        neg_13: "f32[200, 201, 25]" = torch.ops.aten.neg.default(abs_7);  abs_7 = None
        add_36: "f32[200, 201, 25]" = torch.ops.aten.add.Tensor(neg_13, 0.001);  neg_13 = None
        div_18: "f32[200, 201, 25]" = torch.ops.aten.div.Tensor(add_36, 0.001);  add_36 = None
        tanh_5: "f32[200, 201, 25]" = torch.ops.aten.tanh.default(div_18);  div_18 = None
        add_37: "f32[200, 201, 25]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_73: "f32[200, 201, 25]" = torch.ops.aten.mul.Tensor(add_37, 0.5);  add_37 = None
        mul_74: "f32[200, 201, 25]" = torch.ops.aten.mul.Tensor(slice_315, mul_73);  slice_315 = None
        maximum_5: "f32[200, 201, 25]" = torch.ops.aten.maximum.default(full_21, mul_74);  full_21 = mul_74 = None
        mul_75: "f32[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_68, maximum_5);  mul_68 = maximum_5 = None
        add_38: "f32[200, 201, 25]" = torch.ops.aten.add.Tensor(slice_308, mul_75);  slice_308 = mul_75 = None
        convert_element_type_5: "bf16[200, 201, 25]" = torch.ops.prims.convert_element_type.default(add_38, torch.bfloat16);  add_38 = None
        slice_scatter_54: "bf16[200, 201, 26]" = torch.ops.aten.slice_scatter.default(slice_305, convert_element_type_5, 2, 1, 9223372036854775807);  slice_305 = convert_element_type_5 = None
        slice_scatter_55: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_304, slice_scatter_54, 1, 1, -2);  slice_304 = slice_scatter_54 = None
        slice_scatter_56: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_53, slice_scatter_55, 0, 2, -2);  slice_scatter_53 = slice_scatter_55 = None
        slice_340: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_56, 0, 2, -2)
        slice_341: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_340, 1, 1, -2);  slice_340 = None
        slice_342: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_341, 2, 1, 9223372036854775807);  slice_341 = slice_342 = None
        slice_343: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_56, 0, 2, -2)
        slice_344: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_343, 1, 1, -2)
        slice_345: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_56, 0, 2, -2)
        slice_346: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_345, 1, 1, -2);  slice_345 = None
        slice_347: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_346, 2, 1, 9223372036854775807);  slice_346 = None
        slice_scatter_57: "bf16[200, 201, 26]" = torch.ops.aten.slice_scatter.default(slice_344, slice_347, 2, 1, 9223372036854775807);  slice_344 = slice_347 = None
        slice_scatter_58: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_343, slice_scatter_57, 1, 1, -2);  slice_343 = slice_scatter_57 = None
        slice_scatter_59: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_56, slice_scatter_58, 0, 2, -2);  slice_scatter_56 = slice_scatter_58 = None
        slice_348: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_59, 0, 2, -2)
        slice_349: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_59, 0, 2, -2)
        slice_350: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_349, 1, 1, -2);  slice_349 = None
        unsqueeze_30: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg0_1, 0)
        unsqueeze_31: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 1);  unsqueeze_30 = None
        slice_351: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg13_1, 0, 2, -2)
        slice_352: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_351, 1, 1, -2);  slice_351 = None
        mul_76: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_31, slice_352);  unsqueeze_31 = slice_352 = None
        full_23: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_353: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_45, 0, 2, -2)
        slice_354: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_353, 1, 1, -2);  slice_353 = None
        slice_355: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2)
        slice_356: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_355, 1, 1, -2);  slice_355 = None
        slice_357: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_46, 0, 2, -2)
        slice_358: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_357, 1, 1, -2);  slice_357 = None
        mul_77: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_356, slice_358);  slice_356 = slice_358 = None
        slice_359: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2)
        slice_360: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_359, 1, 1, -2);  slice_359 = None
        slice_361: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 2, -2)
        slice_362: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_361, 1, 1, -2);  slice_361 = None
        mul_78: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_360, slice_362);  slice_360 = slice_362 = None
        add_39: "bf16[200, 201, 26]" = torch.ops.aten.add.Tensor(mul_77, mul_78);  mul_77 = mul_78 = None
        neg_14: "bf16[200, 201, 26]" = torch.ops.aten.neg.default(add_39);  add_39 = None
        full_24: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_363: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2)
        slice_364: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_363, 1, 1, -2);  slice_363 = None
        slice_365: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 2, -2)
        slice_366: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_365, 1, 1, -2);  slice_365 = None
        mul_79: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_364, slice_366);  slice_364 = slice_366 = None
        slice_367: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2)
        slice_368: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_367, 1, 1, -2);  slice_367 = None
        slice_369: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_8, 0, 2, -2)
        slice_370: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_369, 1, 1, -2);  slice_369 = None
        mul_80: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_368, slice_370);  slice_368 = slice_370 = None
        add_40: "bf16[200, 201, 26]" = torch.ops.aten.add.Tensor(mul_79, mul_80);  mul_79 = mul_80 = None
        minimum_6: "f32[200, 201, 26]" = torch.ops.aten.minimum.default(full_24, add_40);  full_24 = add_40 = None
        sub_15: "f32[200, 201, 26]" = torch.ops.aten.sub.Tensor(minimum_6, 1e-20);  minimum_6 = None
        div_19: "f32[200, 201, 26]" = torch.ops.aten.div.Tensor(neg_14, sub_15);  neg_14 = sub_15 = None
        abs_8: "f32[200, 201, 26]" = torch.ops.aten.abs.default(div_19)
        neg_15: "f32[200, 201, 26]" = torch.ops.aten.neg.default(abs_8);  abs_8 = None
        add_41: "f32[200, 201, 26]" = torch.ops.aten.add.Tensor(neg_15, 0.001);  neg_15 = None
        div_20: "f32[200, 201, 26]" = torch.ops.aten.div.Tensor(add_41, 0.001);  add_41 = None
        tanh_6: "f32[200, 201, 26]" = torch.ops.aten.tanh.default(div_20);  div_20 = None
        add_42: "f32[200, 201, 26]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_81: "f32[200, 201, 26]" = torch.ops.aten.mul.Tensor(add_42, 0.5);  add_42 = None
        mul_82: "f32[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_354, mul_81);  slice_354 = None
        maximum_6: "f32[200, 201, 26]" = torch.ops.aten.maximum.default(full_23, mul_82);  full_23 = mul_82 = None
        mul_83: "f32[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_76, maximum_6);  mul_76 = maximum_6 = None
        add_43: "f32[200, 201, 26]" = torch.ops.aten.add.Tensor(slice_350, mul_83);  slice_350 = mul_83 = None
        convert_element_type_6: "bf16[200, 201, 26]" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None
        slice_scatter_60: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_348, convert_element_type_6, 1, 1, -2);  slice_348 = convert_element_type_6 = None
        slice_scatter_61: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_59, slice_scatter_60, 0, 2, -2);  slice_scatter_59 = slice_scatter_60 = None
        slice_371: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_61, 0, 2, -2)
        slice_372: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_371, 1, 1, -2);  slice_371 = slice_372 = None
        slice_373: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_61, 0, 2, -2)
        slice_374: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_61, 0, 2, -2)
        slice_375: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_374, 1, 1, -2);  slice_374 = None
        slice_scatter_62: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_373, slice_375, 1, 1, -2);  slice_373 = slice_375 = None
        slice_scatter_63: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_61, slice_scatter_62, 0, 2, -2);  slice_scatter_61 = slice_scatter_62 = None
        slice_376: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_63, 0, 2, -2)
        slice_377: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_63, 0, 2, -2)
        slice_378: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_377, 1, 1, -2);  slice_377 = None
        unsqueeze_32: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
        unsqueeze_33: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, 1);  unsqueeze_32 = None
        slice_379: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg13_1, 0, 2, -2)
        slice_380: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_379, 1, 1, -2);  slice_379 = None
        mul_84: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_33, slice_380);  unsqueeze_33 = slice_380 = None
        full_25: "f32[1]" = torch.ops.aten.full.default([1], 50.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_381: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_45, 0, 2, -2);  slice_scatter_45 = None
        slice_382: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_381, 1, 1, -2);  slice_381 = None
        slice_383: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2)
        slice_384: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_383, 1, 2, -1);  slice_383 = None
        slice_385: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_46, 0, 2, -2)
        slice_386: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_385, 1, 1, -2);  slice_385 = None
        mul_85: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_384, slice_386);  slice_384 = slice_386 = None
        slice_387: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2)
        slice_388: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_387, 1, 2, -1);  slice_387 = None
        slice_389: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 2, -2)
        slice_390: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_389, 1, 1, -2);  slice_389 = None
        mul_86: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_388, slice_390);  slice_388 = slice_390 = None
        add_44: "bf16[200, 201, 26]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        neg_16: "bf16[200, 201, 26]" = torch.ops.aten.neg.default(add_44);  add_44 = None
        full_26: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_391: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2)
        slice_392: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_391, 1, 2, -1);  slice_391 = None
        slice_393: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 2, -2)
        slice_394: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_393, 1, 2, -1);  slice_393 = None
        mul_87: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_392, slice_394);  slice_392 = slice_394 = None
        slice_395: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2)
        slice_396: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_395, 1, 2, -1);  slice_395 = None
        slice_397: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_8, 0, 2, -2)
        slice_398: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_397, 1, 2, -1);  slice_397 = None
        mul_88: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_396, slice_398);  slice_396 = slice_398 = None
        add_45: "bf16[200, 201, 26]" = torch.ops.aten.add.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
        minimum_7: "f32[200, 201, 26]" = torch.ops.aten.minimum.default(full_26, add_45);  full_26 = add_45 = None
        sub_16: "f32[200, 201, 26]" = torch.ops.aten.sub.Tensor(minimum_7, 1e-20);  minimum_7 = None
        div_21: "f32[200, 201, 26]" = torch.ops.aten.div.Tensor(neg_16, sub_16);  neg_16 = sub_16 = None
        abs_9: "f32[200, 201, 26]" = torch.ops.aten.abs.default(div_21)
        neg_17: "f32[200, 201, 26]" = torch.ops.aten.neg.default(abs_9);  abs_9 = None
        add_46: "f32[200, 201, 26]" = torch.ops.aten.add.Tensor(neg_17, 0.001);  neg_17 = None
        div_22: "f32[200, 201, 26]" = torch.ops.aten.div.Tensor(add_46, 0.001);  add_46 = None
        tanh_7: "f32[200, 201, 26]" = torch.ops.aten.tanh.default(div_22);  div_22 = None
        add_47: "f32[200, 201, 26]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_89: "f32[200, 201, 26]" = torch.ops.aten.mul.Tensor(add_47, 0.5);  add_47 = None
        mul_90: "f32[200, 201, 26]" = torch.ops.aten.mul.Tensor(slice_382, mul_89);  slice_382 = None
        maximum_7: "f32[200, 201, 26]" = torch.ops.aten.maximum.default(full_25, mul_90);  full_25 = mul_90 = None
        mul_91: "f32[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_84, maximum_7);  mul_84 = maximum_7 = None
        add_48: "f32[200, 201, 26]" = torch.ops.aten.add.Tensor(slice_378, mul_91);  slice_378 = mul_91 = None
        convert_element_type_7: "bf16[200, 201, 26]" = torch.ops.prims.convert_element_type.default(add_48, torch.bfloat16);  add_48 = None
        slice_scatter_64: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_376, convert_element_type_7, 1, 1, -2);  slice_376 = convert_element_type_7 = None
        slice_scatter_65: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_63, slice_scatter_64, 0, 2, -2);  slice_scatter_63 = slice_scatter_64 = None
        slice_399: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_65, 0, 2, -2)
        slice_400: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_399, 1, 1, -2);  slice_399 = slice_400 = None
        slice_401: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg15_1, 0, 2, -2)
        slice_402: "bf16[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_401, 1, 1, -2)
        slice_403: "bf16[200, 201, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_402, 2, 1, 9223372036854775807)
        select_31: "bf16[200, 201, 25, 2]" = torch.ops.aten.select.int(slice_403, 3, 0)
        slice_404: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg15_1, 0, 2, -2)
        slice_405: "bf16[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_404, 1, 1, -2);  slice_404 = None
        slice_406: "bf16[200, 201, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_405, 2, 1, 9223372036854775807);  slice_405 = None
        select_32: "bf16[200, 201, 25, 2]" = torch.ops.aten.select.int(slice_406, 3, 0);  slice_406 = None
        select_33: "bf16[200, 201, 25]" = torch.ops.aten.select.int(select_32, 3, 0);  select_32 = None
        mul_92: "f32[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_65, div_15);  mul_65 = div_15 = None
        slice_407: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg13_1, 0, 2, -2)
        slice_408: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_407, 1, 1, -2);  slice_407 = None
        slice_409: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_408, 2, 1, 9223372036854775807);  slice_408 = None
        mul_93: "f32[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_92, slice_409);  mul_92 = slice_409 = None
        copy_14: "bf16[200, 201, 25]" = torch.ops.aten.copy.default(select_33, mul_93);  select_33 = mul_93 = None
        select_scatter_10: "bf16[200, 201, 25, 2]" = torch.ops.aten.select_scatter.default(select_31, copy_14, 3, 0);  select_31 = copy_14 = None
        select_scatter_11: "bf16[200, 201, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_403, select_scatter_10, 3, 0);  slice_403 = select_scatter_10 = None
        slice_scatter_66: "bf16[200, 201, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_402, select_scatter_11, 2, 1, 9223372036854775807);  slice_402 = select_scatter_11 = None
        slice_scatter_67: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_401, slice_scatter_66, 1, 1, -2);  slice_401 = slice_scatter_66 = None
        slice_scatter_68: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(arg15_1, slice_scatter_67, 0, 2, -2);  slice_scatter_67 = None
        slice_410: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_68, 0, 2, -2)
        slice_411: "bf16[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_410, 1, 1, -2)
        slice_412: "bf16[200, 201, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_411, 2, 1, 9223372036854775807)
        select_34: "bf16[200, 201, 25, 2]" = torch.ops.aten.select.int(slice_412, 3, 1)
        slice_413: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_68, 0, 2, -2)
        slice_414: "bf16[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_413, 1, 1, -2);  slice_413 = None
        slice_415: "bf16[200, 201, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_414, 2, 1, 9223372036854775807);  slice_414 = None
        select_35: "bf16[200, 201, 25, 2]" = torch.ops.aten.select.int(slice_415, 3, 1);  slice_415 = None
        select_36: "bf16[200, 201, 25]" = torch.ops.aten.select.int(select_35, 3, 0);  select_35 = None
        mul_94: "f32[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_73, div_17);  mul_73 = div_17 = None
        slice_416: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg13_1, 0, 2, -2)
        slice_417: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_416, 1, 1, -2);  slice_416 = None
        slice_418: "bf16[200, 201, 25]" = torch.ops.aten.slice.Tensor(slice_417, 2, 1, 9223372036854775807);  slice_417 = None
        mul_95: "f32[200, 201, 25]" = torch.ops.aten.mul.Tensor(mul_94, slice_418);  mul_94 = slice_418 = None
        copy_15: "bf16[200, 201, 25]" = torch.ops.aten.copy.default(select_36, mul_95);  select_36 = mul_95 = None
        select_scatter_12: "bf16[200, 201, 25, 2]" = torch.ops.aten.select_scatter.default(select_34, copy_15, 3, 0);  select_34 = copy_15 = None
        select_scatter_13: "bf16[200, 201, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_412, select_scatter_12, 3, 1);  slice_412 = select_scatter_12 = None
        slice_scatter_69: "bf16[200, 201, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_411, select_scatter_13, 2, 1, 9223372036854775807);  slice_411 = select_scatter_13 = None
        slice_scatter_70: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_410, slice_scatter_69, 1, 1, -2);  slice_410 = slice_scatter_69 = None
        slice_scatter_71: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_68, slice_scatter_70, 0, 2, -2);  slice_scatter_68 = slice_scatter_70 = None
        slice_419: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_71, 0, 2, -2)
        slice_420: "bf16[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_419, 1, 1, -2)
        select_37: "bf16[200, 201, 26, 2]" = torch.ops.aten.select.int(slice_420, 3, 0)
        slice_421: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_71, 0, 2, -2)
        slice_422: "bf16[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_421, 1, 1, -2);  slice_421 = None
        select_38: "bf16[200, 201, 26, 2]" = torch.ops.aten.select.int(slice_422, 3, 0);  slice_422 = None
        select_39: "bf16[200, 201, 26]" = torch.ops.aten.select.int(select_38, 3, 1);  select_38 = None
        mul_96: "f32[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_81, div_19);  mul_81 = div_19 = None
        slice_423: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg13_1, 0, 2, -2)
        slice_424: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_423, 1, 1, -2);  slice_423 = None
        mul_97: "f32[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_96, slice_424);  mul_96 = slice_424 = None
        copy_16: "bf16[200, 201, 26]" = torch.ops.aten.copy.default(select_39, mul_97);  select_39 = mul_97 = None
        select_scatter_14: "bf16[200, 201, 26, 2]" = torch.ops.aten.select_scatter.default(select_37, copy_16, 3, 1);  select_37 = copy_16 = None
        select_scatter_15: "bf16[200, 201, 26, 2, 2]" = torch.ops.aten.select_scatter.default(slice_420, select_scatter_14, 3, 0);  slice_420 = select_scatter_14 = None
        slice_scatter_72: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_419, select_scatter_15, 1, 1, -2);  slice_419 = select_scatter_15 = None
        slice_scatter_73: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_71, slice_scatter_72, 0, 2, -2);  slice_scatter_71 = slice_scatter_72 = None
        slice_425: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_73, 0, 2, -2)
        slice_426: "bf16[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_425, 1, 1, -2)
        select_40: "bf16[200, 201, 26, 2]" = torch.ops.aten.select.int(slice_426, 3, 1)
        slice_427: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_73, 0, 2, -2)
        slice_428: "bf16[200, 201, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_427, 1, 1, -2);  slice_427 = None
        select_41: "bf16[200, 201, 26, 2]" = torch.ops.aten.select.int(slice_428, 3, 1);  slice_428 = None
        select_42: "bf16[200, 201, 26]" = torch.ops.aten.select.int(select_41, 3, 1);  select_41 = None
        mul_98: "f32[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_89, div_21);  mul_89 = div_21 = None
        slice_429: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg13_1, 0, 2, -2);  arg13_1 = None
        slice_430: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_429, 1, 1, -2);  slice_429 = None
        mul_99: "f32[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_98, slice_430);  mul_98 = slice_430 = None
        copy_17: "bf16[200, 201, 26]" = torch.ops.aten.copy.default(select_42, mul_99);  select_42 = mul_99 = None
        select_scatter_16: "bf16[200, 201, 26, 2]" = torch.ops.aten.select_scatter.default(select_40, copy_17, 3, 1);  select_40 = copy_17 = None
        select_scatter_17: "bf16[200, 201, 26, 2, 2]" = torch.ops.aten.select_scatter.default(slice_426, select_scatter_16, 3, 1);  slice_426 = select_scatter_16 = None
        slice_scatter_74: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_425, select_scatter_17, 1, 1, -2);  slice_425 = select_scatter_17 = None
        slice_scatter_75: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_73, slice_scatter_74, 0, 2, -2);  slice_scatter_73 = slice_scatter_74 = None
        slice_431: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg16_1, 0, 2, -2)
        slice_432: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg16_1, 0, 2, -2)
        slice_433: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_432, 1, 1, -2);  slice_432 = None
        slice_434: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_65, 0, 2, -2)
        slice_435: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_65, 0, 2, -2)
        slice_436: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_435, 1, 1, -2);  slice_435 = None
        slice_scatter_76: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_434, slice_436, 1, 1, -2);  slice_434 = slice_436 = None
        slice_scatter_77: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_65, slice_scatter_76, 0, 2, -2);  slice_scatter_65 = slice_scatter_76 = None
        slice_437: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_77, 0, 2, -2);  slice_scatter_77 = None
        slice_438: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_437, 1, 1, -2);  slice_437 = None
        unsqueeze_34: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg12_1, 0);  arg12_1 = None
        unsqueeze_35: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 1);  unsqueeze_34 = None
        mul_100: "bf16[1, 1, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_35, 4.0);  unsqueeze_35 = None
        div_23: "bf16[200, 201, 26]" = torch.ops.aten.div.Tensor(slice_438, mul_100);  slice_438 = mul_100 = None
        copy_18: "bf16[200, 201, 26]" = torch.ops.aten.copy.default(slice_433, div_23);  slice_433 = div_23 = None
        slice_scatter_78: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_431, copy_18, 1, 1, -2);  slice_431 = copy_18 = None
        slice_scatter_79: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(arg16_1, slice_scatter_78, 0, 2, -2);  slice_scatter_78 = None
        slice_439: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg17_1, 0, 2, -2)
        slice_440: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_439, 1, 2, -2)
        slice_441: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_440, 2, 0, -1)
        select_43: "bf16[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_441, 3, 0)
        slice_442: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg17_1, 0, 2, -2)
        slice_443: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_442, 1, 2, -2);  slice_442 = None
        slice_444: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_443, 2, 0, -1);  slice_443 = None
        select_44: "bf16[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_444, 3, 0);  slice_444 = None
        select_45: "bf16[200, 200, 25]" = torch.ops.aten.select.int(select_44, 3, 0);  select_44 = None
        slice_445: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2)
        slice_446: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_445, 1, 2, -2);  slice_445 = None
        slice_447: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_446, 2, 0, 25);  slice_446 = None
        slice_448: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 1, -3)
        slice_449: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_448, 1, 2, -2);  slice_448 = None
        slice_450: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_449, 2, 0, 25);  slice_449 = None
        mul_101: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_447, slice_450);  slice_447 = slice_450 = None
        slice_451: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2)
        slice_452: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_451, 1, 2, -2);  slice_451 = None
        slice_453: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_452, 2, 0, 25);  slice_452 = None
        slice_454: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_6, 0, 1, -3)
        slice_455: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_454, 1, 2, -2);  slice_454 = None
        slice_456: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_455, 2, 0, 25);  slice_455 = None
        mul_102: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_453, slice_456);  slice_453 = slice_456 = None
        add_49: "bf16[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_101, mul_102);  mul_101 = mul_102 = None
        neg_18: "bf16[200, 200, 25]" = torch.ops.aten.neg.default(add_49);  add_49 = None
        full_27: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_457: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2)
        slice_458: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_457, 1, 2, -2);  slice_457 = None
        slice_459: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_458, 2, 0, 25);  slice_458 = None
        slice_460: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 2, -2)
        slice_461: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_460, 1, 2, -2);  slice_460 = None
        slice_462: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_461, 2, 0, -1);  slice_461 = None
        mul_103: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_459, slice_462);  slice_459 = slice_462 = None
        slice_463: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2)
        slice_464: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_463, 1, 2, -2);  slice_463 = None
        slice_465: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_464, 2, 0, 25);  slice_464 = None
        slice_466: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_8, 0, 2, -2)
        slice_467: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_466, 1, 2, -2);  slice_466 = None
        slice_468: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_467, 2, 0, -1);  slice_467 = None
        mul_104: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_465, slice_468);  slice_465 = slice_468 = None
        add_50: "bf16[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_103, mul_104);  mul_103 = mul_104 = None
        minimum_8: "f32[200, 200, 25]" = torch.ops.aten.minimum.default(full_27, add_50);  full_27 = None
        sub_17: "f32[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_8, 1e-20);  minimum_8 = None
        div_24: "f32[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_18, sub_17);  neg_18 = sub_17 = None
        abs_10: "f32[200, 200, 25]" = torch.ops.aten.abs.default(div_24)
        neg_19: "f32[200, 200, 25]" = torch.ops.aten.neg.default(abs_10);  abs_10 = None
        add_51: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_19, 0.001);  neg_19 = None
        div_25: "f32[200, 200, 25]" = torch.ops.aten.div.Tensor(add_51, 0.001);  add_51 = None
        tanh_8: "f32[200, 200, 25]" = torch.ops.aten.tanh.default(div_25);  div_25 = None
        add_52: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_105: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_52, 0.5);  add_52 = None
        mul_106: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_105, div_24)
        slice_469: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2)
        slice_470: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_469, 1, 2, -2);  slice_469 = None
        slice_471: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_470, 2, 0, -1);  slice_470 = None
        mul_107: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_106, slice_471);  mul_106 = slice_471 = None
        copy_19: "bf16[200, 200, 25]" = torch.ops.aten.copy.default(select_45, mul_107);  select_45 = mul_107 = None
        select_scatter_18: "bf16[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_43, copy_19, 3, 0);  select_43 = copy_19 = None
        select_scatter_19: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_441, select_scatter_18, 3, 0);  slice_441 = select_scatter_18 = None
        slice_scatter_80: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_440, select_scatter_19, 2, 0, -1);  slice_440 = select_scatter_19 = None
        slice_scatter_81: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_439, slice_scatter_80, 1, 2, -2);  slice_439 = slice_scatter_80 = None
        slice_scatter_82: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(arg17_1, slice_scatter_81, 0, 2, -2);  slice_scatter_81 = None
        slice_472: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_82, 0, 2, -2)
        slice_473: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_472, 1, 2, -2)
        slice_474: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_473, 2, 0, -1)
        select_46: "bf16[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_474, 3, 1)
        slice_475: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_82, 0, 2, -2)
        slice_476: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_475, 1, 2, -2);  slice_475 = None
        slice_477: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_476, 2, 0, -1);  slice_476 = None
        select_47: "bf16[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_477, 3, 1);  slice_477 = None
        select_48: "bf16[200, 200, 25]" = torch.ops.aten.select.int(select_47, 3, 0);  select_47 = None
        slice_478: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2)
        slice_479: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_478, 1, 2, -2);  slice_478 = None
        slice_480: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_479, 2, 0, 25);  slice_479 = None
        slice_481: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2)
        slice_482: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_481, 1, 2, -2);  slice_481 = None
        slice_483: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_482, 2, 0, 25);  slice_482 = None
        mul_108: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_480, slice_483);  slice_480 = slice_483 = None
        slice_484: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2)
        slice_485: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_484, 1, 2, -2);  slice_484 = None
        slice_486: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_485, 2, 0, 25);  slice_485 = None
        slice_487: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_6, 0, 2, -2)
        slice_488: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_487, 1, 2, -2);  slice_487 = None
        slice_489: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_488, 2, 0, 25);  slice_488 = None
        mul_109: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_486, slice_489);  slice_486 = slice_489 = None
        add_53: "bf16[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_108, mul_109);  mul_108 = mul_109 = None
        neg_20: "bf16[200, 200, 25]" = torch.ops.aten.neg.default(add_53);  add_53 = None
        full_28: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_9: "f32[200, 200, 25]" = torch.ops.aten.minimum.default(full_28, add_50);  full_28 = None
        sub_18: "f32[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_9, 1e-20);  minimum_9 = None
        div_26: "f32[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_20, sub_18);  neg_20 = sub_18 = None
        abs_11: "f32[200, 200, 25]" = torch.ops.aten.abs.default(div_26)
        neg_21: "f32[200, 200, 25]" = torch.ops.aten.neg.default(abs_11);  abs_11 = None
        add_54: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_21, 0.001);  neg_21 = None
        div_27: "f32[200, 200, 25]" = torch.ops.aten.div.Tensor(add_54, 0.001);  add_54 = None
        tanh_9: "f32[200, 200, 25]" = torch.ops.aten.tanh.default(div_27);  div_27 = None
        add_55: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_110: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_55, 0.5);  add_55 = None
        mul_111: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_110, div_26)
        slice_490: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2)
        slice_491: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_490, 1, 2, -2);  slice_490 = None
        slice_492: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_491, 2, 0, -1);  slice_491 = None
        mul_112: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_111, slice_492);  mul_111 = slice_492 = None
        copy_20: "bf16[200, 200, 25]" = torch.ops.aten.copy.default(select_48, mul_112);  select_48 = mul_112 = None
        select_scatter_20: "bf16[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_46, copy_20, 3, 0);  select_46 = copy_20 = None
        select_scatter_21: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_474, select_scatter_20, 3, 1);  slice_474 = select_scatter_20 = None
        slice_scatter_83: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_473, select_scatter_21, 2, 0, -1);  slice_473 = select_scatter_21 = None
        slice_scatter_84: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_472, slice_scatter_83, 1, 2, -2);  slice_472 = slice_scatter_83 = None
        slice_scatter_85: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_82, slice_scatter_84, 0, 2, -2);  slice_scatter_82 = slice_scatter_84 = None
        slice_493: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_85, 0, 2, -2)
        slice_494: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_493, 1, 2, -2)
        slice_495: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_494, 2, 0, -1)
        select_49: "bf16[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_495, 3, 0)
        slice_496: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_85, 0, 2, -2)
        slice_497: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_496, 1, 2, -2);  slice_496 = None
        slice_498: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_497, 2, 0, -1);  slice_497 = None
        select_50: "bf16[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_498, 3, 0);  slice_498 = None
        select_51: "bf16[200, 200, 25]" = torch.ops.aten.select.int(select_50, 3, 1);  select_50 = None
        slice_499: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2)
        slice_500: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_499, 1, 2, -2);  slice_499 = None
        slice_501: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_500, 2, 1, 26);  slice_500 = None
        slice_502: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 1, -3)
        slice_503: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_502, 1, 2, -2);  slice_502 = None
        slice_504: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_503, 2, 1, 26);  slice_503 = None
        mul_113: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_501, slice_504);  slice_501 = slice_504 = None
        slice_505: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2)
        slice_506: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_505, 1, 2, -2);  slice_505 = None
        slice_507: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_506, 2, 1, 26);  slice_506 = None
        slice_508: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_6, 0, 1, -3)
        slice_509: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_508, 1, 2, -2);  slice_508 = None
        slice_510: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_509, 2, 1, 26);  slice_509 = None
        mul_114: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_507, slice_510);  slice_507 = slice_510 = None
        add_56: "bf16[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        neg_22: "bf16[200, 200, 25]" = torch.ops.aten.neg.default(add_56);  add_56 = None
        full_29: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_511: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2)
        slice_512: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_511, 1, 2, -2);  slice_511 = None
        slice_513: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_512, 2, 1, 26);  slice_512 = None
        slice_514: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 2, -2);  slice_scatter_7 = None
        slice_515: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_514, 1, 2, -2);  slice_514 = None
        slice_516: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_515, 2, 0, -1);  slice_515 = None
        mul_115: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_513, slice_516);  slice_513 = slice_516 = None
        slice_517: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2)
        slice_518: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_517, 1, 2, -2);  slice_517 = None
        slice_519: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_518, 2, 1, 26);  slice_518 = None
        slice_520: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_8, 0, 2, -2);  slice_scatter_8 = None
        slice_521: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_520, 1, 2, -2);  slice_520 = None
        slice_522: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_521, 2, 0, -1);  slice_521 = None
        mul_116: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_519, slice_522);  slice_519 = slice_522 = None
        add_57: "bf16[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None
        minimum_10: "f32[200, 200, 25]" = torch.ops.aten.minimum.default(full_29, add_57);  full_29 = None
        sub_19: "f32[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_10, 1e-20);  minimum_10 = None
        div_28: "f32[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_22, sub_19);  neg_22 = sub_19 = None
        abs_12: "f32[200, 200, 25]" = torch.ops.aten.abs.default(div_28)
        neg_23: "f32[200, 200, 25]" = torch.ops.aten.neg.default(abs_12);  abs_12 = None
        add_58: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_23, 0.001);  neg_23 = None
        div_29: "f32[200, 200, 25]" = torch.ops.aten.div.Tensor(add_58, 0.001);  add_58 = None
        tanh_10: "f32[200, 200, 25]" = torch.ops.aten.tanh.default(div_29);  div_29 = None
        add_59: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_117: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_59, 0.5);  add_59 = None
        mul_118: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_117, div_28)
        slice_523: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2)
        slice_524: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_523, 1, 2, -2);  slice_523 = None
        slice_525: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_524, 2, 0, -1);  slice_524 = None
        mul_119: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_118, slice_525);  mul_118 = slice_525 = None
        copy_21: "bf16[200, 200, 25]" = torch.ops.aten.copy.default(select_51, mul_119);  select_51 = mul_119 = None
        select_scatter_22: "bf16[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_49, copy_21, 3, 1);  select_49 = copy_21 = None
        select_scatter_23: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_495, select_scatter_22, 3, 0);  slice_495 = select_scatter_22 = None
        slice_scatter_86: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_494, select_scatter_23, 2, 0, -1);  slice_494 = select_scatter_23 = None
        slice_scatter_87: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_493, slice_scatter_86, 1, 2, -2);  slice_493 = slice_scatter_86 = None
        slice_scatter_88: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_85, slice_scatter_87, 0, 2, -2);  slice_scatter_85 = slice_scatter_87 = None
        slice_526: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_88, 0, 2, -2)
        slice_527: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_526, 1, 2, -2)
        slice_528: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_527, 2, 0, -1)
        select_52: "bf16[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_528, 3, 1)
        slice_529: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_88, 0, 2, -2)
        slice_530: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_529, 1, 2, -2);  slice_529 = None
        slice_531: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_530, 2, 0, -1);  slice_530 = None
        select_53: "bf16[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_531, 3, 1);  slice_531 = None
        select_54: "bf16[200, 200, 25]" = torch.ops.aten.select.int(select_53, 3, 1);  select_53 = None
        slice_532: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2)
        slice_533: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_532, 1, 2, -2);  slice_532 = None
        slice_534: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_533, 2, 1, 26);  slice_533 = None
        slice_535: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2);  slice_scatter_5 = None
        slice_536: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_535, 1, 2, -2);  slice_535 = None
        slice_537: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_536, 2, 1, 26);  slice_536 = None
        mul_120: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_534, slice_537);  slice_534 = slice_537 = None
        slice_538: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2)
        slice_539: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_538, 1, 2, -2);  slice_538 = None
        slice_540: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_539, 2, 1, 26);  slice_539 = None
        slice_541: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_6, 0, 2, -2);  slice_scatter_6 = None
        slice_542: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_541, 1, 2, -2);  slice_541 = None
        slice_543: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_542, 2, 1, 26);  slice_542 = None
        mul_121: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_540, slice_543);  slice_540 = slice_543 = None
        add_60: "bf16[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
        neg_24: "bf16[200, 200, 25]" = torch.ops.aten.neg.default(add_60);  add_60 = None
        full_30: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_11: "f32[200, 200, 25]" = torch.ops.aten.minimum.default(full_30, add_57);  full_30 = None
        sub_20: "f32[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_11, 1e-20);  minimum_11 = None
        div_30: "f32[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_24, sub_20);  neg_24 = sub_20 = None
        abs_13: "f32[200, 200, 25]" = torch.ops.aten.abs.default(div_30)
        neg_25: "f32[200, 200, 25]" = torch.ops.aten.neg.default(abs_13);  abs_13 = None
        add_61: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_25, 0.001);  neg_25 = None
        div_31: "f32[200, 200, 25]" = torch.ops.aten.div.Tensor(add_61, 0.001);  add_61 = None
        tanh_11: "f32[200, 200, 25]" = torch.ops.aten.tanh.default(div_31);  div_31 = None
        add_62: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_122: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
        mul_123: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_122, div_30)
        slice_544: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2)
        slice_545: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_544, 1, 2, -2);  slice_544 = None
        slice_546: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_545, 2, 0, -1);  slice_545 = None
        mul_124: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_123, slice_546);  mul_123 = slice_546 = None
        copy_22: "bf16[200, 200, 25]" = torch.ops.aten.copy.default(select_54, mul_124);  select_54 = mul_124 = None
        select_scatter_24: "bf16[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_52, copy_22, 3, 1);  select_52 = copy_22 = None
        select_scatter_25: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_528, select_scatter_24, 3, 1);  slice_528 = select_scatter_24 = None
        slice_scatter_89: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_527, select_scatter_25, 2, 0, -1);  slice_527 = select_scatter_25 = None
        slice_scatter_90: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_526, slice_scatter_89, 1, 2, -2);  slice_526 = slice_scatter_89 = None
        slice_scatter_91: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_88, slice_scatter_90, 0, 2, -2);  slice_scatter_88 = slice_scatter_90 = None
        slice_547: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg18_1, 0, 2, -2)
        slice_548: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_547, 1, 2, -2)
        slice_549: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_548, 2, 0, -1)
        select_55: "bf16[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_549, 3, 0)
        slice_550: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(arg18_1, 0, 2, -2)
        slice_551: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_550, 1, 2, -2);  slice_550 = None
        slice_552: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_551, 2, 0, -1);  slice_551 = None
        select_56: "bf16[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_552, 3, 0);  slice_552 = None
        select_57: "bf16[200, 200, 25]" = torch.ops.aten.select.int(select_56, 3, 0);  select_56 = None
        slice_553: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2)
        slice_554: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_553, 1, 2, -2);  slice_553 = None
        slice_555: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_554, 2, 0, 25);  slice_554 = None
        slice_556: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_46, 0, 2, -2)
        slice_557: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_556, 1, 1, -3);  slice_556 = None
        slice_558: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_557, 2, 0, 25);  slice_557 = None
        mul_125: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_555, slice_558);  slice_555 = slice_558 = None
        slice_559: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2)
        slice_560: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_559, 1, 2, -2);  slice_559 = None
        slice_561: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_560, 2, 0, 25);  slice_560 = None
        slice_562: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 2, -2)
        slice_563: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_562, 1, 1, -3);  slice_562 = None
        slice_564: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_563, 2, 0, 25);  slice_563 = None
        mul_126: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_561, slice_564);  slice_561 = slice_564 = None
        add_63: "bf16[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_125, mul_126);  mul_125 = mul_126 = None
        neg_26: "bf16[200, 200, 25]" = torch.ops.aten.neg.default(add_63);  add_63 = None
        full_31: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_12: "f32[200, 200, 25]" = torch.ops.aten.minimum.default(full_31, add_50);  full_31 = None
        sub_21: "f32[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_12, 1e-20);  minimum_12 = None
        div_32: "f32[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_26, sub_21);  neg_26 = sub_21 = None
        abs_14: "f32[200, 200, 25]" = torch.ops.aten.abs.default(div_32)
        neg_27: "f32[200, 200, 25]" = torch.ops.aten.neg.default(abs_14);  abs_14 = None
        add_64: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_27, 0.001);  neg_27 = None
        div_33: "f32[200, 200, 25]" = torch.ops.aten.div.Tensor(add_64, 0.001);  add_64 = None
        tanh_12: "f32[200, 200, 25]" = torch.ops.aten.tanh.default(div_33);  div_33 = None
        add_65: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_127: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_65, 0.5);  add_65 = None
        mul_128: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_127, div_32)
        slice_565: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2)
        slice_566: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_565, 1, 2, -2);  slice_565 = None
        slice_567: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_566, 2, 0, -1);  slice_566 = None
        mul_129: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_128, slice_567);  mul_128 = slice_567 = None
        copy_23: "bf16[200, 200, 25]" = torch.ops.aten.copy.default(select_57, mul_129);  select_57 = mul_129 = None
        select_scatter_26: "bf16[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_55, copy_23, 3, 0);  select_55 = copy_23 = None
        select_scatter_27: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_549, select_scatter_26, 3, 0);  slice_549 = select_scatter_26 = None
        slice_scatter_92: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_548, select_scatter_27, 2, 0, -1);  slice_548 = select_scatter_27 = None
        slice_scatter_93: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_547, slice_scatter_92, 1, 2, -2);  slice_547 = slice_scatter_92 = None
        slice_scatter_94: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(arg18_1, slice_scatter_93, 0, 2, -2);  slice_scatter_93 = None
        slice_568: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_94, 0, 2, -2)
        slice_569: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_568, 1, 2, -2)
        slice_570: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_569, 2, 0, -1)
        select_58: "bf16[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_570, 3, 1)
        slice_571: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_94, 0, 2, -2)
        slice_572: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_571, 1, 2, -2);  slice_571 = None
        slice_573: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_572, 2, 0, -1);  slice_572 = None
        select_59: "bf16[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_573, 3, 1);  slice_573 = None
        select_60: "bf16[200, 200, 25]" = torch.ops.aten.select.int(select_59, 3, 0);  select_59 = None
        slice_574: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2)
        slice_575: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_574, 1, 2, -2);  slice_574 = None
        slice_576: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_575, 2, 0, 25);  slice_575 = None
        slice_577: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_46, 0, 2, -2)
        slice_578: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_577, 1, 2, -2);  slice_577 = None
        slice_579: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_578, 2, 0, 25);  slice_578 = None
        mul_130: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_576, slice_579);  slice_576 = slice_579 = None
        slice_580: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2)
        slice_581: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_580, 1, 2, -2);  slice_580 = None
        slice_582: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_581, 2, 0, 25);  slice_581 = None
        slice_583: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 2, -2)
        slice_584: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_583, 1, 2, -2);  slice_583 = None
        slice_585: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_584, 2, 0, 25);  slice_584 = None
        mul_131: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_582, slice_585);  slice_582 = slice_585 = None
        add_66: "bf16[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        neg_28: "bf16[200, 200, 25]" = torch.ops.aten.neg.default(add_66);  add_66 = None
        full_32: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_13: "f32[200, 200, 25]" = torch.ops.aten.minimum.default(full_32, add_50);  full_32 = add_50 = None
        sub_22: "f32[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_13, 1e-20);  minimum_13 = None
        div_34: "f32[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_28, sub_22);  neg_28 = sub_22 = None
        abs_15: "f32[200, 200, 25]" = torch.ops.aten.abs.default(div_34)
        neg_29: "f32[200, 200, 25]" = torch.ops.aten.neg.default(abs_15);  abs_15 = None
        add_67: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_29, 0.001);  neg_29 = None
        div_35: "f32[200, 200, 25]" = torch.ops.aten.div.Tensor(add_67, 0.001);  add_67 = None
        tanh_13: "f32[200, 200, 25]" = torch.ops.aten.tanh.default(div_35);  div_35 = None
        add_68: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_132: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_68, 0.5);  add_68 = None
        mul_133: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_132, div_34)
        slice_586: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2)
        slice_587: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_586, 1, 2, -2);  slice_586 = None
        slice_588: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_587, 2, 0, -1);  slice_587 = None
        mul_134: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_133, slice_588);  mul_133 = slice_588 = None
        copy_24: "bf16[200, 200, 25]" = torch.ops.aten.copy.default(select_60, mul_134);  select_60 = mul_134 = None
        select_scatter_28: "bf16[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_58, copy_24, 3, 0);  select_58 = copy_24 = None
        select_scatter_29: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_570, select_scatter_28, 3, 1);  slice_570 = select_scatter_28 = None
        slice_scatter_95: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_569, select_scatter_29, 2, 0, -1);  slice_569 = select_scatter_29 = None
        slice_scatter_96: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_568, slice_scatter_95, 1, 2, -2);  slice_568 = slice_scatter_95 = None
        slice_scatter_97: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_94, slice_scatter_96, 0, 2, -2);  slice_scatter_94 = slice_scatter_96 = None
        slice_589: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_97, 0, 2, -2)
        slice_590: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_589, 1, 2, -2)
        slice_591: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_590, 2, 0, -1)
        select_61: "bf16[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_591, 3, 0)
        slice_592: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_97, 0, 2, -2)
        slice_593: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_592, 1, 2, -2);  slice_592 = None
        slice_594: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_593, 2, 0, -1);  slice_593 = None
        select_62: "bf16[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_594, 3, 0);  slice_594 = None
        select_63: "bf16[200, 200, 25]" = torch.ops.aten.select.int(select_62, 3, 1);  select_62 = None
        slice_595: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2)
        slice_596: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_595, 1, 2, -2);  slice_595 = None
        slice_597: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_596, 2, 1, 26);  slice_596 = None
        slice_598: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_46, 0, 2, -2)
        slice_599: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_598, 1, 1, -3);  slice_598 = None
        slice_600: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_599, 2, 1, 26);  slice_599 = None
        mul_135: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_597, slice_600);  slice_597 = slice_600 = None
        slice_601: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2)
        slice_602: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_601, 1, 2, -2);  slice_601 = None
        slice_603: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_602, 2, 1, 26);  slice_602 = None
        slice_604: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 2, -2)
        slice_605: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_604, 1, 1, -3);  slice_604 = None
        slice_606: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_605, 2, 1, 26);  slice_605 = None
        mul_136: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_603, slice_606);  slice_603 = slice_606 = None
        add_69: "bf16[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_135, mul_136);  mul_135 = mul_136 = None
        neg_30: "bf16[200, 200, 25]" = torch.ops.aten.neg.default(add_69);  add_69 = None
        full_33: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_14: "f32[200, 200, 25]" = torch.ops.aten.minimum.default(full_33, add_57);  full_33 = None
        sub_23: "f32[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_14, 1e-20);  minimum_14 = None
        div_36: "f32[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_30, sub_23);  neg_30 = sub_23 = None
        abs_16: "f32[200, 200, 25]" = torch.ops.aten.abs.default(div_36)
        neg_31: "f32[200, 200, 25]" = torch.ops.aten.neg.default(abs_16);  abs_16 = None
        add_70: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_31, 0.001);  neg_31 = None
        div_37: "f32[200, 200, 25]" = torch.ops.aten.div.Tensor(add_70, 0.001);  add_70 = None
        tanh_14: "f32[200, 200, 25]" = torch.ops.aten.tanh.default(div_37);  div_37 = None
        add_71: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_137: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_71, 0.5);  add_71 = None
        mul_138: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_137, div_36)
        slice_607: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2)
        slice_608: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_607, 1, 2, -2);  slice_607 = None
        slice_609: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_608, 2, 0, -1);  slice_608 = None
        mul_139: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_138, slice_609);  mul_138 = slice_609 = None
        copy_25: "bf16[200, 200, 25]" = torch.ops.aten.copy.default(select_63, mul_139);  select_63 = mul_139 = None
        select_scatter_30: "bf16[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_61, copy_25, 3, 1);  select_61 = copy_25 = None
        select_scatter_31: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_591, select_scatter_30, 3, 0);  slice_591 = select_scatter_30 = None
        slice_scatter_98: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_590, select_scatter_31, 2, 0, -1);  slice_590 = select_scatter_31 = None
        slice_scatter_99: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_589, slice_scatter_98, 1, 2, -2);  slice_589 = slice_scatter_98 = None
        slice_scatter_100: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_97, slice_scatter_99, 0, 2, -2);  slice_scatter_97 = slice_scatter_99 = None
        slice_610: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_100, 0, 2, -2)
        slice_611: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_610, 1, 2, -2)
        slice_612: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_611, 2, 0, -1)
        select_64: "bf16[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_612, 3, 1)
        slice_613: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_100, 0, 2, -2)
        slice_614: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice.Tensor(slice_613, 1, 2, -2);  slice_613 = None
        slice_615: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.slice.Tensor(slice_614, 2, 0, -1);  slice_614 = None
        select_65: "bf16[200, 200, 25, 2]" = torch.ops.aten.select.int(slice_615, 3, 1);  slice_615 = None
        select_66: "bf16[200, 200, 25]" = torch.ops.aten.select.int(select_65, 3, 1);  select_65 = None
        slice_616: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_8, 0, 2, -2);  mul_8 = None
        slice_617: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_616, 1, 2, -2);  slice_616 = None
        slice_618: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_617, 2, 1, 26);  slice_617 = None
        slice_619: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_46, 0, 2, -2);  slice_scatter_46 = None
        slice_620: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_619, 1, 2, -2);  slice_619 = None
        slice_621: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_620, 2, 1, 26);  slice_620 = None
        mul_140: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_618, slice_621);  slice_618 = slice_621 = None
        slice_622: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(mul_12, 0, 2, -2);  mul_12 = None
        slice_623: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_622, 1, 2, -2);  slice_622 = None
        slice_624: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_623, 2, 1, 26);  slice_623 = None
        slice_625: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 2, -2);  slice_scatter_47 = None
        slice_626: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_625, 1, 2, -2);  slice_625 = None
        slice_627: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_626, 2, 1, 26);  slice_626 = None
        mul_141: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_624, slice_627);  slice_624 = slice_627 = None
        add_72: "bf16[200, 200, 25]" = torch.ops.aten.add.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None
        neg_32: "bf16[200, 200, 25]" = torch.ops.aten.neg.default(add_72);  add_72 = None
        full_34: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_15: "f32[200, 200, 25]" = torch.ops.aten.minimum.default(full_34, add_57);  full_34 = add_57 = None
        sub_24: "f32[200, 200, 25]" = torch.ops.aten.sub.Tensor(minimum_15, 1e-20);  minimum_15 = None
        div_38: "f32[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_32, sub_24);  neg_32 = sub_24 = None
        abs_17: "f32[200, 200, 25]" = torch.ops.aten.abs.default(div_38)
        neg_33: "f32[200, 200, 25]" = torch.ops.aten.neg.default(abs_17);  abs_17 = None
        add_73: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(neg_33, 0.001);  neg_33 = None
        div_39: "f32[200, 200, 25]" = torch.ops.aten.div.Tensor(add_73, 0.001);  add_73 = None
        tanh_15: "f32[200, 200, 25]" = torch.ops.aten.tanh.default(div_39);  div_39 = None
        add_74: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_142: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(add_74, 0.5);  add_74 = None
        mul_143: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_142, div_38)
        slice_628: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2)
        slice_629: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_628, 1, 2, -2);  slice_628 = None
        slice_630: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_629, 2, 0, -1);  slice_629 = None
        mul_144: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_143, slice_630);  mul_143 = slice_630 = None
        copy_26: "bf16[200, 200, 25]" = torch.ops.aten.copy.default(select_66, mul_144);  select_66 = mul_144 = None
        select_scatter_32: "bf16[200, 200, 25, 2]" = torch.ops.aten.select_scatter.default(select_64, copy_26, 3, 1);  select_64 = copy_26 = None
        select_scatter_33: "bf16[200, 200, 25, 2, 2]" = torch.ops.aten.select_scatter.default(slice_612, select_scatter_32, 3, 1);  slice_612 = select_scatter_32 = None
        slice_scatter_101: "bf16[200, 200, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_611, select_scatter_33, 2, 0, -1);  slice_611 = select_scatter_33 = None
        slice_scatter_102: "bf16[200, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_610, slice_scatter_101, 1, 2, -2);  slice_610 = slice_scatter_101 = None
        slice_scatter_103: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.slice_scatter.default(slice_scatter_100, slice_scatter_102, 0, 2, -2);  slice_scatter_100 = slice_scatter_102 = None
        slice_631: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg19_1, 0, 2, -2)
        slice_632: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_631, 1, 2, -2)
        slice_633: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg19_1, 0, 2, -2)
        slice_634: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_633, 1, 2, -2);  slice_633 = None
        slice_635: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_634, 2, 0, -1);  slice_634 = None
        full_35: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_10, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_10 = None
        slice_636: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_35, 0, 2, -2)
        slice_637: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_636, 1, 2, -2)
        slice_638: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_35, 0, 2, -2)
        slice_639: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_638, 1, 2, -2);  slice_638 = None
        slice_640: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_639, 2, 0, -1);  slice_639 = None
        slice_641: "bf16[200]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 1, -3)
        unsqueeze_36: "bf16[200, 1]" = torch.ops.aten.unsqueeze.default(slice_641, 1);  slice_641 = None
        unsqueeze_37: "bf16[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 2);  unsqueeze_36 = None
        slice_642: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -2)
        slice_643: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_642, 1, 2, -2);  slice_642 = None
        slice_644: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_643, 2, 0, -1);  slice_643 = None
        mul_145: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_37, slice_644);  unsqueeze_37 = slice_644 = None
        mul_146: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_145, mul_105);  mul_145 = mul_105 = None
        pow_1: "f32[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_24, 2);  div_24 = None
        mul_147: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_146, pow_1);  mul_146 = pow_1 = None
        slice_645: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2)
        slice_646: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_645, 1, 2, -2);  slice_645 = None
        slice_647: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_646, 2, 0, -1);  slice_646 = None
        mul_148: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_147, slice_647);  mul_147 = slice_647 = None
        add_75: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_640, mul_148);  slice_640 = mul_148 = None
        convert_element_type_8: "bf16[200, 200, 25]" = torch.ops.prims.convert_element_type.default(add_75, torch.bfloat16);  add_75 = None
        slice_scatter_104: "bf16[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_637, convert_element_type_8, 2, 0, -1);  slice_637 = convert_element_type_8 = None
        slice_scatter_105: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_636, slice_scatter_104, 1, 2, -2);  slice_636 = slice_scatter_104 = None
        slice_scatter_106: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_35, slice_scatter_105, 0, 2, -2);  full_35 = slice_scatter_105 = None
        slice_648: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_106, 0, 2, -2)
        slice_649: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_648, 1, 2, -2)
        slice_650: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_106, 0, 2, -2)
        slice_651: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_650, 1, 2, -2);  slice_650 = None
        slice_652: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_651, 2, 0, -1);  slice_651 = None
        slice_653: "bf16[200]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 2, -2)
        unsqueeze_38: "bf16[200, 1]" = torch.ops.aten.unsqueeze.default(slice_653, 1);  slice_653 = None
        unsqueeze_39: "bf16[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 2);  unsqueeze_38 = None
        slice_654: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -2)
        slice_655: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_654, 1, 2, -2);  slice_654 = None
        slice_656: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_655, 2, 0, -1);  slice_655 = None
        mul_149: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_39, slice_656);  unsqueeze_39 = slice_656 = None
        mul_150: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_149, mul_110);  mul_149 = mul_110 = None
        pow_2: "f32[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_26, 2);  div_26 = None
        mul_151: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_150, pow_2);  mul_150 = pow_2 = None
        slice_657: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2)
        slice_658: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_657, 1, 2, -2);  slice_657 = None
        slice_659: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_658, 2, 0, -1);  slice_658 = None
        mul_152: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_151, slice_659);  mul_151 = slice_659 = None
        add_76: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_652, mul_152);  slice_652 = mul_152 = None
        convert_element_type_9: "bf16[200, 200, 25]" = torch.ops.prims.convert_element_type.default(add_76, torch.bfloat16);  add_76 = None
        slice_scatter_107: "bf16[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_649, convert_element_type_9, 2, 0, -1);  slice_649 = convert_element_type_9 = None
        slice_scatter_108: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_648, slice_scatter_107, 1, 2, -2);  slice_648 = slice_scatter_107 = None
        slice_scatter_109: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_106, slice_scatter_108, 0, 2, -2);  slice_scatter_106 = slice_scatter_108 = None
        slice_660: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_109, 0, 2, -2)
        slice_661: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_660, 1, 2, -2)
        slice_662: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_109, 0, 2, -2)
        slice_663: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_662, 1, 2, -2);  slice_662 = None
        slice_664: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_663, 2, 0, -1);  slice_663 = None
        slice_665: "bf16[200]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 1, -3)
        unsqueeze_40: "bf16[200, 1]" = torch.ops.aten.unsqueeze.default(slice_665, 1);  slice_665 = None
        unsqueeze_41: "bf16[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 2);  unsqueeze_40 = None
        slice_666: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -2)
        slice_667: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_666, 1, 2, -2);  slice_666 = None
        slice_668: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_667, 2, 0, -1);  slice_667 = None
        mul_153: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_41, slice_668);  unsqueeze_41 = slice_668 = None
        mul_154: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_153, mul_117);  mul_153 = mul_117 = None
        pow_3: "f32[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_28, 2);  div_28 = None
        mul_155: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_154, pow_3);  mul_154 = pow_3 = None
        slice_669: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2)
        slice_670: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_669, 1, 2, -2);  slice_669 = None
        slice_671: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_670, 2, 0, -1);  slice_670 = None
        mul_156: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_155, slice_671);  mul_155 = slice_671 = None
        add_77: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_664, mul_156);  slice_664 = mul_156 = None
        convert_element_type_10: "bf16[200, 200, 25]" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None
        slice_scatter_110: "bf16[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_661, convert_element_type_10, 2, 0, -1);  slice_661 = convert_element_type_10 = None
        slice_scatter_111: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_660, slice_scatter_110, 1, 2, -2);  slice_660 = slice_scatter_110 = None
        slice_scatter_112: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_109, slice_scatter_111, 0, 2, -2);  slice_scatter_109 = slice_scatter_111 = None
        slice_672: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_112, 0, 2, -2)
        slice_673: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_672, 1, 2, -2)
        slice_674: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_112, 0, 2, -2)
        slice_675: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_674, 1, 2, -2);  slice_674 = None
        slice_676: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_675, 2, 0, -1);  slice_675 = None
        slice_677: "bf16[200]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 2, -2);  arg6_1 = None
        unsqueeze_42: "bf16[200, 1]" = torch.ops.aten.unsqueeze.default(slice_677, 1);  slice_677 = None
        unsqueeze_43: "bf16[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 2);  unsqueeze_42 = None
        slice_678: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -2)
        slice_679: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_678, 1, 2, -2);  slice_678 = None
        slice_680: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_679, 2, 0, -1);  slice_679 = None
        mul_157: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_43, slice_680);  unsqueeze_43 = slice_680 = None
        mul_158: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_157, mul_122);  mul_157 = mul_122 = None
        pow_4: "f32[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_30, 2);  div_30 = None
        mul_159: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_158, pow_4);  mul_158 = pow_4 = None
        slice_681: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2)
        slice_682: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_681, 1, 2, -2);  slice_681 = None
        slice_683: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_682, 2, 0, -1);  slice_682 = None
        mul_160: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_159, slice_683);  mul_159 = slice_683 = None
        add_78: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_676, mul_160);  slice_676 = mul_160 = None
        convert_element_type_11: "bf16[200, 200, 25]" = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16);  add_78 = None
        slice_scatter_113: "bf16[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_673, convert_element_type_11, 2, 0, -1);  slice_673 = convert_element_type_11 = None
        slice_scatter_114: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_672, slice_scatter_113, 1, 2, -2);  slice_672 = slice_scatter_113 = None
        slice_scatter_115: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_112, slice_scatter_114, 0, 2, -2);  slice_scatter_112 = slice_scatter_114 = None
        slice_684: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_115, 0, 2, -2);  slice_scatter_115 = None
        slice_685: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_684, 1, 2, -2);  slice_684 = None
        slice_686: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_685, 2, 0, -1);  slice_685 = None
        slice_687: "bf16[200]" = torch.ops.aten.slice.Tensor(arg20_1, 0, 2, -2);  arg20_1 = None
        unsqueeze_44: "bf16[200, 1]" = torch.ops.aten.unsqueeze.default(slice_687, 1);  slice_687 = None
        unsqueeze_45: "bf16[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, 2);  unsqueeze_44 = None
        mul_161: "bf16[200, 1, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_45, 4);  unsqueeze_45 = None
        div_40: "bf16[200, 200, 25]" = torch.ops.aten.div.Tensor(slice_686, mul_161);  slice_686 = mul_161 = None
        full_36: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_11, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_11 = None
        slice_688: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_36, 0, 2, -2)
        slice_689: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_688, 1, 2, -2)
        slice_690: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(full_36, 0, 2, -2)
        slice_691: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_690, 1, 2, -2);  slice_690 = None
        slice_692: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_691, 2, 0, -1);  slice_691 = None
        slice_693: "bf16[200]" = torch.ops.aten.slice.Tensor(arg21_1, 0, 1, -3)
        slice_694: "bf16[200]" = torch.ops.aten.slice.Tensor(arg14_1, 0, 1, -3)
        mul_162: "bf16[200]" = torch.ops.aten.mul.Tensor(slice_693, slice_694);  slice_693 = slice_694 = None
        unsqueeze_46: "bf16[1, 200]" = torch.ops.aten.unsqueeze.default(mul_162, 0);  mul_162 = None
        unsqueeze_47: "bf16[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 2);  unsqueeze_46 = None
        slice_695: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -2)
        slice_696: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_695, 1, 2, -2);  slice_695 = None
        slice_697: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_696, 2, 0, -1);  slice_696 = None
        mul_163: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_47, slice_697);  unsqueeze_47 = slice_697 = None
        mul_164: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_163, mul_127);  mul_163 = mul_127 = None
        pow_5: "f32[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_32, 2);  div_32 = None
        mul_165: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_164, pow_5);  mul_164 = pow_5 = None
        slice_698: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2)
        slice_699: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_698, 1, 2, -2);  slice_698 = None
        slice_700: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_699, 2, 0, -1);  slice_699 = None
        mul_166: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_165, slice_700);  mul_165 = slice_700 = None
        add_79: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_692, mul_166);  slice_692 = mul_166 = None
        convert_element_type_12: "bf16[200, 200, 25]" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None
        slice_scatter_116: "bf16[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_689, convert_element_type_12, 2, 0, -1);  slice_689 = convert_element_type_12 = None
        slice_scatter_117: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_688, slice_scatter_116, 1, 2, -2);  slice_688 = slice_scatter_116 = None
        slice_scatter_118: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_36, slice_scatter_117, 0, 2, -2);  full_36 = slice_scatter_117 = None
        slice_701: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_118, 0, 2, -2)
        slice_702: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_701, 1, 2, -2)
        slice_703: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_118, 0, 2, -2)
        slice_704: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_703, 1, 2, -2);  slice_703 = None
        slice_705: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_704, 2, 0, -1);  slice_704 = None
        slice_706: "bf16[200]" = torch.ops.aten.slice.Tensor(arg21_1, 0, 2, -2)
        slice_707: "bf16[200]" = torch.ops.aten.slice.Tensor(arg14_1, 0, 2, -2)
        mul_167: "bf16[200]" = torch.ops.aten.mul.Tensor(slice_706, slice_707);  slice_706 = slice_707 = None
        unsqueeze_48: "bf16[1, 200]" = torch.ops.aten.unsqueeze.default(mul_167, 0);  mul_167 = None
        unsqueeze_49: "bf16[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, 2);  unsqueeze_48 = None
        slice_708: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -2)
        slice_709: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_708, 1, 2, -2);  slice_708 = None
        slice_710: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_709, 2, 0, -1);  slice_709 = None
        mul_168: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_49, slice_710);  unsqueeze_49 = slice_710 = None
        mul_169: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_168, mul_132);  mul_168 = mul_132 = None
        pow_6: "f32[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_34, 2);  div_34 = None
        mul_170: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_169, pow_6);  mul_169 = pow_6 = None
        slice_711: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2)
        slice_712: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_711, 1, 2, -2);  slice_711 = None
        slice_713: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_712, 2, 0, -1);  slice_712 = None
        mul_171: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_170, slice_713);  mul_170 = slice_713 = None
        add_80: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_705, mul_171);  slice_705 = mul_171 = None
        convert_element_type_13: "bf16[200, 200, 25]" = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16);  add_80 = None
        slice_scatter_119: "bf16[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_702, convert_element_type_13, 2, 0, -1);  slice_702 = convert_element_type_13 = None
        slice_scatter_120: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_701, slice_scatter_119, 1, 2, -2);  slice_701 = slice_scatter_119 = None
        slice_scatter_121: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_118, slice_scatter_120, 0, 2, -2);  slice_scatter_118 = slice_scatter_120 = None
        slice_714: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_121, 0, 2, -2)
        slice_715: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_714, 1, 2, -2)
        slice_716: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_121, 0, 2, -2)
        slice_717: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_716, 1, 2, -2);  slice_716 = None
        slice_718: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_717, 2, 0, -1);  slice_717 = None
        slice_719: "bf16[200]" = torch.ops.aten.slice.Tensor(arg21_1, 0, 1, -3)
        slice_720: "bf16[200]" = torch.ops.aten.slice.Tensor(arg14_1, 0, 1, -3)
        mul_172: "bf16[200]" = torch.ops.aten.mul.Tensor(slice_719, slice_720);  slice_719 = slice_720 = None
        unsqueeze_50: "bf16[1, 200]" = torch.ops.aten.unsqueeze.default(mul_172, 0);  mul_172 = None
        unsqueeze_51: "bf16[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, 2);  unsqueeze_50 = None
        slice_721: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -2)
        slice_722: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_721, 1, 2, -2);  slice_721 = None
        slice_723: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_722, 2, 0, -1);  slice_722 = None
        mul_173: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_51, slice_723);  unsqueeze_51 = slice_723 = None
        mul_174: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_173, mul_137);  mul_173 = mul_137 = None
        pow_7: "f32[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_36, 2);  div_36 = None
        mul_175: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_174, pow_7);  mul_174 = pow_7 = None
        slice_724: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2)
        slice_725: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_724, 1, 2, -2);  slice_724 = None
        slice_726: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_725, 2, 0, -1);  slice_725 = None
        mul_176: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_175, slice_726);  mul_175 = slice_726 = None
        add_81: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_718, mul_176);  slice_718 = mul_176 = None
        convert_element_type_14: "bf16[200, 200, 25]" = torch.ops.prims.convert_element_type.default(add_81, torch.bfloat16);  add_81 = None
        slice_scatter_122: "bf16[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_715, convert_element_type_14, 2, 0, -1);  slice_715 = convert_element_type_14 = None
        slice_scatter_123: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_714, slice_scatter_122, 1, 2, -2);  slice_714 = slice_scatter_122 = None
        slice_scatter_124: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_121, slice_scatter_123, 0, 2, -2);  slice_scatter_121 = slice_scatter_123 = None
        slice_727: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_124, 0, 2, -2)
        slice_728: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_727, 1, 2, -2)
        slice_729: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_124, 0, 2, -2)
        slice_730: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_729, 1, 2, -2);  slice_729 = None
        slice_731: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_730, 2, 0, -1);  slice_730 = None
        slice_732: "bf16[200]" = torch.ops.aten.slice.Tensor(arg21_1, 0, 2, -2);  arg21_1 = None
        slice_733: "bf16[200]" = torch.ops.aten.slice.Tensor(arg14_1, 0, 2, -2);  arg14_1 = None
        mul_177: "bf16[200]" = torch.ops.aten.mul.Tensor(slice_732, slice_733);  slice_732 = slice_733 = None
        unsqueeze_52: "bf16[1, 200]" = torch.ops.aten.unsqueeze.default(mul_177, 0);  mul_177 = None
        unsqueeze_53: "bf16[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, 2);  unsqueeze_52 = None
        slice_734: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -2);  arg2_1 = None
        slice_735: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_734, 1, 2, -2);  slice_734 = None
        slice_736: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_735, 2, 0, -1);  slice_735 = None
        mul_178: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(unsqueeze_53, slice_736);  unsqueeze_53 = slice_736 = None
        mul_179: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_178, mul_142);  mul_178 = mul_142 = None
        pow_8: "f32[200, 200, 25]" = torch.ops.aten.pow.Tensor_Scalar(div_38, 2);  div_38 = None
        mul_180: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_179, pow_8);  mul_179 = pow_8 = None
        slice_737: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg9_1, 0, 2, -2);  arg9_1 = None
        slice_738: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_737, 1, 2, -2);  slice_737 = None
        slice_739: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_738, 2, 0, -1);  slice_738 = None
        mul_181: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_180, slice_739);  mul_180 = slice_739 = None
        add_82: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_731, mul_181);  slice_731 = mul_181 = None
        convert_element_type_15: "bf16[200, 200, 25]" = torch.ops.prims.convert_element_type.default(add_82, torch.bfloat16);  add_82 = None
        slice_scatter_125: "bf16[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_728, convert_element_type_15, 2, 0, -1);  slice_728 = convert_element_type_15 = None
        slice_scatter_126: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_727, slice_scatter_125, 1, 2, -2);  slice_727 = slice_scatter_125 = None
        slice_scatter_127: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_124, slice_scatter_126, 0, 2, -2);  slice_scatter_124 = slice_scatter_126 = None
        slice_740: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_127, 0, 2, -2);  slice_scatter_127 = None
        slice_741: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_740, 1, 2, -2);  slice_740 = None
        slice_742: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_741, 2, 0, -1);  slice_741 = None
        unsqueeze_54: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(arg22_1, 0);  arg22_1 = None
        slice_743: "bf16[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_54, 1, 2, -2);  unsqueeze_54 = None
        unsqueeze_55: "bf16[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_743, 2);  slice_743 = None
        mul_182: "bf16[1, 200, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_55, 4);  unsqueeze_55 = None
        unsqueeze_56: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(arg7_1, 0);  arg7_1 = None
        slice_744: "bf16[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_56, 1, 2, -2);  unsqueeze_56 = None
        unsqueeze_57: "bf16[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_744, 2);  slice_744 = None
        mul_183: "bf16[1, 200, 1]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_57);  mul_182 = unsqueeze_57 = None
        div_41: "bf16[200, 200, 25]" = torch.ops.aten.div.Tensor(slice_742, mul_183);  slice_742 = mul_183 = None
        add_83: "bf16[200, 200, 25]" = torch.ops.aten.add.Tensor(div_40, div_41);  div_40 = div_41 = None
        copy_27: "bf16[200, 200, 25]" = torch.ops.aten.copy.default(slice_635, add_83);  slice_635 = add_83 = None
        slice_scatter_128: "bf16[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_632, copy_27, 2, 0, -1);  slice_632 = copy_27 = None
        slice_scatter_129: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_631, slice_scatter_128, 1, 2, -2);  slice_631 = slice_scatter_128 = None
        slice_scatter_130: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(arg19_1, slice_scatter_129, 0, 2, -2);  slice_scatter_129 = None
        slice_745: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_130, 0, 2, -2)
        slice_746: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_745, 1, 2, -2)
        slice_747: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_130, 0, 2, -2)
        slice_748: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_747, 1, 2, -2);  slice_747 = None
        select_67: "bf16[200, 200]" = torch.ops.aten.select.int(slice_748, 2, -1);  slice_748 = None
        full_37: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_28: "bf16[200, 200]" = torch.ops.aten.copy.default(select_67, full_37);  select_67 = full_37 = None
        select_scatter_34: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(slice_746, copy_28, 2, -1);  slice_746 = copy_28 = None
        slice_scatter_131: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_745, select_scatter_34, 1, 2, -2);  slice_745 = select_scatter_34 = None
        slice_scatter_132: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_scatter_130, slice_scatter_131, 0, 2, -2);  slice_scatter_130 = slice_scatter_131 = None
        copy_: "bf16[204, 204, 26]" = torch.ops.aten.copy_.default(arg11_1, slice_scatter_40);  arg11_1 = slice_scatter_40 = None
        copy__1: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.copy_.default(arg10_1, slice_scatter_36);  arg10_1 = slice_scatter_36 = None
        copy__2: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.copy_.default(arg15_1, slice_scatter_75);  arg15_1 = slice_scatter_75 = None
        copy__3: "bf16[204, 204, 26]" = torch.ops.aten.copy_.default(arg16_1, slice_scatter_79);  arg16_1 = slice_scatter_79 = None
        copy__4: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.copy_.default(arg17_1, slice_scatter_91);  arg17_1 = slice_scatter_91 = None
        copy__5: "bf16[204, 204, 26, 2, 2]" = torch.ops.aten.copy_.default(arg18_1, slice_scatter_103);  arg18_1 = slice_scatter_103 = None
        copy__6: "bf16[204, 204, 26]" = torch.ops.aten.copy_.default(arg19_1, slice_scatter_132);  arg19_1 = slice_scatter_132 = None
        return (copy_, copy__1, copy__2, copy__3, copy__4, copy__5, copy__6)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
