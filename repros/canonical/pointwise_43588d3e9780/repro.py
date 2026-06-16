"""
Standalone repro captured via capture_hook.
Label: torchbench_pyhpc_turbulent_kinetic_energy_infer
Pattern hash: 43588d3e9780
Shape hash: f2c38fe6
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
    def forward(self, arg0_1: "bf16[204, 204, 26, 3]", arg1_1: "bf16[204, 204, 26]", arg2_1: "bf16[204, 204]", arg3_1: "bf16[26]", arg4_1: "i64[204, 204]", arg5_1: "bf16[26]", arg6_1: "bf16[204, 204, 26]", arg7_1: "bf16[204, 204, 26]", arg8_1: "bf16[204, 204, 26]", arg9_1: "bf16[204]", arg10_1: "bf16[204]", arg11_1: "bf16[204, 204, 26]", arg12_1: "bf16[204]", arg13_1: "bf16[204]", arg14_1: "bf16[204, 204, 26]", arg15_1: "bf16[204]", arg16_1: "bf16[204]", arg17_1: "bf16[204, 204, 26, 3]", arg18_1: "bf16[204, 204, 26, 3]", arg19_1: "bf16[204, 204, 26, 3]", arg20_1: "bf16[204, 204, 26, 3]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14):
        # No stacktrace found for following nodes
        slice_1: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 2, -2)
        slice_2: "bf16[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_1, 1, 2, -2);  slice_1 = None
        select: "bf16[200, 200, 26]" = torch.ops.aten.select.int(slice_2, 3, 0);  slice_2 = None
        slice_3: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 2, -2);  arg1_1 = None
        slice_4: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_3, 1, 2, -2);  slice_3 = None
        mul: "bf16[200, 200, 26]" = torch.ops.aten.mul.Tensor(slice_4, 1);  slice_4 = None
        add: "bf16[200, 200, 26]" = torch.ops.aten.add.Tensor(select, mul);  select = mul = None
        select_1: "bf16[200, 200]" = torch.ops.aten.select.int(add, 2, -1)
        slice_5: "bf16[200, 204]" = torch.ops.aten.slice.Tensor(arg2_1, 0, 2, -2);  arg2_1 = None
        slice_6: "bf16[200, 200]" = torch.ops.aten.slice.Tensor(slice_5, 1, 2, -2);  slice_5 = None
        mul_1: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(slice_6, 1);  slice_6 = None
        select_2: "bf16[]" = torch.ops.aten.select.int(arg3_1, 0, -1)
        mul_2: "bf16[]" = torch.ops.aten.mul.Tensor(select_2, 0.5);  select_2 = None
        div: "bf16[200, 200]" = torch.ops.aten.div.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        add_1: "bf16[200, 200]" = torch.ops.aten.add.Tensor(select_1, div);  select_1 = div = None
        select_scatter: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(add, add_1, 2, -1);  add = add_1 = None
        select_3: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter, 2, -1);  select_3 = None
        slice_7: "i64[200, 204]" = torch.ops.aten.slice.Tensor(arg4_1, 0, 2, -2);  arg4_1 = None
        slice_8: "i64[200, 200]" = torch.ops.aten.slice.Tensor(slice_7, 1, 2, -2);  slice_7 = None
        sub: "i64[200, 200]" = torch.ops.aten.sub.Tensor(slice_8, 1);  slice_8 = None
        ge: "b8[200, 200]" = torch.ops.aten.ge.Scalar(sub, 0)
        unsqueeze: "b8[200, 200, 1]" = torch.ops.aten.unsqueeze.default(ge, 2);  ge = None
        iota: "i64[26]" = torch.ops.prims.iota.default(26, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1: "i64[1, 26]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        unsqueeze_2: "i64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 1);  unsqueeze_1 = None
        unsqueeze_3: "i64[200, 200, 1]" = torch.ops.aten.unsqueeze.default(sub, 2)
        eq: "b8[200, 200, 26]" = torch.ops.aten.eq.Tensor(unsqueeze_2, unsqueeze_3);  unsqueeze_2 = unsqueeze_3 = None
        bitwise_and: "b8[200, 200, 26]" = torch.ops.aten.bitwise_and.Tensor(unsqueeze, eq);  eq = None
        full: "bf16[200, 200, 26]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        slice_9: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(full, 2, 0, -1)
        unsqueeze_4: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg5_1, 0);  arg5_1 = None
        unsqueeze_5: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        slice_10: "bf16[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_5, 2, 1, 9223372036854775807);  unsqueeze_5 = None
        reciprocal: "bf16[1, 1, 25]" = torch.ops.aten.reciprocal.default(slice_10);  slice_10 = None
        mul_3: "bf16[1, 1, 25]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        mul_4: "bf16[1, 1, 25]" = torch.ops.aten.mul.Tensor(mul_3, 1.0);  mul_3 = None
        mul_5: "bf16[1, 1, 25]" = torch.ops.aten.mul.Tensor(mul_4, 0.5);  mul_4 = None
        slice_11: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 2, -2)
        slice_12: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_11, 1, 2, -2);  slice_11 = None
        slice_13: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_12, 2, 0, -1);  slice_12 = None
        slice_14: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg6_1, 0, 2, -2);  arg6_1 = None
        slice_15: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_14, 1, 2, -2);  slice_14 = None
        slice_16: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_15, 2, 1, 9223372036854775807);  slice_15 = None
        add_2: "bf16[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_13, slice_16);  slice_13 = slice_16 = None
        mul_6: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_5, add_2);  mul_5 = add_2 = None
        copy: "bf16[200, 200, 25]" = torch.ops.aten.copy.default(slice_9, mul_6);  slice_9 = mul_6 = None
        slice_scatter: "bf16[200, 200, 26]" = torch.ops.aten.slice_scatter.default(full, copy, 2, 0, -1);  full = copy = None
        unsqueeze_6: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg3_1, 0)
        unsqueeze_7: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        div_1: "bf16[200, 200, 26]" = torch.ops.aten.div.Tensor(slice_scatter, unsqueeze_7);  unsqueeze_7 = None
        add_3: "bf16[200, 200, 26]" = torch.ops.aten.add.Tensor(div_1, 1);  div_1 = None
        slice_17: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 2, -2)
        slice_18: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_17, 1, 2, -2);  slice_17 = None
        reciprocal_1: "bf16[200, 200, 26]" = torch.ops.aten.reciprocal.default(slice_18);  slice_18 = None
        mul_7: "bf16[200, 200, 26]" = torch.ops.aten.mul.Tensor(reciprocal_1, 0.7);  reciprocal_1 = None
        full_1: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_4: "bf16[204, 204, 26]" = torch.ops.aten.select.int(arg0_1, 3, 0)
        maximum: "f32[204, 204, 26]" = torch.ops.aten.maximum.default(full_1, select_4);  full_1 = select_4 = None
        sqrt: "f32[204, 204, 26]" = torch.ops.aten.sqrt.default(maximum);  maximum = None
        slice_19: "f32[200, 204, 26]" = torch.ops.aten.slice.Tensor(sqrt, 0, 2, -2)
        slice_20: "f32[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_19, 1, 2, -2);  slice_19 = None
        mul_8: "f32[200, 200, 26]" = torch.ops.aten.mul.Tensor(mul_7, slice_20);  mul_7 = slice_20 = None
        add_4: "f32[200, 200, 26]" = torch.ops.aten.add.Tensor(add_3, mul_8);  add_3 = mul_8 = None
        iota_1: "i64[26]" = torch.ops.prims.iota.default(26, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_8: "i64[1, 26]" = torch.ops.aten.unsqueeze.default(iota_1, 0);  iota_1 = None
        unsqueeze_9: "i64[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 1);  unsqueeze_8 = None
        unsqueeze_10: "i64[200, 200, 1]" = torch.ops.aten.unsqueeze.default(sub, 2);  sub = None
        ge_1: "b8[200, 200, 26]" = torch.ops.aten.ge.Tensor(unsqueeze_9, unsqueeze_10);  unsqueeze_9 = unsqueeze_10 = None
        bitwise_and_1: "b8[200, 200, 26]" = torch.ops.aten.bitwise_and.Tensor(unsqueeze, ge_1);  unsqueeze = ge_1 = None
        full_2: "bf16[200, 200, 26]" = torch.ops.aten.full.default(_shape_param_1, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1 = None
        slice_21: "bf16[200, 200, 24]" = torch.ops.aten.slice.Tensor(full_2, 2, 1, -1)
        slice_22: "bf16[200, 200, 24]" = torch.ops.aten.slice.Tensor(slice_scatter, 2, 1, -1)
        slice_23: "bf16[200, 200, 24]" = torch.ops.aten.slice.Tensor(slice_scatter, 2, 0, -2)
        add_5: "bf16[200, 200, 24]" = torch.ops.aten.add.Tensor(slice_22, slice_23);  slice_22 = slice_23 = None
        unsqueeze_11: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg3_1, 0)
        unsqueeze_12: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 1);  unsqueeze_11 = None
        slice_24: "bf16[1, 1, 24]" = torch.ops.aten.slice.Tensor(unsqueeze_12, 2, 1, -1);  unsqueeze_12 = None
        div_2: "bf16[200, 200, 24]" = torch.ops.aten.div.Tensor(add_5, slice_24);  add_5 = slice_24 = None
        add_6: "bf16[200, 200, 24]" = torch.ops.aten.add.Tensor(div_2, 1);  div_2 = None
        slice_25: "f32[200, 204, 26]" = torch.ops.aten.slice.Tensor(sqrt, 0, 2, -2)
        slice_26: "f32[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_25, 1, 2, -2);  slice_25 = None
        slice_27: "f32[200, 200, 24]" = torch.ops.aten.slice.Tensor(slice_26, 2, 1, -1);  slice_26 = None
        mul_9: "f32[200, 200, 24]" = torch.ops.aten.mul.Tensor(slice_27, 0.7);  slice_27 = None
        slice_28: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 2, -2)
        slice_29: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_28, 1, 2, -2);  slice_28 = None
        slice_30: "bf16[200, 200, 24]" = torch.ops.aten.slice.Tensor(slice_29, 2, 1, -1);  slice_29 = None
        div_3: "f32[200, 200, 24]" = torch.ops.aten.div.Tensor(mul_9, slice_30);  mul_9 = slice_30 = None
        add_7: "f32[200, 200, 24]" = torch.ops.aten.add.Tensor(add_6, div_3);  add_6 = div_3 = None
        copy_1: "bf16[200, 200, 24]" = torch.ops.aten.copy.default(slice_21, add_7);  slice_21 = add_7 = None
        slice_scatter_1: "bf16[200, 200, 26]" = torch.ops.aten.slice_scatter.default(full_2, copy_1, 2, 1, -1);  full_2 = copy_1 = None
        select_5: "bf16[200, 200]" = torch.ops.aten.select.int(slice_scatter_1, 2, -1)
        select_6: "bf16[200, 200]" = torch.ops.aten.select.int(slice_scatter, 2, -2)
        select_7: "bf16[]" = torch.ops.aten.select.int(arg3_1, 0, -1)
        mul_10: "bf16[]" = torch.ops.aten.mul.Tensor(select_7, 0.5);  select_7 = None
        div_4: "bf16[200, 200]" = torch.ops.aten.div.Tensor(select_6, mul_10);  select_6 = mul_10 = None
        add_8: "bf16[200, 200]" = torch.ops.aten.add.Tensor(div_4, 1);  div_4 = None
        slice_31: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg7_1, 0, 2, -2);  arg7_1 = None
        slice_32: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_31, 1, 2, -2);  slice_31 = None
        select_8: "bf16[200, 200]" = torch.ops.aten.select.int(slice_32, 2, -1);  slice_32 = None
        reciprocal_2: "bf16[200, 200]" = torch.ops.aten.reciprocal.default(select_8);  select_8 = None
        mul_11: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(reciprocal_2, 0.7);  reciprocal_2 = None
        slice_33: "f32[200, 204, 26]" = torch.ops.aten.slice.Tensor(sqrt, 0, 2, -2);  sqrt = None
        slice_34: "f32[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_33, 1, 2, -2);  slice_33 = None
        select_9: "f32[200, 200]" = torch.ops.aten.select.int(slice_34, 2, -1);  slice_34 = None
        mul_12: "f32[200, 200]" = torch.ops.aten.mul.Tensor(mul_11, select_9);  mul_11 = select_9 = None
        add_9: "f32[200, 200]" = torch.ops.aten.add.Tensor(add_8, mul_12);  add_8 = mul_12 = None
        copy_2: "bf16[200, 200]" = torch.ops.aten.copy.default(select_5, add_9);  select_5 = add_9 = None
        select_scatter_1: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(slice_scatter_1, copy_2, 2, -1);  slice_scatter_1 = copy_2 = None
        full_3: "bf16[]" = torch.ops.aten.full.default([], 1.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[200, 200, 26]" = torch.ops.aten.where.self(bitwise_and_1, select_scatter_1, full_3);  select_scatter_1 = full_3 = None
        where_1: "f32[200, 200, 26]" = torch.ops.aten.where.self(bitwise_and, add_4, where);  add_4 = where = None
        select_10: "f32[200, 200]" = torch.ops.aten.select.int(where_1, 2, 1)
        full_4: "bf16[200, 200, 26]" = torch.ops.aten.full.default(_shape_param_2, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_2 = None
        slice_35: "bf16[200, 200, 24]" = torch.ops.aten.slice.Tensor(full_4, 2, 1, -1)
        slice_36: "bf16[200, 200, 24]" = torch.ops.aten.slice.Tensor(slice_scatter, 2, 0, -2)
        neg: "bf16[200, 200, 24]" = torch.ops.aten.neg.default(slice_36);  slice_36 = None
        unsqueeze_13: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg3_1, 0)
        unsqueeze_14: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 1);  unsqueeze_13 = None
        slice_37: "bf16[1, 1, 24]" = torch.ops.aten.slice.Tensor(unsqueeze_14, 2, 1, -1);  unsqueeze_14 = None
        div_5: "bf16[200, 200, 24]" = torch.ops.aten.div.Tensor(neg, slice_37);  neg = slice_37 = None
        copy_3: "bf16[200, 200, 24]" = torch.ops.aten.copy.default(slice_35, div_5);  slice_35 = div_5 = None
        slice_scatter_2: "bf16[200, 200, 26]" = torch.ops.aten.slice_scatter.default(full_4, copy_3, 2, 1, -1);  full_4 = copy_3 = None
        select_11: "bf16[200, 200]" = torch.ops.aten.select.int(slice_scatter_2, 2, -1)
        select_12: "bf16[200, 200]" = torch.ops.aten.select.int(slice_scatter, 2, -2)
        neg_1: "bf16[200, 200]" = torch.ops.aten.neg.default(select_12);  select_12 = None
        select_13: "bf16[]" = torch.ops.aten.select.int(arg3_1, 0, -1)
        mul_13: "bf16[]" = torch.ops.aten.mul.Tensor(select_13, 0.5);  select_13 = None
        div_6: "bf16[200, 200]" = torch.ops.aten.div.Tensor(neg_1, mul_13);  neg_1 = mul_13 = None
        copy_4: "bf16[200, 200]" = torch.ops.aten.copy.default(select_11, div_6);  select_11 = div_6 = None
        select_scatter_2: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(slice_scatter_2, copy_4, 2, -1);  slice_scatter_2 = copy_4 = None
        mul_14: "bf16[200, 200, 26]" = torch.ops.aten.mul.Tensor(bitwise_and_1, select_scatter_2);  select_scatter_2 = None
        logical_not: "b8[200, 200, 26]" = torch.ops.aten.logical_not.default(bitwise_and);  bitwise_and = None
        mul_15: "bf16[200, 200, 26]" = torch.ops.aten.mul.Tensor(mul_14, logical_not);  mul_14 = logical_not = None
        select_14: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 1)
        select_15: "f32[200, 200]" = torch.ops.aten.select.int(where_1, 2, 0)
        div_7: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_14, select_15);  select_14 = select_15 = None
        neg_2: "f32[200, 200]" = torch.ops.aten.neg.default(div_7)
        full_5: "bf16[200, 200, 26]" = torch.ops.aten.full.default(_shape_param_3, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_3 = None
        slice_38: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(full_5, 2, 0, -1)
        slice_39: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_scatter, 2, 0, -1);  slice_scatter = None
        neg_3: "bf16[200, 200, 25]" = torch.ops.aten.neg.default(slice_39);  slice_39 = None
        unsqueeze_15: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg3_1, 0)
        unsqueeze_16: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 1);  unsqueeze_15 = None
        slice_40: "bf16[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_16, 2, 0, -1);  unsqueeze_16 = None
        div_8: "bf16[200, 200, 25]" = torch.ops.aten.div.Tensor(neg_3, slice_40);  neg_3 = slice_40 = None
        copy_5: "bf16[200, 200, 25]" = torch.ops.aten.copy.default(slice_38, div_8);  slice_38 = div_8 = None
        slice_scatter_3: "bf16[200, 200, 26]" = torch.ops.aten.slice_scatter.default(full_5, copy_5, 2, 0, -1);  full_5 = copy_5 = None
        mul_16: "bf16[200, 200, 26]" = torch.ops.aten.mul.Tensor(bitwise_and_1, slice_scatter_3);  slice_scatter_3 = None
        select_16: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 0)
        mul_17: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_2, select_16);  neg_2 = select_16 = None
        add_10: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_10, mul_17);  select_10 = mul_17 = None
        select_scatter_3: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(where_1, add_10, 2, 1);  where_1 = add_10 = None
        select_17: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_3, 2, 1);  select_17 = None
        select_18: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter, 2, -1)
        select_scatter_4: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter, select_18, 2, -1);  select_scatter = select_18 = None
        mul_18: "bf16[200, 200, 26]" = torch.ops.aten.mul.Tensor(bitwise_and_1, select_scatter_4);  select_scatter_4 = None
        select_19: "bf16[200, 200]" = torch.ops.aten.select.int(mul_18, 2, 1)
        neg_4: "f32[200, 200]" = torch.ops.aten.neg.default(div_7);  div_7 = None
        select_20: "bf16[200, 200]" = torch.ops.aten.select.int(mul_18, 2, 0)
        mul_19: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_4, select_20);  neg_4 = select_20 = None
        add_11: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_19, mul_19);  select_19 = mul_19 = None
        convert_element_type: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None
        select_scatter_5: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(mul_18, convert_element_type, 2, 1);  mul_18 = convert_element_type = None
        select_21: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_5, 2, 1);  select_21 = None
        select_22: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_3, 2, 1)
        select_scatter_6: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_3, select_22, 2, 1);  select_scatter_3 = select_22 = None
        select_23: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_6, 2, 2)
        select_24: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 2)
        select_25: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_6, 2, 1)
        div_9: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_24, select_25);  select_24 = select_25 = None
        neg_5: "f32[200, 200]" = torch.ops.aten.neg.default(div_9)
        select_26: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 1)
        mul_20: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_5, select_26);  neg_5 = select_26 = None
        add_12: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_23, mul_20);  select_23 = mul_20 = None
        select_scatter_7: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_6, add_12, 2, 2);  select_scatter_6 = add_12 = None
        select_27: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_7, 2, 2);  select_27 = None
        select_28: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_5, 2, 1)
        select_scatter_8: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_5, select_28, 2, 1);  select_scatter_5 = select_28 = None
        select_29: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_8, 2, 2)
        neg_6: "f32[200, 200]" = torch.ops.aten.neg.default(div_9);  div_9 = None
        select_30: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_8, 2, 1)
        mul_21: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_6, select_30);  neg_6 = select_30 = None
        add_13: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_29, mul_21);  select_29 = mul_21 = None
        convert_element_type_1: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None
        select_scatter_9: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_8, convert_element_type_1, 2, 2);  select_scatter_8 = convert_element_type_1 = None
        select_31: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_9, 2, 2);  select_31 = None
        select_32: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_7, 2, 2)
        select_scatter_10: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_7, select_32, 2, 2);  select_scatter_7 = select_32 = None
        select_33: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_10, 2, 3)
        select_34: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 3)
        select_35: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_10, 2, 2)
        div_10: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_34, select_35);  select_34 = select_35 = None
        neg_7: "f32[200, 200]" = torch.ops.aten.neg.default(div_10)
        select_36: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 2)
        mul_22: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_7, select_36);  neg_7 = select_36 = None
        add_14: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_33, mul_22);  select_33 = mul_22 = None
        select_scatter_11: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_10, add_14, 2, 3);  select_scatter_10 = add_14 = None
        select_37: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_11, 2, 3);  select_37 = None
        select_38: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_9, 2, 2)
        select_scatter_12: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_9, select_38, 2, 2);  select_scatter_9 = select_38 = None
        select_39: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_12, 2, 3)
        neg_8: "f32[200, 200]" = torch.ops.aten.neg.default(div_10);  div_10 = None
        select_40: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_12, 2, 2)
        mul_23: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_8, select_40);  neg_8 = select_40 = None
        add_15: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_39, mul_23);  select_39 = mul_23 = None
        convert_element_type_2: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None
        select_scatter_13: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_12, convert_element_type_2, 2, 3);  select_scatter_12 = convert_element_type_2 = None
        select_41: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_13, 2, 3);  select_41 = None
        select_42: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_11, 2, 3)
        select_scatter_14: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_11, select_42, 2, 3);  select_scatter_11 = select_42 = None
        select_43: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_14, 2, 4)
        select_44: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 4)
        select_45: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_14, 2, 3)
        div_11: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_44, select_45);  select_44 = select_45 = None
        neg_9: "f32[200, 200]" = torch.ops.aten.neg.default(div_11)
        select_46: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 3)
        mul_24: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_9, select_46);  neg_9 = select_46 = None
        add_16: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_43, mul_24);  select_43 = mul_24 = None
        select_scatter_15: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_14, add_16, 2, 4);  select_scatter_14 = add_16 = None
        select_47: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_15, 2, 4);  select_47 = None
        select_48: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_13, 2, 3)
        select_scatter_16: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_13, select_48, 2, 3);  select_scatter_13 = select_48 = None
        select_49: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_16, 2, 4)
        neg_10: "f32[200, 200]" = torch.ops.aten.neg.default(div_11);  div_11 = None
        select_50: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_16, 2, 3)
        mul_25: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_10, select_50);  neg_10 = select_50 = None
        add_17: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_49, mul_25);  select_49 = mul_25 = None
        convert_element_type_3: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None
        select_scatter_17: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_16, convert_element_type_3, 2, 4);  select_scatter_16 = convert_element_type_3 = None
        select_51: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_17, 2, 4);  select_51 = None
        select_52: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_15, 2, 4)
        select_scatter_18: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_15, select_52, 2, 4);  select_scatter_15 = select_52 = None
        select_53: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_18, 2, 5)
        select_54: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 5)
        select_55: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_18, 2, 4)
        div_12: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_54, select_55);  select_54 = select_55 = None
        neg_11: "f32[200, 200]" = torch.ops.aten.neg.default(div_12)
        select_56: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 4)
        mul_26: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_11, select_56);  neg_11 = select_56 = None
        add_18: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_53, mul_26);  select_53 = mul_26 = None
        select_scatter_19: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_18, add_18, 2, 5);  select_scatter_18 = add_18 = None
        select_57: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_19, 2, 5);  select_57 = None
        select_58: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_17, 2, 4)
        select_scatter_20: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_17, select_58, 2, 4);  select_scatter_17 = select_58 = None
        select_59: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_20, 2, 5)
        neg_12: "f32[200, 200]" = torch.ops.aten.neg.default(div_12);  div_12 = None
        select_60: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_20, 2, 4)
        mul_27: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_12, select_60);  neg_12 = select_60 = None
        add_19: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_59, mul_27);  select_59 = mul_27 = None
        convert_element_type_4: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None
        select_scatter_21: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_20, convert_element_type_4, 2, 5);  select_scatter_20 = convert_element_type_4 = None
        select_61: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_21, 2, 5);  select_61 = None
        select_62: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_19, 2, 5)
        select_scatter_22: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_19, select_62, 2, 5);  select_scatter_19 = select_62 = None
        select_63: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_22, 2, 6)
        select_64: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 6)
        select_65: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_22, 2, 5)
        div_13: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_64, select_65);  select_64 = select_65 = None
        neg_13: "f32[200, 200]" = torch.ops.aten.neg.default(div_13)
        select_66: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 5)
        mul_28: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_13, select_66);  neg_13 = select_66 = None
        add_20: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_63, mul_28);  select_63 = mul_28 = None
        select_scatter_23: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_22, add_20, 2, 6);  select_scatter_22 = add_20 = None
        select_67: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_23, 2, 6);  select_67 = None
        select_68: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_21, 2, 5)
        select_scatter_24: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_21, select_68, 2, 5);  select_scatter_21 = select_68 = None
        select_69: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_24, 2, 6)
        neg_14: "f32[200, 200]" = torch.ops.aten.neg.default(div_13);  div_13 = None
        select_70: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_24, 2, 5)
        mul_29: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_14, select_70);  neg_14 = select_70 = None
        add_21: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_69, mul_29);  select_69 = mul_29 = None
        convert_element_type_5: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16);  add_21 = None
        select_scatter_25: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_24, convert_element_type_5, 2, 6);  select_scatter_24 = convert_element_type_5 = None
        select_71: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_25, 2, 6);  select_71 = None
        select_72: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_23, 2, 6)
        select_scatter_26: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_23, select_72, 2, 6);  select_scatter_23 = select_72 = None
        select_73: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_26, 2, 7)
        select_74: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 7)
        select_75: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_26, 2, 6)
        div_14: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_74, select_75);  select_74 = select_75 = None
        neg_15: "f32[200, 200]" = torch.ops.aten.neg.default(div_14)
        select_76: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 6)
        mul_30: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_15, select_76);  neg_15 = select_76 = None
        add_22: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_73, mul_30);  select_73 = mul_30 = None
        select_scatter_27: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_26, add_22, 2, 7);  select_scatter_26 = add_22 = None
        select_77: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_27, 2, 7);  select_77 = None
        select_78: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_25, 2, 6)
        select_scatter_28: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_25, select_78, 2, 6);  select_scatter_25 = select_78 = None
        select_79: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_28, 2, 7)
        neg_16: "f32[200, 200]" = torch.ops.aten.neg.default(div_14);  div_14 = None
        select_80: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_28, 2, 6)
        mul_31: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_16, select_80);  neg_16 = select_80 = None
        add_23: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_79, mul_31);  select_79 = mul_31 = None
        convert_element_type_6: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None
        select_scatter_29: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_28, convert_element_type_6, 2, 7);  select_scatter_28 = convert_element_type_6 = None
        select_81: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_29, 2, 7);  select_81 = None
        select_82: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_27, 2, 7)
        select_scatter_30: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_27, select_82, 2, 7);  select_scatter_27 = select_82 = None
        select_83: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_30, 2, 8)
        select_84: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 8)
        select_85: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_30, 2, 7)
        div_15: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_84, select_85);  select_84 = select_85 = None
        neg_17: "f32[200, 200]" = torch.ops.aten.neg.default(div_15)
        select_86: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 7)
        mul_32: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_17, select_86);  neg_17 = select_86 = None
        add_24: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_83, mul_32);  select_83 = mul_32 = None
        select_scatter_31: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_30, add_24, 2, 8);  select_scatter_30 = add_24 = None
        select_87: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_31, 2, 8);  select_87 = None
        select_88: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_29, 2, 7)
        select_scatter_32: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_29, select_88, 2, 7);  select_scatter_29 = select_88 = None
        select_89: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_32, 2, 8)
        neg_18: "f32[200, 200]" = torch.ops.aten.neg.default(div_15);  div_15 = None
        select_90: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_32, 2, 7)
        mul_33: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_18, select_90);  neg_18 = select_90 = None
        add_25: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_89, mul_33);  select_89 = mul_33 = None
        convert_element_type_7: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_25, torch.bfloat16);  add_25 = None
        select_scatter_33: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_32, convert_element_type_7, 2, 8);  select_scatter_32 = convert_element_type_7 = None
        select_91: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_33, 2, 8);  select_91 = None
        select_92: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_31, 2, 8)
        select_scatter_34: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_31, select_92, 2, 8);  select_scatter_31 = select_92 = None
        select_93: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_34, 2, 9)
        select_94: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 9)
        select_95: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_34, 2, 8)
        div_16: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_94, select_95);  select_94 = select_95 = None
        neg_19: "f32[200, 200]" = torch.ops.aten.neg.default(div_16)
        select_96: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 8)
        mul_34: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_19, select_96);  neg_19 = select_96 = None
        add_26: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_93, mul_34);  select_93 = mul_34 = None
        select_scatter_35: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_34, add_26, 2, 9);  select_scatter_34 = add_26 = None
        select_97: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_35, 2, 9);  select_97 = None
        select_98: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_33, 2, 8)
        select_scatter_36: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_33, select_98, 2, 8);  select_scatter_33 = select_98 = None
        select_99: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_36, 2, 9)
        neg_20: "f32[200, 200]" = torch.ops.aten.neg.default(div_16);  div_16 = None
        select_100: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_36, 2, 8)
        mul_35: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_20, select_100);  neg_20 = select_100 = None
        add_27: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_99, mul_35);  select_99 = mul_35 = None
        convert_element_type_8: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16);  add_27 = None
        select_scatter_37: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_36, convert_element_type_8, 2, 9);  select_scatter_36 = convert_element_type_8 = None
        select_101: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_37, 2, 9);  select_101 = None
        select_102: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_35, 2, 9)
        select_scatter_38: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_35, select_102, 2, 9);  select_scatter_35 = select_102 = None
        select_103: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_38, 2, 10)
        select_104: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 10)
        select_105: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_38, 2, 9)
        div_17: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_104, select_105);  select_104 = select_105 = None
        neg_21: "f32[200, 200]" = torch.ops.aten.neg.default(div_17)
        select_106: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 9)
        mul_36: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_21, select_106);  neg_21 = select_106 = None
        add_28: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_103, mul_36);  select_103 = mul_36 = None
        select_scatter_39: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_38, add_28, 2, 10);  select_scatter_38 = add_28 = None
        select_107: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_39, 2, 10);  select_107 = None
        select_108: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_37, 2, 9)
        select_scatter_40: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_37, select_108, 2, 9);  select_scatter_37 = select_108 = None
        select_109: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_40, 2, 10)
        neg_22: "f32[200, 200]" = torch.ops.aten.neg.default(div_17);  div_17 = None
        select_110: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_40, 2, 9)
        mul_37: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_22, select_110);  neg_22 = select_110 = None
        add_29: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_109, mul_37);  select_109 = mul_37 = None
        convert_element_type_9: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None
        select_scatter_41: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_40, convert_element_type_9, 2, 10);  select_scatter_40 = convert_element_type_9 = None
        select_111: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_41, 2, 10);  select_111 = None
        select_112: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_39, 2, 10)
        select_scatter_42: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_39, select_112, 2, 10);  select_scatter_39 = select_112 = None
        select_113: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_42, 2, 11)
        select_114: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 11)
        select_115: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_42, 2, 10)
        div_18: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_114, select_115);  select_114 = select_115 = None
        neg_23: "f32[200, 200]" = torch.ops.aten.neg.default(div_18)
        select_116: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 10)
        mul_38: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_23, select_116);  neg_23 = select_116 = None
        add_30: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_113, mul_38);  select_113 = mul_38 = None
        select_scatter_43: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_42, add_30, 2, 11);  select_scatter_42 = add_30 = None
        select_117: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_43, 2, 11);  select_117 = None
        select_118: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_41, 2, 10)
        select_scatter_44: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_41, select_118, 2, 10);  select_scatter_41 = select_118 = None
        select_119: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_44, 2, 11)
        neg_24: "f32[200, 200]" = torch.ops.aten.neg.default(div_18);  div_18 = None
        select_120: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_44, 2, 10)
        mul_39: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_24, select_120);  neg_24 = select_120 = None
        add_31: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_119, mul_39);  select_119 = mul_39 = None
        convert_element_type_10: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None
        select_scatter_45: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_44, convert_element_type_10, 2, 11);  select_scatter_44 = convert_element_type_10 = None
        select_121: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_45, 2, 11);  select_121 = None
        select_122: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_43, 2, 11)
        select_scatter_46: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_43, select_122, 2, 11);  select_scatter_43 = select_122 = None
        select_123: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_46, 2, 12)
        select_124: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 12)
        select_125: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_46, 2, 11)
        div_19: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_124, select_125);  select_124 = select_125 = None
        neg_25: "f32[200, 200]" = torch.ops.aten.neg.default(div_19)
        select_126: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 11)
        mul_40: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_25, select_126);  neg_25 = select_126 = None
        add_32: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_123, mul_40);  select_123 = mul_40 = None
        select_scatter_47: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_46, add_32, 2, 12);  select_scatter_46 = add_32 = None
        select_127: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_47, 2, 12);  select_127 = None
        select_128: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_45, 2, 11)
        select_scatter_48: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_45, select_128, 2, 11);  select_scatter_45 = select_128 = None
        select_129: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_48, 2, 12)
        neg_26: "f32[200, 200]" = torch.ops.aten.neg.default(div_19);  div_19 = None
        select_130: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_48, 2, 11)
        mul_41: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_26, select_130);  neg_26 = select_130 = None
        add_33: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_129, mul_41);  select_129 = mul_41 = None
        convert_element_type_11: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16);  add_33 = None
        select_scatter_49: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_48, convert_element_type_11, 2, 12);  select_scatter_48 = convert_element_type_11 = None
        select_131: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_49, 2, 12);  select_131 = None
        select_132: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_47, 2, 12)
        select_scatter_50: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_47, select_132, 2, 12);  select_scatter_47 = select_132 = None
        select_133: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_50, 2, 13)
        select_134: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 13)
        select_135: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_50, 2, 12)
        div_20: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_134, select_135);  select_134 = select_135 = None
        neg_27: "f32[200, 200]" = torch.ops.aten.neg.default(div_20)
        select_136: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 12)
        mul_42: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_27, select_136);  neg_27 = select_136 = None
        add_34: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_133, mul_42);  select_133 = mul_42 = None
        select_scatter_51: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_50, add_34, 2, 13);  select_scatter_50 = add_34 = None
        select_137: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_51, 2, 13);  select_137 = None
        select_138: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_49, 2, 12)
        select_scatter_52: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_49, select_138, 2, 12);  select_scatter_49 = select_138 = None
        select_139: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_52, 2, 13)
        neg_28: "f32[200, 200]" = torch.ops.aten.neg.default(div_20);  div_20 = None
        select_140: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_52, 2, 12)
        mul_43: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_28, select_140);  neg_28 = select_140 = None
        add_35: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_139, mul_43);  select_139 = mul_43 = None
        convert_element_type_12: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None
        select_scatter_53: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_52, convert_element_type_12, 2, 13);  select_scatter_52 = convert_element_type_12 = None
        select_141: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_53, 2, 13);  select_141 = None
        select_142: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_51, 2, 13)
        select_scatter_54: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_51, select_142, 2, 13);  select_scatter_51 = select_142 = None
        select_143: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_54, 2, 14)
        select_144: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 14)
        select_145: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_54, 2, 13)
        div_21: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_144, select_145);  select_144 = select_145 = None
        neg_29: "f32[200, 200]" = torch.ops.aten.neg.default(div_21)
        select_146: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 13)
        mul_44: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_29, select_146);  neg_29 = select_146 = None
        add_36: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_143, mul_44);  select_143 = mul_44 = None
        select_scatter_55: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_54, add_36, 2, 14);  select_scatter_54 = add_36 = None
        select_147: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_55, 2, 14);  select_147 = None
        select_148: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_53, 2, 13)
        select_scatter_56: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_53, select_148, 2, 13);  select_scatter_53 = select_148 = None
        select_149: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_56, 2, 14)
        neg_30: "f32[200, 200]" = torch.ops.aten.neg.default(div_21);  div_21 = None
        select_150: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_56, 2, 13)
        mul_45: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_30, select_150);  neg_30 = select_150 = None
        add_37: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_149, mul_45);  select_149 = mul_45 = None
        convert_element_type_13: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16);  add_37 = None
        select_scatter_57: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_56, convert_element_type_13, 2, 14);  select_scatter_56 = convert_element_type_13 = None
        select_151: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_57, 2, 14);  select_151 = None
        select_152: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_55, 2, 14)
        select_scatter_58: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_55, select_152, 2, 14);  select_scatter_55 = select_152 = None
        select_153: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_58, 2, 15)
        select_154: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 15)
        select_155: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_58, 2, 14)
        div_22: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_154, select_155);  select_154 = select_155 = None
        neg_31: "f32[200, 200]" = torch.ops.aten.neg.default(div_22)
        select_156: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 14)
        mul_46: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_31, select_156);  neg_31 = select_156 = None
        add_38: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_153, mul_46);  select_153 = mul_46 = None
        select_scatter_59: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_58, add_38, 2, 15);  select_scatter_58 = add_38 = None
        select_157: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_59, 2, 15);  select_157 = None
        select_158: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_57, 2, 14)
        select_scatter_60: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_57, select_158, 2, 14);  select_scatter_57 = select_158 = None
        select_159: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_60, 2, 15)
        neg_32: "f32[200, 200]" = torch.ops.aten.neg.default(div_22);  div_22 = None
        select_160: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_60, 2, 14)
        mul_47: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_32, select_160);  neg_32 = select_160 = None
        add_39: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_159, mul_47);  select_159 = mul_47 = None
        convert_element_type_14: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None
        select_scatter_61: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_60, convert_element_type_14, 2, 15);  select_scatter_60 = convert_element_type_14 = None
        select_161: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_61, 2, 15);  select_161 = None
        select_162: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_59, 2, 15)
        select_scatter_62: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_59, select_162, 2, 15);  select_scatter_59 = select_162 = None
        select_163: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_62, 2, 16)
        select_164: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 16)
        select_165: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_62, 2, 15)
        div_23: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_164, select_165);  select_164 = select_165 = None
        neg_33: "f32[200, 200]" = torch.ops.aten.neg.default(div_23)
        select_166: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 15)
        mul_48: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_33, select_166);  neg_33 = select_166 = None
        add_40: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_163, mul_48);  select_163 = mul_48 = None
        select_scatter_63: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_62, add_40, 2, 16);  select_scatter_62 = add_40 = None
        select_167: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_63, 2, 16);  select_167 = None
        select_168: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_61, 2, 15)
        select_scatter_64: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_61, select_168, 2, 15);  select_scatter_61 = select_168 = None
        select_169: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_64, 2, 16)
        neg_34: "f32[200, 200]" = torch.ops.aten.neg.default(div_23);  div_23 = None
        select_170: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_64, 2, 15)
        mul_49: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_34, select_170);  neg_34 = select_170 = None
        add_41: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_169, mul_49);  select_169 = mul_49 = None
        convert_element_type_15: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None
        select_scatter_65: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_64, convert_element_type_15, 2, 16);  select_scatter_64 = convert_element_type_15 = None
        select_171: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_65, 2, 16);  select_171 = None
        select_172: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_63, 2, 16)
        select_scatter_66: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_63, select_172, 2, 16);  select_scatter_63 = select_172 = None
        select_173: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_66, 2, 17)
        select_174: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 17)
        select_175: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_66, 2, 16)
        div_24: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_174, select_175);  select_174 = select_175 = None
        neg_35: "f32[200, 200]" = torch.ops.aten.neg.default(div_24)
        select_176: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 16)
        mul_50: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_35, select_176);  neg_35 = select_176 = None
        add_42: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_173, mul_50);  select_173 = mul_50 = None
        select_scatter_67: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_66, add_42, 2, 17);  select_scatter_66 = add_42 = None
        select_177: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_67, 2, 17);  select_177 = None
        select_178: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_65, 2, 16)
        select_scatter_68: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_65, select_178, 2, 16);  select_scatter_65 = select_178 = None
        select_179: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_68, 2, 17)
        neg_36: "f32[200, 200]" = torch.ops.aten.neg.default(div_24);  div_24 = None
        select_180: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_68, 2, 16)
        mul_51: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_36, select_180);  neg_36 = select_180 = None
        add_43: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_179, mul_51);  select_179 = mul_51 = None
        convert_element_type_16: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None
        select_scatter_69: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_68, convert_element_type_16, 2, 17);  select_scatter_68 = convert_element_type_16 = None
        select_181: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_69, 2, 17);  select_181 = None
        select_182: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_67, 2, 17)
        select_scatter_70: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_67, select_182, 2, 17);  select_scatter_67 = select_182 = None
        select_183: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_70, 2, 18)
        select_184: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 18)
        select_185: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_70, 2, 17)
        div_25: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_184, select_185);  select_184 = select_185 = None
        neg_37: "f32[200, 200]" = torch.ops.aten.neg.default(div_25)
        select_186: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 17)
        mul_52: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_37, select_186);  neg_37 = select_186 = None
        add_44: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_183, mul_52);  select_183 = mul_52 = None
        select_scatter_71: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_70, add_44, 2, 18);  select_scatter_70 = add_44 = None
        select_187: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_71, 2, 18);  select_187 = None
        select_188: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_69, 2, 17)
        select_scatter_72: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_69, select_188, 2, 17);  select_scatter_69 = select_188 = None
        select_189: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_72, 2, 18)
        neg_38: "f32[200, 200]" = torch.ops.aten.neg.default(div_25);  div_25 = None
        select_190: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_72, 2, 17)
        mul_53: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_38, select_190);  neg_38 = select_190 = None
        add_45: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_189, mul_53);  select_189 = mul_53 = None
        convert_element_type_17: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16);  add_45 = None
        select_scatter_73: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_72, convert_element_type_17, 2, 18);  select_scatter_72 = convert_element_type_17 = None
        select_191: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_73, 2, 18);  select_191 = None
        select_192: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_71, 2, 18)
        select_scatter_74: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_71, select_192, 2, 18);  select_scatter_71 = select_192 = None
        select_193: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_74, 2, 19)
        select_194: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 19)
        select_195: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_74, 2, 18)
        div_26: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_194, select_195);  select_194 = select_195 = None
        neg_39: "f32[200, 200]" = torch.ops.aten.neg.default(div_26)
        select_196: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 18)
        mul_54: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_39, select_196);  neg_39 = select_196 = None
        add_46: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_193, mul_54);  select_193 = mul_54 = None
        select_scatter_75: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_74, add_46, 2, 19);  select_scatter_74 = add_46 = None
        select_197: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_75, 2, 19);  select_197 = None
        select_198: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_73, 2, 18)
        select_scatter_76: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_73, select_198, 2, 18);  select_scatter_73 = select_198 = None
        select_199: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_76, 2, 19)
        neg_40: "f32[200, 200]" = torch.ops.aten.neg.default(div_26);  div_26 = None
        select_200: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_76, 2, 18)
        mul_55: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_40, select_200);  neg_40 = select_200 = None
        add_47: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_199, mul_55);  select_199 = mul_55 = None
        convert_element_type_18: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None
        select_scatter_77: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_76, convert_element_type_18, 2, 19);  select_scatter_76 = convert_element_type_18 = None
        select_201: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_77, 2, 19);  select_201 = None
        select_202: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_75, 2, 19)
        select_scatter_78: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_75, select_202, 2, 19);  select_scatter_75 = select_202 = None
        select_203: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_78, 2, 20)
        select_204: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 20)
        select_205: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_78, 2, 19)
        div_27: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_204, select_205);  select_204 = select_205 = None
        neg_41: "f32[200, 200]" = torch.ops.aten.neg.default(div_27)
        select_206: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 19)
        mul_56: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_41, select_206);  neg_41 = select_206 = None
        add_48: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_203, mul_56);  select_203 = mul_56 = None
        select_scatter_79: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_78, add_48, 2, 20);  select_scatter_78 = add_48 = None
        select_207: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_79, 2, 20);  select_207 = None
        select_208: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_77, 2, 19)
        select_scatter_80: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_77, select_208, 2, 19);  select_scatter_77 = select_208 = None
        select_209: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_80, 2, 20)
        neg_42: "f32[200, 200]" = torch.ops.aten.neg.default(div_27);  div_27 = None
        select_210: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_80, 2, 19)
        mul_57: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_42, select_210);  neg_42 = select_210 = None
        add_49: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_209, mul_57);  select_209 = mul_57 = None
        convert_element_type_19: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16);  add_49 = None
        select_scatter_81: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_80, convert_element_type_19, 2, 20);  select_scatter_80 = convert_element_type_19 = None
        select_211: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_81, 2, 20);  select_211 = None
        select_212: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_79, 2, 20)
        select_scatter_82: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_79, select_212, 2, 20);  select_scatter_79 = select_212 = None
        select_213: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_82, 2, 21)
        select_214: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 21)
        select_215: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_82, 2, 20)
        div_28: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_214, select_215);  select_214 = select_215 = None
        neg_43: "f32[200, 200]" = torch.ops.aten.neg.default(div_28)
        select_216: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 20)
        mul_58: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_43, select_216);  neg_43 = select_216 = None
        add_50: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_213, mul_58);  select_213 = mul_58 = None
        select_scatter_83: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_82, add_50, 2, 21);  select_scatter_82 = add_50 = None
        select_217: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_83, 2, 21);  select_217 = None
        select_218: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_81, 2, 20)
        select_scatter_84: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_81, select_218, 2, 20);  select_scatter_81 = select_218 = None
        select_219: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_84, 2, 21)
        neg_44: "f32[200, 200]" = torch.ops.aten.neg.default(div_28);  div_28 = None
        select_220: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_84, 2, 20)
        mul_59: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_44, select_220);  neg_44 = select_220 = None
        add_51: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_219, mul_59);  select_219 = mul_59 = None
        convert_element_type_20: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16);  add_51 = None
        select_scatter_85: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_84, convert_element_type_20, 2, 21);  select_scatter_84 = convert_element_type_20 = None
        select_221: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_85, 2, 21);  select_221 = None
        select_222: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_83, 2, 21)
        select_scatter_86: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_83, select_222, 2, 21);  select_scatter_83 = select_222 = None
        select_223: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_86, 2, 22)
        select_224: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 22)
        select_225: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_86, 2, 21)
        div_29: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_224, select_225);  select_224 = select_225 = None
        neg_45: "f32[200, 200]" = torch.ops.aten.neg.default(div_29)
        select_226: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 21)
        mul_60: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_45, select_226);  neg_45 = select_226 = None
        add_52: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_223, mul_60);  select_223 = mul_60 = None
        select_scatter_87: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_86, add_52, 2, 22);  select_scatter_86 = add_52 = None
        select_227: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_87, 2, 22);  select_227 = None
        select_228: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_85, 2, 21)
        select_scatter_88: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_85, select_228, 2, 21);  select_scatter_85 = select_228 = None
        select_229: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_88, 2, 22)
        neg_46: "f32[200, 200]" = torch.ops.aten.neg.default(div_29);  div_29 = None
        select_230: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_88, 2, 21)
        mul_61: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_46, select_230);  neg_46 = select_230 = None
        add_53: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_229, mul_61);  select_229 = mul_61 = None
        convert_element_type_21: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16);  add_53 = None
        select_scatter_89: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_88, convert_element_type_21, 2, 22);  select_scatter_88 = convert_element_type_21 = None
        select_231: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_89, 2, 22);  select_231 = None
        select_232: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_87, 2, 22)
        select_scatter_90: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_87, select_232, 2, 22);  select_scatter_87 = select_232 = None
        select_233: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_90, 2, 23)
        select_234: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 23)
        select_235: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_90, 2, 22)
        div_30: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_234, select_235);  select_234 = select_235 = None
        neg_47: "f32[200, 200]" = torch.ops.aten.neg.default(div_30)
        select_236: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 22)
        mul_62: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_47, select_236);  neg_47 = select_236 = None
        add_54: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_233, mul_62);  select_233 = mul_62 = None
        select_scatter_91: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_90, add_54, 2, 23);  select_scatter_90 = add_54 = None
        select_237: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_91, 2, 23);  select_237 = None
        select_238: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_89, 2, 22)
        select_scatter_92: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_89, select_238, 2, 22);  select_scatter_89 = select_238 = None
        select_239: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_92, 2, 23)
        neg_48: "f32[200, 200]" = torch.ops.aten.neg.default(div_30);  div_30 = None
        select_240: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_92, 2, 22)
        mul_63: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_48, select_240);  neg_48 = select_240 = None
        add_55: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_239, mul_63);  select_239 = mul_63 = None
        convert_element_type_22: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None
        select_scatter_93: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_92, convert_element_type_22, 2, 23);  select_scatter_92 = convert_element_type_22 = None
        select_241: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_93, 2, 23);  select_241 = None
        select_242: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_91, 2, 23)
        select_scatter_94: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_91, select_242, 2, 23);  select_scatter_91 = select_242 = None
        select_243: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_94, 2, 24)
        select_244: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 24)
        select_245: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_94, 2, 23)
        div_31: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_244, select_245);  select_244 = select_245 = None
        neg_49: "f32[200, 200]" = torch.ops.aten.neg.default(div_31)
        select_246: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 23)
        mul_64: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_49, select_246);  neg_49 = select_246 = None
        add_56: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_243, mul_64);  select_243 = mul_64 = None
        select_scatter_95: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_94, add_56, 2, 24);  select_scatter_94 = add_56 = None
        select_247: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_95, 2, 24);  select_247 = None
        select_248: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_93, 2, 23)
        select_scatter_96: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_93, select_248, 2, 23);  select_scatter_93 = select_248 = None
        select_249: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_96, 2, 24)
        neg_50: "f32[200, 200]" = torch.ops.aten.neg.default(div_31);  div_31 = None
        select_250: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_96, 2, 23)
        mul_65: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_50, select_250);  neg_50 = select_250 = None
        add_57: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_249, mul_65);  select_249 = mul_65 = None
        convert_element_type_23: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16);  add_57 = None
        select_scatter_97: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_96, convert_element_type_23, 2, 24);  select_scatter_96 = convert_element_type_23 = None
        select_251: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_97, 2, 24);  select_251 = None
        select_252: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_95, 2, 24)
        select_scatter_98: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_95, select_252, 2, 24);  select_scatter_95 = select_252 = None
        select_253: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_98, 2, 25)
        select_254: "bf16[200, 200]" = torch.ops.aten.select.int(mul_15, 2, 25);  mul_15 = None
        select_255: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_98, 2, 24)
        div_32: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_254, select_255);  select_254 = select_255 = None
        neg_51: "f32[200, 200]" = torch.ops.aten.neg.default(div_32)
        select_256: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 24)
        mul_66: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_51, select_256);  neg_51 = select_256 = None
        add_58: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_253, mul_66);  select_253 = mul_66 = None
        select_scatter_99: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_98, add_58, 2, 25);  select_scatter_98 = add_58 = None
        select_257: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_99, 2, 25);  select_257 = None
        select_258: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_97, 2, 24)
        select_scatter_100: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_97, select_258, 2, 24);  select_scatter_97 = select_258 = None
        select_259: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_100, 2, 25)
        neg_52: "f32[200, 200]" = torch.ops.aten.neg.default(div_32);  div_32 = None
        select_260: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_100, 2, 24)
        mul_67: "f32[200, 200]" = torch.ops.aten.mul.Tensor(neg_52, select_260);  neg_52 = select_260 = None
        add_59: "f32[200, 200]" = torch.ops.aten.add.Tensor(select_259, mul_67);  select_259 = mul_67 = None
        convert_element_type_24: "bf16[200, 200]" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None
        select_scatter_101: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_100, convert_element_type_24, 2, 25);  select_scatter_100 = convert_element_type_24 = None
        select_261: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_101, 2, 25);  select_261 = None
        full_6: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_4, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_4 = None
        slice_41: "bf16[203, 204, 26]" = torch.ops.aten.slice.Tensor(full_6, 0, 0, -1);  slice_41 = None
        slice_42: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 2, -2)
        slice_43: "bf16[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_42, 1, 2, -2)
        slice_44: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 2, -2)
        slice_45: "bf16[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_44, 1, 2, -2);  slice_44 = None
        select_262: "bf16[200, 200, 26]" = torch.ops.aten.select.int(slice_45, 3, 1);  slice_45 = None
        empty: "bf16[200, 200, 26]" = torch.ops.aten.empty.memory_format(_shape_param_5, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_5 = None
        select_263: "bf16[200, 200]" = torch.ops.aten.select.int(empty, 2, -1)
        select_264: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_101, 2, 25)
        select_scatter_102: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_101, select_264, 2, 25);  select_scatter_101 = select_264 = None
        select_265: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, -1)
        select_266: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_99, 2, 25)
        select_scatter_103: "f32[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_99, select_266, 2, 25);  select_scatter_99 = select_266 = None
        select_267: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, -1)
        div_33: "f32[200, 200]" = torch.ops.aten.div.Tensor(select_265, select_267);  select_265 = select_267 = None
        copy_6: "bf16[200, 200]" = torch.ops.aten.copy.default(select_263, div_33);  select_263 = div_33 = None
        select_scatter_104: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(empty, copy_6, 2, -1);  empty = copy_6 = None
        select_268: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_104, 2, 24)
        select_269: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 24)
        select_270: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 24)
        select_271: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_104, 2, 25)
        mul_68: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_270, select_271);  select_270 = select_271 = None
        sub_1: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_269, mul_68);  select_269 = mul_68 = None
        select_272: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 24)
        div_34: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_1, select_272);  sub_1 = select_272 = None
        copy_7: "bf16[200, 200]" = torch.ops.aten.copy.default(select_268, div_34);  select_268 = div_34 = None
        select_scatter_105: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_104, copy_7, 2, 24);  select_scatter_104 = copy_7 = None
        select_273: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_105, 2, 23)
        select_274: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 23)
        select_275: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 23)
        select_276: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_105, 2, 24)
        mul_69: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_275, select_276);  select_275 = select_276 = None
        sub_2: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_274, mul_69);  select_274 = mul_69 = None
        select_277: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 23)
        div_35: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_2, select_277);  sub_2 = select_277 = None
        copy_8: "bf16[200, 200]" = torch.ops.aten.copy.default(select_273, div_35);  select_273 = div_35 = None
        select_scatter_106: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_105, copy_8, 2, 23);  select_scatter_105 = copy_8 = None
        select_278: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_106, 2, 22)
        select_279: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 22)
        select_280: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 22)
        select_281: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_106, 2, 23)
        mul_70: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_280, select_281);  select_280 = select_281 = None
        sub_3: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_279, mul_70);  select_279 = mul_70 = None
        select_282: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 22)
        div_36: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_3, select_282);  sub_3 = select_282 = None
        copy_9: "bf16[200, 200]" = torch.ops.aten.copy.default(select_278, div_36);  select_278 = div_36 = None
        select_scatter_107: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_106, copy_9, 2, 22);  select_scatter_106 = copy_9 = None
        select_283: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_107, 2, 21)
        select_284: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 21)
        select_285: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 21)
        select_286: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_107, 2, 22)
        mul_71: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_285, select_286);  select_285 = select_286 = None
        sub_4: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_284, mul_71);  select_284 = mul_71 = None
        select_287: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 21)
        div_37: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_4, select_287);  sub_4 = select_287 = None
        copy_10: "bf16[200, 200]" = torch.ops.aten.copy.default(select_283, div_37);  select_283 = div_37 = None
        select_scatter_108: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_107, copy_10, 2, 21);  select_scatter_107 = copy_10 = None
        select_288: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_108, 2, 20)
        select_289: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 20)
        select_290: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 20)
        select_291: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_108, 2, 21)
        mul_72: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_290, select_291);  select_290 = select_291 = None
        sub_5: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_289, mul_72);  select_289 = mul_72 = None
        select_292: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 20)
        div_38: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_5, select_292);  sub_5 = select_292 = None
        copy_11: "bf16[200, 200]" = torch.ops.aten.copy.default(select_288, div_38);  select_288 = div_38 = None
        select_scatter_109: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_108, copy_11, 2, 20);  select_scatter_108 = copy_11 = None
        select_293: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_109, 2, 19)
        select_294: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 19)
        select_295: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 19)
        select_296: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_109, 2, 20)
        mul_73: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_295, select_296);  select_295 = select_296 = None
        sub_6: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_294, mul_73);  select_294 = mul_73 = None
        select_297: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 19)
        div_39: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_6, select_297);  sub_6 = select_297 = None
        copy_12: "bf16[200, 200]" = torch.ops.aten.copy.default(select_293, div_39);  select_293 = div_39 = None
        select_scatter_110: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_109, copy_12, 2, 19);  select_scatter_109 = copy_12 = None
        select_298: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_110, 2, 18)
        select_299: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 18)
        select_300: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 18)
        select_301: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_110, 2, 19)
        mul_74: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_300, select_301);  select_300 = select_301 = None
        sub_7: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_299, mul_74);  select_299 = mul_74 = None
        select_302: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 18)
        div_40: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_7, select_302);  sub_7 = select_302 = None
        copy_13: "bf16[200, 200]" = torch.ops.aten.copy.default(select_298, div_40);  select_298 = div_40 = None
        select_scatter_111: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_110, copy_13, 2, 18);  select_scatter_110 = copy_13 = None
        select_303: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_111, 2, 17)
        select_304: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 17)
        select_305: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 17)
        select_306: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_111, 2, 18)
        mul_75: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_305, select_306);  select_305 = select_306 = None
        sub_8: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_304, mul_75);  select_304 = mul_75 = None
        select_307: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 17)
        div_41: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_8, select_307);  sub_8 = select_307 = None
        copy_14: "bf16[200, 200]" = torch.ops.aten.copy.default(select_303, div_41);  select_303 = div_41 = None
        select_scatter_112: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_111, copy_14, 2, 17);  select_scatter_111 = copy_14 = None
        select_308: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_112, 2, 16)
        select_309: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 16)
        select_310: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 16)
        select_311: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_112, 2, 17)
        mul_76: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_310, select_311);  select_310 = select_311 = None
        sub_9: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_309, mul_76);  select_309 = mul_76 = None
        select_312: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 16)
        div_42: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_9, select_312);  sub_9 = select_312 = None
        copy_15: "bf16[200, 200]" = torch.ops.aten.copy.default(select_308, div_42);  select_308 = div_42 = None
        select_scatter_113: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_112, copy_15, 2, 16);  select_scatter_112 = copy_15 = None
        select_313: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_113, 2, 15)
        select_314: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 15)
        select_315: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 15)
        select_316: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_113, 2, 16)
        mul_77: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_315, select_316);  select_315 = select_316 = None
        sub_10: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_314, mul_77);  select_314 = mul_77 = None
        select_317: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 15)
        div_43: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_10, select_317);  sub_10 = select_317 = None
        copy_16: "bf16[200, 200]" = torch.ops.aten.copy.default(select_313, div_43);  select_313 = div_43 = None
        select_scatter_114: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_113, copy_16, 2, 15);  select_scatter_113 = copy_16 = None
        select_318: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_114, 2, 14)
        select_319: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 14)
        select_320: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 14)
        select_321: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_114, 2, 15)
        mul_78: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_320, select_321);  select_320 = select_321 = None
        sub_11: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_319, mul_78);  select_319 = mul_78 = None
        select_322: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 14)
        div_44: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_11, select_322);  sub_11 = select_322 = None
        copy_17: "bf16[200, 200]" = torch.ops.aten.copy.default(select_318, div_44);  select_318 = div_44 = None
        select_scatter_115: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_114, copy_17, 2, 14);  select_scatter_114 = copy_17 = None
        select_323: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_115, 2, 13)
        select_324: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 13)
        select_325: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 13)
        select_326: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_115, 2, 14)
        mul_79: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_325, select_326);  select_325 = select_326 = None
        sub_12: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_324, mul_79);  select_324 = mul_79 = None
        select_327: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 13)
        div_45: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_12, select_327);  sub_12 = select_327 = None
        copy_18: "bf16[200, 200]" = torch.ops.aten.copy.default(select_323, div_45);  select_323 = div_45 = None
        select_scatter_116: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_115, copy_18, 2, 13);  select_scatter_115 = copy_18 = None
        select_328: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_116, 2, 12)
        select_329: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 12)
        select_330: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 12)
        select_331: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_116, 2, 13)
        mul_80: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_330, select_331);  select_330 = select_331 = None
        sub_13: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_329, mul_80);  select_329 = mul_80 = None
        select_332: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 12)
        div_46: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_13, select_332);  sub_13 = select_332 = None
        copy_19: "bf16[200, 200]" = torch.ops.aten.copy.default(select_328, div_46);  select_328 = div_46 = None
        select_scatter_117: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_116, copy_19, 2, 12);  select_scatter_116 = copy_19 = None
        select_333: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_117, 2, 11)
        select_334: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 11)
        select_335: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 11)
        select_336: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_117, 2, 12)
        mul_81: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_335, select_336);  select_335 = select_336 = None
        sub_14: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_334, mul_81);  select_334 = mul_81 = None
        select_337: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 11)
        div_47: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_14, select_337);  sub_14 = select_337 = None
        copy_20: "bf16[200, 200]" = torch.ops.aten.copy.default(select_333, div_47);  select_333 = div_47 = None
        select_scatter_118: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_117, copy_20, 2, 11);  select_scatter_117 = copy_20 = None
        select_338: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_118, 2, 10)
        select_339: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 10)
        select_340: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 10)
        select_341: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_118, 2, 11)
        mul_82: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_340, select_341);  select_340 = select_341 = None
        sub_15: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_339, mul_82);  select_339 = mul_82 = None
        select_342: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 10)
        div_48: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_15, select_342);  sub_15 = select_342 = None
        copy_21: "bf16[200, 200]" = torch.ops.aten.copy.default(select_338, div_48);  select_338 = div_48 = None
        select_scatter_119: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_118, copy_21, 2, 10);  select_scatter_118 = copy_21 = None
        select_343: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_119, 2, 9)
        select_344: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 9)
        select_345: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 9)
        select_346: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_119, 2, 10)
        mul_83: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_345, select_346);  select_345 = select_346 = None
        sub_16: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_344, mul_83);  select_344 = mul_83 = None
        select_347: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 9)
        div_49: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_16, select_347);  sub_16 = select_347 = None
        copy_22: "bf16[200, 200]" = torch.ops.aten.copy.default(select_343, div_49);  select_343 = div_49 = None
        select_scatter_120: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_119, copy_22, 2, 9);  select_scatter_119 = copy_22 = None
        select_348: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_120, 2, 8)
        select_349: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 8)
        select_350: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 8)
        select_351: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_120, 2, 9)
        mul_84: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_350, select_351);  select_350 = select_351 = None
        sub_17: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_349, mul_84);  select_349 = mul_84 = None
        select_352: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 8)
        div_50: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_17, select_352);  sub_17 = select_352 = None
        copy_23: "bf16[200, 200]" = torch.ops.aten.copy.default(select_348, div_50);  select_348 = div_50 = None
        select_scatter_121: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_120, copy_23, 2, 8);  select_scatter_120 = copy_23 = None
        select_353: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_121, 2, 7)
        select_354: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 7)
        select_355: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 7)
        select_356: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_121, 2, 8)
        mul_85: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_355, select_356);  select_355 = select_356 = None
        sub_18: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_354, mul_85);  select_354 = mul_85 = None
        select_357: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 7)
        div_51: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_18, select_357);  sub_18 = select_357 = None
        copy_24: "bf16[200, 200]" = torch.ops.aten.copy.default(select_353, div_51);  select_353 = div_51 = None
        select_scatter_122: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_121, copy_24, 2, 7);  select_scatter_121 = copy_24 = None
        select_358: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_122, 2, 6)
        select_359: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 6)
        select_360: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 6)
        select_361: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_122, 2, 7)
        mul_86: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_360, select_361);  select_360 = select_361 = None
        sub_19: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_359, mul_86);  select_359 = mul_86 = None
        select_362: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 6)
        div_52: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_19, select_362);  sub_19 = select_362 = None
        copy_25: "bf16[200, 200]" = torch.ops.aten.copy.default(select_358, div_52);  select_358 = div_52 = None
        select_scatter_123: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_122, copy_25, 2, 6);  select_scatter_122 = copy_25 = None
        select_363: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_123, 2, 5)
        select_364: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 5)
        select_365: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 5)
        select_366: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_123, 2, 6)
        mul_87: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_365, select_366);  select_365 = select_366 = None
        sub_20: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_364, mul_87);  select_364 = mul_87 = None
        select_367: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 5)
        div_53: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_20, select_367);  sub_20 = select_367 = None
        copy_26: "bf16[200, 200]" = torch.ops.aten.copy.default(select_363, div_53);  select_363 = div_53 = None
        select_scatter_124: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_123, copy_26, 2, 5);  select_scatter_123 = copy_26 = None
        select_368: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_124, 2, 4)
        select_369: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 4)
        select_370: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 4)
        select_371: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_124, 2, 5)
        mul_88: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_370, select_371);  select_370 = select_371 = None
        sub_21: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_369, mul_88);  select_369 = mul_88 = None
        select_372: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 4)
        div_54: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_21, select_372);  sub_21 = select_372 = None
        copy_27: "bf16[200, 200]" = torch.ops.aten.copy.default(select_368, div_54);  select_368 = div_54 = None
        select_scatter_125: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_124, copy_27, 2, 4);  select_scatter_124 = copy_27 = None
        select_373: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_125, 2, 3)
        select_374: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 3)
        select_375: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 3)
        select_376: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_125, 2, 4)
        mul_89: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_375, select_376);  select_375 = select_376 = None
        sub_22: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_374, mul_89);  select_374 = mul_89 = None
        select_377: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 3)
        div_55: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_22, select_377);  sub_22 = select_377 = None
        copy_28: "bf16[200, 200]" = torch.ops.aten.copy.default(select_373, div_55);  select_373 = div_55 = None
        select_scatter_126: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_125, copy_28, 2, 3);  select_scatter_125 = copy_28 = None
        select_378: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_126, 2, 2)
        select_379: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 2)
        select_380: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 2)
        select_381: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_126, 2, 3)
        mul_90: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_380, select_381);  select_380 = select_381 = None
        sub_23: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_379, mul_90);  select_379 = mul_90 = None
        select_382: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 2)
        div_56: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_23, select_382);  sub_23 = select_382 = None
        copy_29: "bf16[200, 200]" = torch.ops.aten.copy.default(select_378, div_56);  select_378 = div_56 = None
        select_scatter_127: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_126, copy_29, 2, 2);  select_scatter_126 = copy_29 = None
        select_383: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_127, 2, 1)
        select_384: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 1)
        select_385: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 1)
        select_386: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_127, 2, 2)
        mul_91: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_385, select_386);  select_385 = select_386 = None
        sub_24: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_384, mul_91);  select_384 = mul_91 = None
        select_387: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 1)
        div_57: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_24, select_387);  sub_24 = select_387 = None
        copy_30: "bf16[200, 200]" = torch.ops.aten.copy.default(select_383, div_57);  select_383 = div_57 = None
        select_scatter_128: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_127, copy_30, 2, 1);  select_scatter_127 = copy_30 = None
        select_388: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_128, 2, 0)
        select_389: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_102, 2, 0);  select_scatter_102 = None
        select_390: "bf16[200, 200]" = torch.ops.aten.select.int(mul_16, 2, 0);  mul_16 = None
        select_391: "bf16[200, 200]" = torch.ops.aten.select.int(select_scatter_128, 2, 1)
        mul_92: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(select_390, select_391);  select_390 = select_391 = None
        sub_25: "bf16[200, 200]" = torch.ops.aten.sub.Tensor(select_389, mul_92);  select_389 = mul_92 = None
        select_392: "f32[200, 200]" = torch.ops.aten.select.int(select_scatter_103, 2, 0);  select_scatter_103 = None
        div_58: "f32[200, 200]" = torch.ops.aten.div.Tensor(sub_25, select_392);  sub_25 = select_392 = None
        copy_31: "bf16[200, 200]" = torch.ops.aten.copy.default(select_388, div_58);  select_388 = div_58 = None
        select_scatter_129: "bf16[200, 200, 26]" = torch.ops.aten.select_scatter.default(select_scatter_128, copy_31, 2, 0);  select_scatter_128 = copy_31 = None
        slice_46: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg0_1, 0, 2, -2)
        slice_47: "bf16[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_46, 1, 2, -2);  slice_46 = None
        select_393: "bf16[200, 200, 26]" = torch.ops.aten.select.int(slice_47, 3, 1);  slice_47 = None
        where_2: "bf16[200, 200, 26]" = torch.ops.aten.where.self(bitwise_and_1, select_scatter_129, select_393);  bitwise_and_1 = select_scatter_129 = select_393 = None
        copy_32: "bf16[200, 200, 26]" = torch.ops.aten.copy.default(select_262, where_2);  select_262 = where_2 = None
        select_scatter_130: "bf16[200, 200, 26, 3]" = torch.ops.aten.select_scatter.default(slice_43, copy_32, 3, 1);  slice_43 = copy_32 = None
        slice_scatter_4: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_42, select_scatter_130, 1, 2, -2);  slice_42 = select_scatter_130 = None
        slice_scatter_5: "bf16[204, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(arg0_1, slice_scatter_4, 0, 2, -2);  arg0_1 = slice_scatter_4 = None
        slice_48: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2)
        slice_49: "bf16[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_48, 1, 2, -2)
        select_394: "bf16[200, 200, 3]" = torch.ops.aten.select.int(slice_49, 2, -1)
        slice_50: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2)
        slice_51: "bf16[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_50, 1, 2, -2);  slice_50 = None
        select_395: "bf16[200, 200, 3]" = torch.ops.aten.select.int(slice_51, 2, -1);  slice_51 = None
        select_396: "bf16[200, 200]" = torch.ops.aten.select.int(select_395, 2, 1);  select_395 = None
        full_7: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_52: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2)
        slice_53: "bf16[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_52, 1, 2, -2);  slice_52 = None
        select_397: "bf16[200, 200, 3]" = torch.ops.aten.select.int(slice_53, 2, -1);  slice_53 = None
        select_398: "bf16[200, 200]" = torch.ops.aten.select.int(select_397, 2, 1);  select_397 = None
        maximum_1: "f32[200, 200]" = torch.ops.aten.maximum.default(full_7, select_398);  full_7 = select_398 = None
        copy_33: "bf16[200, 200]" = torch.ops.aten.copy.default(select_396, maximum_1);  select_396 = maximum_1 = None
        select_scatter_131: "bf16[200, 200, 3]" = torch.ops.aten.select_scatter.default(select_394, copy_33, 2, 1);  select_394 = copy_33 = None
        select_scatter_132: "bf16[200, 200, 26, 3]" = torch.ops.aten.select_scatter.default(slice_49, select_scatter_131, 2, -1);  slice_49 = select_scatter_131 = None
        slice_scatter_6: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_48, select_scatter_132, 1, 2, -2);  slice_48 = select_scatter_132 = None
        slice_scatter_7: "bf16[204, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_5, slice_scatter_6, 0, 2, -2);  slice_scatter_6 = None
        slice_54: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 2, -2)
        slice_55: "bf16[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_54, 1, 2, -2)
        slice_56: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 2, -2)
        slice_57: "bf16[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_56, 1, 2, -2);  slice_56 = None
        select_399: "bf16[200, 200, 26]" = torch.ops.aten.select.int(slice_57, 3, 1);  slice_57 = None
        slice_58: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 2, -2)
        slice_59: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_58, 1, 2, -2);  slice_58 = None
        mul_93: "bf16[200, 200, 26]" = torch.ops.aten.mul.Tensor(slice_59, 1);  slice_59 = None
        slice_60: "bf16[203, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 1, 9223372036854775807)
        select_400: "bf16[203, 204, 26]" = torch.ops.aten.select.int(slice_60, 3, 0);  slice_60 = None
        slice_61: "bf16[203, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 0, 0, -1)
        select_401: "bf16[203, 204, 26]" = torch.ops.aten.select.int(slice_61, 3, 0);  slice_61 = None
        sub_26: "bf16[203, 204, 26]" = torch.ops.aten.sub.Tensor(select_400, select_401);  select_400 = select_401 = None
        mul_94: "bf16[203, 204, 26]" = torch.ops.aten.mul.Tensor(sub_26, 2000.0);  sub_26 = None
        unsqueeze_17: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(arg9_1, 0)
        unsqueeze_18: "bf16[1, 204, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 2);  unsqueeze_17 = None
        slice_62: "bf16[203]" = torch.ops.aten.slice.Tensor(arg10_1, 0, 0, -1);  arg10_1 = None
        unsqueeze_19: "bf16[203, 1]" = torch.ops.aten.unsqueeze.default(slice_62, 1);  slice_62 = None
        unsqueeze_20: "bf16[203, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_19, 2);  unsqueeze_19 = None
        mul_95: "bf16[203, 204, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_18, unsqueeze_20);  unsqueeze_18 = unsqueeze_20 = None
        div_59: "bf16[203, 204, 26]" = torch.ops.aten.div.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
        slice_63: "bf16[203, 204, 26]" = torch.ops.aten.slice.Tensor(arg11_1, 0, 0, -1);  arg11_1 = None
        mul_96: "bf16[203, 204, 26]" = torch.ops.aten.mul.Tensor(div_59, slice_63);  div_59 = slice_63 = None
        slice_scatter_8: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_6, mul_96, 0, 0, -1);  full_6 = mul_96 = None
        select_402: "bf16[204, 26]" = torch.ops.aten.select.int(slice_scatter_8, 0, -1)
        full_8: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_34: "bf16[204, 26]" = torch.ops.aten.copy.default(select_402, full_8);  select_402 = full_8 = None
        select_scatter_133: "bf16[204, 204, 26]" = torch.ops.aten.select_scatter.default(slice_scatter_8, copy_34, 0, -1);  slice_scatter_8 = copy_34 = None
        slice_64: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_scatter_133, 0, 2, -2)
        slice_65: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_64, 1, 2, -2);  slice_64 = None
        slice_66: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_scatter_133, 0, 1, -3)
        slice_67: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_66, 1, 2, -2);  slice_66 = None
        sub_27: "bf16[200, 200, 26]" = torch.ops.aten.sub.Tensor(slice_65, slice_67);  slice_65 = slice_67 = None
        unsqueeze_21: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(arg9_1, 0)
        slice_68: "bf16[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_21, 1, 2, -2);  unsqueeze_21 = None
        unsqueeze_22: "bf16[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_68, 2);  slice_68 = None
        slice_69: "bf16[200]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2)
        unsqueeze_23: "bf16[200, 1]" = torch.ops.aten.unsqueeze.default(slice_69, 1);  slice_69 = None
        unsqueeze_24: "bf16[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, 2);  unsqueeze_23 = None
        mul_97: "bf16[200, 200, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_22, unsqueeze_24);  unsqueeze_22 = unsqueeze_24 = None
        div_60: "bf16[200, 200, 26]" = torch.ops.aten.div.Tensor(sub_27, mul_97);  sub_27 = mul_97 = None
        full_9: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_6, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_6 = None
        slice_70: "bf16[204, 203, 26]" = torch.ops.aten.slice.Tensor(full_9, 1, 0, -1)
        slice_71: "bf16[204, 203, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 1, 1, 9223372036854775807)
        select_403: "bf16[204, 203, 26]" = torch.ops.aten.select.int(slice_71, 3, 0);  slice_71 = None
        slice_72: "bf16[204, 203, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 1, 0, -1)
        select_404: "bf16[204, 203, 26]" = torch.ops.aten.select.int(slice_72, 3, 0);  slice_72 = None
        sub_28: "bf16[204, 203, 26]" = torch.ops.aten.sub.Tensor(select_403, select_404);  select_403 = select_404 = None
        mul_98: "bf16[204, 203, 26]" = torch.ops.aten.mul.Tensor(sub_28, 2000.0);  sub_28 = None
        unsqueeze_25: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(arg13_1, 0);  arg13_1 = None
        slice_73: "bf16[1, 203]" = torch.ops.aten.slice.Tensor(unsqueeze_25, 1, 0, -1);  unsqueeze_25 = None
        unsqueeze_26: "bf16[1, 203, 1]" = torch.ops.aten.unsqueeze.default(slice_73, 2);  slice_73 = None
        div_61: "bf16[204, 203, 26]" = torch.ops.aten.div.Tensor(mul_98, unsqueeze_26);  mul_98 = unsqueeze_26 = None
        slice_74: "bf16[204, 203, 26]" = torch.ops.aten.slice.Tensor(arg14_1, 1, 0, -1);  arg14_1 = None
        mul_99: "bf16[204, 203, 26]" = torch.ops.aten.mul.Tensor(div_61, slice_74);  div_61 = slice_74 = None
        unsqueeze_27: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(arg15_1, 0)
        slice_75: "bf16[1, 203]" = torch.ops.aten.slice.Tensor(unsqueeze_27, 1, 0, -1);  unsqueeze_27 = None
        unsqueeze_28: "bf16[1, 203, 1]" = torch.ops.aten.unsqueeze.default(slice_75, 2);  slice_75 = None
        mul_100: "bf16[204, 203, 26]" = torch.ops.aten.mul.Tensor(mul_99, unsqueeze_28);  mul_99 = unsqueeze_28 = None
        copy_35: "bf16[204, 203, 26]" = torch.ops.aten.copy.default(slice_70, mul_100);  slice_70 = mul_100 = None
        slice_scatter_9: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_9, copy_35, 1, 0, -1);  full_9 = copy_35 = None
        select_405: "bf16[204, 26]" = torch.ops.aten.select.int(slice_scatter_9, 1, -1)
        full_10: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_36: "bf16[204, 26]" = torch.ops.aten.copy.default(select_405, full_10);  select_405 = full_10 = None
        select_scatter_134: "bf16[204, 204, 26]" = torch.ops.aten.select_scatter.default(slice_scatter_9, copy_36, 1, -1);  slice_scatter_9 = copy_36 = None
        slice_76: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_scatter_134, 0, 2, -2)
        slice_77: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_76, 1, 2, -2);  slice_76 = None
        slice_78: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_scatter_134, 0, 2, -2)
        slice_79: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_78, 1, 1, -3);  slice_78 = None
        sub_29: "bf16[200, 200, 26]" = torch.ops.aten.sub.Tensor(slice_77, slice_79);  slice_77 = slice_79 = None
        unsqueeze_29: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(arg9_1, 0)
        slice_80: "bf16[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_29, 1, 2, -2);  unsqueeze_29 = None
        unsqueeze_30: "bf16[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_80, 2);  slice_80 = None
        unsqueeze_31: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(arg16_1, 0)
        slice_81: "bf16[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_31, 1, 2, -2);  unsqueeze_31 = None
        unsqueeze_32: "bf16[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_81, 2);  slice_81 = None
        mul_101: "bf16[1, 200, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_30, unsqueeze_32);  unsqueeze_30 = unsqueeze_32 = None
        div_62: "bf16[200, 200, 26]" = torch.ops.aten.div.Tensor(sub_29, mul_101);  sub_29 = mul_101 = None
        add_60: "bf16[200, 200, 26]" = torch.ops.aten.add.Tensor(div_60, div_62);  div_60 = div_62 = None
        mul_102: "bf16[200, 200, 26]" = torch.ops.aten.mul.Tensor(mul_93, add_60);  mul_93 = add_60 = None
        add_61: "bf16[200, 200, 26]" = torch.ops.aten.add.Tensor(select_399, mul_102);  select_399 = mul_102 = None
        select_scatter_135: "bf16[200, 200, 26, 3]" = torch.ops.aten.select_scatter.default(slice_55, add_61, 3, 1);  slice_55 = add_61 = None
        slice_scatter_10: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_54, select_scatter_135, 1, 2, -2);  slice_54 = select_scatter_135 = None
        slice_scatter_11: "bf16[204, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_7, slice_scatter_10, 0, 2, -2);  slice_scatter_7 = slice_scatter_10 = None
        slice_82: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_11, 0, 2, -2)
        slice_83: "bf16[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_82, 1, 2, -2);  slice_82 = None
        select_406: "bf16[200, 200, 26]" = torch.ops.aten.select.int(slice_83, 3, 1);  slice_83 = select_406 = None
        full_11: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_7, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_7 = None
        slice_84: "bf16[203, 204, 26]" = torch.ops.aten.slice.Tensor(full_11, 0, 0, -1);  slice_84 = None
        slice_85: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg17_1, 0, 2, -2)
        slice_86: "bf16[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_85, 1, 2, -2)
        slice_87: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(arg17_1, 0, 2, -2)
        slice_88: "bf16[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_87, 1, 2, -2);  slice_87 = None
        select_407: "bf16[200, 200, 26]" = torch.ops.aten.select.int(slice_88, 3, 0);  slice_88 = None
        slice_89: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 2, -2)
        slice_90: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_89, 1, 2, -2);  slice_89 = None
        full_12: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_37: "bf16[204, 204, 26]" = torch.ops.aten.copy.default(select_scatter_133, full_12);  select_scatter_133 = full_12 = None
        slice_91: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(copy_37, 0, 1, -2)
        slice_92: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(copy_37, 0, 1, -2)
        slice_93: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_92, 1, 2, -2);  slice_92 = None
        select_408: "bf16[204, 204, 26]" = torch.ops.aten.select.int(arg18_1, 3, 0);  arg18_1 = None
        slice_94: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_408, 0, 1, -2)
        slice_95: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_94, 1, 2, -2);  slice_94 = None
        slice_96: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_11, 0, 2, -2)
        slice_97: "bf16[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_96, 1, 2, -2)
        slice_98: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_11, 0, 2, -2)
        slice_99: "bf16[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_98, 1, 2, -2);  slice_98 = None
        select_409: "bf16[200, 200, 26]" = torch.ops.aten.select.int(slice_99, 3, 1);  slice_99 = None
        select_scatter_136: "bf16[200, 200, 26, 3]" = torch.ops.aten.select_scatter.default(slice_97, select_409, 3, 1);  slice_97 = select_409 = None
        slice_scatter_12: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_96, select_scatter_136, 1, 2, -2);  slice_96 = select_scatter_136 = None
        slice_scatter_13: "bf16[204, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_11, slice_scatter_12, 0, 2, -2);  slice_scatter_11 = slice_scatter_12 = None
        select_410: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 0)
        slice_100: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_410, 0, 2, -1);  select_410 = None
        slice_101: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_100, 1, 2, -2);  slice_100 = None
        select_411: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 0)
        slice_102: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_411, 0, 1, -2);  select_411 = None
        slice_103: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_102, 1, 2, -2);  slice_102 = None
        add_62: "bf16[201, 200, 26]" = torch.ops.aten.add.Tensor(slice_101, slice_103);  slice_101 = slice_103 = None
        mul_103: "bf16[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_95, add_62);  slice_95 = add_62 = None
        mul_104: "bf16[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_103, 0.5);  mul_103 = None
        slice_104: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_408, 0, 1, -2)
        slice_105: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_104, 1, 2, -2);  slice_104 = None
        abs_1: "bf16[201, 200, 26]" = torch.ops.aten.abs.default(slice_105);  slice_105 = None
        full_13: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_14: "f32[1]" = torch.ops.aten.full.default([1], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_106: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_408, 0, 1, -2)
        slice_107: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_106, 1, 2, -2);  slice_106 = None
        gt: "b8[201, 200, 26]" = torch.ops.aten.gt.Scalar(slice_107, 0.0);  slice_107 = None
        select_412: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 0)
        slice_108: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_412, 0, 1, -2);  select_412 = None
        slice_109: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_108, 1, 2, -2);  slice_108 = None
        select_413: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 0)
        slice_110: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_413, 0, 0, -3);  select_413 = None
        slice_111: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_110, 1, 2, -2);  slice_110 = None
        sub_30: "bf16[201, 200, 26]" = torch.ops.aten.sub.Tensor(slice_109, slice_111);  slice_109 = slice_111 = None
        slice_112: "bf16[203, 204, 26]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 1, 9223372036854775807)
        slice_113: "bf16[203, 204, 26]" = torch.ops.aten.slice.Tensor(arg8_1, 0, 0, -1)
        mul_105: "bf16[203, 204, 26]" = torch.ops.aten.mul.Tensor(slice_112, slice_113);  slice_112 = slice_113 = None
        slice_scatter_14: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_11, mul_105, 0, 0, -1);  full_11 = mul_105 = None
        slice_114: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_14, 0, 0, -3)
        slice_115: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_114, 1, 2, -2);  slice_114 = None
        mul_106: "bf16[201, 200, 26]" = torch.ops.aten.mul.Tensor(sub_30, slice_115);  sub_30 = slice_115 = None
        select_414: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 0)
        slice_116: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_414, 0, 3, 9223372036854775807);  select_414 = None
        slice_117: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_116, 1, 2, -2);  slice_116 = None
        select_415: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 0)
        slice_118: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_415, 0, 2, -1);  select_415 = None
        slice_119: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_118, 1, 2, -2);  slice_118 = None
        sub_31: "bf16[201, 200, 26]" = torch.ops.aten.sub.Tensor(slice_117, slice_119);  slice_117 = slice_119 = None
        slice_120: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_14, 0, 2, -1)
        slice_121: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_120, 1, 2, -2);  slice_120 = None
        mul_107: "bf16[201, 200, 26]" = torch.ops.aten.mul.Tensor(sub_31, slice_121);  sub_31 = slice_121 = None
        where_3: "bf16[201, 200, 26]" = torch.ops.aten.where.self(gt, mul_106, mul_107);  gt = mul_106 = mul_107 = None
        select_416: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 0)
        slice_122: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_416, 0, 2, -1);  select_416 = None
        slice_123: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_122, 1, 2, -2);  slice_122 = None
        select_417: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 0)
        slice_124: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_417, 0, 1, -2);  select_417 = None
        slice_125: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_124, 1, 2, -2);  slice_124 = None
        sub_32: "bf16[201, 200, 26]" = torch.ops.aten.sub.Tensor(slice_123, slice_125);  slice_123 = slice_125 = None
        slice_126: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_14, 0, 1, -2);  slice_scatter_14 = None
        slice_127: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_126, 1, 2, -2);  slice_126 = None
        mul_108: "bf16[201, 200, 26]" = torch.ops.aten.mul.Tensor(sub_32, slice_127);  sub_32 = slice_127 = None
        abs_2: "bf16[201, 200, 26]" = torch.ops.aten.abs.default(mul_108)
        lt: "b8[201, 200, 26]" = torch.ops.aten.lt.Scalar(abs_2, 1e-20);  abs_2 = None
        full_15: "bf16[]" = torch.ops.aten.full.default([], 1.0005576689441423e-20, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[201, 200, 26]" = torch.ops.aten.where.self(lt, full_15, mul_108);  lt = full_15 = None
        div_63: "bf16[201, 200, 26]" = torch.ops.aten.div.Tensor(where_3, where_4);  where_3 = where_4 = None
        mul_109: "bf16[201, 200, 26]" = torch.ops.aten.mul.Tensor(div_63, 2)
        minimum: "f32[201, 200, 26]" = torch.ops.aten.minimum.default(full_14, mul_109);  full_14 = mul_109 = None
        full_16: "f32[1]" = torch.ops.aten.full.default([1], 2.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_1: "f32[201, 200, 26]" = torch.ops.aten.minimum.default(full_16, div_63);  full_16 = div_63 = None
        maximum_2: "f32[201, 200, 26]" = torch.ops.aten.maximum.default(minimum, minimum_1);  minimum = minimum_1 = None
        maximum_3: "f32[201, 200, 26]" = torch.ops.aten.maximum.default(full_13, maximum_2);  full_13 = maximum_2 = None
        sub_33: "f32[201, 200, 26]" = torch.ops.aten.sub.Tensor(1.0, maximum_3)
        slice_128: "bf16[201, 204, 26]" = torch.ops.aten.slice.Tensor(select_408, 0, 1, -2);  select_408 = None
        slice_129: "bf16[201, 200, 26]" = torch.ops.aten.slice.Tensor(slice_128, 1, 2, -2);  slice_128 = None
        mul_110: "bf16[201, 200, 26]" = torch.ops.aten.mul.Tensor(slice_129, 1.0);  slice_129 = None
        unsqueeze_33: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(arg9_1, 0)
        slice_130: "bf16[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_33, 1, 2, -2);  unsqueeze_33 = None
        unsqueeze_34: "bf16[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_130, 2);  slice_130 = None
        slice_131: "bf16[201]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 1, -2)
        unsqueeze_35: "bf16[201, 1]" = torch.ops.aten.unsqueeze.default(slice_131, 1);  slice_131 = None
        unsqueeze_36: "bf16[201, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_35, 2);  unsqueeze_35 = None
        mul_111: "bf16[201, 200, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_34, unsqueeze_36);  unsqueeze_34 = unsqueeze_36 = None
        div_64: "bf16[201, 200, 26]" = torch.ops.aten.div.Tensor(mul_110, mul_111);  mul_110 = mul_111 = None
        abs_3: "bf16[201, 200, 26]" = torch.ops.aten.abs.default(div_64);  div_64 = None
        mul_112: "f32[201, 200, 26]" = torch.ops.aten.mul.Tensor(abs_3, maximum_3);  abs_3 = maximum_3 = None
        add_63: "f32[201, 200, 26]" = torch.ops.aten.add.Tensor(sub_33, mul_112);  sub_33 = mul_112 = None
        mul_113: "f32[201, 200, 26]" = torch.ops.aten.mul.Tensor(abs_1, add_63);  abs_1 = add_63 = None
        mul_114: "f32[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_113, mul_108);  mul_113 = mul_108 = None
        mul_115: "f32[201, 200, 26]" = torch.ops.aten.mul.Tensor(mul_114, 0.5);  mul_114 = None
        sub_34: "f32[201, 200, 26]" = torch.ops.aten.sub.Tensor(mul_104, mul_115);  mul_104 = mul_115 = None
        copy_38: "bf16[201, 200, 26]" = torch.ops.aten.copy.default(slice_93, sub_34);  slice_93 = sub_34 = None
        slice_scatter_15: "bf16[201, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_91, copy_38, 1, 2, -2);  slice_91 = copy_38 = None
        slice_scatter_16: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(copy_37, slice_scatter_15, 0, 1, -2);  copy_37 = slice_scatter_15 = None
        slice_132: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_16, 0, 2, -2)
        slice_133: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_132, 1, 2, -2);  slice_132 = None
        slice_134: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_16, 0, 1, -3);  slice_scatter_16 = None
        slice_135: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_134, 1, 2, -2);  slice_134 = None
        sub_35: "bf16[200, 200, 26]" = torch.ops.aten.sub.Tensor(slice_133, slice_135);  slice_133 = slice_135 = None
        neg_53: "bf16[200, 200, 26]" = torch.ops.aten.neg.default(sub_35);  sub_35 = None
        unsqueeze_37: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(arg9_1, 0)
        slice_136: "bf16[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_37, 1, 2, -2);  unsqueeze_37 = None
        unsqueeze_38: "bf16[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_136, 2);  slice_136 = None
        slice_137: "bf16[200]" = torch.ops.aten.slice.Tensor(arg12_1, 0, 2, -2);  arg12_1 = None
        unsqueeze_39: "bf16[200, 1]" = torch.ops.aten.unsqueeze.default(slice_137, 1);  slice_137 = None
        unsqueeze_40: "bf16[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_39, 2);  unsqueeze_39 = None
        mul_116: "bf16[200, 200, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_38, unsqueeze_40);  unsqueeze_38 = unsqueeze_40 = None
        div_65: "bf16[200, 200, 26]" = torch.ops.aten.div.Tensor(neg_53, mul_116);  neg_53 = mul_116 = None
        full_17: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_39: "bf16[204, 204, 26]" = torch.ops.aten.copy.default(select_scatter_134, full_17);  select_scatter_134 = full_17 = None
        slice_138: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(copy_39, 0, 2, -2)
        slice_139: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(copy_39, 0, 2, -2)
        slice_140: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_139, 1, 1, -2);  slice_139 = None
        unsqueeze_41: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(arg15_1, 0);  arg15_1 = None
        slice_141: "bf16[1, 201]" = torch.ops.aten.slice.Tensor(unsqueeze_41, 1, 1, -2);  unsqueeze_41 = None
        unsqueeze_42: "bf16[1, 201, 1]" = torch.ops.aten.unsqueeze.default(slice_141, 2);  slice_141 = None
        select_418: "bf16[204, 204, 26]" = torch.ops.aten.select.int(arg19_1, 3, 0);  arg19_1 = None
        slice_142: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_418, 0, 2, -2)
        slice_143: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_142, 1, 1, -2);  slice_142 = None
        mul_117: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_42, slice_143);  slice_143 = None
        select_419: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 0)
        slice_144: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_419, 0, 2, -2);  select_419 = None
        slice_145: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_144, 1, 2, -1);  slice_144 = None
        select_420: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 0)
        slice_146: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_420, 0, 2, -2);  select_420 = None
        slice_147: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_146, 1, 1, -2);  slice_146 = None
        add_64: "bf16[200, 201, 26]" = torch.ops.aten.add.Tensor(slice_145, slice_147);  slice_145 = slice_147 = None
        mul_118: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_117, add_64);  mul_117 = add_64 = None
        mul_119: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_118, 0.5);  mul_118 = None
        slice_148: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_418, 0, 2, -2)
        slice_149: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_148, 1, 1, -2);  slice_148 = None
        mul_120: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_42, slice_149);  slice_149 = None
        abs_4: "bf16[200, 201, 26]" = torch.ops.aten.abs.default(mul_120);  mul_120 = None
        full_18: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_19: "f32[1]" = torch.ops.aten.full.default([1], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_150: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_418, 0, 2, -2)
        slice_151: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_150, 1, 1, -2);  slice_150 = None
        gt_1: "b8[200, 201, 26]" = torch.ops.aten.gt.Scalar(slice_151, 0.0);  slice_151 = None
        select_421: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 0)
        slice_152: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_421, 0, 2, -2);  select_421 = None
        slice_153: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_152, 1, 1, -2);  slice_152 = None
        select_422: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 0)
        slice_154: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_422, 0, 2, -2);  select_422 = None
        slice_155: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_154, 1, 0, -3);  slice_154 = None
        sub_36: "bf16[200, 201, 26]" = torch.ops.aten.sub.Tensor(slice_153, slice_155);  slice_153 = slice_155 = None
        full_20: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_8, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_8 = None
        slice_156: "bf16[204, 203, 26]" = torch.ops.aten.slice.Tensor(full_20, 1, 0, -1)
        slice_157: "bf16[204, 203, 26]" = torch.ops.aten.slice.Tensor(arg8_1, 1, 1, 9223372036854775807)
        slice_158: "bf16[204, 203, 26]" = torch.ops.aten.slice.Tensor(arg8_1, 1, 0, -1)
        mul_121: "bf16[204, 203, 26]" = torch.ops.aten.mul.Tensor(slice_157, slice_158);  slice_157 = slice_158 = None
        copy_40: "bf16[204, 203, 26]" = torch.ops.aten.copy.default(slice_156, mul_121);  slice_156 = mul_121 = None
        slice_scatter_17: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_20, copy_40, 1, 0, -1);  full_20 = copy_40 = None
        slice_159: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_17, 0, 2, -2)
        slice_160: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_159, 1, 0, -3);  slice_159 = None
        mul_122: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(sub_36, slice_160);  sub_36 = slice_160 = None
        select_423: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 0)
        slice_161: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_423, 0, 2, -2);  select_423 = None
        slice_162: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_161, 1, 3, 9223372036854775807);  slice_161 = None
        select_424: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 0)
        slice_163: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_424, 0, 2, -2);  select_424 = None
        slice_164: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_163, 1, 2, -1);  slice_163 = None
        sub_37: "bf16[200, 201, 26]" = torch.ops.aten.sub.Tensor(slice_162, slice_164);  slice_162 = slice_164 = None
        slice_165: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_17, 0, 2, -2)
        slice_166: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_165, 1, 2, -1);  slice_165 = None
        mul_123: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(sub_37, slice_166);  sub_37 = slice_166 = None
        where_5: "bf16[200, 201, 26]" = torch.ops.aten.where.self(gt_1, mul_122, mul_123);  gt_1 = mul_122 = mul_123 = None
        select_425: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 0)
        slice_167: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_425, 0, 2, -2);  select_425 = None
        slice_168: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_167, 1, 2, -1);  slice_167 = None
        select_426: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 0)
        slice_169: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_426, 0, 2, -2);  select_426 = None
        slice_170: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_169, 1, 1, -2);  slice_169 = None
        sub_38: "bf16[200, 201, 26]" = torch.ops.aten.sub.Tensor(slice_168, slice_170);  slice_168 = slice_170 = None
        slice_171: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_17, 0, 2, -2);  slice_scatter_17 = None
        slice_172: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_171, 1, 1, -2);  slice_171 = None
        mul_124: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(sub_38, slice_172);  sub_38 = slice_172 = None
        abs_5: "bf16[200, 201, 26]" = torch.ops.aten.abs.default(mul_124)
        lt_1: "b8[200, 201, 26]" = torch.ops.aten.lt.Scalar(abs_5, 1e-20);  abs_5 = None
        full_21: "bf16[]" = torch.ops.aten.full.default([], 1.0005576689441423e-20, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "bf16[200, 201, 26]" = torch.ops.aten.where.self(lt_1, full_21, mul_124);  lt_1 = full_21 = None
        div_66: "bf16[200, 201, 26]" = torch.ops.aten.div.Tensor(where_5, where_6);  where_5 = where_6 = None
        mul_125: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(div_66, 2)
        minimum_2: "f32[200, 201, 26]" = torch.ops.aten.minimum.default(full_19, mul_125);  full_19 = mul_125 = None
        full_22: "f32[1]" = torch.ops.aten.full.default([1], 2.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_3: "f32[200, 201, 26]" = torch.ops.aten.minimum.default(full_22, div_66);  full_22 = div_66 = None
        maximum_4: "f32[200, 201, 26]" = torch.ops.aten.maximum.default(minimum_2, minimum_3);  minimum_2 = minimum_3 = None
        maximum_5: "f32[200, 201, 26]" = torch.ops.aten.maximum.default(full_18, maximum_4);  full_18 = maximum_4 = None
        sub_39: "f32[200, 201, 26]" = torch.ops.aten.sub.Tensor(1.0, maximum_5)
        slice_173: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(select_418, 0, 2, -2);  select_418 = None
        slice_174: "bf16[200, 201, 26]" = torch.ops.aten.slice.Tensor(slice_173, 1, 1, -2);  slice_173 = None
        mul_126: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(unsqueeze_42, slice_174);  unsqueeze_42 = slice_174 = None
        mul_127: "bf16[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_126, 1.0);  mul_126 = None
        mul_128: "bf16[204]" = torch.ops.aten.mul.Tensor(arg9_1, arg16_1)
        unsqueeze_43: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(mul_128, 0);  mul_128 = None
        slice_175: "bf16[1, 201]" = torch.ops.aten.slice.Tensor(unsqueeze_43, 1, 1, -2);  unsqueeze_43 = None
        unsqueeze_44: "bf16[1, 201, 1]" = torch.ops.aten.unsqueeze.default(slice_175, 2);  slice_175 = None
        div_67: "bf16[200, 201, 26]" = torch.ops.aten.div.Tensor(mul_127, unsqueeze_44);  mul_127 = unsqueeze_44 = None
        abs_6: "bf16[200, 201, 26]" = torch.ops.aten.abs.default(div_67);  div_67 = None
        mul_129: "f32[200, 201, 26]" = torch.ops.aten.mul.Tensor(abs_6, maximum_5);  abs_6 = maximum_5 = None
        add_65: "f32[200, 201, 26]" = torch.ops.aten.add.Tensor(sub_39, mul_129);  sub_39 = mul_129 = None
        mul_130: "f32[200, 201, 26]" = torch.ops.aten.mul.Tensor(abs_4, add_65);  abs_4 = add_65 = None
        mul_131: "f32[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_130, mul_124);  mul_130 = mul_124 = None
        mul_132: "f32[200, 201, 26]" = torch.ops.aten.mul.Tensor(mul_131, 0.5);  mul_131 = None
        sub_40: "f32[200, 201, 26]" = torch.ops.aten.sub.Tensor(mul_119, mul_132);  mul_119 = mul_132 = None
        copy_41: "bf16[200, 201, 26]" = torch.ops.aten.copy.default(slice_140, sub_40);  slice_140 = sub_40 = None
        slice_scatter_18: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_138, copy_41, 1, 1, -2);  slice_138 = copy_41 = None
        slice_scatter_19: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(copy_39, slice_scatter_18, 0, 2, -2);  copy_39 = slice_scatter_18 = None
        slice_176: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_19, 0, 2, -2)
        slice_177: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_176, 1, 2, -2);  slice_176 = None
        slice_178: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_19, 0, 2, -2);  slice_scatter_19 = None
        slice_179: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_178, 1, 1, -3);  slice_178 = None
        sub_41: "bf16[200, 200, 26]" = torch.ops.aten.sub.Tensor(slice_177, slice_179);  slice_177 = slice_179 = None
        unsqueeze_45: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(arg9_1, 0);  arg9_1 = None
        slice_180: "bf16[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_45, 1, 2, -2);  unsqueeze_45 = None
        unsqueeze_46: "bf16[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_180, 2);  slice_180 = None
        unsqueeze_47: "bf16[1, 204]" = torch.ops.aten.unsqueeze.default(arg16_1, 0);  arg16_1 = None
        slice_181: "bf16[1, 200]" = torch.ops.aten.slice.Tensor(unsqueeze_47, 1, 2, -2);  unsqueeze_47 = None
        unsqueeze_48: "bf16[1, 200, 1]" = torch.ops.aten.unsqueeze.default(slice_181, 2);  slice_181 = None
        mul_133: "bf16[1, 200, 1]" = torch.ops.aten.mul.Tensor(unsqueeze_46, unsqueeze_48);  unsqueeze_46 = unsqueeze_48 = None
        div_68: "bf16[200, 200, 26]" = torch.ops.aten.div.Tensor(sub_41, mul_133);  sub_41 = mul_133 = None
        sub_42: "bf16[200, 200, 26]" = torch.ops.aten.sub.Tensor(div_65, div_68);  div_65 = div_68 = None
        mul_134: "bf16[200, 200, 26]" = torch.ops.aten.mul.Tensor(slice_90, sub_42);  slice_90 = sub_42 = None
        copy_42: "bf16[200, 200, 26]" = torch.ops.aten.copy.default(select_407, mul_134);  select_407 = mul_134 = None
        select_scatter_137: "bf16[200, 200, 26, 3]" = torch.ops.aten.select_scatter.default(slice_86, copy_42, 3, 0);  slice_86 = copy_42 = None
        slice_scatter_20: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_85, select_scatter_137, 1, 2, -2);  slice_85 = select_scatter_137 = None
        slice_scatter_21: "bf16[204, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(arg17_1, slice_scatter_20, 0, 2, -2);  arg17_1 = slice_scatter_20 = None
        select_427: "bf16[204, 204, 3]" = torch.ops.aten.select.int(slice_scatter_21, 2, 0)
        select_428: "bf16[204, 204, 3]" = torch.ops.aten.select.int(slice_scatter_21, 2, 0)
        select_429: "bf16[204, 204]" = torch.ops.aten.select.int(select_428, 2, 0);  select_428 = None
        full_23: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_9, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_9 = None
        full_24: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_43: "bf16[204, 204, 26]" = torch.ops.aten.copy.default(full_23, full_24);  full_23 = full_24 = None
        slice_182: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(copy_43, 0, 2, -2)
        slice_183: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_182, 1, 2, -2)
        slice_184: "bf16[200, 204, 26]" = torch.ops.aten.slice.Tensor(copy_43, 0, 2, -2)
        slice_185: "bf16[200, 200, 26]" = torch.ops.aten.slice.Tensor(slice_184, 1, 2, -2);  slice_184 = None
        slice_186: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_185, 2, 0, -1);  slice_185 = None
        full_25: "bf16[204, 204, 28]" = torch.ops.aten.full.default(_shape_param_10, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_10 = None
        slice_187: "bf16[204, 204, 26]" = torch.ops.aten.slice.Tensor(full_25, 2, 1, -1)
        select_430: "bf16[204, 204, 26]" = torch.ops.aten.select.int(arg20_1, 3, 0);  arg20_1 = None
        copy_44: "bf16[204, 204, 26]" = torch.ops.aten.copy.default(slice_187, select_430);  slice_187 = select_430 = None
        slice_scatter_22: "bf16[204, 204, 28]" = torch.ops.aten.slice_scatter.default(full_25, copy_44, 2, 1, -1);  full_25 = copy_44 = None
        slice_188: "bf16[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_22, 0, 2, -2)
        slice_189: "bf16[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_188, 1, 2, -2);  slice_188 = None
        slice_190: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_189, 2, 1, -2);  slice_189 = None
        full_26: "bf16[204, 204, 28]" = torch.ops.aten.full.default(_shape_param_11, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_11 = None
        slice_191: "bf16[204, 204, 26]" = torch.ops.aten.slice.Tensor(full_26, 2, 1, -1)
        select_431: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 0)
        copy_45: "bf16[204, 204, 26]" = torch.ops.aten.copy.default(slice_191, select_431);  slice_191 = select_431 = None
        slice_scatter_23: "bf16[204, 204, 28]" = torch.ops.aten.slice_scatter.default(full_26, copy_45, 2, 1, -1);  full_26 = copy_45 = None
        slice_192: "bf16[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 2, -2)
        slice_193: "bf16[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_192, 1, 2, -2);  slice_192 = None
        slice_194: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_193, 2, 2, -1);  slice_193 = None
        slice_195: "bf16[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 2, -2)
        slice_196: "bf16[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_195, 1, 2, -2);  slice_195 = None
        slice_197: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_196, 2, 1, -2);  slice_196 = None
        add_66: "bf16[200, 200, 25]" = torch.ops.aten.add.Tensor(slice_194, slice_197);  slice_194 = slice_197 = None
        mul_135: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_190, add_66);  slice_190 = add_66 = None
        mul_136: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_135, 0.5);  mul_135 = None
        slice_198: "bf16[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_22, 0, 2, -2)
        slice_199: "bf16[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_198, 1, 2, -2);  slice_198 = None
        slice_200: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_199, 2, 1, -2);  slice_199 = None
        abs_7: "bf16[200, 200, 25]" = torch.ops.aten.abs.default(slice_200);  slice_200 = None
        full_27: "f32[1]" = torch.ops.aten.full.default([1], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_28: "f32[1]" = torch.ops.aten.full.default([1], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_201: "bf16[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_22, 0, 2, -2)
        slice_202: "bf16[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_201, 1, 2, -2);  slice_201 = None
        slice_203: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_202, 2, 1, -2);  slice_202 = None
        gt_2: "b8[200, 200, 25]" = torch.ops.aten.gt.Scalar(slice_203, 0.0);  slice_203 = None
        slice_204: "bf16[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 2, -2)
        slice_205: "bf16[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_204, 1, 2, -2);  slice_204 = None
        slice_206: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_205, 2, 1, -2);  slice_205 = None
        slice_207: "bf16[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 2, -2)
        slice_208: "bf16[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_207, 1, 2, -2);  slice_207 = None
        slice_209: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_208, 2, 0, -3);  slice_208 = None
        sub_43: "bf16[200, 200, 25]" = torch.ops.aten.sub.Tensor(slice_206, slice_209);  slice_206 = slice_209 = None
        full_29: "bf16[204, 204, 28]" = torch.ops.aten.full.default(_shape_param_12, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_12 = None
        slice_210: "bf16[204, 204, 26]" = torch.ops.aten.slice.Tensor(full_29, 2, 1, -1)
        full_30: "bf16[204, 204, 26]" = torch.ops.aten.full.default(_shape_param_13, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_13 = None
        slice_211: "bf16[204, 204, 25]" = torch.ops.aten.slice.Tensor(full_30, 2, 0, -1)
        slice_212: "bf16[204, 204, 25]" = torch.ops.aten.slice.Tensor(arg8_1, 2, 1, 9223372036854775807)
        slice_213: "bf16[204, 204, 25]" = torch.ops.aten.slice.Tensor(arg8_1, 2, 0, -1);  arg8_1 = None
        mul_137: "bf16[204, 204, 25]" = torch.ops.aten.mul.Tensor(slice_212, slice_213);  slice_212 = slice_213 = None
        copy_46: "bf16[204, 204, 25]" = torch.ops.aten.copy.default(slice_211, mul_137);  slice_211 = mul_137 = None
        slice_scatter_24: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(full_30, copy_46, 2, 0, -1);  full_30 = copy_46 = None
        copy_47: "bf16[204, 204, 26]" = torch.ops.aten.copy.default(slice_210, slice_scatter_24);  slice_210 = slice_scatter_24 = None
        slice_scatter_25: "bf16[204, 204, 28]" = torch.ops.aten.slice_scatter.default(full_29, copy_47, 2, 1, -1);  full_29 = copy_47 = None
        slice_214: "bf16[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_25, 0, 2, -2)
        slice_215: "bf16[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_214, 1, 2, -2);  slice_214 = None
        slice_216: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_215, 2, 0, -3);  slice_215 = None
        mul_138: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(sub_43, slice_216);  sub_43 = slice_216 = None
        slice_217: "bf16[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 2, -2)
        slice_218: "bf16[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_217, 1, 2, -2);  slice_217 = None
        slice_219: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_218, 2, 3, 9223372036854775807);  slice_218 = None
        slice_220: "bf16[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 2, -2)
        slice_221: "bf16[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_220, 1, 2, -2);  slice_220 = None
        slice_222: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_221, 2, 2, -1);  slice_221 = None
        sub_44: "bf16[200, 200, 25]" = torch.ops.aten.sub.Tensor(slice_219, slice_222);  slice_219 = slice_222 = None
        slice_223: "bf16[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_25, 0, 2, -2)
        slice_224: "bf16[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_223, 1, 2, -2);  slice_223 = None
        slice_225: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_224, 2, 2, -1);  slice_224 = None
        mul_139: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(sub_44, slice_225);  sub_44 = slice_225 = None
        where_7: "bf16[200, 200, 25]" = torch.ops.aten.where.self(gt_2, mul_138, mul_139);  gt_2 = mul_138 = mul_139 = None
        slice_226: "bf16[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 2, -2)
        slice_227: "bf16[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_226, 1, 2, -2);  slice_226 = None
        slice_228: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_227, 2, 2, -1);  slice_227 = None
        slice_229: "bf16[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 2, -2);  slice_scatter_23 = None
        slice_230: "bf16[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_229, 1, 2, -2);  slice_229 = None
        slice_231: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_230, 2, 1, -2);  slice_230 = None
        sub_45: "bf16[200, 200, 25]" = torch.ops.aten.sub.Tensor(slice_228, slice_231);  slice_228 = slice_231 = None
        slice_232: "bf16[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_25, 0, 2, -2);  slice_scatter_25 = None
        slice_233: "bf16[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_232, 1, 2, -2);  slice_232 = None
        slice_234: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_233, 2, 1, -2);  slice_233 = None
        mul_140: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(sub_45, slice_234);  sub_45 = slice_234 = None
        abs_8: "bf16[200, 200, 25]" = torch.ops.aten.abs.default(mul_140)
        lt_2: "b8[200, 200, 25]" = torch.ops.aten.lt.Scalar(abs_8, 1e-20);  abs_8 = None
        full_31: "bf16[]" = torch.ops.aten.full.default([], 1.0005576689441423e-20, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "bf16[200, 200, 25]" = torch.ops.aten.where.self(lt_2, full_31, mul_140);  lt_2 = full_31 = None
        div_69: "bf16[200, 200, 25]" = torch.ops.aten.div.Tensor(where_7, where_8);  where_7 = where_8 = None
        mul_141: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(div_69, 2)
        minimum_4: "f32[200, 200, 25]" = torch.ops.aten.minimum.default(full_28, mul_141);  full_28 = mul_141 = None
        full_32: "f32[1]" = torch.ops.aten.full.default([1], 2.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum_5: "f32[200, 200, 25]" = torch.ops.aten.minimum.default(full_32, div_69);  full_32 = div_69 = None
        maximum_6: "f32[200, 200, 25]" = torch.ops.aten.maximum.default(minimum_4, minimum_5);  minimum_4 = minimum_5 = None
        maximum_7: "f32[200, 200, 25]" = torch.ops.aten.maximum.default(full_27, maximum_6);  full_27 = maximum_6 = None
        sub_46: "f32[200, 200, 25]" = torch.ops.aten.sub.Tensor(1.0, maximum_7)
        slice_235: "bf16[200, 204, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_22, 0, 2, -2);  slice_scatter_22 = None
        slice_236: "bf16[200, 200, 28]" = torch.ops.aten.slice.Tensor(slice_235, 1, 2, -2);  slice_235 = None
        slice_237: "bf16[200, 200, 25]" = torch.ops.aten.slice.Tensor(slice_236, 2, 1, -2);  slice_236 = None
        mul_142: "bf16[200, 200, 25]" = torch.ops.aten.mul.Tensor(slice_237, 1.0);  slice_237 = None
        unsqueeze_49: "bf16[1, 26]" = torch.ops.aten.unsqueeze.default(arg3_1, 0)
        unsqueeze_50: "bf16[1, 1, 26]" = torch.ops.aten.unsqueeze.default(unsqueeze_49, 1);  unsqueeze_49 = None
        slice_238: "bf16[1, 1, 25]" = torch.ops.aten.slice.Tensor(unsqueeze_50, 2, 0, -1);  unsqueeze_50 = None
        div_70: "bf16[200, 200, 25]" = torch.ops.aten.div.Tensor(mul_142, slice_238);  mul_142 = slice_238 = None
        abs_9: "bf16[200, 200, 25]" = torch.ops.aten.abs.default(div_70);  div_70 = None
        mul_143: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(abs_9, maximum_7);  abs_9 = maximum_7 = None
        add_67: "f32[200, 200, 25]" = torch.ops.aten.add.Tensor(sub_46, mul_143);  sub_46 = mul_143 = None
        mul_144: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(abs_7, add_67);  abs_7 = add_67 = None
        mul_145: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_144, mul_140);  mul_144 = mul_140 = None
        mul_146: "f32[200, 200, 25]" = torch.ops.aten.mul.Tensor(mul_145, 0.5);  mul_145 = None
        sub_47: "f32[200, 200, 25]" = torch.ops.aten.sub.Tensor(mul_136, mul_146);  mul_136 = mul_146 = None
        copy_48: "bf16[200, 200, 25]" = torch.ops.aten.copy.default(slice_186, sub_47);  slice_186 = sub_47 = None
        slice_scatter_26: "bf16[200, 200, 26]" = torch.ops.aten.slice_scatter.default(slice_183, copy_48, 2, 0, -1);  slice_183 = copy_48 = None
        slice_scatter_27: "bf16[200, 204, 26]" = torch.ops.aten.slice_scatter.default(slice_182, slice_scatter_26, 1, 2, -2);  slice_182 = slice_scatter_26 = None
        slice_scatter_28: "bf16[204, 204, 26]" = torch.ops.aten.slice_scatter.default(copy_43, slice_scatter_27, 0, 2, -2);  copy_43 = slice_scatter_27 = None
        select_432: "bf16[204, 204]" = torch.ops.aten.select.int(slice_scatter_28, 2, 0)
        neg_54: "bf16[204, 204]" = torch.ops.aten.neg.default(select_432);  select_432 = None
        select_433: "bf16[]" = torch.ops.aten.select.int(arg3_1, 0, 0)
        div_71: "bf16[204, 204]" = torch.ops.aten.div.Tensor(neg_54, select_433);  neg_54 = select_433 = None
        add_68: "bf16[204, 204]" = torch.ops.aten.add.Tensor(select_429, div_71);  select_429 = div_71 = None
        select_scatter_138: "bf16[204, 204, 3]" = torch.ops.aten.select_scatter.default(select_427, add_68, 2, 0);  select_427 = add_68 = None
        select_scatter_139: "bf16[204, 204, 26, 3]" = torch.ops.aten.select_scatter.default(slice_scatter_21, select_scatter_138, 2, 0);  slice_scatter_21 = select_scatter_138 = None
        select_434: "bf16[204, 204, 3]" = torch.ops.aten.select.int(select_scatter_139, 2, 0)
        select_435: "bf16[204, 204]" = torch.ops.aten.select.int(select_434, 2, 0);  select_434 = select_435 = None
        select_436: "bf16[204, 204, 3]" = torch.ops.aten.select.int(select_scatter_139, 2, 0)
        select_437: "bf16[204, 204, 3]" = torch.ops.aten.select.int(select_scatter_139, 2, 0)
        select_438: "bf16[204, 204]" = torch.ops.aten.select.int(select_437, 2, 0);  select_437 = None
        select_scatter_140: "bf16[204, 204, 3]" = torch.ops.aten.select_scatter.default(select_436, select_438, 2, 0);  select_436 = select_438 = None
        select_scatter_141: "bf16[204, 204, 26, 3]" = torch.ops.aten.select_scatter.default(select_scatter_139, select_scatter_140, 2, 0);  select_scatter_139 = select_scatter_140 = None
        slice_239: "bf16[204, 204, 24, 3]" = torch.ops.aten.slice.Tensor(select_scatter_141, 2, 1, -1)
        slice_240: "bf16[204, 204, 24, 3]" = torch.ops.aten.slice.Tensor(select_scatter_141, 2, 1, -1)
        select_439: "bf16[204, 204, 24]" = torch.ops.aten.select.int(slice_240, 3, 0);  slice_240 = None
        slice_241: "bf16[204, 204, 24]" = torch.ops.aten.slice.Tensor(slice_scatter_28, 2, 1, -1)
        slice_242: "bf16[204, 204, 24]" = torch.ops.aten.slice.Tensor(slice_scatter_28, 2, 0, -2)
        sub_48: "bf16[204, 204, 24]" = torch.ops.aten.sub.Tensor(slice_241, slice_242);  slice_241 = slice_242 = None
        neg_55: "bf16[204, 204, 24]" = torch.ops.aten.neg.default(sub_48);  sub_48 = None
        slice_243: "bf16[24]" = torch.ops.aten.slice.Tensor(arg3_1, 0, 1, -1)
        div_72: "bf16[204, 204, 24]" = torch.ops.aten.div.Tensor(neg_55, slice_243);  neg_55 = slice_243 = None
        add_69: "bf16[204, 204, 24]" = torch.ops.aten.add.Tensor(select_439, div_72);  select_439 = div_72 = None
        select_scatter_142: "bf16[204, 204, 24, 3]" = torch.ops.aten.select_scatter.default(slice_239, add_69, 3, 0);  slice_239 = add_69 = None
        slice_scatter_29: "bf16[204, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(select_scatter_141, select_scatter_142, 2, 1, -1);  select_scatter_141 = select_scatter_142 = None
        slice_244: "bf16[204, 204, 24, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_29, 2, 1, -1)
        select_440: "bf16[204, 204, 24]" = torch.ops.aten.select.int(slice_244, 3, 0);  slice_244 = select_440 = None
        slice_245: "bf16[204, 204, 24, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_29, 2, 1, -1)
        slice_246: "bf16[204, 204, 24, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_29, 2, 1, -1)
        select_441: "bf16[204, 204, 24]" = torch.ops.aten.select.int(slice_246, 3, 0);  slice_246 = None
        select_scatter_143: "bf16[204, 204, 24, 3]" = torch.ops.aten.select_scatter.default(slice_245, select_441, 3, 0);  slice_245 = select_441 = None
        slice_scatter_30: "bf16[204, 204, 26, 3]" = torch.ops.aten.slice_scatter.default(slice_scatter_29, select_scatter_143, 2, 1, -1);  slice_scatter_29 = select_scatter_143 = None
        select_442: "bf16[204, 204, 3]" = torch.ops.aten.select.int(slice_scatter_30, 2, -1)
        select_443: "bf16[204, 204, 3]" = torch.ops.aten.select.int(slice_scatter_30, 2, -1)
        select_444: "bf16[204, 204]" = torch.ops.aten.select.int(select_443, 2, 0);  select_443 = None
        select_445: "bf16[204, 204]" = torch.ops.aten.select.int(slice_scatter_28, 2, -1)
        select_446: "bf16[204, 204]" = torch.ops.aten.select.int(slice_scatter_28, 2, -2);  slice_scatter_28 = None
        sub_49: "bf16[204, 204]" = torch.ops.aten.sub.Tensor(select_445, select_446);  select_445 = select_446 = None
        neg_56: "bf16[204, 204]" = torch.ops.aten.neg.default(sub_49);  sub_49 = None
        select_447: "bf16[]" = torch.ops.aten.select.int(arg3_1, 0, -1)
        mul_147: "bf16[]" = torch.ops.aten.mul.Tensor(select_447, 0.5);  select_447 = None
        div_73: "bf16[204, 204]" = torch.ops.aten.div.Tensor(neg_56, mul_147);  neg_56 = mul_147 = None
        add_70: "bf16[204, 204]" = torch.ops.aten.add.Tensor(select_444, div_73);  select_444 = div_73 = None
        select_scatter_144: "bf16[204, 204, 3]" = torch.ops.aten.select_scatter.default(select_442, add_70, 2, 0);  select_442 = add_70 = None
        select_scatter_145: "bf16[204, 204, 26, 3]" = torch.ops.aten.select_scatter.default(slice_scatter_30, select_scatter_144, 2, -1);  slice_scatter_30 = select_scatter_144 = None
        select_448: "bf16[204, 204, 3]" = torch.ops.aten.select.int(select_scatter_145, 2, -1)
        select_449: "bf16[204, 204]" = torch.ops.aten.select.int(select_448, 2, 0);  select_448 = select_449 = None
        select_450: "bf16[204, 204, 26]" = torch.ops.aten.select.int(slice_scatter_13, 3, 1)
        select_451: "bf16[204, 204, 3]" = torch.ops.aten.select.int(select_scatter_145, 2, -1)
        select_452: "bf16[204, 204, 3]" = torch.ops.aten.select.int(select_scatter_145, 2, -1)
        select_453: "bf16[204, 204]" = torch.ops.aten.select.int(select_452, 2, 0);  select_452 = None
        select_scatter_146: "bf16[204, 204, 3]" = torch.ops.aten.select_scatter.default(select_451, select_453, 2, 0);  select_451 = select_453 = None
        select_scatter_147: "bf16[204, 204, 26, 3]" = torch.ops.aten.select_scatter.default(select_scatter_145, select_scatter_146, 2, -1);  select_scatter_145 = select_scatter_146 = None
        select_454: "bf16[204, 204, 26]" = torch.ops.aten.select.int(select_scatter_147, 3, 0)
        mul_148: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(select_454, 1.6);  select_454 = None
        select_455: "bf16[204, 204, 26]" = torch.ops.aten.select.int(select_scatter_147, 3, 2)
        mul_149: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(select_455, 0.6);  select_455 = None
        sub_50: "bf16[204, 204, 26]" = torch.ops.aten.sub.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
        mul_150: "bf16[204, 204, 26]" = torch.ops.aten.mul.Tensor(sub_50, 1.0);  sub_50 = None
        add_71: "bf16[204, 204, 26]" = torch.ops.aten.add.Tensor(select_450, mul_150);  select_450 = mul_150 = None
        select_scatter_148: "bf16[204, 204, 26, 3]" = torch.ops.aten.select_scatter.default(slice_scatter_13, add_71, 3, 1);  slice_scatter_13 = add_71 = None
        select_456: "bf16[204, 204, 26]" = torch.ops.aten.select.int(select_scatter_148, 3, 1);  select_456 = None
        select_457: "bf16[204, 204, 26]" = torch.ops.aten.select.int(select_scatter_148, 3, 1)
        select_scatter_149: "bf16[204, 204, 26, 3]" = torch.ops.aten.select_scatter.default(select_scatter_148, select_457, 3, 1);  select_scatter_148 = select_457 = None
        full_33: "f32[204, 204]" = torch.ops.aten.full.default(_shape_param_14, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_14 = None
        slice_247: "f32[200, 204]" = torch.ops.aten.slice.Tensor(full_33, 0, 2, -2)
        slice_248: "f32[200, 204]" = torch.ops.aten.slice.Tensor(full_33, 0, 2, -2)
        slice_249: "f32[200, 200]" = torch.ops.aten.slice.Tensor(slice_248, 1, 2, -2);  slice_248 = None
        slice_250: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2)
        slice_251: "bf16[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_250, 1, 2, -2);  slice_250 = None
        select_458: "bf16[200, 200, 3]" = torch.ops.aten.select.int(slice_251, 2, -1);  slice_251 = None
        select_459: "bf16[200, 200]" = torch.ops.aten.select.int(select_458, 2, 1);  select_458 = None
        lt_3: "b8[200, 200]" = torch.ops.aten.lt.Scalar(select_459, 0.0);  select_459 = None
        slice_252: "bf16[200, 204, 26, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 2, -2);  slice_scatter_5 = None
        slice_253: "bf16[200, 200, 26, 3]" = torch.ops.aten.slice.Tensor(slice_252, 1, 2, -2);  slice_252 = None
        select_460: "bf16[200, 200, 3]" = torch.ops.aten.select.int(slice_253, 2, -1);  slice_253 = None
        select_461: "bf16[200, 200]" = torch.ops.aten.select.int(select_460, 2, 1);  select_460 = None
        neg_57: "bf16[200, 200]" = torch.ops.aten.neg.default(select_461);  select_461 = None
        mul_151: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(neg_57, 0.5);  neg_57 = None
        select_462: "bf16[]" = torch.ops.aten.select.int(arg3_1, 0, -1);  arg3_1 = None
        mul_152: "bf16[200, 200]" = torch.ops.aten.mul.Tensor(mul_151, select_462);  mul_151 = select_462 = None
        div_74: "bf16[200, 200]" = torch.ops.aten.div.Tensor(mul_152, 1);  mul_152 = None
        full_34: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "bf16[200, 200]" = torch.ops.aten.where.self(lt_3, div_74, full_34);  lt_3 = div_74 = full_34 = None
        copy_49: "f32[200, 200]" = torch.ops.aten.copy.default(slice_249, where_9);  slice_249 = where_9 = None
        slice_scatter_31: "f32[200, 204]" = torch.ops.aten.slice_scatter.default(slice_247, copy_49, 1, 2, -2);  slice_247 = copy_49 = None
        slice_scatter_32: "f32[204, 204]" = torch.ops.aten.slice_scatter.default(full_33, slice_scatter_31, 0, 2, -2);  full_33 = slice_scatter_31 = None
        return (select_scatter_147, select_scatter_149, slice_scatter_32)



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
