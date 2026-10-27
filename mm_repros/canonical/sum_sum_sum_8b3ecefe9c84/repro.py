"""
Standalone repro captured via capture_hook.
Label: GoogleFnet
Pattern hash: 8b3ecefe9c84
Shape hash: 6fdb2df3
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
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "i64[32, 512]", arg3_1: "f32[32, 512, 32000]", arg4_1: "f32[16384, 1]", arg5_1: "f32[16384, 1]", arg6_1: "f32[32, 512, 32000]", arg7_1: "f32[32000, 768]", arg8_1: "f32[768]", arg9_1: "f32[16384, 768]", arg10_1: "f32[32, 512, 1]", arg11_1: "f32[32, 512, 1]", arg12_1: "f32[768, 768]", arg13_1: "f32[16384, 768]", arg14_1: "f32[768]", arg15_1: "f32[32, 512, 768]", arg16_1: "f32[32, 512, 1]", arg17_1: "b8[32, 512, 768]", arg18_1: "f32[768, 3072]", arg19_1: "f32[16384, 3072]", arg20_1: "f32[16384, 3072]", arg21_1: "f32[3072, 768]", arg22_1: "f32[16384, 768]", arg23_1: "f32[768]", arg24_1: "f32[32, 512, 768]", arg25_1: "f32[32, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "i64[16384]" = torch.ops.aten.view.default(arg2_1, [-1]);  arg2_1 = None
        unsqueeze: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(view, 1);  view = None
        ne: "b8[16384, 1]" = torch.ops.aten.ne.Scalar(unsqueeze, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[16384, 1]" = torch.ops.aten.where.self(ne, unsqueeze, full);  unsqueeze = full = None
        iota: "i64[32000]" = torch.ops.prims.iota.default(32000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_1: "i64[1, 32000]" = torch.ops.aten.view.default(iota, _shape_param_0);  iota = _shape_param_0 = None
        expand: "i64[16384, 32000]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        eq: "b8[16384, 32000]" = torch.ops.aten.eq.Tensor(expand, view_1);  expand = view_1 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[16384, 32000]" = torch.ops.aten.where.self(eq, scalar_tensor_1, scalar_tensor);  eq = scalar_tensor_1 = scalar_tensor = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[16384, 1]" = torch.ops.aten.where.self(ne, div, full_1);  ne = div = full_1 = None
        mul: "f32[16384, 32000]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        view_2: "f32[16384, 32000]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        sub: "f32[16384, 32000]" = torch.ops.aten.sub.Tensor(view_2, arg4_1);  view_2 = arg4_1 = None
        sub_1: "f32[16384, 32000]" = torch.ops.aten.sub.Tensor(sub, arg5_1);  sub = arg5_1 = None
        exp: "f32[16384, 32000]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
        mul_1: "f32[16384, 32000]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub_2: "f32[16384, 32000]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
        view_3: "f32[32, 512, 32000]" = torch.ops.aten.view.default(sub_2, _shape_param_3);  sub_2 = _shape_param_3 = None
        add: "f32[32, 512, 32000]" = torch.ops.aten.add.Tensor(arg6_1, view_3);  arg6_1 = view_3 = None
        view_4: "f32[16384, 32000]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        permute: "f32[768, 32000]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        permute_1: "f32[32000, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm: "f32[16384, 768]" = torch.ops.aten.mm.default(view_4, permute_1);  permute_1 = None
        sum_2: "f32[1, 32000]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True)
        view_5: "f32[32000]" = torch.ops.aten.view.default(sum_2, _shape_param_5);  sum_2 = _shape_param_5 = None
        view_6: "f32[32, 512, 768]" = torch.ops.aten.view.default(mm, _shape_param_6);  mm = _shape_param_6 = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_6, arg8_1);  arg8_1 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, 768)
        sum_3: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        view_7: "f32[32, 512, 768]" = torch.ops.aten.view.default(arg9_1, _shape_param_7);  arg9_1 = _shape_param_7 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_7, 0.5)
        pow_1: "f32[32, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(view_7, 3.0)
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_7, mul_5);  mul_5 = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_1, 0.7978845608028654);  add_1 = None
        tanh: "f32[32, 512, 768]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, add_2)
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_7, arg10_1);  mul_7 = arg10_1 = None
        mul_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, arg11_1);  sub_3 = None
        mul_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, mul_8);  mul_2 = None
        sum_4: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_9, [2], True);  mul_9 = None
        mul_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, sum_4);  sum_4 = None
        sub_4: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_3, sum_3);  mul_3 = sum_3 = None
        sub_5: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_4, mul_10);  sub_4 = mul_10 = None
        div_1: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(arg11_1, 768);  arg11_1 = None
        mul_11: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_12: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_6, mul_8);  mul_8 = None
        sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_12, [0, 1]);  mul_12 = None
        sum_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_6, [0, 1]);  view_6 = None
        mul_13: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_11, mul_4);  mul_4 = None
        mul_14: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_11, add_2);  mul_11 = add_2 = None
        mul_15: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_6: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(1, mul_15);  mul_15 = None
        mul_16: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_13, sub_6);  mul_13 = sub_6 = None
        mul_17: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, 0.7978845608028654);  mul_16 = None
        mul_18: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, 0.044715)
        pow_2: "f32[32, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(view_7, 2.0);  view_7 = None
        mul_19: "f32[32, 512, 768]" = torch.ops.aten.mul.Scalar(pow_2, 3.0);  pow_2 = None
        mul_20: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        add_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_17, mul_20);  mul_17 = mul_20 = None
        mul_21: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_14, 0.5);  mul_14 = None
        add_4: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_3, mul_21);  add_3 = mul_21 = None
        view_8: "f32[16384, 768]" = torch.ops.aten.view.default(add_4, _shape_param_8);  add_4 = _shape_param_8 = None
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_1: "f32[16384, 768]" = torch.ops.aten.mm.default(view_8, permute_3);  permute_3 = None
        permute_4: "f32[768, 16384]" = torch.ops.aten.permute.default(view_8, [1, 0])
        mm_2: "f32[768, 768]" = torch.ops.aten.mm.default(permute_4, arg13_1);  permute_4 = arg13_1 = None
        sum_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True);  view_8 = None
        view_9: "f32[768]" = torch.ops.aten.view.default(sum_7, _shape_param_9);  sum_7 = _shape_param_9 = None
        view_10: "f32[32, 512, 768]" = torch.ops.aten.view.default(mm_1, _shape_param_10);  mm_1 = _shape_param_10 = None
        mul_22: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_10, arg14_1);  arg14_1 = None
        mul_23: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, 768)
        sum_8: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_22, [2], True)
        mul_24: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, arg15_1);  mul_22 = None
        sum_9: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_24, [2], True);  mul_24 = None
        mul_25: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg15_1, sum_9);  sum_9 = None
        sub_7: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_23, sum_8);  mul_23 = sum_8 = None
        sub_8: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_7, mul_25);  sub_7 = mul_25 = None
        mul_26: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg16_1, sub_8);  arg16_1 = sub_8 = None
        mul_27: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_10, arg15_1);  arg15_1 = None
        sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_27, [0, 1]);  mul_27 = None
        sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_10, [0, 1]);  view_10 = None
        convert_element_type: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.float32);  arg17_1 = None
        mul_28: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_29: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_26, mul_28);  mul_28 = None
        view_11: "f32[16384, 768]" = torch.ops.aten.view.default(mul_29, _shape_param_11);  mul_29 = _shape_param_11 = None
        permute_5: "f32[3072, 768]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        permute_6: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        mm_3: "f32[16384, 3072]" = torch.ops.aten.mm.default(view_11, permute_6);  permute_6 = None
        permute_7: "f32[768, 16384]" = torch.ops.aten.permute.default(view_11, [1, 0])
        mm_4: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_7, arg19_1);  permute_7 = arg19_1 = None
        sum_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_11, [0], True);  view_11 = None
        view_12: "f32[768]" = torch.ops.aten.view.default(sum_12, _shape_param_12);  sum_12 = _shape_param_12 = None
        view_13: "f32[32, 512, 3072]" = torch.ops.aten.view.default(mm_3, _shape_param_13);  mm_3 = _shape_param_13 = None
        view_14: "f32[32, 512, 3072]" = torch.ops.aten.view.default(arg20_1, _shape_param_14);  arg20_1 = _shape_param_14 = None
        mul_30: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_14, 0.5)
        mul_31: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_13, mul_30);  mul_30 = None
        pow_3: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_14, 3.0)
        mul_32: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_5: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_14, mul_32);  mul_32 = None
        mul_33: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_5, 0.7978845608028654);  add_5 = None
        tanh_1: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_33);  mul_33 = None
        add_6: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_34: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_13, add_6);  view_13 = add_6 = None
        mul_35: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_9: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_35);  mul_35 = None
        mul_36: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_31, sub_9);  mul_31 = sub_9 = None
        mul_37: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_36, 0.7978845608028654);  mul_36 = None
        mul_38: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_37, 0.044715)
        pow_4: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_14, 2.0);  view_14 = None
        mul_39: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_4, 3.0);  pow_4 = None
        mul_40: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_38, mul_39);  mul_38 = mul_39 = None
        add_7: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_37, mul_40);  mul_37 = mul_40 = None
        mul_41: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_34, 0.5);  mul_34 = None
        add_8: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_7, mul_41);  add_7 = mul_41 = None
        view_15: "f32[16384, 3072]" = torch.ops.aten.view.default(add_8, _shape_param_15);  add_8 = _shape_param_15 = None
        permute_8: "f32[768, 3072]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        permute_9: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_5: "f32[16384, 768]" = torch.ops.aten.mm.default(view_15, permute_9);  permute_9 = None
        permute_10: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_15, [1, 0])
        mm_6: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_10, arg22_1);  permute_10 = arg22_1 = None
        sum_13: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_15, [0], True);  view_15 = None
        view_16: "f32[3072]" = torch.ops.aten.view.default(sum_13, _shape_param_16);  sum_13 = _shape_param_16 = None
        view_17: "f32[32, 512, 768]" = torch.ops.aten.view.default(mm_5, _shape_param_17);  mm_5 = _shape_param_17 = None
        add_9: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_26, view_17);  mul_26 = view_17 = None
        mul_42: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_9, arg23_1);  arg23_1 = None
        mul_43: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_42, 768)
        sum_14: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_42, [2], True)
        mul_44: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_42, arg24_1);  mul_42 = None
        sum_15: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_44, [2], True);  mul_44 = None
        mul_45: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg24_1, sum_15);  sum_15 = None
        sub_10: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_43, sum_14);  mul_43 = sum_14 = None
        sub_11: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_10, mul_45);  sub_10 = mul_45 = None
        mul_46: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg25_1, sub_11);  arg25_1 = sub_11 = None
        mul_47: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_9, arg24_1);  arg24_1 = None
        sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_47, [0, 1]);  mul_47 = None
        sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_9, [0, 1]);  add_9 = None
        full_2: "f32[32, 512, 768, 2]" = torch.ops.aten.full.default(_shape_param_18, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_18 = None
        select_scatter: "f32[32, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_2, mul_46, 3, 0)
        return (view_4, view_5, sum_5, sum_6, mm_2, view_9, sum_10, sum_11, mm_4, view_12, mm_6, view_16, mul_46, sum_16, sum_17, full_2, select_scatter)



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
