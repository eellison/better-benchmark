"""
Standalone repro captured via capture_hook.
Label: gpt-oss-20b
Pattern hash: 441d6e79ae66
Shape hash: 9dfaf485
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
    def forward(self, arg0_1: "i64[4000]", arg1_1: "i64[4000]", arg2_1: "bf16[1000, 2880]", arg3_1: "bf16[32, 2880, 5760]", arg4_1: "i32[32]", arg5_1: "bf16[32, 5760]", arg6_1: "bf16[32, 2880, 2880]", arg7_1: "bf16[32, 2880]", arg8_1: "bf16[1000, 4]", arg9_1: "bf16[1, 1000, 2880]", arg10_1: "bf16[2880]", arg11_1: "bf16[4096, 2880]", arg12_1: "bf16[4096]", arg13_1: "bf16[512, 2880]", arg14_1: "bf16[512]", arg15_1: "bf16[1, 1000, 32]", arg16_1: "bf16[1, 1000, 32]", arg17_1: "bf16[1, 1, 1000, 1000]", arg18_1: "bf16[64]", arg19_1: "bf16[512, 2880]", arg20_1: "bf16[512]", arg21_1: "bf16[2880, 4096]", arg22_1: "bf16[2880]", arg23_1: "bf16[2880]", arg24_1: "bf16[32, 2880]", arg25_1: "bf16[32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30):
        # No stacktrace found for following nodes
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ge: "b8[4000]" = torch.ops.aten.ge.Scalar(arg0_1, 32)
        unsqueeze: "b8[4000, 1]" = torch.ops.aten.unsqueeze.default(ge, -1);  ge = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        div: "i64[4000]" = torch.ops.aten.div.Tensor_mode(arg1_1, 4, rounding_mode = 'floor')
        index: "bf16[4000, 2880]" = torch.ops.aten.index.Tensor(arg2_1, [div]);  arg2_1 = div = None
        where: "bf16[4000, 2880]" = torch.ops.aten.where.self(unsqueeze, full_1, index);  full_1 = index = None
        _grouped_mm: "bf16[4000, 5760]" = torch.ops.aten._grouped_mm.default(where, arg3_1, arg4_1);  where = arg3_1 = None
        clamp_max: "i64[4000]" = torch.ops.aten.clamp_max.default(arg0_1, 31);  arg0_1 = None
        index_1: "bf16[4000, 5760]" = torch.ops.aten.index.Tensor(arg5_1, [clamp_max]);  arg5_1 = None
        add: "bf16[4000, 5760]" = torch.ops.aten.add.Tensor(_grouped_mm, index_1);  _grouped_mm = index_1 = None
        slice_1: "bf16[4000, 2880]" = torch.ops.aten.slice.Tensor(add, 1, 1, 9223372036854775807, 2)
        convert_element_type: "f32[4000, 2880]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        clamp_min: "f32[4000, 2880]" = torch.ops.aten.clamp_min.default(convert_element_type, -7.0);  convert_element_type = None
        clamp_max_1: "f32[4000, 2880]" = torch.ops.aten.clamp_max.default(clamp_min, 7.0);  clamp_min = None
        convert_element_type_1: "bf16[4000, 2880]" = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.bfloat16);  clamp_max_1 = None
        add_1: "bf16[4000, 2880]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1);  convert_element_type_1 = None
        slice_2: "bf16[4000, 2880]" = torch.ops.aten.slice.Tensor(add, 1, 0, 9223372036854775807, 2);  add = None
        convert_element_type_2: "f32[4000, 2880]" = torch.ops.prims.convert_element_type.default(slice_2, torch.float32);  slice_2 = None
        clamp_max_2: "f32[4000, 2880]" = torch.ops.aten.clamp_max.default(convert_element_type_2, 7.0);  convert_element_type_2 = None
        convert_element_type_3: "bf16[4000, 2880]" = torch.ops.prims.convert_element_type.default(clamp_max_2, torch.bfloat16);  clamp_max_2 = None
        mul: "bf16[4000, 2880]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.702)
        sigmoid: "bf16[4000, 2880]" = torch.ops.aten.sigmoid.default(mul);  mul = None
        mul_1: "bf16[4000, 2880]" = torch.ops.aten.mul.Tensor(convert_element_type_3, sigmoid);  convert_element_type_3 = sigmoid = None
        mul_2: "bf16[4000, 2880]" = torch.ops.aten.mul.Tensor(add_1, mul_1);  add_1 = mul_1 = None
        _grouped_mm_1: "bf16[4000, 2880]" = torch.ops.aten._grouped_mm.default(mul_2, arg6_1, arg4_1);  mul_2 = arg6_1 = arg4_1 = None
        index_2: "bf16[4000, 2880]" = torch.ops.aten.index.Tensor(arg7_1, [clamp_max]);  arg7_1 = clamp_max = None
        add_2: "bf16[4000, 2880]" = torch.ops.aten.add.Tensor(_grouped_mm_1, index_2);  _grouped_mm_1 = index_2 = None
        convert_element_type_4: "f32[1000, 4]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        amax: "f32[1000, 1]" = torch.ops.aten.amax.default(convert_element_type_4, [1], True)
        sub: "f32[1000, 4]" = torch.ops.aten.sub.Tensor(convert_element_type_4, amax);  convert_element_type_4 = amax = None
        exp: "f32[1000, 4]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[1000, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div_1: "f32[1000, 4]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_5: "bf16[1000, 4]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        view: "bf16[4000]" = torch.ops.aten.view.default(convert_element_type_5, [-1]);  convert_element_type_5 = None
        index_3: "bf16[4000]" = torch.ops.aten.index.Tensor(view, [arg1_1]);  view = None
        unsqueeze_1: "bf16[4000, 1]" = torch.ops.aten.unsqueeze.default(index_3, -1);  index_3 = None
        mul_3: "bf16[4000, 2880]" = torch.ops.aten.mul.Tensor(add_2, unsqueeze_1);  add_2 = unsqueeze_1 = None
        where_1: "bf16[4000, 2880]" = torch.ops.aten.where.self(unsqueeze, full, mul_3);  unsqueeze = full = mul_3 = None
        empty: "i64[4000]" = torch.ops.aten.empty.memory_format(_shape_param_0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        iota: "i64[4000]" = torch.ops.prims.iota.default(4000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_put: "i64[4000]" = torch.ops.aten.index_put.default(empty, [arg1_1], iota);  empty = arg1_1 = iota = None
        index_4: "bf16[4000, 2880]" = torch.ops.aten.index.Tensor(where_1, [index_put]);  where_1 = index_put = None
        view_1: "bf16[1000, 4, 2880]" = torch.ops.aten.view.default(index_4, _shape_param_1);  index_4 = _shape_param_1 = None
        sum_2: "bf16[1000, 2880]" = torch.ops.aten.sum.dim_IntList(view_1, [1]);  view_1 = None
        view_2: "bf16[1, 1000, 2880]" = torch.ops.aten.view.default(sum_2, _shape_param_2);  sum_2 = _shape_param_2 = None
        add_3: "bf16[1, 1000, 2880]" = torch.ops.aten.add.Tensor(arg9_1, view_2);  arg9_1 = view_2 = None
        convert_element_type_6: "f32[1, 1000, 2880]" = torch.ops.prims.convert_element_type.default(add_3, torch.float32)
        pow_1: "f32[1, 1000, 2880]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_6, 2)
        mean: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_4: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean, 1e-05);  mean = None
        rsqrt: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_4: "f32[1, 1000, 2880]" = torch.ops.aten.mul.Tensor(convert_element_type_6, rsqrt);  convert_element_type_6 = rsqrt = None
        mul_5: "f32[1, 1000, 2880]" = torch.ops.aten.mul.Tensor(arg10_1, mul_4);  arg10_1 = mul_4 = None
        convert_element_type_7: "bf16[1, 1000, 2880]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        view_3: "bf16[1000, 2880]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_3);  _shape_param_3 = None
        permute: "bf16[2880, 4096]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm: "bf16[1000, 4096]" = torch.ops.aten.addmm.default(arg12_1, view_3, permute);  arg12_1 = view_3 = permute = None
        view_4: "bf16[1, 1000, 4096]" = torch.ops.aten.view.default(addmm, _shape_param_4);  addmm = _shape_param_4 = None
        view_5: "bf16[1, 1000, 64, 64]" = torch.ops.aten.view.default(view_4, _shape_param_5);  view_4 = _shape_param_5 = None
        permute_1: "bf16[1, 64, 1000, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        split = torch.ops.aten.split.Tensor(permute_1, 32, -1);  permute_1 = None
        getitem: "bf16[1, 64, 1000, 32]" = split[0]
        getitem_1: "bf16[1, 64, 1000, 32]" = split[1];  split = None
        view_6: "bf16[1000, 2880]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_6);  _shape_param_6 = None
        permute_2: "bf16[2880, 512]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_1: "bf16[1000, 512]" = torch.ops.aten.addmm.default(arg14_1, view_6, permute_2);  arg14_1 = view_6 = permute_2 = None
        view_7: "bf16[1, 1000, 512]" = torch.ops.aten.view.default(addmm_1, _shape_param_7);  addmm_1 = _shape_param_7 = None
        view_8: "bf16[1, 1000, 8, 64]" = torch.ops.aten.view.default(view_7, _shape_param_8);  view_7 = _shape_param_8 = None
        permute_3: "bf16[1, 8, 1000, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        split_1 = torch.ops.aten.split.Tensor(permute_3, 32, -1);  permute_3 = None
        getitem_2: "bf16[1, 8, 1000, 32]" = split_1[0]
        getitem_3: "bf16[1, 8, 1000, 32]" = split_1[1];  split_1 = None
        unsqueeze_2: "bf16[1, 1, 1000, 32]" = torch.ops.aten.unsqueeze.default(arg15_1, 1);  arg15_1 = None
        mul_6: "bf16[1, 64, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_2)
        unsqueeze_3: "bf16[1, 1, 1000, 32]" = torch.ops.aten.unsqueeze.default(arg16_1, 1);  arg16_1 = None
        mul_7: "bf16[1, 64, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze_3)
        sub_1: "bf16[1, 64, 1000, 32]" = torch.ops.aten.sub.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        mul_8: "bf16[1, 64, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze_2);  getitem_1 = None
        mul_9: "bf16[1, 64, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_3);  getitem = None
        add_5: "bf16[1, 64, 1000, 32]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        cat: "bf16[1, 64, 1000, 64]" = torch.ops.aten.cat.default([sub_1, add_5], -1);  sub_1 = add_5 = None
        expand: "bf16[1, 64, 1000, 64]" = torch.ops.aten.expand.default(cat, _shape_param_9);  cat = _shape_param_9 = None
        view_9: "bf16[64, 1000, 64]" = torch.ops.aten.view.default(expand, _shape_param_10);  expand = _shape_param_10 = None
        mul_10: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_2, unsqueeze_2)
        mul_11: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_3, unsqueeze_3)
        sub_2: "bf16[1, 8, 1000, 32]" = torch.ops.aten.sub.Tensor(mul_10, mul_11);  mul_10 = mul_11 = None
        mul_12: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_3, unsqueeze_2);  getitem_3 = unsqueeze_2 = None
        mul_13: "bf16[1, 8, 1000, 32]" = torch.ops.aten.mul.Tensor(getitem_2, unsqueeze_3);  getitem_2 = unsqueeze_3 = None
        add_6: "bf16[1, 8, 1000, 32]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
        cat_1: "bf16[1, 8, 1000, 64]" = torch.ops.aten.cat.default([sub_2, add_6], -1);  sub_2 = add_6 = None
        unsqueeze_4: "bf16[1, 8, 1, 1000, 64]" = torch.ops.aten.unsqueeze.default(cat_1, 2)
        expand_1: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.expand.default(unsqueeze_4, _shape_param_11);  unsqueeze_4 = _shape_param_11 = None
        clone: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_10: "bf16[1, 64, 1000, 64]" = torch.ops.aten.view.default(clone, _shape_param_12);  clone = _shape_param_12 = None
        permute_4: "bf16[1, 64, 64, 1000]" = torch.ops.aten.permute.default(view_10, [0, 1, 3, 2]);  view_10 = None
        expand_2: "bf16[1, 64, 64, 1000]" = torch.ops.aten.expand.default(permute_4, _shape_param_13);  permute_4 = _shape_param_13 = None
        view_11: "bf16[64, 64, 1000]" = torch.ops.aten.view.default(expand_2, _shape_param_14);  expand_2 = _shape_param_14 = None
        bmm: "bf16[64, 1000, 1000]" = torch.ops.aten.bmm.default(view_9, view_11);  view_9 = view_11 = None
        view_12: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.view.default(bmm, _shape_param_15);  bmm = _shape_param_15 = None
        mul_14: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.mul.Tensor(view_12, 0.125);  view_12 = None
        add_7: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.add.Tensor(mul_14, arg17_1);  mul_14 = arg17_1 = None
        view_13: "bf16[1, 64, 1, 1]" = torch.ops.aten.view.default(arg18_1, [1, -1, 1, 1]);  arg18_1 = None
        expand_3: "bf16[1, 64, 1000, 1]" = torch.ops.aten.expand.default(view_13, _shape_param_16);  view_13 = _shape_param_16 = None
        cat_2: "bf16[1, 64, 1000, 1001]" = torch.ops.aten.cat.default([add_7, expand_3], -1);  add_7 = expand_3 = None
        max_1 = torch.ops.aten.max.dim(cat_2, -1, True)
        getitem_4: "bf16[1, 64, 1000, 1]" = max_1[0]
        getitem_5: "i64[1, 64, 1000, 1]" = max_1[1];  max_1 = getitem_5 = None
        sub_3: "bf16[1, 64, 1000, 1001]" = torch.ops.aten.sub.Tensor(cat_2, getitem_4);  cat_2 = getitem_4 = None
        convert_element_type_8: "f32[1, 64, 1000, 1001]" = torch.ops.prims.convert_element_type.default(sub_3, torch.float32);  sub_3 = None
        amax_1: "f32[1, 64, 1000, 1]" = torch.ops.aten.amax.default(convert_element_type_8, [-1], True)
        sub_4: "f32[1, 64, 1000, 1001]" = torch.ops.aten.sub.Tensor(convert_element_type_8, amax_1);  convert_element_type_8 = amax_1 = None
        exp_1: "f32[1, 64, 1000, 1001]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_3: "f32[1, 64, 1000, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_2: "f32[1, 64, 1000, 1001]" = torch.ops.aten.div.Tensor(exp_1, sum_3);  exp_1 = sum_3 = None
        convert_element_type_9: "bf16[1, 64, 1000, 1001]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        slice_3: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.slice.Tensor(convert_element_type_9, 3, 0, -1);  convert_element_type_9 = None
        clone_1: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.clone.default(slice_3);  slice_3 = None
        expand_4: "bf16[1, 64, 1000, 1000]" = torch.ops.aten.expand.default(clone_1, _shape_param_17);  clone_1 = _shape_param_17 = None
        view_14: "bf16[64, 1000, 1000]" = torch.ops.aten.view.default(expand_4, _shape_param_18);  expand_4 = _shape_param_18 = None
        view_15: "bf16[1000, 2880]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_19);  convert_element_type_7 = _shape_param_19 = None
        permute_5: "bf16[2880, 512]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_2: "bf16[1000, 512]" = torch.ops.aten.addmm.default(arg20_1, view_15, permute_5);  arg20_1 = view_15 = permute_5 = None
        view_16: "bf16[1, 1000, 512]" = torch.ops.aten.view.default(addmm_2, _shape_param_20);  addmm_2 = _shape_param_20 = None
        view_17: "bf16[1, 1000, 8, 64]" = torch.ops.aten.view.default(view_16, _shape_param_21);  view_16 = _shape_param_21 = None
        permute_6: "bf16[1, 8, 1000, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        unsqueeze_5: "bf16[1, 8, 1, 1000, 64]" = torch.ops.aten.unsqueeze.default(permute_6, 2)
        expand_5: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.expand.default(unsqueeze_5, _shape_param_22);  unsqueeze_5 = _shape_param_22 = None
        clone_2: "bf16[1, 8, 8, 1000, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_18: "bf16[1, 64, 1000, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_23);  clone_2 = _shape_param_23 = None
        expand_6: "bf16[1, 64, 1000, 64]" = torch.ops.aten.expand.default(view_18, _shape_param_24);  view_18 = _shape_param_24 = None
        view_19: "bf16[64, 1000, 64]" = torch.ops.aten.view.default(expand_6, _shape_param_25);  expand_6 = _shape_param_25 = None
        bmm_1: "bf16[64, 1000, 64]" = torch.ops.aten.bmm.default(view_14, view_19);  view_14 = view_19 = None
        view_20: "bf16[1, 64, 1000, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_26);  bmm_1 = _shape_param_26 = None
        permute_7: "bf16[1, 1000, 64, 64]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
        clone_3: "bf16[1, 1000, 64, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_21: "bf16[1, 1000, 4096]" = torch.ops.aten.view.default(clone_3, _shape_param_27);  clone_3 = _shape_param_27 = None
        view_22: "bf16[1000, 4096]" = torch.ops.aten.view.default(view_21, _shape_param_28);  view_21 = _shape_param_28 = None
        permute_8: "bf16[4096, 2880]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_3: "bf16[1000, 2880]" = torch.ops.aten.addmm.default(arg22_1, view_22, permute_8);  arg22_1 = view_22 = permute_8 = None
        view_23: "bf16[1, 1000, 2880]" = torch.ops.aten.view.default(addmm_3, _shape_param_29);  addmm_3 = _shape_param_29 = None
        add_8: "bf16[1, 1000, 2880]" = torch.ops.aten.add.Tensor(add_3, view_23);  add_3 = view_23 = None
        convert_element_type_10: "f32[1, 1000, 2880]" = torch.ops.prims.convert_element_type.default(add_8, torch.float32)
        pow_2: "f32[1, 1000, 2880]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_10, 2)
        mean_1: "f32[1, 1000, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_9: "f32[1, 1000, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-05);  mean_1 = None
        rsqrt_1: "f32[1, 1000, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_15: "f32[1, 1000, 2880]" = torch.ops.aten.mul.Tensor(convert_element_type_10, rsqrt_1);  convert_element_type_10 = rsqrt_1 = None
        mul_16: "f32[1, 1000, 2880]" = torch.ops.aten.mul.Tensor(arg23_1, mul_15);  arg23_1 = mul_15 = None
        convert_element_type_11: "bf16[1, 1000, 2880]" = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None
        view_24: "bf16[1000, 2880]" = torch.ops.aten.view.default(convert_element_type_11, _shape_param_30);  convert_element_type_11 = _shape_param_30 = None
        permute_9: "bf16[2880, 32]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_4: "bf16[1000, 32]" = torch.ops.aten.addmm.default(arg25_1, view_24, permute_9);  arg25_1 = permute_9 = None
        topk = torch.ops.aten.topk.default(addmm_4, 4);  addmm_4 = None
        getitem_6: "bf16[1000, 4]" = topk[0]
        getitem_7: "i64[1000, 4]" = topk[1];  topk = None
        view_25: "i64[4000]" = torch.ops.aten.view.default(getitem_7, [-1]);  getitem_7 = None
        sort = torch.ops.aten.sort.default(view_25);  view_25 = None
        getitem_8: "i64[4000]" = sort[0]
        getitem_9: "i64[4000]" = sort[1];  sort = None
        convert_element_type_12: "i32[4000]" = torch.ops.prims.convert_element_type.default(getitem_8, torch.int32)
        return (cat_1, permute_6, add_8, view_24, getitem_6, getitem_8, getitem_9, convert_element_type_12)



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
