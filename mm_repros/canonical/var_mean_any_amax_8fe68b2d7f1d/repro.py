"""
Standalone repro captured via capture_hook.
Label: BlenderbotForConditionalGeneration
Pattern hash: 8fe68b2d7f1d
Shape hash: 39e8da5d
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
    def forward(self, arg0_1: "bf16[8008, 2560]", arg1_1: "i64[16, 128]", arg2_1: "bf16[128, 2560]", arg3_1: "bf16[2560]", arg4_1: "bf16[2560]", arg5_1: "bf16[2560, 2560]", arg6_1: "bf16[2560]", arg7_1: "bf16[2560, 2560]", arg8_1: "bf16[2560]", arg9_1: "bf16[2560, 2560]", arg10_1: "bf16[2560]", arg11_1: "bf16[2560, 2560]", arg12_1: "bf16[2560]", arg13_1: "bf16[2560]", arg14_1: "bf16[2560]", arg15_1: "bf16[10240, 2560]", arg16_1: "bf16[10240]", arg17_1: "bf16[2560, 10240]", arg18_1: "bf16[2560]", arg19_1: "bf16[2560]", arg20_1: "bf16[2560]", arg21_1: "bf16[2560, 2560]", arg22_1: "bf16[2560]", arg23_1: "bf16[2560, 2560]", arg24_1: "bf16[2560]", arg25_1: "bf16[2560, 2560]", arg26_1: "bf16[2560]", arg27_1: "bf16[2560, 2560]", arg28_1: "bf16[2560]", arg29_1: "bf16[2560]", arg30_1: "bf16[2560]", arg31_1: "bf16[10240, 2560]", arg32_1: "bf16[10240]", arg33_1: "bf16[2560, 10240]", arg34_1: "bf16[2560]", arg35_1: "bf16[16, 32, 128, 80]", arg36_1: "bf16[2560, 2560]", arg37_1: "bf16[2560]", arg38_1: "bf16[16, 128, 2560]", arg39_1: "bf16[2560]", arg40_1: "bf16[2560]", arg41_1: "bf16[2560, 2560]", arg42_1: "bf16[2560]", arg43_1: "bf16[2560]", arg44_1: "bf16[2560]", arg45_1: "bf16[2560, 2560]", arg46_1: "bf16[2560]", arg47_1: "bf16[2560, 2560]", arg48_1: "bf16[2560]", arg49_1: "bf16[2560, 2560]", arg50_1: "bf16[2560]", arg51_1: "bf16[2560]", arg52_1: "bf16[2560]", arg53_1: "bf16[10240, 2560]", arg54_1: "bf16[10240]", arg55_1: "bf16[2560, 10240]", arg56_1: "bf16[2560]", arg57_1: "bf16[2560]", arg58_1: "bf16[2560]", arg59_1: "bf16[2560, 2560]", arg60_1: "bf16[2560]", arg61_1: "bf16[2560, 2560]", arg62_1: "bf16[2560]", arg63_1: "bf16[2560, 2560]", arg64_1: "bf16[2560]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92):
        # No stacktrace found for following nodes
        embedding: "bf16[16, 128, 2560]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg0_1 = arg1_1 = None
        mul: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        embedding_1: "bf16[128, 2560]" = torch.ops.aten.embedding.default(arg2_1, iota);  arg2_1 = iota = None
        add: "bf16[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
        convert_element_type: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 128, 1]" = var_mean[0]
        getitem_1: "f32[16, 128, 1]" = var_mean[1];  var_mean = None
        sub: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul_1: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_2: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        convert_element_type_1: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  _shape_param_0 = None
        permute: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg6_1, view, permute);  arg6_1 = view = permute = None
        view_1: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        view_2: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute_1: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        mul_3: "bf16[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_1, 0.334370152488211);  permute_1 = None
        expand: "bf16[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_3, _shape_param_3);  mul_3 = _shape_param_3 = None
        clone: "bf16[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_3: "bf16[512, 128, 80]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        view_4: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_5);  _shape_param_5 = None
        permute_2: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_1: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg8_1, view_4, permute_2);  arg8_1 = view_4 = permute_2 = None
        view_5: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_1, _shape_param_6);  addmm_1 = _shape_param_6 = None
        view_6: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_5, _shape_param_7);  view_5 = _shape_param_7 = None
        permute_3: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_4: "bf16[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        mul_4: "bf16[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_4, 0.334370152488211);  permute_4 = None
        expand_1: "bf16[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_4, _shape_param_8);  mul_4 = _shape_param_8 = None
        clone_1: "bf16[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_7: "bf16[512, 80, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_9);  clone_1 = _shape_param_9 = None
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_3, view_7);  view_3 = view_7 = None
        view_8: "bf16[16, 32, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_10);  bmm = _shape_param_10 = None
        eq: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(view_8, -inf)
        logical_not: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full: "bf16[16, 32, 128, 128]" = torch.ops.aten.full.default(_shape_param_11, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_11 = None
        convert_element_type_2: "f32[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        amax: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
        sub_1: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: "bf16[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        where: "bf16[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full, convert_element_type_3);  logical_not_1 = full = convert_element_type_3 = None
        expand_2: "bf16[16, 32, 128, 128]" = torch.ops.aten.expand.default(where, _shape_param_12);  where = _shape_param_12 = None
        view_9: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_2, _shape_param_13);  expand_2 = _shape_param_13 = None
        view_10: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_14);  convert_element_type_1 = _shape_param_14 = None
        permute_5: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_2: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg10_1, view_10, permute_5);  arg10_1 = view_10 = permute_5 = None
        view_11: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_2, _shape_param_15);  addmm_2 = _shape_param_15 = None
        view_12: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_11, _shape_param_16);  view_11 = _shape_param_16 = None
        permute_6: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        expand_3: "bf16[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_6, _shape_param_17);  permute_6 = _shape_param_17 = None
        clone_2: "bf16[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_13: "bf16[512, 128, 80]" = torch.ops.aten.view.default(clone_2, _shape_param_18);  clone_2 = _shape_param_18 = None
        bmm_1: "bf16[512, 128, 80]" = torch.ops.aten.bmm.default(view_9, view_13);  view_9 = view_13 = None
        view_14: "bf16[16, 32, 128, 80]" = torch.ops.aten.view.default(bmm_1, _shape_param_19);  bmm_1 = _shape_param_19 = None
        permute_7: "bf16[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_3: "bf16[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_15: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(clone_3, _shape_param_20);  clone_3 = _shape_param_20 = None
        view_16: "bf16[2048, 2560]" = torch.ops.aten.view.default(view_15, _shape_param_21);  view_15 = _shape_param_21 = None
        permute_8: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_3: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg12_1, view_16, permute_8);  arg12_1 = view_16 = permute_8 = None
        view_17: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_3, _shape_param_22);  addmm_3 = _shape_param_22 = None
        add_3: "bf16[16, 128, 2560]" = torch.ops.aten.add.Tensor(add, view_17);  add = view_17 = None
        convert_element_type_4: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_3, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_4, [2], correction = 0, keepdim = True)
        getitem_2: "f32[16, 128, 1]" = var_mean_1[0]
        getitem_3: "f32[16, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type_4, getitem_3);  convert_element_type_4 = getitem_3 = None
        add_4: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_5: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_6: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_5, arg13_1);  mul_5 = arg13_1 = None
        add_5: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_6, arg14_1);  mul_6 = arg14_1 = None
        convert_element_type_5: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        view_18: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_23);  convert_element_type_5 = _shape_param_23 = None
        permute_9: "bf16[2560, 10240]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_4: "bf16[2048, 10240]" = torch.ops.aten.addmm.default(arg16_1, view_18, permute_9);  arg16_1 = view_18 = permute_9 = None
        view_19: "bf16[16, 128, 10240]" = torch.ops.aten.view.default(addmm_4, _shape_param_24);  addmm_4 = _shape_param_24 = None
        convert_element_type_6: "f32[16, 128, 10240]" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_7: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.5)
        mul_8: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476);  convert_element_type_6 = None
        erf: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_6: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_7, add_6);  mul_7 = add_6 = None
        convert_element_type_7: "bf16[16, 128, 10240]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        view_20: "bf16[2048, 10240]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_25);  convert_element_type_7 = _shape_param_25 = None
        permute_10: "bf16[10240, 2560]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_5: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg18_1, view_20, permute_10);  arg18_1 = view_20 = permute_10 = None
        view_21: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_5, _shape_param_26);  addmm_5 = _shape_param_26 = None
        add_7: "bf16[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_3, view_21);  add_3 = view_21 = None
        convert_element_type_8: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_7, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_8, [2], correction = 0, keepdim = True)
        getitem_4: "f32[16, 128, 1]" = var_mean_2[0]
        getitem_5: "f32[16, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type_8, getitem_5);  convert_element_type_8 = getitem_5 = None
        add_8: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_10: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_11: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_10, arg19_1);  mul_10 = arg19_1 = None
        add_9: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_11, arg20_1);  mul_11 = arg20_1 = None
        convert_element_type_9: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        view_22: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_27);  _shape_param_27 = None
        permute_11: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_6: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg22_1, view_22, permute_11);  arg22_1 = view_22 = permute_11 = None
        view_23: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_6, _shape_param_28);  addmm_6 = _shape_param_28 = None
        view_24: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_23, _shape_param_29);  view_23 = _shape_param_29 = None
        permute_12: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        mul_12: "bf16[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_12, 0.334370152488211);  permute_12 = None
        expand_4: "bf16[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_12, _shape_param_30);  mul_12 = _shape_param_30 = None
        clone_4: "bf16[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_25: "bf16[512, 128, 80]" = torch.ops.aten.view.default(clone_4, _shape_param_31);  clone_4 = _shape_param_31 = None
        view_26: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_32);  _shape_param_32 = None
        permute_13: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_7: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg24_1, view_26, permute_13);  arg24_1 = view_26 = permute_13 = None
        view_27: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_7, _shape_param_33);  addmm_7 = _shape_param_33 = None
        view_28: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_27, _shape_param_34);  view_27 = _shape_param_34 = None
        permute_14: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        permute_15: "bf16[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
        mul_13: "bf16[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_15, 0.334370152488211);  permute_15 = None
        expand_5: "bf16[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_13, _shape_param_35);  mul_13 = _shape_param_35 = None
        clone_5: "bf16[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_29: "bf16[512, 80, 128]" = torch.ops.aten.view.default(clone_5, _shape_param_36);  clone_5 = _shape_param_36 = None
        bmm_2: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_25, view_29);  view_25 = view_29 = None
        view_30: "bf16[16, 32, 128, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_37);  bmm_2 = _shape_param_37 = None
        eq_1: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(view_30, -inf)
        logical_not_2: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        full_1: "bf16[16, 32, 128, 128]" = torch.ops.aten.full.default(_shape_param_38, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_38 = None
        convert_element_type_10: "f32[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None
        amax_1: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_10, [-1], True)
        sub_4: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_10, amax_1);  convert_element_type_10 = amax_1 = None
        exp_1: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_11: "bf16[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        where_1: "bf16[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_3, full_1, convert_element_type_11);  logical_not_3 = full_1 = convert_element_type_11 = None
        expand_6: "bf16[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_1, _shape_param_39);  where_1 = _shape_param_39 = None
        view_31: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_6, _shape_param_40);  expand_6 = _shape_param_40 = None
        view_32: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_41);  convert_element_type_9 = _shape_param_41 = None
        permute_16: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_8: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg26_1, view_32, permute_16);  arg26_1 = view_32 = permute_16 = None
        view_33: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_8, _shape_param_42);  addmm_8 = _shape_param_42 = None
        view_34: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_33, _shape_param_43);  view_33 = _shape_param_43 = None
        permute_17: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        expand_7: "bf16[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_17, _shape_param_44);  permute_17 = _shape_param_44 = None
        clone_6: "bf16[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_35: "bf16[512, 128, 80]" = torch.ops.aten.view.default(clone_6, _shape_param_45);  clone_6 = _shape_param_45 = None
        bmm_3: "bf16[512, 128, 80]" = torch.ops.aten.bmm.default(view_31, view_35);  view_31 = view_35 = None
        view_36: "bf16[16, 32, 128, 80]" = torch.ops.aten.view.default(bmm_3, _shape_param_46);  bmm_3 = _shape_param_46 = None
        permute_18: "bf16[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_7: "bf16[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_37: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(clone_7, _shape_param_47);  clone_7 = _shape_param_47 = None
        view_38: "bf16[2048, 2560]" = torch.ops.aten.view.default(view_37, _shape_param_48);  view_37 = _shape_param_48 = None
        permute_19: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_9: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg28_1, view_38, permute_19);  arg28_1 = view_38 = permute_19 = None
        view_39: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_9, _shape_param_49);  addmm_9 = _shape_param_49 = None
        add_10: "bf16[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_7, view_39);  add_7 = view_39 = None
        convert_element_type_12: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_10, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_12, [2], correction = 0, keepdim = True)
        getitem_6: "f32[16, 128, 1]" = var_mean_3[0]
        getitem_7: "f32[16, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type_12, getitem_7);  convert_element_type_12 = getitem_7 = None
        add_11: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_14: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_15: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_14, arg29_1);  mul_14 = arg29_1 = None
        add_12: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_15, arg30_1);  mul_15 = arg30_1 = None
        convert_element_type_13: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16);  add_12 = None
        view_40: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_13, _shape_param_50);  convert_element_type_13 = _shape_param_50 = None
        permute_20: "bf16[2560, 10240]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_10: "bf16[2048, 10240]" = torch.ops.aten.addmm.default(arg32_1, view_40, permute_20);  arg32_1 = view_40 = permute_20 = None
        view_41: "bf16[16, 128, 10240]" = torch.ops.aten.view.default(addmm_10, _shape_param_51);  addmm_10 = _shape_param_51 = None
        convert_element_type_14: "f32[16, 128, 10240]" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_16: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.5)
        mul_17: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.7071067811865476);  convert_element_type_14 = None
        erf_1: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
        add_13: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_18: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_16, add_13);  mul_16 = add_13 = None
        convert_element_type_15: "bf16[16, 128, 10240]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        view_42: "bf16[2048, 10240]" = torch.ops.aten.view.default(convert_element_type_15, _shape_param_52);  convert_element_type_15 = _shape_param_52 = None
        permute_21: "bf16[10240, 2560]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_11: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg34_1, view_42, permute_21);  arg34_1 = view_42 = permute_21 = None
        view_43: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_11, _shape_param_53);  addmm_11 = _shape_param_53 = None
        add_14: "bf16[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_10, view_43);  add_10 = view_43 = None
        convert_element_type_16: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_14, torch.float32);  add_14 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_16, [2], correction = 0, keepdim = True)
        getitem_8: "f32[16, 128, 1]" = var_mean_4[0]
        getitem_9: "f32[16, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        permute_22: "bf16[16, 128, 32, 80]" = torch.ops.aten.permute.default(arg35_1, [0, 2, 1, 3]);  arg35_1 = None
        view_44: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(permute_22, _shape_param_54);  permute_22 = _shape_param_54 = None
        view_45: "bf16[2048, 2560]" = torch.ops.aten.view.default(view_44, _shape_param_55);  view_44 = _shape_param_55 = None
        permute_23: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        addmm_12: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg37_1, view_45, permute_23);  arg37_1 = view_45 = permute_23 = None
        view_46: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_12, _shape_param_56);  addmm_12 = _shape_param_56 = None
        add_15: "bf16[16, 128, 2560]" = torch.ops.aten.add.Tensor(arg38_1, view_46);  arg38_1 = view_46 = None
        convert_element_type_17: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_15, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_17, [2], correction = 0, keepdim = True)
        getitem_10: "f32[16, 128, 1]" = var_mean_5[0]
        getitem_11: "f32[16, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_6: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type_17, getitem_11);  convert_element_type_17 = getitem_11 = None
        add_16: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_4: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_19: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_20: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_19, arg39_1);  mul_19 = arg39_1 = None
        add_17: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_20, arg40_1);  mul_20 = arg40_1 = None
        convert_element_type_18: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None
        view_47: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_18, _shape_param_57);  convert_element_type_18 = _shape_param_57 = None
        permute_24: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_13: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg42_1, view_47, permute_24);  arg42_1 = view_47 = permute_24 = None
        view_48: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_13, _shape_param_58);  addmm_13 = _shape_param_58 = None
        view_49: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_48, _shape_param_59);  view_48 = _shape_param_59 = None
        permute_25: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
        mul_21: "bf16[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_25, 0.334370152488211);  permute_25 = None
        expand_8: "bf16[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_21, _shape_param_60);  mul_21 = _shape_param_60 = None
        clone_8: "bf16[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_50: "bf16[512, 128, 80]" = torch.ops.aten.view.default(clone_8, _shape_param_61);  clone_8 = _shape_param_61 = None
        sub_7: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type_16, getitem_9);  convert_element_type_16 = getitem_9 = None
        add_18: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_5: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_22: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = rsqrt_5 = None
        mul_23: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_22, arg43_1);  mul_22 = arg43_1 = None
        add_19: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_23, arg44_1);  mul_23 = arg44_1 = None
        convert_element_type_19: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None
        view_51: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_19, _shape_param_62);  _shape_param_62 = None
        permute_26: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_14: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg46_1, view_51, permute_26);  arg46_1 = view_51 = permute_26 = None
        view_52: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_14, _shape_param_63);  addmm_14 = _shape_param_63 = None
        view_53: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_52, _shape_param_64);  view_52 = _shape_param_64 = None
        permute_27: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        permute_28: "bf16[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_27, [0, 1, 3, 2]);  permute_27 = None
        mul_24: "bf16[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_28, 0.334370152488211);  permute_28 = None
        expand_9: "bf16[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_24, _shape_param_65);  mul_24 = _shape_param_65 = None
        clone_9: "bf16[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_54: "bf16[512, 80, 128]" = torch.ops.aten.view.default(clone_9, _shape_param_66);  clone_9 = _shape_param_66 = None
        bmm_4: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_50, view_54);  view_50 = view_54 = None
        view_55: "bf16[16, 32, 128, 128]" = torch.ops.aten.view.default(bmm_4, _shape_param_67);  bmm_4 = _shape_param_67 = None
        eq_2: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(view_55, -inf)
        logical_not_4: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        full_2: "bf16[16, 32, 128, 128]" = torch.ops.aten.full.default(_shape_param_68, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_68 = None
        convert_element_type_20: "f32[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(view_55, torch.float32);  view_55 = None
        amax_2: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_20, [-1], True)
        sub_8: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_20, amax_2);  convert_element_type_20 = amax_2 = None
        exp_2: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_3: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_21: "bf16[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        where_2: "bf16[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_5, full_2, convert_element_type_21);  logical_not_5 = full_2 = convert_element_type_21 = None
        expand_10: "bf16[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_2, _shape_param_69);  where_2 = _shape_param_69 = None
        view_56: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_10, _shape_param_70);  expand_10 = _shape_param_70 = None
        view_57: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_19, _shape_param_71);  _shape_param_71 = None
        permute_29: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_15: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg48_1, view_57, permute_29);  arg48_1 = view_57 = permute_29 = None
        view_58: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_15, _shape_param_72);  addmm_15 = _shape_param_72 = None
        view_59: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_58, _shape_param_73);  view_58 = _shape_param_73 = None
        permute_30: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
        expand_11: "bf16[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_30, _shape_param_74);  permute_30 = _shape_param_74 = None
        clone_10: "bf16[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_60: "bf16[512, 128, 80]" = torch.ops.aten.view.default(clone_10, _shape_param_75);  clone_10 = _shape_param_75 = None
        bmm_5: "bf16[512, 128, 80]" = torch.ops.aten.bmm.default(view_56, view_60);  view_56 = view_60 = None
        view_61: "bf16[16, 32, 128, 80]" = torch.ops.aten.view.default(bmm_5, _shape_param_76);  bmm_5 = _shape_param_76 = None
        permute_31: "bf16[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
        clone_11: "bf16[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
        view_62: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(clone_11, _shape_param_77);  clone_11 = _shape_param_77 = None
        view_63: "bf16[2048, 2560]" = torch.ops.aten.view.default(view_62, _shape_param_78);  view_62 = _shape_param_78 = None
        permute_32: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_16: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg50_1, view_63, permute_32);  arg50_1 = view_63 = permute_32 = None
        view_64: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_16, _shape_param_79);  addmm_16 = _shape_param_79 = None
        add_20: "bf16[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_15, view_64);  add_15 = view_64 = None
        convert_element_type_22: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_20, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_22, [2], correction = 0, keepdim = True)
        getitem_12: "f32[16, 128, 1]" = var_mean_6[0]
        getitem_13: "f32[16, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_9: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type_22, getitem_13);  convert_element_type_22 = getitem_13 = None
        add_21: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_25: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_26: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_25, arg51_1);  mul_25 = arg51_1 = None
        add_22: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_26, arg52_1);  mul_26 = arg52_1 = None
        convert_element_type_23: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16);  add_22 = None
        view_65: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_23, _shape_param_80);  convert_element_type_23 = _shape_param_80 = None
        permute_33: "bf16[2560, 10240]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_17: "bf16[2048, 10240]" = torch.ops.aten.addmm.default(arg54_1, view_65, permute_33);  arg54_1 = view_65 = permute_33 = None
        view_66: "bf16[16, 128, 10240]" = torch.ops.aten.view.default(addmm_17, _shape_param_81);  addmm_17 = _shape_param_81 = None
        convert_element_type_24: "f32[16, 128, 10240]" = torch.ops.prims.convert_element_type.default(view_66, torch.float32);  view_66 = None
        mul_27: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.5)
        mul_28: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.7071067811865476);  convert_element_type_24 = None
        erf_2: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_23: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_29: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_27, add_23);  mul_27 = add_23 = None
        convert_element_type_25: "bf16[16, 128, 10240]" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        view_67: "bf16[2048, 10240]" = torch.ops.aten.view.default(convert_element_type_25, _shape_param_82);  convert_element_type_25 = _shape_param_82 = None
        permute_34: "bf16[10240, 2560]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_18: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg56_1, view_67, permute_34);  arg56_1 = view_67 = permute_34 = None
        view_68: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_18, _shape_param_83);  addmm_18 = _shape_param_83 = None
        add_24: "bf16[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_20, view_68);  add_20 = view_68 = None
        convert_element_type_26: "f32[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_24, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_26, [2], correction = 0, keepdim = True)
        getitem_14: "f32[16, 128, 1]" = var_mean_7[0]
        getitem_15: "f32[16, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_10: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(convert_element_type_26, getitem_15);  convert_element_type_26 = getitem_15 = None
        add_25: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_30: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_7);  sub_10 = rsqrt_7 = None
        mul_31: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_30, arg57_1);  mul_30 = arg57_1 = None
        add_26: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_31, arg58_1);  mul_31 = arg58_1 = None
        convert_element_type_27: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_26, torch.bfloat16);  add_26 = None
        view_69: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_27, _shape_param_84);  _shape_param_84 = None
        permute_35: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_19: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg60_1, view_69, permute_35);  arg60_1 = view_69 = permute_35 = None
        view_70: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_19, _shape_param_85);  addmm_19 = _shape_param_85 = None
        view_71: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_70, _shape_param_86);  view_70 = _shape_param_86 = None
        permute_36: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        view_72: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_27, _shape_param_87);  _shape_param_87 = None
        permute_37: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_20: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg62_1, view_72, permute_37);  arg62_1 = view_72 = permute_37 = None
        view_73: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_20, _shape_param_88);  addmm_20 = _shape_param_88 = None
        view_74: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_73, _shape_param_89);  view_73 = _shape_param_89 = None
        permute_38: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        view_75: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_27, _shape_param_90);  convert_element_type_27 = _shape_param_90 = None
        permute_39: "bf16[2560, 2560]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_21: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(arg64_1, view_75, permute_39);  arg64_1 = view_75 = permute_39 = None
        view_76: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_21, _shape_param_91);  addmm_21 = _shape_param_91 = None
        view_77: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_76, _shape_param_92);  view_76 = _shape_param_92 = None
        permute_40: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
        return (convert_element_type_19, add_24, permute_36, permute_38, permute_40)



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
