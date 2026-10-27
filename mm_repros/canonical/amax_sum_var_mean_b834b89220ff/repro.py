"""
Standalone repro captured via capture_hook.
Label: YituTechConvBert
Pattern hash: b834b89220ff
Shape hash: 718d9f31
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
    def forward(self, arg0_1: "bf16[32, 384, 512]", arg1_1: "f32[384, 1]", arg2_1: "bf16[32, 512, 384]", arg3_1: "f32[54]", arg4_1: "f32[54, 384]", arg5_1: "f32[384]", arg6_1: "f32[384, 768]", arg7_1: "bf16[16384, 768]", arg8_1: "i64[9, 512, 1, 1]", arg9_1: "i64[1, 1]", arg10_1: "bf16[32, 6, 512, 64]", arg11_1: "f32[768]", arg12_1: "f32[768, 768]", arg13_1: "i64[25]", arg14_1: "f32[32, 512, 768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[3072]", arg18_1: "f32[3072, 768]", arg19_1: "f32[768]", arg20_1: "f32[768, 3072]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[768, 768]", arg25_1: "f32[768]", arg26_1: "f32[768]", arg27_1: "f32[30522]", arg28_1: "f32[30522, 768]", arg29_1: "i64[32, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27):
        # No stacktrace found for following nodes
        add: "f32[32, 384, 512]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "bf16[32, 384, 512]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        permute: "bf16[32, 512, 384]" = torch.ops.aten.permute.default(convert_element_type, [0, 2, 1]);  convert_element_type = None
        mul: "bf16[32, 512, 384]" = torch.ops.aten.mul.Tensor(permute, arg2_1);  permute = arg2_1 = None
        convert_element_type_1: "bf16[54]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_2: "bf16[54, 384]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        permute_1: "bf16[384, 54]" = torch.ops.aten.permute.default(convert_element_type_2, [1, 0]);  convert_element_type_2 = None
        clone: "bf16[32, 512, 384]" = torch.ops.aten.clone.default(mul, memory_format = torch.contiguous_format);  mul = None
        view: "bf16[16384, 384]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        mm: "bf16[16384, 54]" = torch.ops.aten.mm.default(view, permute_1)
        view_1: "bf16[32, 512, 54]" = torch.ops.aten.view.default(mm, _shape_param_1);  _shape_param_1 = None
        add_1: "bf16[32, 512, 54]" = torch.ops.aten.add.Tensor(view_1, convert_element_type_1);  view_1 = convert_element_type_1 = None
        view_2: "bf16[98304, 9, 1]" = torch.ops.aten.view.default(add_1, _shape_param_2);  add_1 = _shape_param_2 = None
        convert_element_type_3: "f32[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        amax: "f32[98304, 1, 1]" = torch.ops.aten.amax.default(convert_element_type_3, [1], True)
        sub: "f32[98304, 9, 1]" = torch.ops.aten.sub.Tensor(convert_element_type_3, amax);  convert_element_type_3 = None
        exp: "f32[98304, 9, 1]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[98304, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
        div: "f32[98304, 9, 1]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_4: "bf16[384]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_5: "bf16[384, 768]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        permute_2: "bf16[768, 384]" = torch.ops.aten.permute.default(convert_element_type_5, [1, 0]);  convert_element_type_5 = None
        addmm: "bf16[16384, 384]" = torch.ops.aten.addmm.default(convert_element_type_4, arg7_1, permute_2);  convert_element_type_4 = arg7_1 = None
        view_3: "bf16[32, 512, 384]" = torch.ops.aten.view.default(addmm, _shape_param_3);  addmm = _shape_param_3 = None
        permute_3: "bf16[32, 384, 512]" = torch.ops.aten.permute.default(view_3, [0, 2, 1]);  view_3 = None
        clone_1: "bf16[32, 384, 512]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        unsqueeze: "bf16[32, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_1, -1);  clone_1 = None
        constant_pad_nd: "bf16[32, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze, _shape_param_4, 0.0);  unsqueeze = _shape_param_4 = None
        index: "bf16[32, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd, [None, None, arg8_1, arg9_1]);  constant_pad_nd = arg8_1 = arg9_1 = None
        permute_4: "bf16[32, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index, [0, 1, 2, 4, 3, 5]);  index = None
        view_4: "bf16[32, 3456, 512]" = torch.ops.aten.view.default(permute_4, _shape_param_5);  permute_4 = _shape_param_5 = None
        permute_5: "bf16[32, 512, 3456]" = torch.ops.aten.permute.default(view_4, [0, 2, 1]);  view_4 = None
        view_5: "bf16[32, 512, 384, 9]" = torch.ops.aten.view.default(permute_5, _shape_param_6);  permute_5 = _shape_param_6 = None
        clone_2: "bf16[32, 512, 384, 9]" = torch.ops.aten.clone.default(view_5, memory_format = torch.contiguous_format);  view_5 = None
        view_6: "bf16[98304, 64, 9]" = torch.ops.aten.view.default(clone_2, _shape_param_7);  clone_2 = _shape_param_7 = None
        convert_element_type_6: "bf16[98304, 9, 1]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand: "bf16[98304, 64, 9]" = torch.ops.aten.expand.default(view_6, _shape_param_8);  view_6 = _shape_param_8 = None
        expand_1: "bf16[98304, 9, 1]" = torch.ops.aten.expand.default(convert_element_type_6, _shape_param_9);  convert_element_type_6 = _shape_param_9 = None
        bmm: "bf16[98304, 64, 1]" = torch.ops.aten.bmm.default(expand, expand_1);  expand_1 = None
        view_7: "bf16[16384, 384]" = torch.ops.aten.view.default(bmm, _shape_param_10);  bmm = _shape_param_10 = None
        permute_6: "bf16[32, 512, 6, 64]" = torch.ops.aten.permute.default(arg10_1, [0, 2, 1, 3]);  arg10_1 = None
        clone_3: "bf16[32, 512, 6, 64]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_8: "bf16[32, 512, 6, 64]" = torch.ops.aten.view.default(view_7, _shape_param_11);  view_7 = _shape_param_11 = None
        cat: "bf16[32, 512, 12, 64]" = torch.ops.aten.cat.default([clone_3, view_8], 2);  clone_3 = view_8 = None
        view_9: "bf16[32, 512, 768]" = torch.ops.aten.view.default(cat, _shape_param_12);  cat = _shape_param_12 = None
        convert_element_type_7: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.bfloat16);  arg11_1 = None
        convert_element_type_8: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.bfloat16);  arg12_1 = None
        view_10: "bf16[16384, 768]" = torch.ops.aten.view.default(view_9, _shape_param_13);  view_9 = _shape_param_13 = None
        permute_7: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_8, [1, 0]);  convert_element_type_8 = None
        addmm_1: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_7, view_10, permute_7);  convert_element_type_7 = None
        view_11: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_1, _shape_param_14);  addmm_1 = _shape_param_14 = None
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg13_1, 23)
        inductor_random: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_15, inductor_lookup_seed, 'rand');  _shape_param_15 = inductor_lookup_seed = None
        convert_element_type_9: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_9, 0.1);  convert_element_type_9 = None
        mul_1: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt, view_11);  view_11 = None
        mul_2: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, 1.1111111111111112);  mul_1 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, arg14_1);  mul_2 = arg14_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        add_3: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_2, getitem_1);  add_2 = getitem_1 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, arg15_1);  arg15_1 = None
        add_4: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_4, arg16_1);  mul_4 = arg16_1 = None
        convert_element_type_10: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        convert_element_type_11: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.bfloat16);  arg18_1 = None
        convert_element_type_12: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16)
        view_12: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_12, _shape_param_16);  convert_element_type_12 = _shape_param_16 = None
        permute_8: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_11, [1, 0]);  convert_element_type_11 = None
        addmm_2: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(convert_element_type_10, view_12, permute_8);  convert_element_type_10 = None
        view_13: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_2, _shape_param_17);  _shape_param_17 = None
        convert_element_type_13: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        mul_5: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 0.5)
        mul_6: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 0.7071067811865476);  convert_element_type_13 = None
        erf: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_5: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_7: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_5, add_5);  mul_5 = add_5 = None
        convert_element_type_14: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        convert_element_type_15: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg19_1, torch.bfloat16);  arg19_1 = None
        convert_element_type_16: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.bfloat16);  arg20_1 = None
        view_14: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_14, _shape_param_18);  convert_element_type_14 = _shape_param_18 = None
        permute_9: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_16, [1, 0]);  convert_element_type_16 = None
        addmm_3: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_15, view_14, permute_9);  convert_element_type_15 = None
        view_15: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_3, _shape_param_19);  addmm_3 = _shape_param_19 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(arg13_1, 24);  arg13_1 = None
        inductor_random_1: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_20, inductor_lookup_seed_1, 'rand');  _shape_param_20 = inductor_lookup_seed_1 = None
        convert_element_type_17: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_1, torch.bfloat16);  inductor_random_1 = None
        gt_1: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_17, 0.1);  convert_element_type_17 = None
        mul_8: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_1, view_15);  view_15 = None
        mul_9: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, 1.1111111111111112);  mul_8 = None
        add_6: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_9, add_4);  mul_9 = add_4 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_7: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_6, getitem_3);  add_6 = getitem_3 = None
        mul_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_11: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, arg21_1);  arg21_1 = None
        add_8: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_11, arg22_1);  mul_11 = arg22_1 = None
        convert_element_type_18: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg23_1, torch.bfloat16);  arg23_1 = None
        convert_element_type_19: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg24_1, torch.bfloat16);  arg24_1 = None
        convert_element_type_20: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None
        view_16: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_20, _shape_param_21);  convert_element_type_20 = _shape_param_21 = None
        permute_10: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_19, [1, 0]);  convert_element_type_19 = None
        addmm_4: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_18, view_16, permute_10);  convert_element_type_18 = None
        view_17: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_4, _shape_param_22);  _shape_param_22 = None
        convert_element_type_21: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None
        mul_12: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.5)
        mul_13: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.7071067811865476);  convert_element_type_21 = None
        erf_1: "f32[32, 512, 768]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_9: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_14: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_12, add_9);  mul_12 = add_9 = None
        convert_element_type_22: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        convert_element_type_23: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_22, torch.float32);  convert_element_type_22 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_23, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[32, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_10: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_23, getitem_5);  convert_element_type_23 = None
        mul_15: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_16: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_15, arg25_1);  mul_15 = arg25_1 = None
        add_11: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_16, arg26_1);  mul_16 = arg26_1 = None
        convert_element_type_24: "bf16[30522]" = torch.ops.prims.convert_element_type.default(arg27_1, torch.bfloat16);  arg27_1 = None
        convert_element_type_25: "bf16[30522, 768]" = torch.ops.prims.convert_element_type.default(arg28_1, torch.bfloat16);  arg28_1 = None
        convert_element_type_26: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None
        view_18: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_26, _shape_param_23);  convert_element_type_26 = _shape_param_23 = None
        permute_11: "bf16[768, 30522]" = torch.ops.aten.permute.default(convert_element_type_25, [1, 0]);  convert_element_type_25 = None
        constant_pad_nd_1: "bf16[768, 30528]" = torch.ops.aten.constant_pad_nd.default(permute_11, _shape_param_24);  _shape_param_24 = None
        full: "bf16[6]" = torch.ops.aten.full.default(_shape_param_25, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_25 = None
        cat_1: "bf16[30528]" = torch.ops.aten.cat.default([convert_element_type_24, full]);  convert_element_type_24 = full = None
        addmm_5: "bf16[16384, 30528]" = torch.ops.aten.addmm.default(cat_1, view_18, constant_pad_nd_1);  cat_1 = constant_pad_nd_1 = None
        slice_1: "bf16[16384, 30522]" = torch.ops.aten.slice.Tensor(addmm_5, 1, 0, -6);  addmm_5 = None
        view_19: "bf16[32, 512, 30522]" = torch.ops.aten.view.default(slice_1, _shape_param_26);  slice_1 = _shape_param_26 = None
        view_20: "bf16[16384, 30522]" = torch.ops.aten.view.default(view_19, _shape_param_27);  _shape_param_27 = None
        view_21: "i64[16384]" = torch.ops.aten.view.default(arg29_1, [-1]);  arg29_1 = None
        convert_element_type_27: "f32[16384, 30522]" = torch.ops.prims.convert_element_type.default(view_20, torch.float32);  view_20 = None
        amax_1: "f32[16384, 1]" = torch.ops.aten.amax.default(convert_element_type_27, [1], True)
        sub_4: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(convert_element_type_27, amax_1);  convert_element_type_27 = None
        exp_1: "f32[16384, 30522]" = torch.ops.aten.exp.default(sub_4)
        sum_2: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [1], True);  exp_1 = None
        log: "f32[16384, 1]" = torch.ops.aten.log.default(sum_2);  sum_2 = None
        sub_5: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(sub_4, log);  sub_4 = None
        convert_element_type_28: "bf16[16384, 30522]" = torch.ops.prims.convert_element_type.default(sub_5, torch.bfloat16);  sub_5 = None
        convert_element_type_29: "f32[16384, 30522]" = torch.ops.prims.convert_element_type.default(convert_element_type_28, torch.float32);  convert_element_type_28 = None
        ne: "b8[16384]" = torch.ops.aten.ne.Scalar(view_21, -100)
        full_1: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[16384]" = torch.ops.aten.where.self(ne, view_21, full_1);  view_21 = full_1 = None
        unsqueeze_1: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[16384, 1]" = torch.ops.aten.gather.default(convert_element_type_29, 1, unsqueeze_1);  convert_element_type_29 = unsqueeze_1 = None
        squeeze: "f32[16384]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[16384]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[16384]" = torch.ops.aten.where.self(ne, neg, full_2);  neg = full_2 = None
        sum_3: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_30: "f32[]" = torch.ops.prims.convert_element_type.default(sum_3, torch.float32);  sum_3 = None
        sum_4: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div_1: "f32[]" = torch.ops.aten.div.Tensor(sum_4, convert_element_type_30);  sum_4 = None
        permute_12: "bf16[30522, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        permute_13: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        div_2: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        permute_14: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        permute_15: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        div_3: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        permute_16: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        permute_17: "bf16[98304, 9, 64]" = torch.ops.aten.permute.default(expand, [0, 2, 1]);  expand = None
        permute_18: "bf16[384, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_19: "bf16[54, 384]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (view, mm, amax, sum_1, view_10, gt, mul_3, view_12, addmm_2, view_14, gt_1, mul_10, view_16, addmm_4, getitem_5, rsqrt_2, view_18, view_19, amax_1, log, convert_element_type_30, div_1, permute_12, permute_13, div_2, permute_14, permute_15, div_3, permute_16, permute_17, permute_18, permute_19)



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
