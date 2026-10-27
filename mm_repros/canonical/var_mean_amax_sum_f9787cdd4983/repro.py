"""
Standalone repro captured via capture_hook.
Label: BlenderbotForConditionalGeneration
Pattern hash: f9787cdd4983
Shape hash: 3b152dd3
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
    def forward(self, arg0_1: "bf16[16, 32, 128, 80]", arg1_1: "f32[2560]", arg2_1: "f32[2560, 2560]", arg3_1: "f32[16, 128, 2560]", arg4_1: "f32[2560]", arg5_1: "f32[2560]", arg6_1: "f32[2560]", arg7_1: "f32[2560, 2560]", arg8_1: "f32[2560]", arg9_1: "f32[2560, 2560]", arg10_1: "f32[16, 128, 2560]", arg11_1: "f32[2560]", arg12_1: "f32[2560, 2560]", arg13_1: "b8[16, 1, 128, 128]", arg14_1: "bf16[]", arg15_1: "bf16[]", arg16_1: "f32[2560]", arg17_1: "f32[2560, 2560]", arg18_1: "f32[2560]", arg19_1: "f32[2560]", arg20_1: "f32[10240]", arg21_1: "f32[10240, 2560]", arg22_1: "f32[2560]", arg23_1: "f32[2560, 10240]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31):
        # No stacktrace found for following nodes
        permute: "bf16[16, 128, 32, 80]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        convert_element_type: "bf16[2560]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_1: "bf16[2560, 2560]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        view_1: "bf16[2048, 2560]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[2560, 2560]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(convert_element_type, view_1, permute_1);  convert_element_type = view_1 = None
        view_2: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm, _shape_param_2);  _shape_param_2 = None
        inductor_seeds: "i64[3]" = torch.ops.prims.inductor_seeds.default(3, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[16, 128, 2560]" = torch.ops.prims.inductor_random.default(_shape_param_3, inductor_lookup_seed, 'rand');  _shape_param_3 = inductor_lookup_seed = None
        convert_element_type_2: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[16, 128, 2560]" = torch.ops.aten.gt.Scalar(convert_element_type_2, 0.1);  convert_element_type_2 = None
        mul: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(gt, view_2);  view_2 = None
        mul_1: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        add: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(arg3_1, mul_1);  arg3_1 = mul_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 128, 1]" = var_mean[0]
        getitem_1: "f32[16, 128, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add, getitem_1)
        mul_2: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_3: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        add_2: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_3, arg5_1);  mul_3 = arg5_1 = None
        convert_element_type_3: "bf16[2560]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        convert_element_type_4: "bf16[2560, 2560]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        convert_element_type_5: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view_3: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_4);  convert_element_type_5 = _shape_param_4 = None
        permute_2: "bf16[2560, 2560]" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        addmm_1: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(convert_element_type_3, view_3, permute_2);  convert_element_type_3 = None
        view_4: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_1, _shape_param_5);  addmm_1 = _shape_param_5 = None
        view_5: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_4, _shape_param_6);  view_4 = _shape_param_6 = None
        permute_3: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        convert_element_type_6: "bf16[2560]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        convert_element_type_7: "bf16[2560, 2560]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        convert_element_type_8: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        view_6: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_8, _shape_param_7);  convert_element_type_8 = _shape_param_7 = None
        permute_4: "bf16[2560, 2560]" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        addmm_2: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(convert_element_type_6, view_6, permute_4);  convert_element_type_6 = None
        view_7: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_2, _shape_param_8);  addmm_2 = _shape_param_8 = None
        convert_element_type_9: "bf16[2560]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.bfloat16);  arg11_1 = None
        convert_element_type_10: "bf16[2560, 2560]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.bfloat16);  arg12_1 = None
        permute_5: "bf16[2560, 2560]" = torch.ops.aten.permute.default(convert_element_type_10, [1, 0]);  convert_element_type_10 = None
        addmm_3: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(convert_element_type_9, view_6, permute_5);  convert_element_type_9 = None
        view_8: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_3, _shape_param_9);  addmm_3 = _shape_param_9 = None
        view_9: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_7, _shape_param_10);  view_7 = _shape_param_10 = None
        permute_6: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        view_10: "bf16[16, 128, 32, 80]" = torch.ops.aten.view.default(view_8, _shape_param_11);  view_8 = _shape_param_11 = None
        permute_7: "bf16[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        where: "bf16[16, 1, 128, 128]" = torch.ops.aten.where.self(arg13_1, arg14_1, arg15_1);  arg13_1 = arg14_1 = arg15_1 = None
        mul_4: "bf16[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_3, 0.334370152488211);  permute_3 = None
        permute_8: "bf16[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_6, [0, 1, 3, 2]);  permute_6 = None
        mul_5: "bf16[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_8, 0.334370152488211);  permute_8 = None
        expand: "bf16[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_4, _shape_param_12);  mul_4 = _shape_param_12 = None
        clone: "bf16[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_11: "bf16[512, 128, 80]" = torch.ops.aten.view.default(clone, _shape_param_13);  clone = _shape_param_13 = None
        expand_1: "bf16[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_5, _shape_param_14);  mul_5 = _shape_param_14 = None
        clone_1: "bf16[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_12: "bf16[512, 80, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_15);  clone_1 = _shape_param_15 = None
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_11, view_12)
        view_13: "bf16[16, 32, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_16);  bmm = _shape_param_16 = None
        add_3: "bf16[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_13, where);  view_13 = where = None
        convert_element_type_11: "f32[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(add_3, torch.float32)
        amax: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_11, [-1], True)
        sub_1: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_11, amax);  convert_element_type_11 = amax = None
        exp: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_12: "bf16[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        eq: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_3, -inf);  add_3 = None
        logical_not: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full: "bf16[16, 32, 128, 128]" = torch.ops.aten.full.default(_shape_param_17, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_17 = None
        where_1: "bf16[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full, convert_element_type_12);  logical_not_1 = full = convert_element_type_12 = None
        expand_2: "bf16[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_1, _shape_param_18);  _shape_param_18 = None
        view_14: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_2, _shape_param_19);  expand_2 = _shape_param_19 = None
        expand_3: "bf16[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_7, _shape_param_20);  permute_7 = _shape_param_20 = None
        clone_2: "bf16[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_15: "bf16[512, 128, 80]" = torch.ops.aten.view.default(clone_2, _shape_param_21);  clone_2 = _shape_param_21 = None
        bmm_1: "bf16[512, 128, 80]" = torch.ops.aten.bmm.default(view_14, view_15);  view_14 = None
        view_16: "bf16[16, 32, 128, 80]" = torch.ops.aten.view.default(bmm_1, _shape_param_22);  bmm_1 = _shape_param_22 = None
        permute_9: "bf16[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        clone_3: "bf16[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
        view_17: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(clone_3, _shape_param_23);  clone_3 = _shape_param_23 = None
        convert_element_type_13: "bf16[2560]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.bfloat16);  arg16_1 = None
        convert_element_type_14: "bf16[2560, 2560]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        view_18: "bf16[2048, 2560]" = torch.ops.aten.view.default(view_17, _shape_param_24);  view_17 = _shape_param_24 = None
        permute_10: "bf16[2560, 2560]" = torch.ops.aten.permute.default(convert_element_type_14, [1, 0]);  convert_element_type_14 = None
        addmm_4: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(convert_element_type_13, view_18, permute_10);  convert_element_type_13 = None
        view_19: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_4, _shape_param_25);  addmm_4 = _shape_param_25 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1)
        inductor_random_1: "f32[16, 128, 2560]" = torch.ops.prims.inductor_random.default(_shape_param_26, inductor_lookup_seed_1, 'rand');  _shape_param_26 = inductor_lookup_seed_1 = None
        convert_element_type_15: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(inductor_random_1, torch.bfloat16);  inductor_random_1 = None
        gt_1: "b8[16, 128, 2560]" = torch.ops.aten.gt.Scalar(convert_element_type_15, 0.1);  convert_element_type_15 = None
        mul_6: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(gt_1, view_19);  view_19 = None
        mul_7: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_6, 1.1111111111111112);  mul_6 = None
        add_4: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add, mul_7);  add = mul_7 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_2: "f32[16, 128, 1]" = var_mean_1[0]
        getitem_3: "f32[16, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_2: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_4, getitem_3);  getitem_3 = None
        mul_8: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_9: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_8, arg18_1);  arg18_1 = None
        add_6: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_9, arg19_1);  mul_9 = arg19_1 = None
        convert_element_type_16: "bf16[10240]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.bfloat16);  arg20_1 = None
        convert_element_type_17: "bf16[10240, 2560]" = torch.ops.prims.convert_element_type.default(arg21_1, torch.bfloat16);  arg21_1 = None
        convert_element_type_18: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        view_20: "bf16[2048, 2560]" = torch.ops.aten.view.default(convert_element_type_18, _shape_param_27);  convert_element_type_18 = _shape_param_27 = None
        permute_11: "bf16[2560, 10240]" = torch.ops.aten.permute.default(convert_element_type_17, [1, 0]);  convert_element_type_17 = None
        addmm_5: "bf16[2048, 10240]" = torch.ops.aten.addmm.default(convert_element_type_16, view_20, permute_11);  convert_element_type_16 = None
        view_21: "bf16[16, 128, 10240]" = torch.ops.aten.view.default(addmm_5, _shape_param_28);  _shape_param_28 = None
        convert_element_type_19: "f32[16, 128, 10240]" = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None
        mul_10: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 0.5)
        mul_11: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 0.7071067811865476);  convert_element_type_19 = None
        erf: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
        add_7: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_12: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_10, add_7);  mul_10 = add_7 = None
        convert_element_type_20: "bf16[16, 128, 10240]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        convert_element_type_21: "bf16[2560]" = torch.ops.prims.convert_element_type.default(arg22_1, torch.bfloat16);  arg22_1 = None
        convert_element_type_22: "bf16[2560, 10240]" = torch.ops.prims.convert_element_type.default(arg23_1, torch.bfloat16);  arg23_1 = None
        view_22: "bf16[2048, 10240]" = torch.ops.aten.view.default(convert_element_type_20, _shape_param_29);  convert_element_type_20 = _shape_param_29 = None
        permute_12: "bf16[10240, 2560]" = torch.ops.aten.permute.default(convert_element_type_22, [1, 0]);  convert_element_type_22 = None
        addmm_6: "bf16[2048, 2560]" = torch.ops.aten.addmm.default(convert_element_type_21, view_22, permute_12);  convert_element_type_21 = None
        view_23: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(addmm_6, _shape_param_30);  addmm_6 = _shape_param_30 = None
        inductor_lookup_seed_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 2);  inductor_seeds = None
        inductor_random_2: "f32[16, 128, 2560]" = torch.ops.prims.inductor_random.default(_shape_param_31, inductor_lookup_seed_2, 'rand');  _shape_param_31 = inductor_lookup_seed_2 = None
        convert_element_type_23: "bf16[16, 128, 2560]" = torch.ops.prims.convert_element_type.default(inductor_random_2, torch.bfloat16);  inductor_random_2 = None
        gt_2: "b8[16, 128, 2560]" = torch.ops.aten.gt.Scalar(convert_element_type_23, 0.1);  convert_element_type_23 = None
        mul_13: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(gt_2, view_23);  view_23 = None
        mul_14: "bf16[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_13, 1.1111111111111112);  mul_13 = None
        add_8: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_4, mul_14);  add_4 = mul_14 = None
        permute_13: "bf16[2560, 10240]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_14: "bf16[10240, 2560]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        div_1: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 2560);  rsqrt_1 = None
        permute_15: "bf16[2560, 2560]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_16: "bf16[512, 80, 128]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        permute_17: "bf16[512, 80, 128]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        permute_18: "bf16[512, 128, 80]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_19: "bf16[2560, 2560]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        permute_20: "bf16[2560, 2560]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        permute_21: "bf16[2560, 2560]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_22: "bf16[2560, 2560]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (addmm, gt, getitem_1, rsqrt, view_3, view_6, where_1, view_18, gt_1, mul_8, view_20, addmm_5, view_22, gt_2, add_8, permute_13, permute_14, div_1, permute_15, permute_16, permute_17, permute_18, permute_19, permute_20, permute_21, permute_22)



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
