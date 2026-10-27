"""
Standalone repro captured via capture_hook.
Label: DistilBertForMaskedLM
Pattern hash: 5f7406fd8702
Shape hash: 8f29aa96
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
    def forward(self, arg0_1: "bf16[30522, 768]", arg1_1: "i64[256, 128]", arg2_1: "i64[1, 512]", arg3_1: "bf16[512, 768]", arg4_1: "bf16[768]", arg5_1: "bf16[768]", arg6_1: "bf16[768, 768]", arg7_1: "bf16[768]", arg8_1: "bf16[768, 768]", arg9_1: "bf16[768]", arg10_1: "bf16[768, 768]", arg11_1: "bf16[768]", arg12_1: "bf16[768, 768]", arg13_1: "bf16[768]", arg14_1: "bf16[768]", arg15_1: "bf16[768]", arg16_1: "bf16[3072, 768]", arg17_1: "bf16[3072]", arg18_1: "bf16[768, 3072]", arg19_1: "bf16[768]", arg20_1: "bf16[768]", arg21_1: "bf16[768]", arg22_1: "bf16[768, 768]", arg23_1: "bf16[768]", arg24_1: "bf16[768, 768]", arg25_1: "bf16[768]", arg26_1: "bf16[768, 768]", arg27_1: "bf16[768]", arg28_1: "bf16[768, 768]", arg29_1: "bf16[768]", arg30_1: "bf16[768]", arg31_1: "bf16[768]", arg32_1: "bf16[3072, 768]", arg33_1: "bf16[3072]", arg34_1: "bf16[768, 3072]", arg35_1: "bf16[768]", arg36_1: "bf16[768]", arg37_1: "bf16[768]", arg38_1: "bf16[768, 768]", arg39_1: "bf16[768]", arg40_1: "bf16[768, 768]", arg41_1: "bf16[768]", arg42_1: "bf16[768, 768]", arg43_1: "bf16[768]", arg44_1: "bf16[768, 768]", arg45_1: "bf16[768]", arg46_1: "bf16[768]", arg47_1: "bf16[768]", arg48_1: "bf16[3072, 768]", arg49_1: "bf16[3072]", arg50_1: "bf16[768, 3072]", arg51_1: "bf16[768]", arg52_1: "bf16[768]", arg53_1: "bf16[768]", arg54_1: "bf16[768, 768]", arg55_1: "bf16[768]", arg56_1: "bf16[768, 768]", arg57_1: "bf16[768]", arg58_1: "bf16[768, 768]", arg59_1: "bf16[768]", arg60_1: "bf16[768, 768]", arg61_1: "bf16[768]", arg62_1: "bf16[768]", arg63_1: "bf16[768]", arg64_1: "bf16[3072, 768]", arg65_1: "bf16[3072]", arg66_1: "bf16[768, 3072]", arg67_1: "bf16[768]", arg68_1: "bf16[768]", arg69_1: "bf16[768]", arg70_1: "bf16[768, 768]", arg71_1: "bf16[768]", arg72_1: "bf16[768, 768]", arg73_1: "bf16[768]", arg74_1: "bf16[768, 768]", arg75_1: "bf16[768]", arg76_1: "bf16[768, 768]", arg77_1: "bf16[768]", arg78_1: "bf16[768]", arg79_1: "bf16[768]", arg80_1: "bf16[3072, 768]", arg81_1: "bf16[3072]", arg82_1: "bf16[768, 3072]", arg83_1: "bf16[768]", arg84_1: "bf16[768]", arg85_1: "bf16[768]", arg86_1: "bf16[768, 768]", arg87_1: "bf16[768]", arg88_1: "bf16[768, 768]", arg89_1: "bf16[768]", arg90_1: "bf16[768, 768]", arg91_1: "bf16[768]", arg92_1: "bf16[768, 768]", arg93_1: "bf16[768]", arg94_1: "bf16[768]", arg95_1: "bf16[768]", arg96_1: "bf16[3072, 768]", arg97_1: "bf16[3072]", arg98_1: "bf16[768, 3072]", arg99_1: "bf16[768]", arg100_1: "bf16[768]", arg101_1: "bf16[768]", arg102_1: "bf16[768, 768]", arg103_1: "bf16[768]", arg104_1: "i64[256, 128]", arg105_1: "bf16[30522]", arg106_1: "bf16[768]", arg107_1: "bf16[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168):
        # No stacktrace found for following nodes
        embedding: "bf16[256, 128, 768]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg1_1 = None
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 128);  arg2_1 = None
        embedding_1: "bf16[1, 128, 768]" = torch.ops.aten.embedding.default(arg3_1, slice_1);  arg3_1 = slice_1 = None
        add: "bf16[256, 128, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        convert_element_type: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[256, 128, 1]" = var_mean[0]
        getitem_1: "f32[256, 128, 1]" = var_mean[1];  var_mean = None
        sub: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type_1: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  _shape_param_0 = None
        permute: "bf16[768, 768]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg7_1, view, permute);  arg7_1 = view = permute = None
        view_1: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        view_2: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute_1: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        mul_2: "bf16[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(permute_1, 0.3535533905932738);  permute_1 = None
        expand: "bf16[256, 12, 128, 64]" = torch.ops.aten.expand.default(mul_2, _shape_param_3);  mul_2 = _shape_param_3 = None
        clone: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_3: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        view_4: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_5);  _shape_param_5 = None
        permute_2: "bf16[768, 768]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg9_1, view_4, permute_2);  arg9_1 = view_4 = permute_2 = None
        view_5: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_1, _shape_param_6);  addmm_1 = _shape_param_6 = None
        view_6: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_5, _shape_param_7);  view_5 = _shape_param_7 = None
        permute_3: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_4: "bf16[256, 12, 64, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        mul_3: "bf16[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(permute_4, 0.3535533905932738);  permute_4 = None
        expand_1: "bf16[256, 12, 64, 128]" = torch.ops.aten.expand.default(mul_3, _shape_param_8);  mul_3 = _shape_param_8 = None
        clone_1: "bf16[256, 12, 64, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_7: "bf16[3072, 64, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_9);  clone_1 = _shape_param_9 = None
        bmm: "bf16[3072, 128, 128]" = torch.ops.aten.bmm.default(view_3, view_7);  view_3 = view_7 = None
        view_8: "bf16[256, 12, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_10);  bmm = _shape_param_10 = None
        eq: "b8[256, 12, 128, 128]" = torch.ops.aten.eq.Scalar(view_8, -inf)
        logical_not: "b8[256, 12, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[256, 12, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[256, 12, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full: "bf16[256, 12, 128, 128]" = torch.ops.aten.full.default(_shape_param_11, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_11 = None
        convert_element_type_2: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        amax: "f32[256, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
        sub_1: "f32[256, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[256, 12, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[256, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        where: "bf16[256, 12, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full, convert_element_type_3);  logical_not_1 = full = convert_element_type_3 = None
        expand_2: "bf16[256, 12, 128, 128]" = torch.ops.aten.expand.default(where, _shape_param_12);  where = _shape_param_12 = None
        view_9: "bf16[3072, 128, 128]" = torch.ops.aten.view.default(expand_2, _shape_param_13);  expand_2 = _shape_param_13 = None
        view_10: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_14);  _shape_param_14 = None
        permute_5: "bf16[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_2: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg11_1, view_10, permute_5);  arg11_1 = view_10 = permute_5 = None
        view_11: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_2, _shape_param_15);  addmm_2 = _shape_param_15 = None
        view_12: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_11, _shape_param_16);  view_11 = _shape_param_16 = None
        permute_6: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        expand_3: "bf16[256, 12, 128, 64]" = torch.ops.aten.expand.default(permute_6, _shape_param_17);  permute_6 = _shape_param_17 = None
        clone_2: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_13: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_18);  clone_2 = _shape_param_18 = None
        bmm_1: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(view_9, view_13);  view_9 = view_13 = None
        view_14: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_19);  bmm_1 = _shape_param_19 = None
        permute_7: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_3: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_15: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_3, _shape_param_20);  clone_3 = _shape_param_20 = None
        view_16: "bf16[32768, 768]" = torch.ops.aten.view.default(view_15, _shape_param_21);  view_15 = _shape_param_21 = None
        permute_8: "bf16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_3: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg13_1, view_16, permute_8);  arg13_1 = view_16 = permute_8 = None
        view_17: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_3, _shape_param_22);  addmm_3 = _shape_param_22 = None
        add_3: "bf16[256, 128, 768]" = torch.ops.aten.add.Tensor(view_17, convert_element_type_1);  view_17 = convert_element_type_1 = None
        convert_element_type_4: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_3, torch.float32);  add_3 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_4, [2], correction = 0, keepdim = True)
        getitem_2: "f32[256, 128, 1]" = var_mean_1[0]
        getitem_3: "f32[256, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_4, getitem_3);  convert_element_type_4 = getitem_3 = None
        add_4: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_4: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_5: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_4, arg14_1);  mul_4 = arg14_1 = None
        add_5: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_5, arg15_1);  mul_5 = arg15_1 = None
        convert_element_type_5: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        view_18: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_23);  _shape_param_23 = None
        permute_9: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_4: "bf16[32768, 3072]" = torch.ops.aten.addmm.default(arg17_1, view_18, permute_9);  arg17_1 = view_18 = permute_9 = None
        view_19: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(addmm_4, _shape_param_24);  addmm_4 = _shape_param_24 = None
        convert_element_type_6: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_6: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.5)
        mul_7: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476);  convert_element_type_6 = None
        erf: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_6: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_6);  mul_6 = add_6 = None
        convert_element_type_7: "bf16[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        view_20: "bf16[32768, 3072]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_25);  convert_element_type_7 = _shape_param_25 = None
        permute_10: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_5: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg19_1, view_20, permute_10);  arg19_1 = view_20 = permute_10 = None
        view_21: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_5, _shape_param_26);  addmm_5 = _shape_param_26 = None
        add_7: "bf16[256, 128, 768]" = torch.ops.aten.add.Tensor(view_21, convert_element_type_5);  view_21 = convert_element_type_5 = None
        convert_element_type_8: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_7, torch.float32);  add_7 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_8, [2], correction = 0, keepdim = True)
        getitem_4: "f32[256, 128, 1]" = var_mean_2[0]
        getitem_5: "f32[256, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_8, getitem_5);  convert_element_type_8 = getitem_5 = None
        add_8: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_9: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_10: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg20_1);  mul_9 = arg20_1 = None
        add_9: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_10, arg21_1);  mul_10 = arg21_1 = None
        convert_element_type_9: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        view_22: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_27);  _shape_param_27 = None
        permute_11: "bf16[768, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_6: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg23_1, view_22, permute_11);  arg23_1 = view_22 = permute_11 = None
        view_23: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_6, _shape_param_28);  addmm_6 = _shape_param_28 = None
        view_24: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_23, _shape_param_29);  view_23 = _shape_param_29 = None
        permute_12: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        mul_11: "bf16[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(permute_12, 0.3535533905932738);  permute_12 = None
        expand_4: "bf16[256, 12, 128, 64]" = torch.ops.aten.expand.default(mul_11, _shape_param_30);  mul_11 = _shape_param_30 = None
        clone_4: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_25: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_31);  clone_4 = _shape_param_31 = None
        view_26: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_32);  _shape_param_32 = None
        permute_13: "bf16[768, 768]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_7: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg25_1, view_26, permute_13);  arg25_1 = view_26 = permute_13 = None
        view_27: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_7, _shape_param_33);  addmm_7 = _shape_param_33 = None
        view_28: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_27, _shape_param_34);  view_27 = _shape_param_34 = None
        permute_14: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        permute_15: "bf16[256, 12, 64, 128]" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
        mul_12: "bf16[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(permute_15, 0.3535533905932738);  permute_15 = None
        expand_5: "bf16[256, 12, 64, 128]" = torch.ops.aten.expand.default(mul_12, _shape_param_35);  mul_12 = _shape_param_35 = None
        clone_5: "bf16[256, 12, 64, 128]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_29: "bf16[3072, 64, 128]" = torch.ops.aten.view.default(clone_5, _shape_param_36);  clone_5 = _shape_param_36 = None
        bmm_2: "bf16[3072, 128, 128]" = torch.ops.aten.bmm.default(view_25, view_29);  view_25 = view_29 = None
        view_30: "bf16[256, 12, 128, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_37);  bmm_2 = _shape_param_37 = None
        eq_1: "b8[256, 12, 128, 128]" = torch.ops.aten.eq.Scalar(view_30, -inf)
        logical_not_2: "b8[256, 12, 128, 128]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[256, 12, 128, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[256, 12, 128, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        full_1: "bf16[256, 12, 128, 128]" = torch.ops.aten.full.default(_shape_param_38, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_38 = None
        convert_element_type_10: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None
        amax_1: "f32[256, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_10, [-1], True)
        sub_4: "f32[256, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_10, amax_1);  convert_element_type_10 = amax_1 = None
        exp_1: "f32[256, 12, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[256, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_11: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        where_1: "bf16[256, 12, 128, 128]" = torch.ops.aten.where.self(logical_not_3, full_1, convert_element_type_11);  logical_not_3 = full_1 = convert_element_type_11 = None
        expand_6: "bf16[256, 12, 128, 128]" = torch.ops.aten.expand.default(where_1, _shape_param_39);  where_1 = _shape_param_39 = None
        view_31: "bf16[3072, 128, 128]" = torch.ops.aten.view.default(expand_6, _shape_param_40);  expand_6 = _shape_param_40 = None
        view_32: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_41);  _shape_param_41 = None
        permute_16: "bf16[768, 768]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_8: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg27_1, view_32, permute_16);  arg27_1 = view_32 = permute_16 = None
        view_33: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_8, _shape_param_42);  addmm_8 = _shape_param_42 = None
        view_34: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_33, _shape_param_43);  view_33 = _shape_param_43 = None
        permute_17: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        expand_7: "bf16[256, 12, 128, 64]" = torch.ops.aten.expand.default(permute_17, _shape_param_44);  permute_17 = _shape_param_44 = None
        clone_6: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_35: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone_6, _shape_param_45);  clone_6 = _shape_param_45 = None
        bmm_3: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(view_31, view_35);  view_31 = view_35 = None
        view_36: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_46);  bmm_3 = _shape_param_46 = None
        permute_18: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_7: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_37: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_7, _shape_param_47);  clone_7 = _shape_param_47 = None
        view_38: "bf16[32768, 768]" = torch.ops.aten.view.default(view_37, _shape_param_48);  view_37 = _shape_param_48 = None
        permute_19: "bf16[768, 768]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_9: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg29_1, view_38, permute_19);  arg29_1 = view_38 = permute_19 = None
        view_39: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_9, _shape_param_49);  addmm_9 = _shape_param_49 = None
        add_10: "bf16[256, 128, 768]" = torch.ops.aten.add.Tensor(view_39, convert_element_type_9);  view_39 = convert_element_type_9 = None
        convert_element_type_12: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_10, torch.float32);  add_10 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_12, [2], correction = 0, keepdim = True)
        getitem_6: "f32[256, 128, 1]" = var_mean_3[0]
        getitem_7: "f32[256, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_12, getitem_7);  convert_element_type_12 = getitem_7 = None
        add_11: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_13: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_14: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_13, arg30_1);  mul_13 = arg30_1 = None
        add_12: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_14, arg31_1);  mul_14 = arg31_1 = None
        convert_element_type_13: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16);  add_12 = None
        view_40: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_13, _shape_param_50);  _shape_param_50 = None
        permute_20: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_10: "bf16[32768, 3072]" = torch.ops.aten.addmm.default(arg33_1, view_40, permute_20);  arg33_1 = view_40 = permute_20 = None
        view_41: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(addmm_10, _shape_param_51);  addmm_10 = _shape_param_51 = None
        convert_element_type_14: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_15: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.5)
        mul_16: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.7071067811865476);  convert_element_type_14 = None
        erf_1: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_16);  mul_16 = None
        add_13: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_17: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_15, add_13);  mul_15 = add_13 = None
        convert_element_type_15: "bf16[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        view_42: "bf16[32768, 3072]" = torch.ops.aten.view.default(convert_element_type_15, _shape_param_52);  convert_element_type_15 = _shape_param_52 = None
        permute_21: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_11: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg35_1, view_42, permute_21);  arg35_1 = view_42 = permute_21 = None
        view_43: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_11, _shape_param_53);  addmm_11 = _shape_param_53 = None
        add_14: "bf16[256, 128, 768]" = torch.ops.aten.add.Tensor(view_43, convert_element_type_13);  view_43 = convert_element_type_13 = None
        convert_element_type_16: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_14, torch.float32);  add_14 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_16, [2], correction = 0, keepdim = True)
        getitem_8: "f32[256, 128, 1]" = var_mean_4[0]
        getitem_9: "f32[256, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_6: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_16, getitem_9);  convert_element_type_16 = getitem_9 = None
        add_15: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_18: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_19: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg36_1);  mul_18 = arg36_1 = None
        add_16: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_19, arg37_1);  mul_19 = arg37_1 = None
        convert_element_type_17: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16);  add_16 = None
        view_44: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_17, _shape_param_54);  _shape_param_54 = None
        permute_22: "bf16[768, 768]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        addmm_12: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg39_1, view_44, permute_22);  arg39_1 = view_44 = permute_22 = None
        view_45: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_12, _shape_param_55);  addmm_12 = _shape_param_55 = None
        view_46: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_45, _shape_param_56);  view_45 = _shape_param_56 = None
        permute_23: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        mul_20: "bf16[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(permute_23, 0.3535533905932738);  permute_23 = None
        expand_8: "bf16[256, 12, 128, 64]" = torch.ops.aten.expand.default(mul_20, _shape_param_57);  mul_20 = _shape_param_57 = None
        clone_8: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_47: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_58);  clone_8 = _shape_param_58 = None
        view_48: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_17, _shape_param_59);  _shape_param_59 = None
        permute_24: "bf16[768, 768]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_13: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg41_1, view_48, permute_24);  arg41_1 = view_48 = permute_24 = None
        view_49: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_13, _shape_param_60);  addmm_13 = _shape_param_60 = None
        view_50: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_49, _shape_param_61);  view_49 = _shape_param_61 = None
        permute_25: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        permute_26: "bf16[256, 12, 64, 128]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
        mul_21: "bf16[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(permute_26, 0.3535533905932738);  permute_26 = None
        expand_9: "bf16[256, 12, 64, 128]" = torch.ops.aten.expand.default(mul_21, _shape_param_62);  mul_21 = _shape_param_62 = None
        clone_9: "bf16[256, 12, 64, 128]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_51: "bf16[3072, 64, 128]" = torch.ops.aten.view.default(clone_9, _shape_param_63);  clone_9 = _shape_param_63 = None
        bmm_4: "bf16[3072, 128, 128]" = torch.ops.aten.bmm.default(view_47, view_51);  view_47 = view_51 = None
        view_52: "bf16[256, 12, 128, 128]" = torch.ops.aten.view.default(bmm_4, _shape_param_64);  bmm_4 = _shape_param_64 = None
        eq_2: "b8[256, 12, 128, 128]" = torch.ops.aten.eq.Scalar(view_52, -inf)
        logical_not_4: "b8[256, 12, 128, 128]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[256, 12, 128, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[256, 12, 128, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        full_2: "bf16[256, 12, 128, 128]" = torch.ops.aten.full.default(_shape_param_65, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_65 = None
        convert_element_type_18: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_52, torch.float32);  view_52 = None
        amax_2: "f32[256, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_18, [-1], True)
        sub_7: "f32[256, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_18, amax_2);  convert_element_type_18 = amax_2 = None
        exp_2: "f32[256, 12, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[256, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_19: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        where_2: "bf16[256, 12, 128, 128]" = torch.ops.aten.where.self(logical_not_5, full_2, convert_element_type_19);  logical_not_5 = full_2 = convert_element_type_19 = None
        expand_10: "bf16[256, 12, 128, 128]" = torch.ops.aten.expand.default(where_2, _shape_param_66);  where_2 = _shape_param_66 = None
        view_53: "bf16[3072, 128, 128]" = torch.ops.aten.view.default(expand_10, _shape_param_67);  expand_10 = _shape_param_67 = None
        view_54: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_17, _shape_param_68);  _shape_param_68 = None
        permute_27: "bf16[768, 768]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_14: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg43_1, view_54, permute_27);  arg43_1 = view_54 = permute_27 = None
        view_55: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_14, _shape_param_69);  addmm_14 = _shape_param_69 = None
        view_56: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_55, _shape_param_70);  view_55 = _shape_param_70 = None
        permute_28: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        expand_11: "bf16[256, 12, 128, 64]" = torch.ops.aten.expand.default(permute_28, _shape_param_71);  permute_28 = _shape_param_71 = None
        clone_10: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_57: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone_10, _shape_param_72);  clone_10 = _shape_param_72 = None
        bmm_5: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(view_53, view_57);  view_53 = view_57 = None
        view_58: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_5, _shape_param_73);  bmm_5 = _shape_param_73 = None
        permute_29: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_11: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_59: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_11, _shape_param_74);  clone_11 = _shape_param_74 = None
        view_60: "bf16[32768, 768]" = torch.ops.aten.view.default(view_59, _shape_param_75);  view_59 = _shape_param_75 = None
        permute_30: "bf16[768, 768]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_15: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg45_1, view_60, permute_30);  arg45_1 = view_60 = permute_30 = None
        view_61: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_15, _shape_param_76);  addmm_15 = _shape_param_76 = None
        add_17: "bf16[256, 128, 768]" = torch.ops.aten.add.Tensor(view_61, convert_element_type_17);  view_61 = convert_element_type_17 = None
        convert_element_type_20: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_17, torch.float32);  add_17 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_20, [2], correction = 0, keepdim = True)
        getitem_10: "f32[256, 128, 1]" = var_mean_5[0]
        getitem_11: "f32[256, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_20, getitem_11);  convert_element_type_20 = getitem_11 = None
        add_18: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_22: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_23: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_22, arg46_1);  mul_22 = arg46_1 = None
        add_19: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_23, arg47_1);  mul_23 = arg47_1 = None
        convert_element_type_21: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None
        view_62: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_21, _shape_param_77);  _shape_param_77 = None
        permute_31: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        addmm_16: "bf16[32768, 3072]" = torch.ops.aten.addmm.default(arg49_1, view_62, permute_31);  arg49_1 = view_62 = permute_31 = None
        view_63: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(addmm_16, _shape_param_78);  addmm_16 = _shape_param_78 = None
        convert_element_type_22: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        mul_24: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.5)
        mul_25: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.7071067811865476);  convert_element_type_22 = None
        erf_2: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_20: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_26: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_24, add_20);  mul_24 = add_20 = None
        convert_element_type_23: "bf16[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_26, torch.bfloat16);  mul_26 = None
        view_64: "bf16[32768, 3072]" = torch.ops.aten.view.default(convert_element_type_23, _shape_param_79);  convert_element_type_23 = _shape_param_79 = None
        permute_32: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_17: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg51_1, view_64, permute_32);  arg51_1 = view_64 = permute_32 = None
        view_65: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_17, _shape_param_80);  addmm_17 = _shape_param_80 = None
        add_21: "bf16[256, 128, 768]" = torch.ops.aten.add.Tensor(view_65, convert_element_type_21);  view_65 = convert_element_type_21 = None
        convert_element_type_24: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_21, torch.float32);  add_21 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_24, [2], correction = 0, keepdim = True)
        getitem_12: "f32[256, 128, 1]" = var_mean_6[0]
        getitem_13: "f32[256, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_9: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_24, getitem_13);  convert_element_type_24 = getitem_13 = None
        add_22: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_27: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_28: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_27, arg52_1);  mul_27 = arg52_1 = None
        add_23: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_28, arg53_1);  mul_28 = arg53_1 = None
        convert_element_type_25: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None
        view_66: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_25, _shape_param_81);  _shape_param_81 = None
        permute_33: "bf16[768, 768]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        addmm_18: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg55_1, view_66, permute_33);  arg55_1 = view_66 = permute_33 = None
        view_67: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_18, _shape_param_82);  addmm_18 = _shape_param_82 = None
        view_68: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_67, _shape_param_83);  view_67 = _shape_param_83 = None
        permute_34: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        mul_29: "bf16[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(permute_34, 0.3535533905932738);  permute_34 = None
        expand_12: "bf16[256, 12, 128, 64]" = torch.ops.aten.expand.default(mul_29, _shape_param_84);  mul_29 = _shape_param_84 = None
        clone_12: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_69: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_85);  clone_12 = _shape_param_85 = None
        view_70: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_25, _shape_param_86);  _shape_param_86 = None
        permute_35: "bf16[768, 768]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_19: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg57_1, view_70, permute_35);  arg57_1 = view_70 = permute_35 = None
        view_71: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_19, _shape_param_87);  addmm_19 = _shape_param_87 = None
        view_72: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_71, _shape_param_88);  view_71 = _shape_param_88 = None
        permute_36: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        permute_37: "bf16[256, 12, 64, 128]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
        mul_30: "bf16[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(permute_37, 0.3535533905932738);  permute_37 = None
        expand_13: "bf16[256, 12, 64, 128]" = torch.ops.aten.expand.default(mul_30, _shape_param_89);  mul_30 = _shape_param_89 = None
        clone_13: "bf16[256, 12, 64, 128]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_73: "bf16[3072, 64, 128]" = torch.ops.aten.view.default(clone_13, _shape_param_90);  clone_13 = _shape_param_90 = None
        bmm_6: "bf16[3072, 128, 128]" = torch.ops.aten.bmm.default(view_69, view_73);  view_69 = view_73 = None
        view_74: "bf16[256, 12, 128, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_91);  bmm_6 = _shape_param_91 = None
        eq_3: "b8[256, 12, 128, 128]" = torch.ops.aten.eq.Scalar(view_74, -inf)
        logical_not_6: "b8[256, 12, 128, 128]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[256, 12, 128, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[256, 12, 128, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        full_3: "bf16[256, 12, 128, 128]" = torch.ops.aten.full.default(_shape_param_92, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_92 = None
        convert_element_type_26: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_74, torch.float32);  view_74 = None
        amax_3: "f32[256, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_26, [-1], True)
        sub_10: "f32[256, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_26, amax_3);  convert_element_type_26 = amax_3 = None
        exp_3: "f32[256, 12, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[256, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_27: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        where_3: "bf16[256, 12, 128, 128]" = torch.ops.aten.where.self(logical_not_7, full_3, convert_element_type_27);  logical_not_7 = full_3 = convert_element_type_27 = None
        expand_14: "bf16[256, 12, 128, 128]" = torch.ops.aten.expand.default(where_3, _shape_param_93);  where_3 = _shape_param_93 = None
        view_75: "bf16[3072, 128, 128]" = torch.ops.aten.view.default(expand_14, _shape_param_94);  expand_14 = _shape_param_94 = None
        view_76: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_25, _shape_param_95);  _shape_param_95 = None
        permute_38: "bf16[768, 768]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_20: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg59_1, view_76, permute_38);  arg59_1 = view_76 = permute_38 = None
        view_77: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_20, _shape_param_96);  addmm_20 = _shape_param_96 = None
        view_78: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_77, _shape_param_97);  view_77 = _shape_param_97 = None
        permute_39: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
        expand_15: "bf16[256, 12, 128, 64]" = torch.ops.aten.expand.default(permute_39, _shape_param_98);  permute_39 = _shape_param_98 = None
        clone_14: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_79: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone_14, _shape_param_99);  clone_14 = _shape_param_99 = None
        bmm_7: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(view_75, view_79);  view_75 = view_79 = None
        view_80: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_100);  bmm_7 = _shape_param_100 = None
        permute_40: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        clone_15: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_81: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_15, _shape_param_101);  clone_15 = _shape_param_101 = None
        view_82: "bf16[32768, 768]" = torch.ops.aten.view.default(view_81, _shape_param_102);  view_81 = _shape_param_102 = None
        permute_41: "bf16[768, 768]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_21: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg61_1, view_82, permute_41);  arg61_1 = view_82 = permute_41 = None
        view_83: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_21, _shape_param_103);  addmm_21 = _shape_param_103 = None
        add_24: "bf16[256, 128, 768]" = torch.ops.aten.add.Tensor(view_83, convert_element_type_25);  view_83 = convert_element_type_25 = None
        convert_element_type_28: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_24, torch.float32);  add_24 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_28, [2], correction = 0, keepdim = True)
        getitem_14: "f32[256, 128, 1]" = var_mean_7[0]
        getitem_15: "f32[256, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_28, getitem_15);  convert_element_type_28 = getitem_15 = None
        add_25: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_31: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_32: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_31, arg62_1);  mul_31 = arg62_1 = None
        add_26: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_32, arg63_1);  mul_32 = arg63_1 = None
        convert_element_type_29: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_26, torch.bfloat16);  add_26 = None
        view_84: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_29, _shape_param_104);  _shape_param_104 = None
        permute_42: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        addmm_22: "bf16[32768, 3072]" = torch.ops.aten.addmm.default(arg65_1, view_84, permute_42);  arg65_1 = view_84 = permute_42 = None
        view_85: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(addmm_22, _shape_param_105);  addmm_22 = _shape_param_105 = None
        convert_element_type_30: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_33: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 0.5)
        mul_34: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 0.7071067811865476);  convert_element_type_30 = None
        erf_3: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_27: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_35: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_33, add_27);  mul_33 = add_27 = None
        convert_element_type_31: "bf16[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None
        view_86: "bf16[32768, 3072]" = torch.ops.aten.view.default(convert_element_type_31, _shape_param_106);  convert_element_type_31 = _shape_param_106 = None
        permute_43: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_23: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg67_1, view_86, permute_43);  arg67_1 = view_86 = permute_43 = None
        view_87: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_23, _shape_param_107);  addmm_23 = _shape_param_107 = None
        add_28: "bf16[256, 128, 768]" = torch.ops.aten.add.Tensor(view_87, convert_element_type_29);  view_87 = convert_element_type_29 = None
        convert_element_type_32: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_28, torch.float32);  add_28 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_32, [2], correction = 0, keepdim = True)
        getitem_16: "f32[256, 128, 1]" = var_mean_8[0]
        getitem_17: "f32[256, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_12: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_32, getitem_17);  convert_element_type_32 = getitem_17 = None
        add_29: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        mul_36: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_37: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_36, arg68_1);  mul_36 = arg68_1 = None
        add_30: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_37, arg69_1);  mul_37 = arg69_1 = None
        convert_element_type_33: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_30, torch.bfloat16);  add_30 = None
        view_88: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_33, _shape_param_108);  _shape_param_108 = None
        permute_44: "bf16[768, 768]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        addmm_24: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg71_1, view_88, permute_44);  arg71_1 = view_88 = permute_44 = None
        view_89: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_24, _shape_param_109);  addmm_24 = _shape_param_109 = None
        view_90: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_89, _shape_param_110);  view_89 = _shape_param_110 = None
        permute_45: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        mul_38: "bf16[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(permute_45, 0.3535533905932738);  permute_45 = None
        expand_16: "bf16[256, 12, 128, 64]" = torch.ops.aten.expand.default(mul_38, _shape_param_111);  mul_38 = _shape_param_111 = None
        clone_16: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_91: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_112);  clone_16 = _shape_param_112 = None
        view_92: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_33, _shape_param_113);  _shape_param_113 = None
        permute_46: "bf16[768, 768]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_25: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg73_1, view_92, permute_46);  arg73_1 = view_92 = permute_46 = None
        view_93: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_25, _shape_param_114);  addmm_25 = _shape_param_114 = None
        view_94: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_93, _shape_param_115);  view_93 = _shape_param_115 = None
        permute_47: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        permute_48: "bf16[256, 12, 64, 128]" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
        mul_39: "bf16[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(permute_48, 0.3535533905932738);  permute_48 = None
        expand_17: "bf16[256, 12, 64, 128]" = torch.ops.aten.expand.default(mul_39, _shape_param_116);  mul_39 = _shape_param_116 = None
        clone_17: "bf16[256, 12, 64, 128]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_95: "bf16[3072, 64, 128]" = torch.ops.aten.view.default(clone_17, _shape_param_117);  clone_17 = _shape_param_117 = None
        bmm_8: "bf16[3072, 128, 128]" = torch.ops.aten.bmm.default(view_91, view_95);  view_91 = view_95 = None
        view_96: "bf16[256, 12, 128, 128]" = torch.ops.aten.view.default(bmm_8, _shape_param_118);  bmm_8 = _shape_param_118 = None
        eq_4: "b8[256, 12, 128, 128]" = torch.ops.aten.eq.Scalar(view_96, -inf)
        logical_not_8: "b8[256, 12, 128, 128]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[256, 12, 128, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[256, 12, 128, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        full_4: "bf16[256, 12, 128, 128]" = torch.ops.aten.full.default(_shape_param_119, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_119 = None
        convert_element_type_34: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_96, torch.float32);  view_96 = None
        amax_4: "f32[256, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_34, [-1], True)
        sub_13: "f32[256, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_34, amax_4);  convert_element_type_34 = amax_4 = None
        exp_4: "f32[256, 12, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[256, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_35: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        where_4: "bf16[256, 12, 128, 128]" = torch.ops.aten.where.self(logical_not_9, full_4, convert_element_type_35);  logical_not_9 = full_4 = convert_element_type_35 = None
        expand_18: "bf16[256, 12, 128, 128]" = torch.ops.aten.expand.default(where_4, _shape_param_120);  where_4 = _shape_param_120 = None
        view_97: "bf16[3072, 128, 128]" = torch.ops.aten.view.default(expand_18, _shape_param_121);  expand_18 = _shape_param_121 = None
        view_98: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_33, _shape_param_122);  _shape_param_122 = None
        permute_49: "bf16[768, 768]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_26: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg75_1, view_98, permute_49);  arg75_1 = view_98 = permute_49 = None
        view_99: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_26, _shape_param_123);  addmm_26 = _shape_param_123 = None
        view_100: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_99, _shape_param_124);  view_99 = _shape_param_124 = None
        permute_50: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        expand_19: "bf16[256, 12, 128, 64]" = torch.ops.aten.expand.default(permute_50, _shape_param_125);  permute_50 = _shape_param_125 = None
        clone_18: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_101: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone_18, _shape_param_126);  clone_18 = _shape_param_126 = None
        bmm_9: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(view_97, view_101);  view_97 = view_101 = None
        view_102: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_9, _shape_param_127);  bmm_9 = _shape_param_127 = None
        permute_51: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_19: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_103: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_19, _shape_param_128);  clone_19 = _shape_param_128 = None
        view_104: "bf16[32768, 768]" = torch.ops.aten.view.default(view_103, _shape_param_129);  view_103 = _shape_param_129 = None
        permute_52: "bf16[768, 768]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_27: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg77_1, view_104, permute_52);  arg77_1 = view_104 = permute_52 = None
        view_105: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_27, _shape_param_130);  addmm_27 = _shape_param_130 = None
        add_31: "bf16[256, 128, 768]" = torch.ops.aten.add.Tensor(view_105, convert_element_type_33);  view_105 = convert_element_type_33 = None
        convert_element_type_36: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_31, torch.float32);  add_31 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_36, [2], correction = 0, keepdim = True)
        getitem_18: "f32[256, 128, 1]" = var_mean_9[0]
        getitem_19: "f32[256, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_36, getitem_19);  convert_element_type_36 = getitem_19 = None
        add_32: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_40: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_41: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg78_1);  mul_40 = arg78_1 = None
        add_33: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_41, arg79_1);  mul_41 = arg79_1 = None
        convert_element_type_37: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16);  add_33 = None
        view_106: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_37, _shape_param_131);  _shape_param_131 = None
        permute_53: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        addmm_28: "bf16[32768, 3072]" = torch.ops.aten.addmm.default(arg81_1, view_106, permute_53);  arg81_1 = view_106 = permute_53 = None
        view_107: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(addmm_28, _shape_param_132);  addmm_28 = _shape_param_132 = None
        convert_element_type_38: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        mul_42: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.5)
        mul_43: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.7071067811865476);  convert_element_type_38 = None
        erf_4: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_34: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_44: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_42, add_34);  mul_42 = add_34 = None
        convert_element_type_39: "bf16[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_44, torch.bfloat16);  mul_44 = None
        view_108: "bf16[32768, 3072]" = torch.ops.aten.view.default(convert_element_type_39, _shape_param_133);  convert_element_type_39 = _shape_param_133 = None
        permute_54: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_29: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg83_1, view_108, permute_54);  arg83_1 = view_108 = permute_54 = None
        view_109: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_29, _shape_param_134);  addmm_29 = _shape_param_134 = None
        add_35: "bf16[256, 128, 768]" = torch.ops.aten.add.Tensor(view_109, convert_element_type_37);  view_109 = convert_element_type_37 = None
        convert_element_type_40: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_35, torch.float32);  add_35 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_40, [2], correction = 0, keepdim = True)
        getitem_20: "f32[256, 128, 1]" = var_mean_10[0]
        getitem_21: "f32[256, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_15: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_40, getitem_21);  convert_element_type_40 = getitem_21 = None
        add_36: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_45: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_46: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_45, arg84_1);  mul_45 = arg84_1 = None
        add_37: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_46, arg85_1);  mul_46 = arg85_1 = None
        convert_element_type_41: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16);  add_37 = None
        view_110: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_41, _shape_param_135);  _shape_param_135 = None
        permute_55: "bf16[768, 768]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_30: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg87_1, view_110, permute_55);  arg87_1 = view_110 = permute_55 = None
        view_111: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_30, _shape_param_136);  addmm_30 = _shape_param_136 = None
        view_112: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_111, _shape_param_137);  view_111 = _shape_param_137 = None
        permute_56: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        mul_47: "bf16[256, 12, 128, 64]" = torch.ops.aten.mul.Scalar(permute_56, 0.3535533905932738);  permute_56 = None
        expand_20: "bf16[256, 12, 128, 64]" = torch.ops.aten.expand.default(mul_47, _shape_param_138);  mul_47 = _shape_param_138 = None
        clone_20: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_113: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_139);  clone_20 = _shape_param_139 = None
        view_114: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_41, _shape_param_140);  _shape_param_140 = None
        permute_57: "bf16[768, 768]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_31: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg89_1, view_114, permute_57);  arg89_1 = view_114 = permute_57 = None
        view_115: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_31, _shape_param_141);  addmm_31 = _shape_param_141 = None
        view_116: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_115, _shape_param_142);  view_115 = _shape_param_142 = None
        permute_58: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        permute_59: "bf16[256, 12, 64, 128]" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
        mul_48: "bf16[256, 12, 64, 128]" = torch.ops.aten.mul.Scalar(permute_59, 0.3535533905932738);  permute_59 = None
        expand_21: "bf16[256, 12, 64, 128]" = torch.ops.aten.expand.default(mul_48, _shape_param_143);  mul_48 = _shape_param_143 = None
        clone_21: "bf16[256, 12, 64, 128]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_117: "bf16[3072, 64, 128]" = torch.ops.aten.view.default(clone_21, _shape_param_144);  clone_21 = _shape_param_144 = None
        bmm_10: "bf16[3072, 128, 128]" = torch.ops.aten.bmm.default(view_113, view_117);  view_113 = view_117 = None
        view_118: "bf16[256, 12, 128, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_145);  bmm_10 = _shape_param_145 = None
        eq_5: "b8[256, 12, 128, 128]" = torch.ops.aten.eq.Scalar(view_118, -inf)
        logical_not_10: "b8[256, 12, 128, 128]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[256, 12, 128, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[256, 12, 128, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        full_5: "bf16[256, 12, 128, 128]" = torch.ops.aten.full.default(_shape_param_146, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_146 = None
        convert_element_type_42: "f32[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_118, torch.float32);  view_118 = None
        amax_5: "f32[256, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_42, [-1], True)
        sub_16: "f32[256, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_42, amax_5);  convert_element_type_42 = amax_5 = None
        exp_5: "f32[256, 12, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[256, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[256, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_43: "bf16[256, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        where_5: "bf16[256, 12, 128, 128]" = torch.ops.aten.where.self(logical_not_11, full_5, convert_element_type_43);  logical_not_11 = full_5 = convert_element_type_43 = None
        expand_22: "bf16[256, 12, 128, 128]" = torch.ops.aten.expand.default(where_5, _shape_param_147);  where_5 = _shape_param_147 = None
        view_119: "bf16[3072, 128, 128]" = torch.ops.aten.view.default(expand_22, _shape_param_148);  expand_22 = _shape_param_148 = None
        view_120: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_41, _shape_param_149);  _shape_param_149 = None
        permute_60: "bf16[768, 768]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_32: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg91_1, view_120, permute_60);  arg91_1 = view_120 = permute_60 = None
        view_121: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_32, _shape_param_150);  addmm_32 = _shape_param_150 = None
        view_122: "bf16[256, 128, 12, 64]" = torch.ops.aten.view.default(view_121, _shape_param_151);  view_121 = _shape_param_151 = None
        permute_61: "bf16[256, 12, 128, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        expand_23: "bf16[256, 12, 128, 64]" = torch.ops.aten.expand.default(permute_61, _shape_param_152);  permute_61 = _shape_param_152 = None
        clone_22: "bf16[256, 12, 128, 64]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_123: "bf16[3072, 128, 64]" = torch.ops.aten.view.default(clone_22, _shape_param_153);  clone_22 = _shape_param_153 = None
        bmm_11: "bf16[3072, 128, 64]" = torch.ops.aten.bmm.default(view_119, view_123);  view_119 = view_123 = None
        view_124: "bf16[256, 12, 128, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_154);  bmm_11 = _shape_param_154 = None
        permute_62: "bf16[256, 128, 12, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        clone_23: "bf16[256, 128, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_125: "bf16[256, 128, 768]" = torch.ops.aten.view.default(clone_23, _shape_param_155);  clone_23 = _shape_param_155 = None
        view_126: "bf16[32768, 768]" = torch.ops.aten.view.default(view_125, _shape_param_156);  view_125 = _shape_param_156 = None
        permute_63: "bf16[768, 768]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_33: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg93_1, view_126, permute_63);  arg93_1 = view_126 = permute_63 = None
        view_127: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_33, _shape_param_157);  addmm_33 = _shape_param_157 = None
        add_38: "bf16[256, 128, 768]" = torch.ops.aten.add.Tensor(view_127, convert_element_type_41);  view_127 = convert_element_type_41 = None
        convert_element_type_44: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_38, torch.float32);  add_38 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_44, [2], correction = 0, keepdim = True)
        getitem_22: "f32[256, 128, 1]" = var_mean_11[0]
        getitem_23: "f32[256, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_44, getitem_23);  convert_element_type_44 = getitem_23 = None
        add_39: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_49: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_50: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg94_1);  mul_49 = arg94_1 = None
        add_40: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_50, arg95_1);  mul_50 = arg95_1 = None
        convert_element_type_45: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16);  add_40 = None
        view_128: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_45, _shape_param_158);  _shape_param_158 = None
        permute_64: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        addmm_34: "bf16[32768, 3072]" = torch.ops.aten.addmm.default(arg97_1, view_128, permute_64);  arg97_1 = view_128 = permute_64 = None
        view_129: "bf16[256, 128, 3072]" = torch.ops.aten.view.default(addmm_34, _shape_param_159);  addmm_34 = _shape_param_159 = None
        convert_element_type_46: "f32[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_51: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 0.5)
        mul_52: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 0.7071067811865476);  convert_element_type_46 = None
        erf_5: "f32[256, 128, 3072]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_41: "f32[256, 128, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_53: "f32[256, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_51, add_41);  mul_51 = add_41 = None
        convert_element_type_47: "bf16[256, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16);  mul_53 = None
        view_130: "bf16[32768, 3072]" = torch.ops.aten.view.default(convert_element_type_47, _shape_param_160);  convert_element_type_47 = _shape_param_160 = None
        permute_65: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_35: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg99_1, view_130, permute_65);  arg99_1 = view_130 = permute_65 = None
        view_131: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_35, _shape_param_161);  addmm_35 = _shape_param_161 = None
        add_42: "bf16[256, 128, 768]" = torch.ops.aten.add.Tensor(view_131, convert_element_type_45);  view_131 = convert_element_type_45 = None
        convert_element_type_48: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_42, torch.float32);  add_42 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_48, [2], correction = 0, keepdim = True)
        getitem_24: "f32[256, 128, 1]" = var_mean_12[0]
        getitem_25: "f32[256, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_18: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_48, getitem_25);  convert_element_type_48 = getitem_25 = None
        add_43: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_54: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_55: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_54, arg100_1);  mul_54 = arg100_1 = None
        add_44: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_55, arg101_1);  mul_55 = arg101_1 = None
        convert_element_type_49: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None
        view_132: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_49, _shape_param_162);  convert_element_type_49 = _shape_param_162 = None
        permute_66: "bf16[768, 768]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_36: "bf16[32768, 768]" = torch.ops.aten.addmm.default(arg103_1, view_132, permute_66);  arg103_1 = view_132 = permute_66 = None
        view_133: "bf16[256, 128, 768]" = torch.ops.aten.view.default(addmm_36, _shape_param_163);  addmm_36 = _shape_param_163 = None
        convert_element_type_50: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(view_133, torch.float32);  view_133 = None
        mul_56: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 0.5)
        mul_57: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 0.7071067811865476);  convert_element_type_50 = None
        erf_6: "f32[256, 128, 768]" = torch.ops.aten.erf.default(mul_57);  mul_57 = None
        add_45: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_58: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_56, add_45);  mul_56 = add_45 = None
        convert_element_type_51: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_58, torch.bfloat16);  mul_58 = None
        convert_element_type_52: "f32[256, 128, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_51, torch.float32);  convert_element_type_51 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_52, [2], correction = 0, keepdim = True)
        getitem_26: "f32[256, 128, 1]" = var_mean_13[0]
        getitem_27: "f32[256, 128, 1]" = var_mean_13[1];  var_mean_13 = None
        view_134: "i64[32768]" = torch.ops.aten.view.default(arg104_1, [-1]);  arg104_1 = None
        ne: "b8[32768]" = torch.ops.aten.ne.Scalar(view_134, -100)
        full_6: "bf16[6]" = torch.ops.aten.full.default(_shape_param_164, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_164 = None
        cat: "bf16[30528]" = torch.ops.aten.cat.default([arg105_1, full_6]);  arg105_1 = full_6 = None
        sub_19: "f32[256, 128, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_52, getitem_27);  convert_element_type_52 = getitem_27 = None
        add_46: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_59: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_13);  sub_19 = rsqrt_13 = None
        mul_60: "f32[256, 128, 768]" = torch.ops.aten.mul.Tensor(mul_59, arg106_1);  mul_59 = arg106_1 = None
        add_47: "f32[256, 128, 768]" = torch.ops.aten.add.Tensor(mul_60, arg107_1);  mul_60 = arg107_1 = None
        convert_element_type_53: "bf16[256, 128, 768]" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None
        view_135: "bf16[32768, 768]" = torch.ops.aten.view.default(convert_element_type_53, _shape_param_165);  convert_element_type_53 = _shape_param_165 = None
        permute_67: "bf16[768, 30522]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        constant_pad_nd: "bf16[768, 30528]" = torch.ops.aten.constant_pad_nd.default(permute_67, _shape_param_166);  permute_67 = _shape_param_166 = None
        addmm_37: "bf16[32768, 30528]" = torch.ops.aten.addmm.default(cat, view_135, constant_pad_nd);  cat = view_135 = constant_pad_nd = None
        slice_2: "bf16[32768, 30522]" = torch.ops.aten.slice.Tensor(addmm_37, 1, 0, -6);  addmm_37 = None
        view_136: "bf16[256, 128, 30522]" = torch.ops.aten.view.default(slice_2, _shape_param_167);  slice_2 = _shape_param_167 = None
        view_137: "bf16[32768, 30522]" = torch.ops.aten.view.default(view_136, _shape_param_168);  _shape_param_168 = None
        convert_element_type_54: "f32[32768, 30522]" = torch.ops.prims.convert_element_type.default(view_137, torch.float32);  view_137 = None
        amax_6: "f32[32768, 1]" = torch.ops.aten.amax.default(convert_element_type_54, [1], True)
        sub_20: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(convert_element_type_54, amax_6);  convert_element_type_54 = amax_6 = None
        exp_6: "f32[32768, 30522]" = torch.ops.aten.exp.default(sub_20)
        sum_7: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True);  exp_6 = None
        log: "f32[32768, 1]" = torch.ops.aten.log.default(sum_7);  sum_7 = None
        sub_21: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(sub_20, log);  sub_20 = log = None
        convert_element_type_55: "bf16[32768, 30522]" = torch.ops.prims.convert_element_type.default(sub_21, torch.bfloat16);  sub_21 = None
        ne_1: "b8[32768]" = torch.ops.aten.ne.Scalar(view_134, -100)
        full_7: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "i64[32768]" = torch.ops.aten.where.self(ne_1, view_134, full_7);  ne_1 = full_7 = None
        unsqueeze: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(where_6, 1);  where_6 = None
        gather: "bf16[32768, 1]" = torch.ops.aten.gather.default(convert_element_type_55, 1, unsqueeze);  convert_element_type_55 = unsqueeze = None
        squeeze: "bf16[32768]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[32768]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_8: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "bf16[32768]" = torch.ops.aten.where.self(ne, neg, full_8);  ne = neg = full_8 = None
        sum_8: "bf16[]" = torch.ops.aten.sum.default(where_7);  where_7 = None
        ne_2: "b8[32768]" = torch.ops.aten.ne.Scalar(view_134, -100);  view_134 = None
        sum_9: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_56: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_9, torch.bfloat16);  sum_9 = None
        div_6: "bf16[]" = torch.ops.aten.div.Tensor(sum_8, convert_element_type_56);  sum_8 = convert_element_type_56 = None
        return (view_136, div_6)



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
