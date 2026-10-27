"""
Standalone repro captured via capture_hook.
Label: AlbertForMaskedLM
Pattern hash: a101513f8554
Shape hash: 437d13a9
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
    def forward(self, arg0_1: "i64[1, 512]", arg1_1: "i64[1, 512]", arg2_1: "f32[30000, 128]", arg3_1: "i64[8, 512]", arg4_1: "f32[2, 128]", arg5_1: "f32[512, 128]", arg6_1: "f32[128]", arg7_1: "f32[128]", arg8_1: "f32[4096]", arg9_1: "f32[4096, 128]", arg10_1: "f32[4096]", arg11_1: "f32[4096, 4096]", arg12_1: "f32[4096]", arg13_1: "f32[4096, 4096]", arg14_1: "f32[4096]", arg15_1: "f32[4096, 4096]", arg16_1: "f32[4096]", arg17_1: "f32[4096, 4096]", arg18_1: "f32[4096]", arg19_1: "f32[4096]", arg20_1: "f32[16384]", arg21_1: "f32[16384, 4096]", arg22_1: "f32[4096]", arg23_1: "f32[4096, 16384]", arg24_1: "f32[4096]", arg25_1: "f32[4096]", arg26_1: "f32[128]", arg27_1: "f32[128, 4096]", arg28_1: "f32[128]", arg29_1: "f32[128]", arg30_1: "f32[30000]", arg31_1: "i64[8, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296):
        # No stacktrace found for following nodes
        expand: "i64[1, 512]" = torch.ops.aten.expand.default(arg0_1, [1, -1]);  arg0_1 = None
        gather: "i64[1, 512]" = torch.ops.aten.gather.default(expand, 1, arg1_1);  expand = None
        expand_1: "i64[8, 512]" = torch.ops.aten.expand.default(gather, _shape_param_0);  _shape_param_0 = None
        embedding: "f32[8, 512, 128]" = torch.ops.aten.embedding.default(arg2_1, arg3_1, 0);  arg3_1 = None
        embedding_1: "f32[8, 512, 128]" = torch.ops.aten.embedding.default(arg4_1, expand_1);  arg4_1 = expand_1 = None
        add: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2: "f32[1, 512, 128]" = torch.ops.aten.embedding.default(arg5_1, arg1_1);  arg5_1 = arg1_1 = None
        add_1: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1]" = var_mean[0]
        getitem_1: "f32[8, 512, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  arg6_1 = None
        add_3: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        convert_element_type: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        convert_element_type_1: "bf16[4096, 128]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        convert_element_type_2: "bf16[8, 512, 128]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view: "bf16[4096, 128]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_1);  convert_element_type_2 = _shape_param_1 = None
        permute: "bf16[128, 4096]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        convert_element_type_3: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        convert_element_type_4: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.bfloat16);  arg11_1 = None
        view_2: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_1, _shape_param_3);  _shape_param_3 = None
        permute_1: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        addmm_1: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_3, view_2, permute_1)
        view_3: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_1, _shape_param_4);  addmm_1 = _shape_param_4 = None
        view_4: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_3, _shape_param_5);  view_3 = _shape_param_5 = None
        permute_2: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        convert_element_type_5: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.bfloat16);  arg12_1 = None
        convert_element_type_6: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        permute_3: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_6, [1, 0]);  convert_element_type_6 = None
        addmm_2: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_5, view_2, permute_3)
        view_5: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_2, _shape_param_6);  addmm_2 = _shape_param_6 = None
        view_6: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_5, _shape_param_7);  view_5 = _shape_param_7 = None
        permute_4: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        convert_element_type_7: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.bfloat16);  arg14_1 = None
        convert_element_type_8: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg15_1, torch.bfloat16);  arg15_1 = None
        permute_5: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_8, [1, 0]);  convert_element_type_8 = None
        addmm_3: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_7, view_2, permute_5)
        view_7: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_3, _shape_param_8);  addmm_3 = _shape_param_8 = None
        view_8: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_7, _shape_param_9);  view_7 = _shape_param_9 = None
        permute_6: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        mul_2: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_2, 0.3535533905932738);  permute_2 = None
        permute_7: "bf16[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_4, [0, 1, 3, 2]);  permute_4 = None
        mul_3: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_7, 0.3535533905932738);  permute_7 = None
        expand_2: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_2, _shape_param_10);  mul_2 = _shape_param_10 = None
        clone: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_9: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_11);  clone = _shape_param_11 = None
        expand_3: "bf16[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_3, _shape_param_12);  mul_3 = _shape_param_12 = None
        clone_1: "bf16[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_10: "bf16[512, 64, 512]" = torch.ops.aten.view.default(clone_1, _shape_param_13);  clone_1 = _shape_param_13 = None
        bmm: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_9, view_10)
        view_11: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_14);  bmm = _shape_param_14 = None
        convert_element_type_9: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_11, torch.float32)
        amax: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_9, [-1], True)
        sub_1: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_9, amax);  convert_element_type_9 = amax = None
        exp: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_10: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        eq: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(view_11, -inf);  view_11 = None
        logical_not: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full: "bf16[8, 64, 512, 512]" = torch.ops.aten.full.default(_shape_param_15, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_15 = None
        where: "bf16[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full, convert_element_type_10);  logical_not_1 = convert_element_type_10 = None
        expand_4: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(where, _shape_param_16);  _shape_param_16 = None
        view_12: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_4, _shape_param_17);  expand_4 = _shape_param_17 = None
        expand_5: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_6, _shape_param_18);  permute_6 = _shape_param_18 = None
        clone_2: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_13: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_19);  clone_2 = _shape_param_19 = None
        bmm_1: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = None
        view_14: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_20);  bmm_1 = _shape_param_20 = None
        permute_8: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_3: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_15: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_3, _shape_param_21);  clone_3 = _shape_param_21 = None
        convert_element_type_11: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.bfloat16);  arg16_1 = None
        convert_element_type_12: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        view_16: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_15, _shape_param_22);  view_15 = _shape_param_22 = None
        permute_9: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_12, [1, 0]);  convert_element_type_12 = None
        addmm_4: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_11, view_16, permute_9)
        view_17: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_4, _shape_param_23);  addmm_4 = _shape_param_23 = None
        add_4: "bf16[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_1, view_17);  view_1 = view_17 = None
        convert_element_type_13: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_4, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_13, [2], correction = 0, keepdim = True)
        getitem_2: "f32[8, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[8, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_2: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_13, getitem_3);  convert_element_type_13 = None
        mul_4: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_5: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_4, arg18_1);  mul_4 = None
        add_6: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_5, arg19_1);  mul_5 = None
        convert_element_type_14: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.bfloat16);  arg20_1 = None
        convert_element_type_15: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg21_1, torch.bfloat16);  arg21_1 = None
        convert_element_type_16: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16)
        view_18: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_16, _shape_param_24);  convert_element_type_16 = _shape_param_24 = None
        permute_10: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_15, [1, 0]);  convert_element_type_15 = None
        addmm_5: "bf16[4096, 16384]" = torch.ops.aten.addmm.default(convert_element_type_14, view_18, permute_10)
        view_19: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(addmm_5, _shape_param_25);  _shape_param_25 = None
        mul_6: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        convert_element_type_17: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_19, torch.float32)
        pow_1: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_17, 3.0);  convert_element_type_17 = None
        mul_7: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_7: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_19, mul_7);  view_19 = mul_7 = None
        mul_8: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
        add_8: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_9: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_6, add_8);  mul_6 = add_8 = None
        convert_element_type_18: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg22_1, torch.bfloat16);  arg22_1 = None
        convert_element_type_19: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg23_1, torch.bfloat16);  arg23_1 = None
        convert_element_type_20: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        view_20: "bf16[4096, 16384]" = torch.ops.aten.view.default(convert_element_type_20, _shape_param_26);  convert_element_type_20 = _shape_param_26 = None
        permute_11: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_19, [1, 0]);  convert_element_type_19 = None
        addmm_6: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_18, view_20, permute_11)
        view_21: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_6, _shape_param_27);  addmm_6 = _shape_param_27 = None
        add_9: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_21, add_6);  view_21 = add_6 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_4: "f32[8, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[8, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_10: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_3: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_9, getitem_5);  add_9 = getitem_5 = None
        mul_10: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_11: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_10, arg24_1)
        add_11: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_11, arg25_1);  mul_11 = None
        convert_element_type_21: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16)
        view_22: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_21, _shape_param_28);  convert_element_type_21 = _shape_param_28 = None
        addmm_7: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_3, view_22, permute_1)
        view_23: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_7, _shape_param_29);  addmm_7 = _shape_param_29 = None
        view_24: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_23, _shape_param_30);  view_23 = _shape_param_30 = None
        permute_12: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        addmm_8: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_5, view_22, permute_3)
        view_25: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_8, _shape_param_31);  addmm_8 = _shape_param_31 = None
        view_26: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_25, _shape_param_32);  view_25 = _shape_param_32 = None
        permute_13: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        addmm_9: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_7, view_22, permute_5)
        view_27: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_9, _shape_param_33);  addmm_9 = _shape_param_33 = None
        view_28: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_27, _shape_param_34);  view_27 = _shape_param_34 = None
        permute_14: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        mul_12: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_12, 0.3535533905932738);  permute_12 = None
        permute_15: "bf16[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_13, [0, 1, 3, 2]);  permute_13 = None
        mul_13: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_15, 0.3535533905932738);  permute_15 = None
        expand_6: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_12, _shape_param_35);  mul_12 = _shape_param_35 = None
        clone_4: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_29: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_36);  clone_4 = _shape_param_36 = None
        expand_7: "bf16[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_13, _shape_param_37);  mul_13 = _shape_param_37 = None
        clone_5: "bf16[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_30: "bf16[512, 64, 512]" = torch.ops.aten.view.default(clone_5, _shape_param_38);  clone_5 = _shape_param_38 = None
        bmm_2: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_29, view_30)
        view_31: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_2, _shape_param_39);  bmm_2 = _shape_param_39 = None
        convert_element_type_22: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_31, torch.float32)
        amax_1: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_22, [-1], True)
        sub_4: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_22, amax_1);  convert_element_type_22 = amax_1 = None
        exp_1: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_23: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        eq_1: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(view_31, -inf);  view_31 = None
        logical_not_2: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        where_1: "bf16[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_3, full, convert_element_type_23);  logical_not_3 = convert_element_type_23 = None
        expand_8: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_1, _shape_param_40);  _shape_param_40 = None
        view_32: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_8, _shape_param_41);  expand_8 = _shape_param_41 = None
        expand_9: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_14, _shape_param_42);  permute_14 = _shape_param_42 = None
        clone_6: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_33: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_6, _shape_param_43);  clone_6 = _shape_param_43 = None
        bmm_3: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_32, view_33);  view_32 = None
        view_34: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_44);  bmm_3 = _shape_param_44 = None
        permute_16: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        clone_7: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_35: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_7, _shape_param_45);  clone_7 = _shape_param_45 = None
        view_36: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_35, _shape_param_46);  view_35 = _shape_param_46 = None
        addmm_10: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_11, view_36, permute_9)
        view_37: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_10, _shape_param_47);  addmm_10 = _shape_param_47 = None
        add_12: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_11, view_37);  add_11 = view_37 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_12, [2], correction = 0, keepdim = True)
        getitem_6: "f32[8, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[8, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        add_13: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        sub_5: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_12, getitem_7);  add_12 = getitem_7 = None
        mul_14: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_15: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_14, arg18_1)
        add_14: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_15, arg19_1);  mul_15 = None
        convert_element_type_24: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16)
        view_38: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_24, _shape_param_48);  convert_element_type_24 = _shape_param_48 = None
        addmm_11: "bf16[4096, 16384]" = torch.ops.aten.addmm.default(convert_element_type_14, view_38, permute_10)
        view_39: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(addmm_11, _shape_param_49);  _shape_param_49 = None
        mul_16: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_39, 0.5)
        convert_element_type_25: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_39, torch.float32)
        pow_2: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_25, 3.0);  convert_element_type_25 = None
        mul_17: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_15: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_39, mul_17);  view_39 = mul_17 = None
        mul_18: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        tanh_1: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
        add_16: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_19: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_16, add_16);  mul_16 = add_16 = None
        convert_element_type_26: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16);  mul_19 = None
        view_40: "bf16[4096, 16384]" = torch.ops.aten.view.default(convert_element_type_26, _shape_param_50);  convert_element_type_26 = _shape_param_50 = None
        addmm_12: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_18, view_40, permute_11)
        view_41: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_12, _shape_param_51);  addmm_12 = _shape_param_51 = None
        add_17: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_41, add_14);  view_41 = add_14 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_8: "f32[8, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[8, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        add_18: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_6: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_17, getitem_9);  add_17 = getitem_9 = None
        mul_20: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_21: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_20, arg24_1)
        add_19: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_21, arg25_1);  mul_21 = None
        convert_element_type_27: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16)
        view_42: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_27, _shape_param_52);  convert_element_type_27 = _shape_param_52 = None
        addmm_13: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_3, view_42, permute_1)
        view_43: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_13, _shape_param_53);  addmm_13 = _shape_param_53 = None
        view_44: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_43, _shape_param_54);  view_43 = _shape_param_54 = None
        permute_17: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None
        addmm_14: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_5, view_42, permute_3)
        view_45: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_14, _shape_param_55);  addmm_14 = _shape_param_55 = None
        view_46: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_45, _shape_param_56);  view_45 = _shape_param_56 = None
        permute_18: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        addmm_15: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_7, view_42, permute_5)
        view_47: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_15, _shape_param_57);  addmm_15 = _shape_param_57 = None
        view_48: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_47, _shape_param_58);  view_47 = _shape_param_58 = None
        permute_19: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        mul_22: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_17, 0.3535533905932738);  permute_17 = None
        permute_20: "bf16[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_18, [0, 1, 3, 2]);  permute_18 = None
        mul_23: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_20, 0.3535533905932738);  permute_20 = None
        expand_10: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_22, _shape_param_59);  mul_22 = _shape_param_59 = None
        clone_8: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_49: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_60);  clone_8 = _shape_param_60 = None
        expand_11: "bf16[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_23, _shape_param_61);  mul_23 = _shape_param_61 = None
        clone_9: "bf16[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_50: "bf16[512, 64, 512]" = torch.ops.aten.view.default(clone_9, _shape_param_62);  clone_9 = _shape_param_62 = None
        bmm_4: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_49, view_50)
        view_51: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_4, _shape_param_63);  bmm_4 = _shape_param_63 = None
        convert_element_type_28: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_51, torch.float32)
        amax_2: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_28, [-1], True)
        sub_7: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_28, amax_2);  convert_element_type_28 = amax_2 = None
        exp_2: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_29: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        eq_2: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(view_51, -inf);  view_51 = None
        logical_not_4: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        where_2: "bf16[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_5, full, convert_element_type_29);  logical_not_5 = convert_element_type_29 = None
        expand_12: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_2, _shape_param_64);  _shape_param_64 = None
        view_52: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_12, _shape_param_65);  expand_12 = _shape_param_65 = None
        expand_13: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_19, _shape_param_66);  permute_19 = _shape_param_66 = None
        clone_10: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_53: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_10, _shape_param_67);  clone_10 = _shape_param_67 = None
        bmm_5: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_52, view_53);  view_52 = None
        view_54: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_5, _shape_param_68);  bmm_5 = _shape_param_68 = None
        permute_21: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        clone_11: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
        view_55: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_11, _shape_param_69);  clone_11 = _shape_param_69 = None
        view_56: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_55, _shape_param_70);  view_55 = _shape_param_70 = None
        addmm_16: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_11, view_56, permute_9)
        view_57: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_16, _shape_param_71);  addmm_16 = _shape_param_71 = None
        add_20: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_19, view_57);  add_19 = view_57 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_10: "f32[8, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[8, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        add_21: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_8: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_20, getitem_11);  add_20 = getitem_11 = None
        mul_24: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
        mul_25: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_24, arg18_1)
        add_22: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_25, arg19_1);  mul_25 = None
        convert_element_type_30: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16)
        view_58: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_30, _shape_param_72);  convert_element_type_30 = _shape_param_72 = None
        addmm_17: "bf16[4096, 16384]" = torch.ops.aten.addmm.default(convert_element_type_14, view_58, permute_10)
        view_59: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(addmm_17, _shape_param_73);  _shape_param_73 = None
        mul_26: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_59, 0.5)
        convert_element_type_31: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_59, torch.float32)
        pow_3: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_31, 3.0);  convert_element_type_31 = None
        mul_27: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_23: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_59, mul_27);  view_59 = mul_27 = None
        mul_28: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        tanh_2: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_28);  mul_28 = None
        add_24: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_29: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_26, add_24);  mul_26 = add_24 = None
        convert_element_type_32: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        view_60: "bf16[4096, 16384]" = torch.ops.aten.view.default(convert_element_type_32, _shape_param_74);  convert_element_type_32 = _shape_param_74 = None
        addmm_18: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_18, view_60, permute_11)
        view_61: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_18, _shape_param_75);  addmm_18 = _shape_param_75 = None
        add_25: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_61, add_22);  view_61 = add_22 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_12: "f32[8, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[8, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        add_26: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_9: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_25, getitem_13);  add_25 = getitem_13 = None
        mul_30: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
        mul_31: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_30, arg24_1)
        add_27: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_31, arg25_1);  mul_31 = None
        convert_element_type_33: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16)
        view_62: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_33, _shape_param_76);  convert_element_type_33 = _shape_param_76 = None
        addmm_19: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_3, view_62, permute_1)
        view_63: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_19, _shape_param_77);  addmm_19 = _shape_param_77 = None
        view_64: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_63, _shape_param_78);  view_63 = _shape_param_78 = None
        permute_22: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        addmm_20: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_5, view_62, permute_3)
        view_65: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_20, _shape_param_79);  addmm_20 = _shape_param_79 = None
        view_66: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_65, _shape_param_80);  view_65 = _shape_param_80 = None
        permute_23: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        addmm_21: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_7, view_62, permute_5)
        view_67: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_21, _shape_param_81);  addmm_21 = _shape_param_81 = None
        view_68: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_67, _shape_param_82);  view_67 = _shape_param_82 = None
        permute_24: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        mul_32: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_22, 0.3535533905932738);  permute_22 = None
        permute_25: "bf16[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_23, [0, 1, 3, 2]);  permute_23 = None
        mul_33: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_25, 0.3535533905932738);  permute_25 = None
        expand_14: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_32, _shape_param_83);  mul_32 = _shape_param_83 = None
        clone_12: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_69: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_84);  clone_12 = _shape_param_84 = None
        expand_15: "bf16[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_33, _shape_param_85);  mul_33 = _shape_param_85 = None
        clone_13: "bf16[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_70: "bf16[512, 64, 512]" = torch.ops.aten.view.default(clone_13, _shape_param_86);  clone_13 = _shape_param_86 = None
        bmm_6: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_69, view_70)
        view_71: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_6, _shape_param_87);  bmm_6 = _shape_param_87 = None
        convert_element_type_34: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_71, torch.float32)
        amax_3: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_34, [-1], True)
        sub_10: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_34, amax_3);  convert_element_type_34 = amax_3 = None
        exp_3: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_35: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        eq_3: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(view_71, -inf);  view_71 = None
        logical_not_6: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        where_3: "bf16[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_7, full, convert_element_type_35);  logical_not_7 = convert_element_type_35 = None
        expand_16: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_3, _shape_param_88);  _shape_param_88 = None
        view_72: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_16, _shape_param_89);  expand_16 = _shape_param_89 = None
        expand_17: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_24, _shape_param_90);  permute_24 = _shape_param_90 = None
        clone_14: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_73: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_14, _shape_param_91);  clone_14 = _shape_param_91 = None
        bmm_7: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_72, view_73);  view_72 = None
        view_74: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_92);  bmm_7 = _shape_param_92 = None
        permute_26: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        clone_15: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_75: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_15, _shape_param_93);  clone_15 = _shape_param_93 = None
        view_76: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_75, _shape_param_94);  view_75 = _shape_param_94 = None
        addmm_22: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_11, view_76, permute_9)
        view_77: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_22, _shape_param_95);  addmm_22 = _shape_param_95 = None
        add_28: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_27, view_77);  add_27 = view_77 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_14: "f32[8, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[8, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        add_29: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_11: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_28, getitem_15);  add_28 = getitem_15 = None
        mul_34: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
        mul_35: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_34, arg18_1)
        add_30: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_35, arg19_1);  mul_35 = None
        convert_element_type_36: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_30, torch.bfloat16)
        view_78: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_36, _shape_param_96);  convert_element_type_36 = _shape_param_96 = None
        addmm_23: "bf16[4096, 16384]" = torch.ops.aten.addmm.default(convert_element_type_14, view_78, permute_10)
        view_79: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(addmm_23, _shape_param_97);  _shape_param_97 = None
        mul_36: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_79, 0.5)
        convert_element_type_37: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_79, torch.float32)
        pow_4: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_37, 3.0);  convert_element_type_37 = None
        mul_37: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_31: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_79, mul_37);  view_79 = mul_37 = None
        mul_38: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_31, 0.7978845608028654);  add_31 = None
        tanh_3: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_32: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_39: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_36, add_32);  mul_36 = add_32 = None
        convert_element_type_38: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None
        view_80: "bf16[4096, 16384]" = torch.ops.aten.view.default(convert_element_type_38, _shape_param_98);  convert_element_type_38 = _shape_param_98 = None
        addmm_24: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_18, view_80, permute_11)
        view_81: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_24, _shape_param_99);  addmm_24 = _shape_param_99 = None
        add_33: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_81, add_30);  view_81 = add_30 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_16: "f32[8, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[8, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        add_34: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_12: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_33, getitem_17);  add_33 = getitem_17 = None
        mul_40: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
        mul_41: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_40, arg24_1)
        add_35: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_41, arg25_1);  mul_41 = None
        convert_element_type_39: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16)
        view_82: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_39, _shape_param_100);  convert_element_type_39 = _shape_param_100 = None
        addmm_25: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_3, view_82, permute_1)
        view_83: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_25, _shape_param_101);  addmm_25 = _shape_param_101 = None
        view_84: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_83, _shape_param_102);  view_83 = _shape_param_102 = None
        permute_27: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        addmm_26: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_5, view_82, permute_3)
        view_85: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_26, _shape_param_103);  addmm_26 = _shape_param_103 = None
        view_86: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_85, _shape_param_104);  view_85 = _shape_param_104 = None
        permute_28: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        addmm_27: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_7, view_82, permute_5)
        view_87: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_27, _shape_param_105);  addmm_27 = _shape_param_105 = None
        view_88: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_87, _shape_param_106);  view_87 = _shape_param_106 = None
        permute_29: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        mul_42: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_27, 0.3535533905932738);  permute_27 = None
        permute_30: "bf16[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_28, [0, 1, 3, 2]);  permute_28 = None
        mul_43: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_30, 0.3535533905932738);  permute_30 = None
        expand_18: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_42, _shape_param_107);  mul_42 = _shape_param_107 = None
        clone_16: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_89: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_108);  clone_16 = _shape_param_108 = None
        expand_19: "bf16[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_43, _shape_param_109);  mul_43 = _shape_param_109 = None
        clone_17: "bf16[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_90: "bf16[512, 64, 512]" = torch.ops.aten.view.default(clone_17, _shape_param_110);  clone_17 = _shape_param_110 = None
        bmm_8: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_89, view_90)
        view_91: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_8, _shape_param_111);  bmm_8 = _shape_param_111 = None
        convert_element_type_40: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_91, torch.float32)
        amax_4: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_40, [-1], True)
        sub_13: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_40, amax_4);  convert_element_type_40 = amax_4 = None
        exp_4: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_41: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        eq_4: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(view_91, -inf);  view_91 = None
        logical_not_8: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        where_4: "bf16[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_9, full, convert_element_type_41);  logical_not_9 = convert_element_type_41 = None
        expand_20: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_4, _shape_param_112);  _shape_param_112 = None
        view_92: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_20, _shape_param_113);  expand_20 = _shape_param_113 = None
        expand_21: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_29, _shape_param_114);  permute_29 = _shape_param_114 = None
        clone_18: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_93: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_18, _shape_param_115);  clone_18 = _shape_param_115 = None
        bmm_9: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_92, view_93);  view_92 = None
        view_94: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_9, _shape_param_116);  bmm_9 = _shape_param_116 = None
        permute_31: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        clone_19: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
        view_95: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_19, _shape_param_117);  clone_19 = _shape_param_117 = None
        view_96: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_95, _shape_param_118);  view_95 = _shape_param_118 = None
        addmm_28: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_11, view_96, permute_9)
        view_97: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_28, _shape_param_119);  addmm_28 = _shape_param_119 = None
        add_36: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_35, view_97);  add_35 = view_97 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_18: "f32[8, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[8, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        add_37: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_14: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_36, getitem_19);  add_36 = getitem_19 = None
        mul_44: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
        mul_45: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_44, arg18_1)
        add_38: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_45, arg19_1);  mul_45 = None
        convert_element_type_42: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_38, torch.bfloat16)
        view_98: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_42, _shape_param_120);  convert_element_type_42 = _shape_param_120 = None
        addmm_29: "bf16[4096, 16384]" = torch.ops.aten.addmm.default(convert_element_type_14, view_98, permute_10)
        view_99: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(addmm_29, _shape_param_121);  _shape_param_121 = None
        mul_46: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_99, 0.5)
        convert_element_type_43: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_99, torch.float32)
        pow_5: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_43, 3.0);  convert_element_type_43 = None
        mul_47: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_39: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_99, mul_47);  view_99 = mul_47 = None
        mul_48: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
        add_40: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_49: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_46, add_40);  mul_46 = add_40 = None
        convert_element_type_44: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        view_100: "bf16[4096, 16384]" = torch.ops.aten.view.default(convert_element_type_44, _shape_param_122);  convert_element_type_44 = _shape_param_122 = None
        addmm_30: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_18, view_100, permute_11)
        view_101: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_30, _shape_param_123);  addmm_30 = _shape_param_123 = None
        add_41: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_101, add_38);  view_101 = add_38 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_20: "f32[8, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[8, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        add_42: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_15: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_41, getitem_21);  add_41 = getitem_21 = None
        mul_50: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
        mul_51: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_50, arg24_1)
        add_43: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_51, arg25_1);  mul_51 = None
        convert_element_type_45: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16)
        view_102: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_45, _shape_param_124);  convert_element_type_45 = _shape_param_124 = None
        addmm_31: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_3, view_102, permute_1)
        view_103: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_31, _shape_param_125);  addmm_31 = _shape_param_125 = None
        view_104: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_103, _shape_param_126);  view_103 = _shape_param_126 = None
        permute_32: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        addmm_32: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_5, view_102, permute_3)
        view_105: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_32, _shape_param_127);  addmm_32 = _shape_param_127 = None
        view_106: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_105, _shape_param_128);  view_105 = _shape_param_128 = None
        permute_33: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        addmm_33: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_7, view_102, permute_5)
        view_107: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_33, _shape_param_129);  addmm_33 = _shape_param_129 = None
        view_108: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_107, _shape_param_130);  view_107 = _shape_param_130 = None
        permute_34: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
        mul_52: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_32, 0.3535533905932738);  permute_32 = None
        permute_35: "bf16[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_33, [0, 1, 3, 2]);  permute_33 = None
        mul_53: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_35, 0.3535533905932738);  permute_35 = None
        expand_22: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_52, _shape_param_131);  mul_52 = _shape_param_131 = None
        clone_20: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_109: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_132);  clone_20 = _shape_param_132 = None
        expand_23: "bf16[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_53, _shape_param_133);  mul_53 = _shape_param_133 = None
        clone_21: "bf16[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_110: "bf16[512, 64, 512]" = torch.ops.aten.view.default(clone_21, _shape_param_134);  clone_21 = _shape_param_134 = None
        bmm_10: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_109, view_110)
        view_111: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_10, _shape_param_135);  bmm_10 = _shape_param_135 = None
        convert_element_type_46: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_111, torch.float32)
        amax_5: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_46, [-1], True)
        sub_16: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_46, amax_5);  convert_element_type_46 = amax_5 = None
        exp_5: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_47: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        eq_5: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(view_111, -inf);  view_111 = None
        logical_not_10: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        where_5: "bf16[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_11, full, convert_element_type_47);  logical_not_11 = convert_element_type_47 = None
        expand_24: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_5, _shape_param_136);  _shape_param_136 = None
        view_112: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_24, _shape_param_137);  expand_24 = _shape_param_137 = None
        expand_25: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_34, _shape_param_138);  permute_34 = _shape_param_138 = None
        clone_22: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_113: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_22, _shape_param_139);  clone_22 = _shape_param_139 = None
        bmm_11: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_112, view_113);  view_112 = None
        view_114: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_140);  bmm_11 = _shape_param_140 = None
        permute_36: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        clone_23: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        view_115: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_23, _shape_param_141);  clone_23 = _shape_param_141 = None
        view_116: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_115, _shape_param_142);  view_115 = _shape_param_142 = None
        addmm_34: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_11, view_116, permute_9)
        view_117: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_34, _shape_param_143);  addmm_34 = _shape_param_143 = None
        add_44: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_43, view_117);  add_43 = view_117 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_22: "f32[8, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[8, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        add_45: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_17: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_44, getitem_23);  add_44 = getitem_23 = None
        mul_54: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
        mul_55: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_54, arg18_1)
        add_46: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_55, arg19_1);  mul_55 = None
        convert_element_type_48: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16)
        view_118: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_48, _shape_param_144);  convert_element_type_48 = _shape_param_144 = None
        addmm_35: "bf16[4096, 16384]" = torch.ops.aten.addmm.default(convert_element_type_14, view_118, permute_10)
        view_119: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(addmm_35, _shape_param_145);  _shape_param_145 = None
        mul_56: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_119, 0.5)
        convert_element_type_49: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_119, torch.float32)
        pow_6: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_49, 3.0);  convert_element_type_49 = None
        mul_57: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_47: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_119, mul_57);  view_119 = mul_57 = None
        mul_58: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
        tanh_5: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_48: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_59: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_56, add_48);  mul_56 = add_48 = None
        convert_element_type_50: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None
        view_120: "bf16[4096, 16384]" = torch.ops.aten.view.default(convert_element_type_50, _shape_param_146);  convert_element_type_50 = _shape_param_146 = None
        addmm_36: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_18, view_120, permute_11)
        view_121: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_36, _shape_param_147);  addmm_36 = _shape_param_147 = None
        add_49: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_121, add_46);  view_121 = add_46 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_24: "f32[8, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[8, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        add_50: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_18: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_49, getitem_25);  add_49 = getitem_25 = None
        mul_60: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
        mul_61: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_60, arg24_1)
        add_51: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_61, arg25_1);  mul_61 = None
        convert_element_type_51: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16)
        view_122: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_51, _shape_param_148);  convert_element_type_51 = _shape_param_148 = None
        addmm_37: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_3, view_122, permute_1)
        view_123: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_37, _shape_param_149);  addmm_37 = _shape_param_149 = None
        view_124: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_123, _shape_param_150);  view_123 = _shape_param_150 = None
        permute_37: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        addmm_38: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_5, view_122, permute_3)
        view_125: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_38, _shape_param_151);  addmm_38 = _shape_param_151 = None
        view_126: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_125, _shape_param_152);  view_125 = _shape_param_152 = None
        permute_38: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        addmm_39: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_7, view_122, permute_5)
        view_127: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_39, _shape_param_153);  addmm_39 = _shape_param_153 = None
        view_128: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_127, _shape_param_154);  view_127 = _shape_param_154 = None
        permute_39: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
        mul_62: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_37, 0.3535533905932738);  permute_37 = None
        permute_40: "bf16[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_38, [0, 1, 3, 2]);  permute_38 = None
        mul_63: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_40, 0.3535533905932738);  permute_40 = None
        expand_26: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_62, _shape_param_155);  mul_62 = _shape_param_155 = None
        clone_24: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_129: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_156);  clone_24 = _shape_param_156 = None
        expand_27: "bf16[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_63, _shape_param_157);  mul_63 = _shape_param_157 = None
        clone_25: "bf16[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_130: "bf16[512, 64, 512]" = torch.ops.aten.view.default(clone_25, _shape_param_158);  clone_25 = _shape_param_158 = None
        bmm_12: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_129, view_130)
        view_131: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_12, _shape_param_159);  bmm_12 = _shape_param_159 = None
        convert_element_type_52: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_131, torch.float32)
        amax_6: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_52, [-1], True)
        sub_19: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_52, amax_6);  convert_element_type_52 = amax_6 = None
        exp_6: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_53: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        eq_6: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(view_131, -inf);  view_131 = None
        logical_not_12: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        where_6: "bf16[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_13, full, convert_element_type_53);  logical_not_13 = convert_element_type_53 = None
        expand_28: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_6, _shape_param_160);  _shape_param_160 = None
        view_132: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_28, _shape_param_161);  expand_28 = _shape_param_161 = None
        expand_29: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_39, _shape_param_162);  permute_39 = _shape_param_162 = None
        clone_26: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_133: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_26, _shape_param_163);  clone_26 = _shape_param_163 = None
        bmm_13: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_132, view_133);  view_132 = None
        view_134: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_13, _shape_param_164);  bmm_13 = _shape_param_164 = None
        permute_41: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        clone_27: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
        view_135: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_27, _shape_param_165);  clone_27 = _shape_param_165 = None
        view_136: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_135, _shape_param_166);  view_135 = _shape_param_166 = None
        addmm_40: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_11, view_136, permute_9)
        view_137: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_40, _shape_param_167);  addmm_40 = _shape_param_167 = None
        add_52: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_51, view_137);  add_51 = view_137 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_26: "f32[8, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[8, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        add_53: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_20: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_52, getitem_27);  add_52 = getitem_27 = None
        mul_64: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
        mul_65: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_64, arg18_1)
        add_54: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_65, arg19_1);  mul_65 = None
        convert_element_type_54: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16)
        view_138: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_54, _shape_param_168);  convert_element_type_54 = _shape_param_168 = None
        addmm_41: "bf16[4096, 16384]" = torch.ops.aten.addmm.default(convert_element_type_14, view_138, permute_10)
        view_139: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(addmm_41, _shape_param_169);  _shape_param_169 = None
        mul_66: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_139, 0.5)
        convert_element_type_55: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_139, torch.float32)
        pow_7: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_55, 3.0);  convert_element_type_55 = None
        mul_67: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_55: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_139, mul_67);  view_139 = mul_67 = None
        mul_68: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_6: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_68);  mul_68 = None
        add_56: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_69: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_66, add_56);  mul_66 = add_56 = None
        convert_element_type_56: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        view_140: "bf16[4096, 16384]" = torch.ops.aten.view.default(convert_element_type_56, _shape_param_170);  convert_element_type_56 = _shape_param_170 = None
        addmm_42: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_18, view_140, permute_11)
        view_141: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_42, _shape_param_171);  addmm_42 = _shape_param_171 = None
        add_57: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_141, add_54);  view_141 = add_54 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_28: "f32[8, 512, 1]" = var_mean_14[0]
        getitem_29: "f32[8, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        add_58: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_21: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_57, getitem_29);  add_57 = getitem_29 = None
        mul_70: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
        mul_71: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_70, arg24_1)
        add_59: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_71, arg25_1);  mul_71 = None
        convert_element_type_57: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16)
        view_142: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_57, _shape_param_172);  convert_element_type_57 = _shape_param_172 = None
        addmm_43: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_3, view_142, permute_1)
        view_143: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_43, _shape_param_173);  addmm_43 = _shape_param_173 = None
        view_144: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_143, _shape_param_174);  view_143 = _shape_param_174 = None
        permute_42: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        addmm_44: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_5, view_142, permute_3)
        view_145: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_44, _shape_param_175);  addmm_44 = _shape_param_175 = None
        view_146: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_145, _shape_param_176);  view_145 = _shape_param_176 = None
        permute_43: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        addmm_45: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_7, view_142, permute_5)
        view_147: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_45, _shape_param_177);  addmm_45 = _shape_param_177 = None
        view_148: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_147, _shape_param_178);  view_147 = _shape_param_178 = None
        permute_44: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        mul_72: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_42, 0.3535533905932738);  permute_42 = None
        permute_45: "bf16[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_43, [0, 1, 3, 2]);  permute_43 = None
        mul_73: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_45, 0.3535533905932738);  permute_45 = None
        expand_30: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_72, _shape_param_179);  mul_72 = _shape_param_179 = None
        clone_28: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_149: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_180);  clone_28 = _shape_param_180 = None
        expand_31: "bf16[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_73, _shape_param_181);  mul_73 = _shape_param_181 = None
        clone_29: "bf16[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_150: "bf16[512, 64, 512]" = torch.ops.aten.view.default(clone_29, _shape_param_182);  clone_29 = _shape_param_182 = None
        bmm_14: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_149, view_150)
        view_151: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_14, _shape_param_183);  bmm_14 = _shape_param_183 = None
        convert_element_type_58: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_151, torch.float32)
        amax_7: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_58, [-1], True)
        sub_22: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_58, amax_7);  convert_element_type_58 = amax_7 = None
        exp_7: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_59: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        eq_7: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(view_151, -inf);  view_151 = None
        logical_not_14: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        where_7: "bf16[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_15, full, convert_element_type_59);  logical_not_15 = convert_element_type_59 = None
        expand_32: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_7, _shape_param_184);  _shape_param_184 = None
        view_152: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_32, _shape_param_185);  expand_32 = _shape_param_185 = None
        expand_33: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_44, _shape_param_186);  permute_44 = _shape_param_186 = None
        clone_30: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_153: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_30, _shape_param_187);  clone_30 = _shape_param_187 = None
        bmm_15: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_152, view_153);  view_152 = None
        view_154: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_188);  bmm_15 = _shape_param_188 = None
        permute_46: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
        clone_31: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        view_155: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_31, _shape_param_189);  clone_31 = _shape_param_189 = None
        view_156: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_155, _shape_param_190);  view_155 = _shape_param_190 = None
        addmm_46: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_11, view_156, permute_9)
        view_157: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_46, _shape_param_191);  addmm_46 = _shape_param_191 = None
        add_60: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_59, view_157);  add_59 = view_157 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_30: "f32[8, 512, 1]" = var_mean_15[0]
        getitem_31: "f32[8, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        add_61: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_23: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_60, getitem_31);  add_60 = getitem_31 = None
        mul_74: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
        mul_75: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_74, arg18_1)
        add_62: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_75, arg19_1);  mul_75 = None
        convert_element_type_60: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_62, torch.bfloat16)
        view_158: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_60, _shape_param_192);  convert_element_type_60 = _shape_param_192 = None
        addmm_47: "bf16[4096, 16384]" = torch.ops.aten.addmm.default(convert_element_type_14, view_158, permute_10)
        view_159: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(addmm_47, _shape_param_193);  _shape_param_193 = None
        mul_76: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_159, 0.5)
        convert_element_type_61: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_159, torch.float32)
        pow_8: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_61, 3.0);  convert_element_type_61 = None
        mul_77: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_63: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_159, mul_77);  view_159 = mul_77 = None
        mul_78: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_63, 0.7978845608028654);  add_63 = None
        tanh_7: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_64: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_79: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_76, add_64);  mul_76 = add_64 = None
        convert_element_type_62: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None
        view_160: "bf16[4096, 16384]" = torch.ops.aten.view.default(convert_element_type_62, _shape_param_194);  convert_element_type_62 = _shape_param_194 = None
        addmm_48: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_18, view_160, permute_11)
        view_161: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_48, _shape_param_195);  addmm_48 = _shape_param_195 = None
        add_65: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_161, add_62);  view_161 = add_62 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_32: "f32[8, 512, 1]" = var_mean_16[0]
        getitem_33: "f32[8, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        add_66: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_24: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_65, getitem_33);  add_65 = getitem_33 = None
        mul_80: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
        mul_81: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_80, arg24_1)
        add_67: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_81, arg25_1);  mul_81 = None
        convert_element_type_63: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_67, torch.bfloat16)
        view_162: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_63, _shape_param_196);  convert_element_type_63 = _shape_param_196 = None
        addmm_49: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_3, view_162, permute_1)
        view_163: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_49, _shape_param_197);  addmm_49 = _shape_param_197 = None
        view_164: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_163, _shape_param_198);  view_163 = _shape_param_198 = None
        permute_47: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        addmm_50: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_5, view_162, permute_3)
        view_165: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_50, _shape_param_199);  addmm_50 = _shape_param_199 = None
        view_166: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_165, _shape_param_200);  view_165 = _shape_param_200 = None
        permute_48: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        addmm_51: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_7, view_162, permute_5)
        view_167: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_51, _shape_param_201);  addmm_51 = _shape_param_201 = None
        view_168: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_167, _shape_param_202);  view_167 = _shape_param_202 = None
        permute_49: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        mul_82: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_47, 0.3535533905932738);  permute_47 = None
        permute_50: "bf16[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        mul_83: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_50, 0.3535533905932738);  permute_50 = None
        expand_34: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_82, _shape_param_203);  mul_82 = _shape_param_203 = None
        clone_32: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_169: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_32, _shape_param_204);  clone_32 = _shape_param_204 = None
        expand_35: "bf16[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_83, _shape_param_205);  mul_83 = _shape_param_205 = None
        clone_33: "bf16[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_170: "bf16[512, 64, 512]" = torch.ops.aten.view.default(clone_33, _shape_param_206);  clone_33 = _shape_param_206 = None
        bmm_16: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_169, view_170)
        view_171: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_16, _shape_param_207);  bmm_16 = _shape_param_207 = None
        convert_element_type_64: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_171, torch.float32)
        amax_8: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_64, [-1], True)
        sub_25: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_64, amax_8);  convert_element_type_64 = amax_8 = None
        exp_8: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_65: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        eq_8: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(view_171, -inf);  view_171 = None
        logical_not_16: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        where_8: "bf16[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_17, full, convert_element_type_65);  logical_not_17 = convert_element_type_65 = None
        expand_36: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_8, _shape_param_208);  _shape_param_208 = None
        view_172: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_36, _shape_param_209);  expand_36 = _shape_param_209 = None
        expand_37: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_49, _shape_param_210);  permute_49 = _shape_param_210 = None
        clone_34: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_173: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_34, _shape_param_211);  clone_34 = _shape_param_211 = None
        bmm_17: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_172, view_173);  view_172 = None
        view_174: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_17, _shape_param_212);  bmm_17 = _shape_param_212 = None
        permute_51: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
        clone_35: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_175: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_35, _shape_param_213);  clone_35 = _shape_param_213 = None
        view_176: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_175, _shape_param_214);  view_175 = _shape_param_214 = None
        addmm_52: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_11, view_176, permute_9)
        view_177: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_52, _shape_param_215);  addmm_52 = _shape_param_215 = None
        add_68: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_67, view_177);  add_67 = view_177 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
        getitem_34: "f32[8, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[8, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        add_69: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_26: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_68, getitem_35);  add_68 = getitem_35 = None
        mul_84: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
        mul_85: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_84, arg18_1)
        add_70: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_85, arg19_1);  mul_85 = None
        convert_element_type_66: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_70, torch.bfloat16)
        view_178: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_66, _shape_param_216);  convert_element_type_66 = _shape_param_216 = None
        addmm_53: "bf16[4096, 16384]" = torch.ops.aten.addmm.default(convert_element_type_14, view_178, permute_10)
        view_179: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(addmm_53, _shape_param_217);  _shape_param_217 = None
        mul_86: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_179, 0.5)
        convert_element_type_67: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_179, torch.float32)
        pow_9: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_67, 3.0);  convert_element_type_67 = None
        mul_87: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_71: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_179, mul_87);  view_179 = mul_87 = None
        mul_88: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_8: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_88);  mul_88 = None
        add_72: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_89: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_86, add_72);  mul_86 = add_72 = None
        convert_element_type_68: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None
        view_180: "bf16[4096, 16384]" = torch.ops.aten.view.default(convert_element_type_68, _shape_param_218);  convert_element_type_68 = _shape_param_218 = None
        addmm_54: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_18, view_180, permute_11)
        view_181: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_54, _shape_param_219);  addmm_54 = _shape_param_219 = None
        add_73: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_181, add_70);  view_181 = add_70 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_36: "f32[8, 512, 1]" = var_mean_18[0]
        getitem_37: "f32[8, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        add_74: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_27: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_73, getitem_37);  add_73 = getitem_37 = None
        mul_90: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
        mul_91: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_90, arg24_1)
        add_75: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_91, arg25_1);  mul_91 = None
        convert_element_type_69: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_75, torch.bfloat16)
        view_182: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_69, _shape_param_220);  convert_element_type_69 = _shape_param_220 = None
        addmm_55: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_3, view_182, permute_1)
        view_183: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_55, _shape_param_221);  addmm_55 = _shape_param_221 = None
        view_184: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_183, _shape_param_222);  view_183 = _shape_param_222 = None
        permute_52: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        addmm_56: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_5, view_182, permute_3)
        view_185: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_56, _shape_param_223);  addmm_56 = _shape_param_223 = None
        view_186: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_185, _shape_param_224);  view_185 = _shape_param_224 = None
        permute_53: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        addmm_57: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_7, view_182, permute_5)
        view_187: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_57, _shape_param_225);  addmm_57 = _shape_param_225 = None
        view_188: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_187, _shape_param_226);  view_187 = _shape_param_226 = None
        permute_54: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
        mul_92: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_52, 0.3535533905932738);  permute_52 = None
        permute_55: "bf16[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_53, [0, 1, 3, 2]);  permute_53 = None
        mul_93: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_55, 0.3535533905932738);  permute_55 = None
        expand_38: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_92, _shape_param_227);  mul_92 = _shape_param_227 = None
        clone_36: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_189: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_228);  clone_36 = _shape_param_228 = None
        expand_39: "bf16[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_93, _shape_param_229);  mul_93 = _shape_param_229 = None
        clone_37: "bf16[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_190: "bf16[512, 64, 512]" = torch.ops.aten.view.default(clone_37, _shape_param_230);  clone_37 = _shape_param_230 = None
        bmm_18: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_189, view_190)
        view_191: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_18, _shape_param_231);  bmm_18 = _shape_param_231 = None
        convert_element_type_70: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_191, torch.float32)
        amax_9: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_70, [-1], True)
        sub_28: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_70, amax_9);  convert_element_type_70 = amax_9 = None
        exp_9: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_71: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        eq_9: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(view_191, -inf);  view_191 = None
        logical_not_18: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        where_9: "bf16[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_19, full, convert_element_type_71);  logical_not_19 = convert_element_type_71 = None
        expand_40: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_9, _shape_param_232);  _shape_param_232 = None
        view_192: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_40, _shape_param_233);  expand_40 = _shape_param_233 = None
        expand_41: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_54, _shape_param_234);  permute_54 = _shape_param_234 = None
        clone_38: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_193: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_38, _shape_param_235);  clone_38 = _shape_param_235 = None
        bmm_19: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_192, view_193);  view_192 = None
        view_194: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_236);  bmm_19 = _shape_param_236 = None
        permute_56: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
        clone_39: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_195: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_39, _shape_param_237);  clone_39 = _shape_param_237 = None
        view_196: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_195, _shape_param_238);  view_195 = _shape_param_238 = None
        addmm_58: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_11, view_196, permute_9)
        view_197: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_58, _shape_param_239);  addmm_58 = _shape_param_239 = None
        add_76: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_75, view_197);  add_75 = view_197 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
        getitem_38: "f32[8, 512, 1]" = var_mean_19[0]
        getitem_39: "f32[8, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        add_77: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_29: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_76, getitem_39);  add_76 = getitem_39 = None
        mul_94: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
        mul_95: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_94, arg18_1)
        add_78: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_95, arg19_1);  mul_95 = None
        convert_element_type_72: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16)
        view_198: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_72, _shape_param_240);  convert_element_type_72 = _shape_param_240 = None
        addmm_59: "bf16[4096, 16384]" = torch.ops.aten.addmm.default(convert_element_type_14, view_198, permute_10)
        view_199: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(addmm_59, _shape_param_241);  _shape_param_241 = None
        mul_96: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_199, 0.5)
        convert_element_type_73: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_199, torch.float32)
        pow_10: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_73, 3.0);  convert_element_type_73 = None
        mul_97: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_79: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_199, mul_97);  view_199 = mul_97 = None
        mul_98: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
        tanh_9: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_98);  mul_98 = None
        add_80: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_99: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_96, add_80);  mul_96 = add_80 = None
        convert_element_type_74: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_99, torch.bfloat16);  mul_99 = None
        view_200: "bf16[4096, 16384]" = torch.ops.aten.view.default(convert_element_type_74, _shape_param_242);  convert_element_type_74 = _shape_param_242 = None
        addmm_60: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_18, view_200, permute_11)
        view_201: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_60, _shape_param_243);  addmm_60 = _shape_param_243 = None
        add_81: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_201, add_78);  view_201 = add_78 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_40: "f32[8, 512, 1]" = var_mean_20[0]
        getitem_41: "f32[8, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        add_82: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_30: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_81, getitem_41);  add_81 = getitem_41 = None
        mul_100: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
        mul_101: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_100, arg24_1)
        add_83: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_101, arg25_1);  mul_101 = None
        convert_element_type_75: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16)
        view_202: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_75, _shape_param_244);  convert_element_type_75 = _shape_param_244 = None
        addmm_61: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_3, view_202, permute_1)
        view_203: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_61, _shape_param_245);  addmm_61 = _shape_param_245 = None
        view_204: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_203, _shape_param_246);  view_203 = _shape_param_246 = None
        permute_57: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
        addmm_62: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_5, view_202, permute_3)
        view_205: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_62, _shape_param_247);  addmm_62 = _shape_param_247 = None
        view_206: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_205, _shape_param_248);  view_205 = _shape_param_248 = None
        permute_58: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        addmm_63: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_7, view_202, permute_5)
        view_207: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_63, _shape_param_249);  addmm_63 = _shape_param_249 = None
        view_208: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_207, _shape_param_250);  view_207 = _shape_param_250 = None
        permute_59: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        mul_102: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_57, 0.3535533905932738);  permute_57 = None
        permute_60: "bf16[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
        mul_103: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_60, 0.3535533905932738);  permute_60 = None
        expand_42: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_102, _shape_param_251);  mul_102 = _shape_param_251 = None
        clone_40: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_209: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_252);  clone_40 = _shape_param_252 = None
        expand_43: "bf16[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_103, _shape_param_253);  mul_103 = _shape_param_253 = None
        clone_41: "bf16[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_210: "bf16[512, 64, 512]" = torch.ops.aten.view.default(clone_41, _shape_param_254);  clone_41 = _shape_param_254 = None
        bmm_20: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_209, view_210)
        view_211: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_20, _shape_param_255);  bmm_20 = _shape_param_255 = None
        convert_element_type_76: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_211, torch.float32)
        amax_10: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_76, [-1], True)
        sub_31: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_76, amax_10);  convert_element_type_76 = amax_10 = None
        exp_10: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_77: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        eq_10: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(view_211, -inf);  view_211 = None
        logical_not_20: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        where_10: "bf16[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_21, full, convert_element_type_77);  logical_not_21 = convert_element_type_77 = None
        expand_44: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_10, _shape_param_256);  _shape_param_256 = None
        view_212: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_44, _shape_param_257);  expand_44 = _shape_param_257 = None
        expand_45: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_59, _shape_param_258);  permute_59 = _shape_param_258 = None
        clone_42: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_213: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_42, _shape_param_259);  clone_42 = _shape_param_259 = None
        bmm_21: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_212, view_213);  view_212 = None
        view_214: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_21, _shape_param_260);  bmm_21 = _shape_param_260 = None
        permute_61: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
        clone_43: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
        view_215: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_43, _shape_param_261);  clone_43 = _shape_param_261 = None
        view_216: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_215, _shape_param_262);  view_215 = _shape_param_262 = None
        addmm_64: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_11, view_216, permute_9)
        view_217: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_64, _shape_param_263);  addmm_64 = _shape_param_263 = None
        add_84: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_83, view_217);  add_83 = view_217 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_42: "f32[8, 512, 1]" = var_mean_21[0]
        getitem_43: "f32[8, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        add_85: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_32: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_84, getitem_43);  add_84 = getitem_43 = None
        mul_104: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
        mul_105: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_104, arg18_1)
        add_86: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_105, arg19_1);  mul_105 = None
        convert_element_type_78: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_86, torch.bfloat16)
        view_218: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_78, _shape_param_264);  convert_element_type_78 = _shape_param_264 = None
        addmm_65: "bf16[4096, 16384]" = torch.ops.aten.addmm.default(convert_element_type_14, view_218, permute_10)
        view_219: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(addmm_65, _shape_param_265);  _shape_param_265 = None
        mul_106: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
        convert_element_type_79: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_219, torch.float32)
        pow_11: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_79, 3.0);  convert_element_type_79 = None
        mul_107: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_87: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_219, mul_107);  view_219 = mul_107 = None
        mul_108: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_87, 0.7978845608028654);  add_87 = None
        tanh_10: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_108);  mul_108 = None
        add_88: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_109: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_106, add_88);  mul_106 = add_88 = None
        convert_element_type_80: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_109, torch.bfloat16);  mul_109 = None
        view_220: "bf16[4096, 16384]" = torch.ops.aten.view.default(convert_element_type_80, _shape_param_266);  convert_element_type_80 = _shape_param_266 = None
        addmm_66: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_18, view_220, permute_11)
        view_221: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_66, _shape_param_267);  addmm_66 = _shape_param_267 = None
        add_89: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_221, add_86);  view_221 = add_86 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_44: "f32[8, 512, 1]" = var_mean_22[0]
        getitem_45: "f32[8, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        add_90: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_33: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_89, getitem_45);  add_89 = getitem_45 = None
        mul_110: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
        mul_111: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_110, arg24_1)
        add_91: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_111, arg25_1);  mul_111 = None
        convert_element_type_81: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16)
        view_222: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_81, _shape_param_268);  convert_element_type_81 = _shape_param_268 = None
        addmm_67: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_3, view_222, permute_1);  convert_element_type_3 = None
        view_223: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_67, _shape_param_269);  addmm_67 = _shape_param_269 = None
        view_224: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_223, _shape_param_270);  view_223 = _shape_param_270 = None
        permute_62: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
        addmm_68: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_5, view_222, permute_3);  convert_element_type_5 = None
        view_225: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_68, _shape_param_271);  addmm_68 = _shape_param_271 = None
        view_226: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_225, _shape_param_272);  view_225 = _shape_param_272 = None
        permute_63: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        addmm_69: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_7, view_222, permute_5);  convert_element_type_7 = None
        view_227: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_69, _shape_param_273);  addmm_69 = _shape_param_273 = None
        view_228: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_227, _shape_param_274);  view_227 = _shape_param_274 = None
        permute_64: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
        mul_112: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(permute_62, 0.3535533905932738);  permute_62 = None
        permute_65: "bf16[8, 64, 64, 512]" = torch.ops.aten.permute.default(permute_63, [0, 1, 3, 2]);  permute_63 = None
        mul_113: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(permute_65, 0.3535533905932738);  permute_65 = None
        expand_46: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(mul_112, _shape_param_275);  mul_112 = _shape_param_275 = None
        clone_44: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_229: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_44, _shape_param_276);  clone_44 = _shape_param_276 = None
        expand_47: "bf16[8, 64, 64, 512]" = torch.ops.aten.expand.default(mul_113, _shape_param_277);  mul_113 = _shape_param_277 = None
        clone_45: "bf16[8, 64, 64, 512]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_230: "bf16[512, 64, 512]" = torch.ops.aten.view.default(clone_45, _shape_param_278);  clone_45 = _shape_param_278 = None
        bmm_22: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_229, view_230)
        view_231: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_22, _shape_param_279);  bmm_22 = _shape_param_279 = None
        convert_element_type_82: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_231, torch.float32)
        amax_11: "f32[8, 64, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_82, [-1], True)
        sub_34: "f32[8, 64, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_82, amax_11);  convert_element_type_82 = amax_11 = None
        exp_11: "f32[8, 64, 512, 512]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[8, 64, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_83: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        eq_11: "b8[8, 64, 512, 512]" = torch.ops.aten.eq.Scalar(view_231, -inf);  view_231 = None
        logical_not_22: "b8[8, 64, 512, 512]" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[8, 64, 512, 1]" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[8, 64, 512, 1]" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        where_11: "bf16[8, 64, 512, 512]" = torch.ops.aten.where.self(logical_not_23, full, convert_element_type_83);  logical_not_23 = full = convert_element_type_83 = None
        expand_48: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(where_11, _shape_param_280);  _shape_param_280 = None
        view_232: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_48, _shape_param_281);  expand_48 = _shape_param_281 = None
        expand_49: "bf16[8, 64, 512, 64]" = torch.ops.aten.expand.default(permute_64, _shape_param_282);  permute_64 = _shape_param_282 = None
        clone_46: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_233: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_46, _shape_param_283);  clone_46 = _shape_param_283 = None
        bmm_23: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_232, view_233);  view_232 = None
        view_234: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_284);  bmm_23 = _shape_param_284 = None
        permute_66: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_47: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
        view_235: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_47, _shape_param_285);  clone_47 = _shape_param_285 = None
        view_236: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_235, _shape_param_286);  view_235 = _shape_param_286 = None
        addmm_70: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_11, view_236, permute_9);  convert_element_type_11 = None
        view_237: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_70, _shape_param_287);  addmm_70 = _shape_param_287 = None
        add_92: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_91, view_237);  add_91 = view_237 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_46: "f32[8, 512, 1]" = var_mean_23[0]
        getitem_47: "f32[8, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        add_93: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_35: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_92, getitem_47);  add_92 = getitem_47 = None
        mul_114: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
        mul_115: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_114, arg18_1);  arg18_1 = None
        add_94: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_115, arg19_1);  mul_115 = arg19_1 = None
        convert_element_type_84: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16)
        view_238: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_84, _shape_param_288);  convert_element_type_84 = _shape_param_288 = None
        addmm_71: "bf16[4096, 16384]" = torch.ops.aten.addmm.default(convert_element_type_14, view_238, permute_10);  convert_element_type_14 = None
        view_239: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(addmm_71, _shape_param_289);  _shape_param_289 = None
        mul_116: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        convert_element_type_85: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_239, torch.float32)
        pow_12: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_85, 3.0);  convert_element_type_85 = None
        mul_117: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_95: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_239, mul_117);  view_239 = mul_117 = None
        mul_118: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_95, 0.7978845608028654);  add_95 = None
        tanh_11: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_96: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_119: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_116, add_96);  mul_116 = add_96 = None
        convert_element_type_86: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_119, torch.bfloat16);  mul_119 = None
        view_240: "bf16[4096, 16384]" = torch.ops.aten.view.default(convert_element_type_86, _shape_param_290);  convert_element_type_86 = _shape_param_290 = None
        addmm_72: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(convert_element_type_18, view_240, permute_11);  convert_element_type_18 = None
        view_241: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(addmm_72, _shape_param_291);  addmm_72 = _shape_param_291 = None
        add_97: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_241, add_94);  view_241 = add_94 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_48: "f32[8, 512, 1]" = var_mean_24[0]
        getitem_49: "f32[8, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        add_98: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_36: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_97, getitem_49);  add_97 = getitem_49 = None
        mul_120: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
        mul_121: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_120, arg24_1);  arg24_1 = None
        add_99: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_121, arg25_1);  mul_121 = arg25_1 = None
        convert_element_type_87: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg26_1, torch.bfloat16);  arg26_1 = None
        convert_element_type_88: "bf16[128, 4096]" = torch.ops.prims.convert_element_type.default(arg27_1, torch.bfloat16);  arg27_1 = None
        convert_element_type_89: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None
        view_242: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_89, _shape_param_292);  convert_element_type_89 = _shape_param_292 = None
        permute_67: "bf16[4096, 128]" = torch.ops.aten.permute.default(convert_element_type_88, [1, 0]);  convert_element_type_88 = None
        addmm_73: "bf16[4096, 128]" = torch.ops.aten.addmm.default(convert_element_type_87, view_242, permute_67);  convert_element_type_87 = None
        view_243: "bf16[8, 512, 128]" = torch.ops.aten.view.default(addmm_73, _shape_param_293);  _shape_param_293 = None
        mul_122: "bf16[8, 512, 128]" = torch.ops.aten.mul.Tensor(view_243, 0.5)
        convert_element_type_90: "f32[8, 512, 128]" = torch.ops.prims.convert_element_type.default(view_243, torch.float32)
        pow_13: "f32[8, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_90, 3.0);  convert_element_type_90 = None
        mul_123: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_100: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(view_243, mul_123);  view_243 = mul_123 = None
        mul_124: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_12: "f32[8, 512, 128]" = torch.ops.aten.tanh.default(mul_124);  mul_124 = None
        add_101: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_125: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_122, add_101);  mul_122 = add_101 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(mul_125, [2], correction = 0, keepdim = True)
        getitem_50: "f32[8, 512, 1]" = var_mean_25[0]
        getitem_51: "f32[8, 512, 1]" = var_mean_25[1];  var_mean_25 = None
        add_102: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_37: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_125, getitem_51);  mul_125 = None
        mul_126: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = None
        mul_127: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_126, arg28_1);  mul_126 = arg28_1 = None
        add_103: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(mul_127, arg29_1);  mul_127 = arg29_1 = None
        convert_element_type_91: "bf16[30000]" = torch.ops.prims.convert_element_type.default(arg30_1, torch.bfloat16);  arg30_1 = None
        convert_element_type_92: "bf16[30000, 128]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        convert_element_type_93: "bf16[8, 512, 128]" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16);  add_103 = None
        view_244: "bf16[4096, 128]" = torch.ops.aten.view.default(convert_element_type_93, _shape_param_294);  convert_element_type_93 = _shape_param_294 = None
        permute_68: "bf16[128, 30000]" = torch.ops.aten.permute.default(convert_element_type_92, [1, 0]);  convert_element_type_92 = None
        addmm_74: "bf16[4096, 30000]" = torch.ops.aten.addmm.default(convert_element_type_91, view_244, permute_68);  convert_element_type_91 = None
        view_245: "bf16[8, 512, 30000]" = torch.ops.aten.view.default(addmm_74, _shape_param_295);  addmm_74 = _shape_param_295 = None
        view_246: "bf16[4096, 30000]" = torch.ops.aten.view.default(view_245, _shape_param_296);  _shape_param_296 = None
        view_247: "i64[4096]" = torch.ops.aten.view.default(arg31_1, [-1]);  arg31_1 = None
        convert_element_type_94: "f32[4096, 30000]" = torch.ops.prims.convert_element_type.default(view_246, torch.float32);  view_246 = None
        amax_12: "f32[4096, 1]" = torch.ops.aten.amax.default(convert_element_type_94, [1], True)
        sub_38: "f32[4096, 30000]" = torch.ops.aten.sub.Tensor(convert_element_type_94, amax_12);  convert_element_type_94 = None
        exp_12: "f32[4096, 30000]" = torch.ops.aten.exp.default(sub_38)
        sum_13: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log: "f32[4096, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_39: "f32[4096, 30000]" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = None
        convert_element_type_95: "bf16[4096, 30000]" = torch.ops.prims.convert_element_type.default(sub_39, torch.bfloat16);  sub_39 = None
        convert_element_type_96: "f32[4096, 30000]" = torch.ops.prims.convert_element_type.default(convert_element_type_95, torch.float32);  convert_element_type_95 = None
        ne: "b8[4096]" = torch.ops.aten.ne.Scalar(view_247, -100)
        full_1: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "i64[4096]" = torch.ops.aten.where.self(ne, view_247, full_1);  view_247 = full_1 = None
        unsqueeze: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather_1: "f32[4096, 1]" = torch.ops.aten.gather.default(convert_element_type_96, 1, unsqueeze);  convert_element_type_96 = unsqueeze = None
        squeeze: "f32[4096]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg: "f32[4096]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "f32[4096]" = torch.ops.aten.where.self(ne, neg, full_2);  neg = full_2 = None
        sum_14: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_97: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        sum_15: "f32[]" = torch.ops.aten.sum.default(where_13);  where_13 = None
        div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type_97);  sum_15 = None
        permute_69: "bf16[30000, 128]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        permute_70: "bf16[128, 4096]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        div_13: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 4096);  rsqrt_24 = None
        permute_71: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        permute_72: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        div_14: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 4096);  rsqrt_23 = None
        permute_73: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        permute_74: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
        permute_75: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        permute_76: "bf16[512, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
        permute_77: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        permute_78: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_79: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        div_15: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 4096);  rsqrt_22 = None
        div_16: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 4096);  rsqrt_21 = None
        permute_80: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
        permute_81: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
        permute_82: "bf16[512, 512, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        div_17: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 4096);  rsqrt_20 = None
        div_18: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 4096);  rsqrt_19 = None
        permute_83: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
        permute_84: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
        permute_85: "bf16[512, 512, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
        div_19: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 4096);  rsqrt_18 = None
        div_20: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 4096);  rsqrt_17 = None
        permute_86: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
        permute_87: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
        permute_88: "bf16[512, 512, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1]);  view_170 = None
        div_21: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 4096);  rsqrt_16 = None
        div_22: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 4096);  rsqrt_15 = None
        permute_89: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
        permute_90: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_149, [0, 2, 1]);  view_149 = None
        permute_91: "bf16[512, 512, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1]);  view_150 = None
        div_23: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 4096);  rsqrt_14 = None
        div_24: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 4096);  rsqrt_13 = None
        permute_92: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
        permute_93: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_129, [0, 2, 1]);  view_129 = None
        permute_94: "bf16[512, 512, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
        div_25: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 4096);  rsqrt_12 = None
        div_26: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 4096);  rsqrt_11 = None
        permute_95: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
        permute_96: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_109, [0, 2, 1]);  view_109 = None
        permute_97: "bf16[512, 512, 64]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
        div_27: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 4096);  rsqrt_10 = None
        div_28: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 4096);  rsqrt_9 = None
        permute_98: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_93, [0, 2, 1]);  view_93 = None
        permute_99: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_89, [0, 2, 1]);  view_89 = None
        permute_100: "bf16[512, 512, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1]);  view_90 = None
        div_29: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 4096);  rsqrt_8 = None
        div_30: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 4096);  rsqrt_7 = None
        permute_101: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_73, [0, 2, 1]);  view_73 = None
        permute_102: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
        permute_103: "bf16[512, 512, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None
        div_31: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 4096);  rsqrt_6 = None
        div_32: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 4096);  rsqrt_5 = None
        permute_104: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_105: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_49, [0, 2, 1]);  view_49 = None
        permute_106: "bf16[512, 512, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
        div_33: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 4096);  rsqrt_4 = None
        div_34: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 4096);  rsqrt_3 = None
        permute_107: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
        permute_108: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
        permute_109: "bf16[512, 512, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
        div_35: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 4096);  rsqrt_2 = None
        permute_110: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        permute_111: "bf16[512, 64, 512]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_112: "bf16[512, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_113: "bf16[4096, 128]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        div_36: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        return (gather, mul, view, view_2, where, view_16, add_4, getitem_3, rsqrt_1, view_18, addmm_5, view_20, mul_10, view_22, where_1, view_36, mul_14, view_38, addmm_11, view_40, mul_20, view_42, where_2, view_56, mul_24, view_58, addmm_17, view_60, mul_30, view_62, where_3, view_76, mul_34, view_78, addmm_23, view_80, mul_40, view_82, where_4, view_96, mul_44, view_98, addmm_29, view_100, mul_50, view_102, where_5, view_116, mul_54, view_118, addmm_35, view_120, mul_60, view_122, where_6, view_136, mul_64, view_138, addmm_41, view_140, mul_70, view_142, where_7, view_156, mul_74, view_158, addmm_47, view_160, mul_80, view_162, where_8, view_176, mul_84, view_178, addmm_53, view_180, mul_90, view_182, where_9, view_196, mul_94, view_198, addmm_59, view_200, mul_100, view_202, where_10, view_216, mul_104, view_218, addmm_65, view_220, mul_110, view_222, where_11, view_236, mul_114, view_238, addmm_71, view_240, mul_120, view_242, addmm_73, getitem_51, rsqrt_25, view_244, view_245, amax_12, log, convert_element_type_97, div_12, permute_69, permute_70, div_13, permute_71, permute_72, div_14, permute_73, permute_74, permute_75, permute_76, permute_77, permute_78, permute_79, div_15, div_16, permute_80, permute_81, permute_82, div_17, div_18, permute_83, permute_84, permute_85, div_19, div_20, permute_86, permute_87, permute_88, div_21, div_22, permute_89, permute_90, permute_91, div_23, div_24, permute_92, permute_93, permute_94, div_25, div_26, permute_95, permute_96, permute_97, div_27, div_28, permute_98, permute_99, permute_100, div_29, div_30, permute_101, permute_102, permute_103, div_31, div_32, permute_104, permute_105, permute_106, div_33, div_34, permute_107, permute_108, permute_109, div_35, permute_110, permute_111, permute_112, permute_113, div_36)



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
