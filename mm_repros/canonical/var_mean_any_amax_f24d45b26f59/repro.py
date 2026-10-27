"""
Standalone repro captured via capture_hook.
Label: BertForMaskedLM
Pattern hash: f24d45b26f59
Shape hash: 57cb83f0
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
    def forward(self, arg0_1: "bf16[30522, 768]", arg1_1: "i64[32, 512]", arg2_1: "i64[1, 512]", arg3_1: "i64[1, 512]", arg4_1: "bf16[2, 768]", arg5_1: "bf16[512, 768]", arg6_1: "bf16[768]", arg7_1: "bf16[768]", arg8_1: "bf16[768, 768]", arg9_1: "bf16[768]", arg10_1: "bf16[768, 768]", arg11_1: "bf16[768]", arg12_1: "bf16[768, 768]", arg13_1: "bf16[768]", arg14_1: "bf16[768, 768]", arg15_1: "bf16[768]", arg16_1: "bf16[768]", arg17_1: "bf16[768]", arg18_1: "bf16[3072, 768]", arg19_1: "bf16[3072]", arg20_1: "bf16[768, 3072]", arg21_1: "bf16[768]", arg22_1: "bf16[768]", arg23_1: "bf16[768]", arg24_1: "bf16[768, 768]", arg25_1: "bf16[768]", arg26_1: "bf16[768, 768]", arg27_1: "bf16[768]", arg28_1: "bf16[768, 768]", arg29_1: "bf16[768]", arg30_1: "bf16[768, 768]", arg31_1: "bf16[768]", arg32_1: "bf16[768]", arg33_1: "bf16[768]", arg34_1: "bf16[3072, 768]", arg35_1: "bf16[3072]", arg36_1: "bf16[768, 3072]", arg37_1: "bf16[768]", arg38_1: "bf16[768]", arg39_1: "bf16[768]", arg40_1: "bf16[768, 768]", arg41_1: "bf16[768]", arg42_1: "bf16[768, 768]", arg43_1: "bf16[768]", arg44_1: "bf16[768, 768]", arg45_1: "bf16[768]", arg46_1: "bf16[768, 768]", arg47_1: "bf16[768]", arg48_1: "bf16[768]", arg49_1: "bf16[768]", arg50_1: "bf16[3072, 768]", arg51_1: "bf16[3072]", arg52_1: "bf16[768, 3072]", arg53_1: "bf16[768]", arg54_1: "bf16[768]", arg55_1: "bf16[768]", arg56_1: "bf16[768, 768]", arg57_1: "bf16[768]", arg58_1: "bf16[768, 768]", arg59_1: "bf16[768]", arg60_1: "bf16[768, 768]", arg61_1: "bf16[768]", arg62_1: "bf16[768, 768]", arg63_1: "bf16[768]", arg64_1: "bf16[768]", arg65_1: "bf16[768]", arg66_1: "bf16[3072, 768]", arg67_1: "bf16[3072]", arg68_1: "bf16[768, 3072]", arg69_1: "bf16[768]", arg70_1: "bf16[768]", arg71_1: "bf16[768]", arg72_1: "bf16[768, 768]", arg73_1: "bf16[768]", arg74_1: "bf16[768, 768]", arg75_1: "bf16[768]", arg76_1: "bf16[768, 768]", arg77_1: "bf16[768]", arg78_1: "bf16[768, 768]", arg79_1: "bf16[768]", arg80_1: "bf16[768]", arg81_1: "bf16[768]", arg82_1: "bf16[3072, 768]", arg83_1: "bf16[3072]", arg84_1: "bf16[768, 3072]", arg85_1: "bf16[768]", arg86_1: "bf16[768]", arg87_1: "bf16[768]", arg88_1: "bf16[768, 768]", arg89_1: "bf16[768]", arg90_1: "bf16[768, 768]", arg91_1: "bf16[768]", arg92_1: "bf16[768, 768]", arg93_1: "bf16[768]", arg94_1: "bf16[768, 768]", arg95_1: "bf16[768]", arg96_1: "bf16[768]", arg97_1: "bf16[768]", arg98_1: "bf16[3072, 768]", arg99_1: "bf16[3072]", arg100_1: "bf16[768, 3072]", arg101_1: "bf16[768]", arg102_1: "bf16[768]", arg103_1: "bf16[768]", arg104_1: "bf16[768, 768]", arg105_1: "bf16[768]", arg106_1: "bf16[768, 768]", arg107_1: "bf16[768]", arg108_1: "bf16[768, 768]", arg109_1: "bf16[768]", arg110_1: "bf16[768, 768]", arg111_1: "bf16[768]", arg112_1: "bf16[768]", arg113_1: "bf16[768]", arg114_1: "bf16[3072, 768]", arg115_1: "bf16[3072]", arg116_1: "bf16[768, 3072]", arg117_1: "bf16[768]", arg118_1: "bf16[768]", arg119_1: "bf16[768]", arg120_1: "bf16[768, 768]", arg121_1: "bf16[768]", arg122_1: "bf16[768, 768]", arg123_1: "bf16[768]", arg124_1: "bf16[768, 768]", arg125_1: "bf16[768]", arg126_1: "bf16[768, 768]", arg127_1: "bf16[768]", arg128_1: "bf16[768]", arg129_1: "bf16[768]", arg130_1: "bf16[3072, 768]", arg131_1: "bf16[3072]", arg132_1: "bf16[768, 3072]", arg133_1: "bf16[768]", arg134_1: "bf16[768]", arg135_1: "bf16[768]", arg136_1: "bf16[768, 768]", arg137_1: "bf16[768]", arg138_1: "bf16[768, 768]", arg139_1: "bf16[768]", arg140_1: "bf16[768, 768]", arg141_1: "bf16[768]", arg142_1: "bf16[768, 768]", arg143_1: "bf16[768]", arg144_1: "bf16[768]", arg145_1: "bf16[768]", arg146_1: "bf16[3072, 768]", arg147_1: "bf16[3072]", arg148_1: "bf16[768, 3072]", arg149_1: "bf16[768]", arg150_1: "bf16[768]", arg151_1: "bf16[768]", arg152_1: "bf16[768, 768]", arg153_1: "bf16[768]", arg154_1: "bf16[768, 768]", arg155_1: "bf16[768]", arg156_1: "bf16[768, 768]", arg157_1: "bf16[768]", arg158_1: "bf16[768, 768]", arg159_1: "bf16[768]", arg160_1: "bf16[768]", arg161_1: "bf16[768]", arg162_1: "bf16[3072, 768]", arg163_1: "bf16[3072]", arg164_1: "bf16[768, 3072]", arg165_1: "bf16[768]", arg166_1: "bf16[768]", arg167_1: "bf16[768]", arg168_1: "bf16[768, 768]", arg169_1: "bf16[768]", arg170_1: "bf16[768, 768]", arg171_1: "bf16[768]", arg172_1: "bf16[768, 768]", arg173_1: "bf16[768]", arg174_1: "bf16[768, 768]", arg175_1: "bf16[768]", arg176_1: "bf16[768]", arg177_1: "bf16[768]", arg178_1: "bf16[3072, 768]", arg179_1: "bf16[3072]", arg180_1: "bf16[768, 3072]", arg181_1: "bf16[768]", arg182_1: "bf16[768]", arg183_1: "bf16[768]", arg184_1: "bf16[768, 768]", arg185_1: "bf16[768]", arg186_1: "bf16[768, 768]", arg187_1: "bf16[768]", arg188_1: "bf16[768, 768]", arg189_1: "bf16[768]", arg190_1: "bf16[768, 768]", arg191_1: "bf16[768]", arg192_1: "bf16[768]", arg193_1: "bf16[768]", arg194_1: "bf16[3072, 768]", arg195_1: "bf16[3072]", arg196_1: "bf16[768, 3072]", arg197_1: "bf16[768]", arg198_1: "bf16[768]", arg199_1: "bf16[768]", arg200_1: "bf16[768, 768]", arg201_1: "bf16[768]", arg202_1: "i64[32, 512]", arg203_1: "bf16[30522]", arg204_1: "bf16[768]", arg205_1: "bf16[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331):
        # No stacktrace found for following nodes
        embedding: "bf16[32, 512, 768]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg1_1 = None
        expand: "i64[1, 512]" = torch.ops.aten.expand.default(arg2_1, [1, -1]);  arg2_1 = None
        gather: "i64[1, 512]" = torch.ops.aten.gather.default(expand, 1, arg3_1);  expand = None
        expand_1: "i64[32, 512]" = torch.ops.aten.expand.default(gather, _shape_param_0);  gather = _shape_param_0 = None
        embedding_1: "bf16[32, 512, 768]" = torch.ops.aten.embedding.default(arg4_1, expand_1);  arg4_1 = expand_1 = None
        add: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2: "bf16[1, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, arg3_1);  arg5_1 = arg3_1 = None
        add_1: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        convert_element_type: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_2: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg6_1);  mul = arg6_1 = None
        add_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg7_1);  mul_1 = arg7_1 = None
        convert_element_type_1: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  _shape_param_1 = None
        permute: "bf16[768, 768]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg9_1, view, permute);  arg9_1 = view = permute = None
        view_1: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        view_2: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_1, _shape_param_3);  view_1 = _shape_param_3 = None
        permute_1: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        mul_2: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_1, 0.3535533905932738);  permute_1 = None
        expand_2: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_2, _shape_param_4);  mul_2 = _shape_param_4 = None
        clone: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_3: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        view_4: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_6);  _shape_param_6 = None
        permute_2: "bf16[768, 768]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_1: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg11_1, view_4, permute_2);  arg11_1 = view_4 = permute_2 = None
        view_5: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_1, _shape_param_7);  addmm_1 = _shape_param_7 = None
        view_6: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_5, _shape_param_8);  view_5 = _shape_param_8 = None
        permute_3: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_4: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        mul_3: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_4, 0.3535533905932738);  permute_4 = None
        expand_3: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_3, _shape_param_9);  mul_3 = _shape_param_9 = None
        clone_1: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_7: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_1, _shape_param_10);  clone_1 = _shape_param_10 = None
        bmm: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_3, view_7);  view_3 = view_7 = None
        view_8: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_11);  bmm = _shape_param_11 = None
        eq: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_8, -inf)
        logical_not: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full: "bf16[32, 12, 512, 512]" = torch.ops.aten.full.default(_shape_param_12, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_12 = None
        convert_element_type_2: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        amax: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
        sub_1: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        where: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full, convert_element_type_3);  logical_not_1 = full = convert_element_type_3 = None
        expand_4: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(where, _shape_param_13);  where = _shape_param_13 = None
        view_9: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_4, _shape_param_14);  expand_4 = _shape_param_14 = None
        view_10: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_15);  _shape_param_15 = None
        permute_5: "bf16[768, 768]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_2: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg13_1, view_10, permute_5);  arg13_1 = view_10 = permute_5 = None
        view_11: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_2, _shape_param_16);  addmm_2 = _shape_param_16 = None
        view_12: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_11, _shape_param_17);  view_11 = _shape_param_17 = None
        permute_6: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        expand_5: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_6, _shape_param_18);  permute_6 = _shape_param_18 = None
        clone_2: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_13: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_19);  clone_2 = _shape_param_19 = None
        bmm_1: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_9, view_13);  view_9 = view_13 = None
        view_14: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_20);  bmm_1 = _shape_param_20 = None
        permute_7: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_3: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_15: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_3, _shape_param_21);  clone_3 = _shape_param_21 = None
        view_16: "bf16[16384, 768]" = torch.ops.aten.view.default(view_15, _shape_param_22);  view_15 = _shape_param_22 = None
        permute_8: "bf16[768, 768]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        addmm_3: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg15_1, view_16, permute_8);  arg15_1 = view_16 = permute_8 = None
        view_17: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_3, _shape_param_23);  addmm_3 = _shape_param_23 = None
        add_4: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_17, convert_element_type_1);  view_17 = convert_element_type_1 = None
        convert_element_type_4: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_4, torch.float32);  add_4 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_4, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_4, getitem_3);  convert_element_type_4 = getitem_3 = None
        add_5: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, arg16_1);  mul_4 = arg16_1 = None
        add_6: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_5, arg17_1);  mul_5 = arg17_1 = None
        convert_element_type_5: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        view_18: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_24);  _shape_param_24 = None
        permute_9: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_4: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(arg19_1, view_18, permute_9);  arg19_1 = view_18 = permute_9 = None
        view_19: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_4, _shape_param_25);  addmm_4 = _shape_param_25 = None
        convert_element_type_6: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_6: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.5)
        mul_7: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476);  convert_element_type_6 = None
        erf: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_7: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_7);  mul_6 = add_7 = None
        convert_element_type_7: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        view_20: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_26);  convert_element_type_7 = _shape_param_26 = None
        permute_10: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        addmm_5: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg21_1, view_20, permute_10);  arg21_1 = view_20 = permute_10 = None
        view_21: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_5, _shape_param_27);  addmm_5 = _shape_param_27 = None
        add_8: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_21, convert_element_type_5);  view_21 = convert_element_type_5 = None
        convert_element_type_8: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_8, torch.float32);  add_8 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_8, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[32, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_8, getitem_5);  convert_element_type_8 = getitem_5 = None
        add_9: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg22_1);  mul_9 = arg22_1 = None
        add_10: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_10, arg23_1);  mul_10 = arg23_1 = None
        convert_element_type_9: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None
        view_22: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_28);  _shape_param_28 = None
        permute_11: "bf16[768, 768]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_6: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg25_1, view_22, permute_11);  arg25_1 = view_22 = permute_11 = None
        view_23: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_6, _shape_param_29);  addmm_6 = _shape_param_29 = None
        view_24: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_23, _shape_param_30);  view_23 = _shape_param_30 = None
        permute_12: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        mul_11: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_12, 0.3535533905932738);  permute_12 = None
        expand_6: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_11, _shape_param_31);  mul_11 = _shape_param_31 = None
        clone_4: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_25: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_32);  clone_4 = _shape_param_32 = None
        view_26: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_33);  _shape_param_33 = None
        permute_13: "bf16[768, 768]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_7: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg27_1, view_26, permute_13);  arg27_1 = view_26 = permute_13 = None
        view_27: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_7, _shape_param_34);  addmm_7 = _shape_param_34 = None
        view_28: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_27, _shape_param_35);  view_27 = _shape_param_35 = None
        permute_14: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        permute_15: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
        mul_12: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_15, 0.3535533905932738);  permute_15 = None
        expand_7: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_12, _shape_param_36);  mul_12 = _shape_param_36 = None
        clone_5: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_29: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_5, _shape_param_37);  clone_5 = _shape_param_37 = None
        bmm_2: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_25, view_29);  view_25 = view_29 = None
        view_30: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_2, _shape_param_38);  bmm_2 = _shape_param_38 = None
        eq_1: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_30, -inf)
        logical_not_2: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        full_1: "bf16[32, 12, 512, 512]" = torch.ops.aten.full.default(_shape_param_39, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_39 = None
        convert_element_type_10: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None
        amax_1: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_10, [-1], True)
        sub_4: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_10, amax_1);  convert_element_type_10 = amax_1 = None
        exp_1: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_11: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        where_1: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_3, full_1, convert_element_type_11);  logical_not_3 = full_1 = convert_element_type_11 = None
        expand_8: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_1, _shape_param_40);  where_1 = _shape_param_40 = None
        view_31: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_8, _shape_param_41);  expand_8 = _shape_param_41 = None
        view_32: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_42);  _shape_param_42 = None
        permute_16: "bf16[768, 768]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_8: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg29_1, view_32, permute_16);  arg29_1 = view_32 = permute_16 = None
        view_33: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_8, _shape_param_43);  addmm_8 = _shape_param_43 = None
        view_34: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_33, _shape_param_44);  view_33 = _shape_param_44 = None
        permute_17: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        expand_9: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_17, _shape_param_45);  permute_17 = _shape_param_45 = None
        clone_6: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_35: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_6, _shape_param_46);  clone_6 = _shape_param_46 = None
        bmm_3: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_31, view_35);  view_31 = view_35 = None
        view_36: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_47);  bmm_3 = _shape_param_47 = None
        permute_18: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_7: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_37: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_7, _shape_param_48);  clone_7 = _shape_param_48 = None
        view_38: "bf16[16384, 768]" = torch.ops.aten.view.default(view_37, _shape_param_49);  view_37 = _shape_param_49 = None
        permute_19: "bf16[768, 768]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        addmm_9: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg31_1, view_38, permute_19);  arg31_1 = view_38 = permute_19 = None
        view_39: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_9, _shape_param_50);  addmm_9 = _shape_param_50 = None
        add_11: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_39, convert_element_type_9);  view_39 = convert_element_type_9 = None
        convert_element_type_12: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_11, torch.float32);  add_11 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_12, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[32, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_12, getitem_7);  convert_element_type_12 = getitem_7 = None
        add_12: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_13: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_14: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_13, arg32_1);  mul_13 = arg32_1 = None
        add_13: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_14, arg33_1);  mul_14 = arg33_1 = None
        convert_element_type_13: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None
        view_40: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_13, _shape_param_51);  _shape_param_51 = None
        permute_20: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_10: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(arg35_1, view_40, permute_20);  arg35_1 = view_40 = permute_20 = None
        view_41: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_10, _shape_param_52);  addmm_10 = _shape_param_52 = None
        convert_element_type_14: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_15: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.5)
        mul_16: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.7071067811865476);  convert_element_type_14 = None
        erf_1: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_16);  mul_16 = None
        add_14: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_17: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_15, add_14);  mul_15 = add_14 = None
        convert_element_type_15: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        view_42: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_15, _shape_param_53);  convert_element_type_15 = _shape_param_53 = None
        permute_21: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        addmm_11: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg37_1, view_42, permute_21);  arg37_1 = view_42 = permute_21 = None
        view_43: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_11, _shape_param_54);  addmm_11 = _shape_param_54 = None
        add_15: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_43, convert_element_type_13);  view_43 = convert_element_type_13 = None
        convert_element_type_16: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_15, torch.float32);  add_15 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_16, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[32, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_6: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_16, getitem_9);  convert_element_type_16 = getitem_9 = None
        add_16: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_18: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_19: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg38_1);  mul_18 = arg38_1 = None
        add_17: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_19, arg39_1);  mul_19 = arg39_1 = None
        convert_element_type_17: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None
        view_44: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_17, _shape_param_55);  _shape_param_55 = None
        permute_22: "bf16[768, 768]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        addmm_12: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg41_1, view_44, permute_22);  arg41_1 = view_44 = permute_22 = None
        view_45: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_12, _shape_param_56);  addmm_12 = _shape_param_56 = None
        view_46: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_45, _shape_param_57);  view_45 = _shape_param_57 = None
        permute_23: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        mul_20: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_23, 0.3535533905932738);  permute_23 = None
        expand_10: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_20, _shape_param_58);  mul_20 = _shape_param_58 = None
        clone_8: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_47: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_59);  clone_8 = _shape_param_59 = None
        view_48: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_17, _shape_param_60);  _shape_param_60 = None
        permute_24: "bf16[768, 768]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        addmm_13: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg43_1, view_48, permute_24);  arg43_1 = view_48 = permute_24 = None
        view_49: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_13, _shape_param_61);  addmm_13 = _shape_param_61 = None
        view_50: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_49, _shape_param_62);  view_49 = _shape_param_62 = None
        permute_25: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        permute_26: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
        mul_21: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_26, 0.3535533905932738);  permute_26 = None
        expand_11: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_21, _shape_param_63);  mul_21 = _shape_param_63 = None
        clone_9: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_51: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_9, _shape_param_64);  clone_9 = _shape_param_64 = None
        bmm_4: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_47, view_51);  view_47 = view_51 = None
        view_52: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_4, _shape_param_65);  bmm_4 = _shape_param_65 = None
        eq_2: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_52, -inf)
        logical_not_4: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        full_2: "bf16[32, 12, 512, 512]" = torch.ops.aten.full.default(_shape_param_66, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_66 = None
        convert_element_type_18: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_52, torch.float32);  view_52 = None
        amax_2: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_18, [-1], True)
        sub_7: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_18, amax_2);  convert_element_type_18 = amax_2 = None
        exp_2: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_19: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        where_2: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_5, full_2, convert_element_type_19);  logical_not_5 = full_2 = convert_element_type_19 = None
        expand_12: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_2, _shape_param_67);  where_2 = _shape_param_67 = None
        view_53: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_12, _shape_param_68);  expand_12 = _shape_param_68 = None
        view_54: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_17, _shape_param_69);  _shape_param_69 = None
        permute_27: "bf16[768, 768]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_14: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg45_1, view_54, permute_27);  arg45_1 = view_54 = permute_27 = None
        view_55: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_14, _shape_param_70);  addmm_14 = _shape_param_70 = None
        view_56: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_55, _shape_param_71);  view_55 = _shape_param_71 = None
        permute_28: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        expand_13: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_28, _shape_param_72);  permute_28 = _shape_param_72 = None
        clone_10: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_57: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_10, _shape_param_73);  clone_10 = _shape_param_73 = None
        bmm_5: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_53, view_57);  view_53 = view_57 = None
        view_58: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_5, _shape_param_74);  bmm_5 = _shape_param_74 = None
        permute_29: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_11: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_59: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_11, _shape_param_75);  clone_11 = _shape_param_75 = None
        view_60: "bf16[16384, 768]" = torch.ops.aten.view.default(view_59, _shape_param_76);  view_59 = _shape_param_76 = None
        permute_30: "bf16[768, 768]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        addmm_15: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg47_1, view_60, permute_30);  arg47_1 = view_60 = permute_30 = None
        view_61: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_15, _shape_param_77);  addmm_15 = _shape_param_77 = None
        add_18: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_61, convert_element_type_17);  view_61 = convert_element_type_17 = None
        convert_element_type_20: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_18, torch.float32);  add_18 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_20, [2], correction = 0, keepdim = True)
        getitem_10: "f32[32, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[32, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_20, getitem_11);  convert_element_type_20 = getitem_11 = None
        add_19: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_22: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_23: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, arg48_1);  mul_22 = arg48_1 = None
        add_20: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_23, arg49_1);  mul_23 = arg49_1 = None
        convert_element_type_21: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_20, torch.bfloat16);  add_20 = None
        view_62: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_21, _shape_param_78);  _shape_param_78 = None
        permute_31: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        addmm_16: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(arg51_1, view_62, permute_31);  arg51_1 = view_62 = permute_31 = None
        view_63: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_16, _shape_param_79);  addmm_16 = _shape_param_79 = None
        convert_element_type_22: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        mul_24: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.5)
        mul_25: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.7071067811865476);  convert_element_type_22 = None
        erf_2: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_21: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_26: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_24, add_21);  mul_24 = add_21 = None
        convert_element_type_23: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_26, torch.bfloat16);  mul_26 = None
        view_64: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_23, _shape_param_80);  convert_element_type_23 = _shape_param_80 = None
        permute_32: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        addmm_17: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg53_1, view_64, permute_32);  arg53_1 = view_64 = permute_32 = None
        view_65: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_17, _shape_param_81);  addmm_17 = _shape_param_81 = None
        add_22: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_65, convert_element_type_21);  view_65 = convert_element_type_21 = None
        convert_element_type_24: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_22, torch.float32);  add_22 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_24, [2], correction = 0, keepdim = True)
        getitem_12: "f32[32, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[32, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_9: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_24, getitem_13);  convert_element_type_24 = getitem_13 = None
        add_23: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_27: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_28: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_27, arg54_1);  mul_27 = arg54_1 = None
        add_24: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_28, arg55_1);  mul_28 = arg55_1 = None
        convert_element_type_25: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None
        view_66: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_25, _shape_param_82);  _shape_param_82 = None
        permute_33: "bf16[768, 768]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_18: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg57_1, view_66, permute_33);  arg57_1 = view_66 = permute_33 = None
        view_67: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_18, _shape_param_83);  addmm_18 = _shape_param_83 = None
        view_68: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_67, _shape_param_84);  view_67 = _shape_param_84 = None
        permute_34: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        mul_29: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_34, 0.3535533905932738);  permute_34 = None
        expand_14: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_29, _shape_param_85);  mul_29 = _shape_param_85 = None
        clone_12: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_69: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_86);  clone_12 = _shape_param_86 = None
        view_70: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_25, _shape_param_87);  _shape_param_87 = None
        permute_35: "bf16[768, 768]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_19: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg59_1, view_70, permute_35);  arg59_1 = view_70 = permute_35 = None
        view_71: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_19, _shape_param_88);  addmm_19 = _shape_param_88 = None
        view_72: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_71, _shape_param_89);  view_71 = _shape_param_89 = None
        permute_36: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        permute_37: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
        mul_30: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_37, 0.3535533905932738);  permute_37 = None
        expand_15: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_30, _shape_param_90);  mul_30 = _shape_param_90 = None
        clone_13: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_73: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_13, _shape_param_91);  clone_13 = _shape_param_91 = None
        bmm_6: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_69, view_73);  view_69 = view_73 = None
        view_74: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_6, _shape_param_92);  bmm_6 = _shape_param_92 = None
        eq_3: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_74, -inf)
        logical_not_6: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        full_3: "bf16[32, 12, 512, 512]" = torch.ops.aten.full.default(_shape_param_93, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_93 = None
        convert_element_type_26: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_74, torch.float32);  view_74 = None
        amax_3: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_26, [-1], True)
        sub_10: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_26, amax_3);  convert_element_type_26 = amax_3 = None
        exp_3: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_27: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        where_3: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_7, full_3, convert_element_type_27);  logical_not_7 = full_3 = convert_element_type_27 = None
        expand_16: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_3, _shape_param_94);  where_3 = _shape_param_94 = None
        view_75: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_16, _shape_param_95);  expand_16 = _shape_param_95 = None
        view_76: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_25, _shape_param_96);  _shape_param_96 = None
        permute_38: "bf16[768, 768]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_20: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg61_1, view_76, permute_38);  arg61_1 = view_76 = permute_38 = None
        view_77: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_20, _shape_param_97);  addmm_20 = _shape_param_97 = None
        view_78: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_77, _shape_param_98);  view_77 = _shape_param_98 = None
        permute_39: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
        expand_17: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_39, _shape_param_99);  permute_39 = _shape_param_99 = None
        clone_14: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_79: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_14, _shape_param_100);  clone_14 = _shape_param_100 = None
        bmm_7: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_75, view_79);  view_75 = view_79 = None
        view_80: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_101);  bmm_7 = _shape_param_101 = None
        permute_40: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        clone_15: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_81: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_15, _shape_param_102);  clone_15 = _shape_param_102 = None
        view_82: "bf16[16384, 768]" = torch.ops.aten.view.default(view_81, _shape_param_103);  view_81 = _shape_param_103 = None
        permute_41: "bf16[768, 768]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_21: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg63_1, view_82, permute_41);  arg63_1 = view_82 = permute_41 = None
        view_83: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_21, _shape_param_104);  addmm_21 = _shape_param_104 = None
        add_25: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_83, convert_element_type_25);  view_83 = convert_element_type_25 = None
        convert_element_type_28: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_25, torch.float32);  add_25 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_28, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[32, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_28, getitem_15);  convert_element_type_28 = getitem_15 = None
        add_26: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_31: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_32: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_31, arg64_1);  mul_31 = arg64_1 = None
        add_27: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_32, arg65_1);  mul_32 = arg65_1 = None
        convert_element_type_29: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16);  add_27 = None
        view_84: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_29, _shape_param_105);  _shape_param_105 = None
        permute_42: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_22: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(arg67_1, view_84, permute_42);  arg67_1 = view_84 = permute_42 = None
        view_85: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_22, _shape_param_106);  addmm_22 = _shape_param_106 = None
        convert_element_type_30: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_33: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 0.5)
        mul_34: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 0.7071067811865476);  convert_element_type_30 = None
        erf_3: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_28: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_35: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_33, add_28);  mul_33 = add_28 = None
        convert_element_type_31: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None
        view_86: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_31, _shape_param_107);  convert_element_type_31 = _shape_param_107 = None
        permute_43: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_23: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg69_1, view_86, permute_43);  arg69_1 = view_86 = permute_43 = None
        view_87: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_23, _shape_param_108);  addmm_23 = _shape_param_108 = None
        add_29: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_87, convert_element_type_29);  view_87 = convert_element_type_29 = None
        convert_element_type_32: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_29, torch.float32);  add_29 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_32, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[32, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_12: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_32, getitem_17);  convert_element_type_32 = getitem_17 = None
        add_30: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_36: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_37: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, arg70_1);  mul_36 = arg70_1 = None
        add_31: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, arg71_1);  mul_37 = arg71_1 = None
        convert_element_type_33: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None
        view_88: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_33, _shape_param_109);  _shape_param_109 = None
        permute_44: "bf16[768, 768]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        addmm_24: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg73_1, view_88, permute_44);  arg73_1 = view_88 = permute_44 = None
        view_89: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_24, _shape_param_110);  addmm_24 = _shape_param_110 = None
        view_90: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_89, _shape_param_111);  view_89 = _shape_param_111 = None
        permute_45: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        mul_38: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_45, 0.3535533905932738);  permute_45 = None
        expand_18: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_38, _shape_param_112);  mul_38 = _shape_param_112 = None
        clone_16: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_91: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_113);  clone_16 = _shape_param_113 = None
        view_92: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_33, _shape_param_114);  _shape_param_114 = None
        permute_46: "bf16[768, 768]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        addmm_25: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg75_1, view_92, permute_46);  arg75_1 = view_92 = permute_46 = None
        view_93: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_25, _shape_param_115);  addmm_25 = _shape_param_115 = None
        view_94: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_93, _shape_param_116);  view_93 = _shape_param_116 = None
        permute_47: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        permute_48: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
        mul_39: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_48, 0.3535533905932738);  permute_48 = None
        expand_19: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_39, _shape_param_117);  mul_39 = _shape_param_117 = None
        clone_17: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_95: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_17, _shape_param_118);  clone_17 = _shape_param_118 = None
        bmm_8: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_91, view_95);  view_91 = view_95 = None
        view_96: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_8, _shape_param_119);  bmm_8 = _shape_param_119 = None
        eq_4: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_96, -inf)
        logical_not_8: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        full_4: "bf16[32, 12, 512, 512]" = torch.ops.aten.full.default(_shape_param_120, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_120 = None
        convert_element_type_34: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_96, torch.float32);  view_96 = None
        amax_4: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_34, [-1], True)
        sub_13: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_34, amax_4);  convert_element_type_34 = amax_4 = None
        exp_4: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_35: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        where_4: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_9, full_4, convert_element_type_35);  logical_not_9 = full_4 = convert_element_type_35 = None
        expand_20: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_4, _shape_param_121);  where_4 = _shape_param_121 = None
        view_97: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_20, _shape_param_122);  expand_20 = _shape_param_122 = None
        view_98: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_33, _shape_param_123);  _shape_param_123 = None
        permute_49: "bf16[768, 768]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        addmm_26: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg77_1, view_98, permute_49);  arg77_1 = view_98 = permute_49 = None
        view_99: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_26, _shape_param_124);  addmm_26 = _shape_param_124 = None
        view_100: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_99, _shape_param_125);  view_99 = _shape_param_125 = None
        permute_50: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        expand_21: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_50, _shape_param_126);  permute_50 = _shape_param_126 = None
        clone_18: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_101: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_18, _shape_param_127);  clone_18 = _shape_param_127 = None
        bmm_9: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_97, view_101);  view_97 = view_101 = None
        view_102: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_9, _shape_param_128);  bmm_9 = _shape_param_128 = None
        permute_51: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_19: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_103: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_19, _shape_param_129);  clone_19 = _shape_param_129 = None
        view_104: "bf16[16384, 768]" = torch.ops.aten.view.default(view_103, _shape_param_130);  view_103 = _shape_param_130 = None
        permute_52: "bf16[768, 768]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        addmm_27: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg79_1, view_104, permute_52);  arg79_1 = view_104 = permute_52 = None
        view_105: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_27, _shape_param_131);  addmm_27 = _shape_param_131 = None
        add_32: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_105, convert_element_type_33);  view_105 = convert_element_type_33 = None
        convert_element_type_36: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_32, torch.float32);  add_32 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_36, [2], correction = 0, keepdim = True)
        getitem_18: "f32[32, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[32, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_36, getitem_19);  convert_element_type_36 = getitem_19 = None
        add_33: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        mul_40: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_41: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_40, arg80_1);  mul_40 = arg80_1 = None
        add_34: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_41, arg81_1);  mul_41 = arg81_1 = None
        convert_element_type_37: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16);  add_34 = None
        view_106: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_37, _shape_param_132);  _shape_param_132 = None
        permute_53: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        addmm_28: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(arg83_1, view_106, permute_53);  arg83_1 = view_106 = permute_53 = None
        view_107: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_28, _shape_param_133);  addmm_28 = _shape_param_133 = None
        convert_element_type_38: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        mul_42: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.5)
        mul_43: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.7071067811865476);  convert_element_type_38 = None
        erf_4: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_35: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_44: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_42, add_35);  mul_42 = add_35 = None
        convert_element_type_39: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_44, torch.bfloat16);  mul_44 = None
        view_108: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_39, _shape_param_134);  convert_element_type_39 = _shape_param_134 = None
        permute_54: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        addmm_29: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg85_1, view_108, permute_54);  arg85_1 = view_108 = permute_54 = None
        view_109: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_29, _shape_param_135);  addmm_29 = _shape_param_135 = None
        add_36: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_109, convert_element_type_37);  view_109 = convert_element_type_37 = None
        convert_element_type_40: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_36, torch.float32);  add_36 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_40, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[32, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_15: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_40, getitem_21);  convert_element_type_40 = getitem_21 = None
        add_37: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_45: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_46: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_45, arg86_1);  mul_45 = arg86_1 = None
        add_38: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_46, arg87_1);  mul_46 = arg87_1 = None
        convert_element_type_41: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_38, torch.bfloat16);  add_38 = None
        view_110: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_41, _shape_param_136);  _shape_param_136 = None
        permute_55: "bf16[768, 768]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_30: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg89_1, view_110, permute_55);  arg89_1 = view_110 = permute_55 = None
        view_111: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_30, _shape_param_137);  addmm_30 = _shape_param_137 = None
        view_112: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_111, _shape_param_138);  view_111 = _shape_param_138 = None
        permute_56: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        mul_47: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_56, 0.3535533905932738);  permute_56 = None
        expand_22: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_47, _shape_param_139);  mul_47 = _shape_param_139 = None
        clone_20: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_113: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_140);  clone_20 = _shape_param_140 = None
        view_114: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_41, _shape_param_141);  _shape_param_141 = None
        permute_57: "bf16[768, 768]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_31: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg91_1, view_114, permute_57);  arg91_1 = view_114 = permute_57 = None
        view_115: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_31, _shape_param_142);  addmm_31 = _shape_param_142 = None
        view_116: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_115, _shape_param_143);  view_115 = _shape_param_143 = None
        permute_58: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        permute_59: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
        mul_48: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_59, 0.3535533905932738);  permute_59 = None
        expand_23: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_48, _shape_param_144);  mul_48 = _shape_param_144 = None
        clone_21: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_117: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_21, _shape_param_145);  clone_21 = _shape_param_145 = None
        bmm_10: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_113, view_117);  view_113 = view_117 = None
        view_118: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_10, _shape_param_146);  bmm_10 = _shape_param_146 = None
        eq_5: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_118, -inf)
        logical_not_10: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        full_5: "bf16[32, 12, 512, 512]" = torch.ops.aten.full.default(_shape_param_147, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_147 = None
        convert_element_type_42: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_118, torch.float32);  view_118 = None
        amax_5: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_42, [-1], True)
        sub_16: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_42, amax_5);  convert_element_type_42 = amax_5 = None
        exp_5: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_43: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        where_5: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_11, full_5, convert_element_type_43);  logical_not_11 = full_5 = convert_element_type_43 = None
        expand_24: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_5, _shape_param_148);  where_5 = _shape_param_148 = None
        view_119: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_24, _shape_param_149);  expand_24 = _shape_param_149 = None
        view_120: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_41, _shape_param_150);  _shape_param_150 = None
        permute_60: "bf16[768, 768]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_32: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg93_1, view_120, permute_60);  arg93_1 = view_120 = permute_60 = None
        view_121: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_32, _shape_param_151);  addmm_32 = _shape_param_151 = None
        view_122: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_121, _shape_param_152);  view_121 = _shape_param_152 = None
        permute_61: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        expand_25: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_61, _shape_param_153);  permute_61 = _shape_param_153 = None
        clone_22: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_123: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_22, _shape_param_154);  clone_22 = _shape_param_154 = None
        bmm_11: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_119, view_123);  view_119 = view_123 = None
        view_124: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_155);  bmm_11 = _shape_param_155 = None
        permute_62: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        clone_23: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_125: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_23, _shape_param_156);  clone_23 = _shape_param_156 = None
        view_126: "bf16[16384, 768]" = torch.ops.aten.view.default(view_125, _shape_param_157);  view_125 = _shape_param_157 = None
        permute_63: "bf16[768, 768]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_33: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg95_1, view_126, permute_63);  arg95_1 = view_126 = permute_63 = None
        view_127: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_33, _shape_param_158);  addmm_33 = _shape_param_158 = None
        add_39: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_127, convert_element_type_41);  view_127 = convert_element_type_41 = None
        convert_element_type_44: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_39, torch.float32);  add_39 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_44, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[32, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_44, getitem_23);  convert_element_type_44 = getitem_23 = None
        add_40: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_49: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_50: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg96_1);  mul_49 = arg96_1 = None
        add_41: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_50, arg97_1);  mul_50 = arg97_1 = None
        convert_element_type_45: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None
        view_128: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_45, _shape_param_159);  _shape_param_159 = None
        permute_64: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_34: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(arg99_1, view_128, permute_64);  arg99_1 = view_128 = permute_64 = None
        view_129: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_34, _shape_param_160);  addmm_34 = _shape_param_160 = None
        convert_element_type_46: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_51: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 0.5)
        mul_52: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 0.7071067811865476);  convert_element_type_46 = None
        erf_5: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_42: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_53: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_51, add_42);  mul_51 = add_42 = None
        convert_element_type_47: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16);  mul_53 = None
        view_130: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_47, _shape_param_161);  convert_element_type_47 = _shape_param_161 = None
        permute_65: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        addmm_35: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg101_1, view_130, permute_65);  arg101_1 = view_130 = permute_65 = None
        view_131: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_35, _shape_param_162);  addmm_35 = _shape_param_162 = None
        add_43: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_131, convert_element_type_45);  view_131 = convert_element_type_45 = None
        convert_element_type_48: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_43, torch.float32);  add_43 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_48, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[32, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_18: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_48, getitem_25);  convert_element_type_48 = getitem_25 = None
        add_44: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_54: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_55: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_54, arg102_1);  mul_54 = arg102_1 = None
        add_45: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_55, arg103_1);  mul_55 = arg103_1 = None
        convert_element_type_49: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16);  add_45 = None
        view_132: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_49, _shape_param_163);  _shape_param_163 = None
        permute_66: "bf16[768, 768]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        addmm_36: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg105_1, view_132, permute_66);  arg105_1 = view_132 = permute_66 = None
        view_133: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_36, _shape_param_164);  addmm_36 = _shape_param_164 = None
        view_134: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_133, _shape_param_165);  view_133 = _shape_param_165 = None
        permute_67: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        mul_56: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_67, 0.3535533905932738);  permute_67 = None
        expand_26: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_56, _shape_param_166);  mul_56 = _shape_param_166 = None
        clone_24: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_135: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_167);  clone_24 = _shape_param_167 = None
        view_136: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_49, _shape_param_168);  _shape_param_168 = None
        permute_68: "bf16[768, 768]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_37: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg107_1, view_136, permute_68);  arg107_1 = view_136 = permute_68 = None
        view_137: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_37, _shape_param_169);  addmm_37 = _shape_param_169 = None
        view_138: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_137, _shape_param_170);  view_137 = _shape_param_170 = None
        permute_69: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        permute_70: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_69, [0, 1, 3, 2]);  permute_69 = None
        mul_57: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_70, 0.3535533905932738);  permute_70 = None
        expand_27: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_57, _shape_param_171);  mul_57 = _shape_param_171 = None
        clone_25: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_139: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_25, _shape_param_172);  clone_25 = _shape_param_172 = None
        bmm_12: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_135, view_139);  view_135 = view_139 = None
        view_140: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_12, _shape_param_173);  bmm_12 = _shape_param_173 = None
        eq_6: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_140, -inf)
        logical_not_12: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        full_6: "bf16[32, 12, 512, 512]" = torch.ops.aten.full.default(_shape_param_174, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_174 = None
        convert_element_type_50: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_140, torch.float32);  view_140 = None
        amax_6: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_50, [-1], True)
        sub_19: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_50, amax_6);  convert_element_type_50 = amax_6 = None
        exp_6: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_51: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        where_6: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_13, full_6, convert_element_type_51);  logical_not_13 = full_6 = convert_element_type_51 = None
        expand_28: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_6, _shape_param_175);  where_6 = _shape_param_175 = None
        view_141: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_28, _shape_param_176);  expand_28 = _shape_param_176 = None
        view_142: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_49, _shape_param_177);  _shape_param_177 = None
        permute_71: "bf16[768, 768]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_38: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg109_1, view_142, permute_71);  arg109_1 = view_142 = permute_71 = None
        view_143: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_38, _shape_param_178);  addmm_38 = _shape_param_178 = None
        view_144: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_143, _shape_param_179);  view_143 = _shape_param_179 = None
        permute_72: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        expand_29: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_72, _shape_param_180);  permute_72 = _shape_param_180 = None
        clone_26: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_145: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_26, _shape_param_181);  clone_26 = _shape_param_181 = None
        bmm_13: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_141, view_145);  view_141 = view_145 = None
        view_146: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_13, _shape_param_182);  bmm_13 = _shape_param_182 = None
        permute_73: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_27: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_147: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_27, _shape_param_183);  clone_27 = _shape_param_183 = None
        view_148: "bf16[16384, 768]" = torch.ops.aten.view.default(view_147, _shape_param_184);  view_147 = _shape_param_184 = None
        permute_74: "bf16[768, 768]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_39: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg111_1, view_148, permute_74);  arg111_1 = view_148 = permute_74 = None
        view_149: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_39, _shape_param_185);  addmm_39 = _shape_param_185 = None
        add_46: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_149, convert_element_type_49);  view_149 = convert_element_type_49 = None
        convert_element_type_52: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_46, torch.float32);  add_46 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_52, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[32, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_20: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_52, getitem_27);  convert_element_type_52 = getitem_27 = None
        add_47: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_58: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_59: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_58, arg112_1);  mul_58 = arg112_1 = None
        add_48: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_59, arg113_1);  mul_59 = arg113_1 = None
        convert_element_type_53: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_48, torch.bfloat16);  add_48 = None
        view_150: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_53, _shape_param_186);  _shape_param_186 = None
        permute_75: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        addmm_40: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(arg115_1, view_150, permute_75);  arg115_1 = view_150 = permute_75 = None
        view_151: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_40, _shape_param_187);  addmm_40 = _shape_param_187 = None
        convert_element_type_54: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        mul_60: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 0.5)
        mul_61: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 0.7071067811865476);  convert_element_type_54 = None
        erf_6: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_49: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_62: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_49);  mul_60 = add_49 = None
        convert_element_type_55: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        view_152: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_55, _shape_param_188);  convert_element_type_55 = _shape_param_188 = None
        permute_76: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_41: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg117_1, view_152, permute_76);  arg117_1 = view_152 = permute_76 = None
        view_153: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_41, _shape_param_189);  addmm_41 = _shape_param_189 = None
        add_50: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_153, convert_element_type_53);  view_153 = convert_element_type_53 = None
        convert_element_type_56: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_50, torch.float32);  add_50 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_56, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 512, 1]" = var_mean_14[0]
        getitem_29: "f32[32, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_21: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_56, getitem_29);  convert_element_type_56 = getitem_29 = None
        add_51: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_63: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_64: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_63, arg118_1);  mul_63 = arg118_1 = None
        add_52: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_64, arg119_1);  mul_64 = arg119_1 = None
        convert_element_type_57: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_52, torch.bfloat16);  add_52 = None
        view_154: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_57, _shape_param_190);  _shape_param_190 = None
        permute_77: "bf16[768, 768]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        addmm_42: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg121_1, view_154, permute_77);  arg121_1 = view_154 = permute_77 = None
        view_155: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_42, _shape_param_191);  addmm_42 = _shape_param_191 = None
        view_156: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_155, _shape_param_192);  view_155 = _shape_param_192 = None
        permute_78: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
        mul_65: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_78, 0.3535533905932738);  permute_78 = None
        expand_30: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_65, _shape_param_193);  mul_65 = _shape_param_193 = None
        clone_28: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_157: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_194);  clone_28 = _shape_param_194 = None
        view_158: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_57, _shape_param_195);  _shape_param_195 = None
        permute_79: "bf16[768, 768]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        addmm_43: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg123_1, view_158, permute_79);  arg123_1 = view_158 = permute_79 = None
        view_159: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_43, _shape_param_196);  addmm_43 = _shape_param_196 = None
        view_160: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_159, _shape_param_197);  view_159 = _shape_param_197 = None
        permute_80: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
        permute_81: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2]);  permute_80 = None
        mul_66: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_81, 0.3535533905932738);  permute_81 = None
        expand_31: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_66, _shape_param_198);  mul_66 = _shape_param_198 = None
        clone_29: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_161: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_29, _shape_param_199);  clone_29 = _shape_param_199 = None
        bmm_14: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_157, view_161);  view_157 = view_161 = None
        view_162: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_14, _shape_param_200);  bmm_14 = _shape_param_200 = None
        eq_7: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_162, -inf)
        logical_not_14: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        full_7: "bf16[32, 12, 512, 512]" = torch.ops.aten.full.default(_shape_param_201, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_201 = None
        convert_element_type_58: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_162, torch.float32);  view_162 = None
        amax_7: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_58, [-1], True)
        sub_22: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_58, amax_7);  convert_element_type_58 = amax_7 = None
        exp_7: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_59: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        where_7: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_15, full_7, convert_element_type_59);  logical_not_15 = full_7 = convert_element_type_59 = None
        expand_32: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_7, _shape_param_202);  where_7 = _shape_param_202 = None
        view_163: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_32, _shape_param_203);  expand_32 = _shape_param_203 = None
        view_164: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_57, _shape_param_204);  _shape_param_204 = None
        permute_82: "bf16[768, 768]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        addmm_44: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg125_1, view_164, permute_82);  arg125_1 = view_164 = permute_82 = None
        view_165: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_44, _shape_param_205);  addmm_44 = _shape_param_205 = None
        view_166: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_165, _shape_param_206);  view_165 = _shape_param_206 = None
        permute_83: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        expand_33: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_83, _shape_param_207);  permute_83 = _shape_param_207 = None
        clone_30: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_167: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_30, _shape_param_208);  clone_30 = _shape_param_208 = None
        bmm_15: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_163, view_167);  view_163 = view_167 = None
        view_168: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_209);  bmm_15 = _shape_param_209 = None
        permute_84: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_31: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_169: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_31, _shape_param_210);  clone_31 = _shape_param_210 = None
        view_170: "bf16[16384, 768]" = torch.ops.aten.view.default(view_169, _shape_param_211);  view_169 = _shape_param_211 = None
        permute_85: "bf16[768, 768]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        addmm_45: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg127_1, view_170, permute_85);  arg127_1 = view_170 = permute_85 = None
        view_171: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_45, _shape_param_212);  addmm_45 = _shape_param_212 = None
        add_53: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_171, convert_element_type_57);  view_171 = convert_element_type_57 = None
        convert_element_type_60: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_53, torch.float32);  add_53 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_60, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 512, 1]" = var_mean_15[0]
        getitem_31: "f32[32, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_23: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_60, getitem_31);  convert_element_type_60 = getitem_31 = None
        add_54: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_67: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_68: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_67, arg128_1);  mul_67 = arg128_1 = None
        add_55: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_68, arg129_1);  mul_68 = arg129_1 = None
        convert_element_type_61: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None
        view_172: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_61, _shape_param_213);  _shape_param_213 = None
        permute_86: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        addmm_46: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(arg131_1, view_172, permute_86);  arg131_1 = view_172 = permute_86 = None
        view_173: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_46, _shape_param_214);  addmm_46 = _shape_param_214 = None
        convert_element_type_62: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_69: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 0.5)
        mul_70: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 0.7071067811865476);  convert_element_type_62 = None
        erf_7: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_70);  mul_70 = None
        add_56: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_71: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_69, add_56);  mul_69 = add_56 = None
        convert_element_type_63: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_71, torch.bfloat16);  mul_71 = None
        view_174: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_63, _shape_param_215);  convert_element_type_63 = _shape_param_215 = None
        permute_87: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        addmm_47: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg133_1, view_174, permute_87);  arg133_1 = view_174 = permute_87 = None
        view_175: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_47, _shape_param_216);  addmm_47 = _shape_param_216 = None
        add_57: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_175, convert_element_type_61);  view_175 = convert_element_type_61 = None
        convert_element_type_64: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_57, torch.float32);  add_57 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_64, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 512, 1]" = var_mean_16[0]
        getitem_33: "f32[32, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_24: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_64, getitem_33);  convert_element_type_64 = getitem_33 = None
        add_58: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_72: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_73: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_72, arg134_1);  mul_72 = arg134_1 = None
        add_59: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_73, arg135_1);  mul_73 = arg135_1 = None
        convert_element_type_65: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None
        view_176: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_65, _shape_param_217);  _shape_param_217 = None
        permute_88: "bf16[768, 768]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        addmm_48: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg137_1, view_176, permute_88);  arg137_1 = view_176 = permute_88 = None
        view_177: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_48, _shape_param_218);  addmm_48 = _shape_param_218 = None
        view_178: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_177, _shape_param_219);  view_177 = _shape_param_219 = None
        permute_89: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        mul_74: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_89, 0.3535533905932738);  permute_89 = None
        expand_34: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_74, _shape_param_220);  mul_74 = _shape_param_220 = None
        clone_32: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_179: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_32, _shape_param_221);  clone_32 = _shape_param_221 = None
        view_180: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_65, _shape_param_222);  _shape_param_222 = None
        permute_90: "bf16[768, 768]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        addmm_49: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg139_1, view_180, permute_90);  arg139_1 = view_180 = permute_90 = None
        view_181: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_49, _shape_param_223);  addmm_49 = _shape_param_223 = None
        view_182: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_181, _shape_param_224);  view_181 = _shape_param_224 = None
        permute_91: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        permute_92: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
        mul_75: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_92, 0.3535533905932738);  permute_92 = None
        expand_35: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_75, _shape_param_225);  mul_75 = _shape_param_225 = None
        clone_33: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_183: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_33, _shape_param_226);  clone_33 = _shape_param_226 = None
        bmm_16: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_179, view_183);  view_179 = view_183 = None
        view_184: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_16, _shape_param_227);  bmm_16 = _shape_param_227 = None
        eq_8: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_184, -inf)
        logical_not_16: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        full_8: "bf16[32, 12, 512, 512]" = torch.ops.aten.full.default(_shape_param_228, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_228 = None
        convert_element_type_66: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_184, torch.float32);  view_184 = None
        amax_8: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_66, [-1], True)
        sub_25: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_66, amax_8);  convert_element_type_66 = amax_8 = None
        exp_8: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_67: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        where_8: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_17, full_8, convert_element_type_67);  logical_not_17 = full_8 = convert_element_type_67 = None
        expand_36: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_8, _shape_param_229);  where_8 = _shape_param_229 = None
        view_185: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_36, _shape_param_230);  expand_36 = _shape_param_230 = None
        view_186: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_65, _shape_param_231);  _shape_param_231 = None
        permute_93: "bf16[768, 768]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        addmm_50: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg141_1, view_186, permute_93);  arg141_1 = view_186 = permute_93 = None
        view_187: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_50, _shape_param_232);  addmm_50 = _shape_param_232 = None
        view_188: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_187, _shape_param_233);  view_187 = _shape_param_233 = None
        permute_94: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
        expand_37: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_94, _shape_param_234);  permute_94 = _shape_param_234 = None
        clone_34: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_189: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_34, _shape_param_235);  clone_34 = _shape_param_235 = None
        bmm_17: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_185, view_189);  view_185 = view_189 = None
        view_190: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_17, _shape_param_236);  bmm_17 = _shape_param_236 = None
        permute_95: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_35: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_191: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_35, _shape_param_237);  clone_35 = _shape_param_237 = None
        view_192: "bf16[16384, 768]" = torch.ops.aten.view.default(view_191, _shape_param_238);  view_191 = _shape_param_238 = None
        permute_96: "bf16[768, 768]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        addmm_51: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg143_1, view_192, permute_96);  arg143_1 = view_192 = permute_96 = None
        view_193: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_51, _shape_param_239);  addmm_51 = _shape_param_239 = None
        add_60: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_193, convert_element_type_65);  view_193 = convert_element_type_65 = None
        convert_element_type_68: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_60, torch.float32);  add_60 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_68, [2], correction = 0, keepdim = True)
        getitem_34: "f32[32, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[32, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_26: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_68, getitem_35);  convert_element_type_68 = getitem_35 = None
        add_61: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_76: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_77: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_76, arg144_1);  mul_76 = arg144_1 = None
        add_62: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_77, arg145_1);  mul_77 = arg145_1 = None
        convert_element_type_69: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_62, torch.bfloat16);  add_62 = None
        view_194: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_69, _shape_param_240);  _shape_param_240 = None
        permute_97: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_52: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(arg147_1, view_194, permute_97);  arg147_1 = view_194 = permute_97 = None
        view_195: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_52, _shape_param_241);  addmm_52 = _shape_param_241 = None
        convert_element_type_70: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        mul_78: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 0.5)
        mul_79: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 0.7071067811865476);  convert_element_type_70 = None
        erf_8: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
        add_63: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_80: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_78, add_63);  mul_78 = add_63 = None
        convert_element_type_71: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_80, torch.bfloat16);  mul_80 = None
        view_196: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_71, _shape_param_242);  convert_element_type_71 = _shape_param_242 = None
        permute_98: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_53: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg149_1, view_196, permute_98);  arg149_1 = view_196 = permute_98 = None
        view_197: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_53, _shape_param_243);  addmm_53 = _shape_param_243 = None
        add_64: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_197, convert_element_type_69);  view_197 = convert_element_type_69 = None
        convert_element_type_72: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_64, torch.float32);  add_64 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_72, [2], correction = 0, keepdim = True)
        getitem_36: "f32[32, 512, 1]" = var_mean_18[0]
        getitem_37: "f32[32, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_27: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_72, getitem_37);  convert_element_type_72 = getitem_37 = None
        add_65: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_81: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_82: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_81, arg150_1);  mul_81 = arg150_1 = None
        add_66: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_82, arg151_1);  mul_82 = arg151_1 = None
        convert_element_type_73: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_66, torch.bfloat16);  add_66 = None
        view_198: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_73, _shape_param_244);  _shape_param_244 = None
        permute_99: "bf16[768, 768]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_54: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg153_1, view_198, permute_99);  arg153_1 = view_198 = permute_99 = None
        view_199: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_54, _shape_param_245);  addmm_54 = _shape_param_245 = None
        view_200: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_199, _shape_param_246);  view_199 = _shape_param_246 = None
        permute_100: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
        mul_83: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_100, 0.3535533905932738);  permute_100 = None
        expand_38: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_83, _shape_param_247);  mul_83 = _shape_param_247 = None
        clone_36: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_201: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_248);  clone_36 = _shape_param_248 = None
        view_202: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_73, _shape_param_249);  _shape_param_249 = None
        permute_101: "bf16[768, 768]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_55: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg155_1, view_202, permute_101);  arg155_1 = view_202 = permute_101 = None
        view_203: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_55, _shape_param_250);  addmm_55 = _shape_param_250 = None
        view_204: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_203, _shape_param_251);  view_203 = _shape_param_251 = None
        permute_102: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
        permute_103: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_102, [0, 1, 3, 2]);  permute_102 = None
        mul_84: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_103, 0.3535533905932738);  permute_103 = None
        expand_39: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_84, _shape_param_252);  mul_84 = _shape_param_252 = None
        clone_37: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_205: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_37, _shape_param_253);  clone_37 = _shape_param_253 = None
        bmm_18: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_201, view_205);  view_201 = view_205 = None
        view_206: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_18, _shape_param_254);  bmm_18 = _shape_param_254 = None
        eq_9: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_206, -inf)
        logical_not_18: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        full_9: "bf16[32, 12, 512, 512]" = torch.ops.aten.full.default(_shape_param_255, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_255 = None
        convert_element_type_74: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_206, torch.float32);  view_206 = None
        amax_9: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_74, [-1], True)
        sub_28: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_74, amax_9);  convert_element_type_74 = amax_9 = None
        exp_9: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_75: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        where_9: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_19, full_9, convert_element_type_75);  logical_not_19 = full_9 = convert_element_type_75 = None
        expand_40: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_9, _shape_param_256);  where_9 = _shape_param_256 = None
        view_207: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_40, _shape_param_257);  expand_40 = _shape_param_257 = None
        view_208: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_73, _shape_param_258);  _shape_param_258 = None
        permute_104: "bf16[768, 768]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_56: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg157_1, view_208, permute_104);  arg157_1 = view_208 = permute_104 = None
        view_209: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_56, _shape_param_259);  addmm_56 = _shape_param_259 = None
        view_210: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_209, _shape_param_260);  view_209 = _shape_param_260 = None
        permute_105: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
        expand_41: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_105, _shape_param_261);  permute_105 = _shape_param_261 = None
        clone_38: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_211: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_38, _shape_param_262);  clone_38 = _shape_param_262 = None
        bmm_19: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_207, view_211);  view_207 = view_211 = None
        view_212: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_263);  bmm_19 = _shape_param_263 = None
        permute_106: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        clone_39: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_213: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_39, _shape_param_264);  clone_39 = _shape_param_264 = None
        view_214: "bf16[16384, 768]" = torch.ops.aten.view.default(view_213, _shape_param_265);  view_213 = _shape_param_265 = None
        permute_107: "bf16[768, 768]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_57: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg159_1, view_214, permute_107);  arg159_1 = view_214 = permute_107 = None
        view_215: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_57, _shape_param_266);  addmm_57 = _shape_param_266 = None
        add_67: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_215, convert_element_type_73);  view_215 = convert_element_type_73 = None
        convert_element_type_76: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_67, torch.float32);  add_67 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_76, [2], correction = 0, keepdim = True)
        getitem_38: "f32[32, 512, 1]" = var_mean_19[0]
        getitem_39: "f32[32, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_29: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_76, getitem_39);  convert_element_type_76 = getitem_39 = None
        add_68: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_85: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_86: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, arg160_1);  mul_85 = arg160_1 = None
        add_69: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_86, arg161_1);  mul_86 = arg161_1 = None
        convert_element_type_77: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16);  add_69 = None
        view_216: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_77, _shape_param_267);  _shape_param_267 = None
        permute_108: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        addmm_58: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(arg163_1, view_216, permute_108);  arg163_1 = view_216 = permute_108 = None
        view_217: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_58, _shape_param_268);  addmm_58 = _shape_param_268 = None
        convert_element_type_78: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_87: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_78, 0.5)
        mul_88: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_78, 0.7071067811865476);  convert_element_type_78 = None
        erf_9: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
        add_70: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_89: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_87, add_70);  mul_87 = add_70 = None
        convert_element_type_79: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None
        view_218: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_79, _shape_param_269);  convert_element_type_79 = _shape_param_269 = None
        permute_109: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_59: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg165_1, view_218, permute_109);  arg165_1 = view_218 = permute_109 = None
        view_219: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_59, _shape_param_270);  addmm_59 = _shape_param_270 = None
        add_71: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_219, convert_element_type_77);  view_219 = convert_element_type_77 = None
        convert_element_type_80: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_71, torch.float32);  add_71 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_80, [2], correction = 0, keepdim = True)
        getitem_40: "f32[32, 512, 1]" = var_mean_20[0]
        getitem_41: "f32[32, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_30: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_80, getitem_41);  convert_element_type_80 = getitem_41 = None
        add_72: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_90: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_91: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg166_1);  mul_90 = arg166_1 = None
        add_73: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_91, arg167_1);  mul_91 = arg167_1 = None
        convert_element_type_81: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16);  add_73 = None
        view_220: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_81, _shape_param_271);  _shape_param_271 = None
        permute_110: "bf16[768, 768]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        addmm_60: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg169_1, view_220, permute_110);  arg169_1 = view_220 = permute_110 = None
        view_221: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_60, _shape_param_272);  addmm_60 = _shape_param_272 = None
        view_222: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_221, _shape_param_273);  view_221 = _shape_param_273 = None
        permute_111: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
        mul_92: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_111, 0.3535533905932738);  permute_111 = None
        expand_42: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_92, _shape_param_274);  mul_92 = _shape_param_274 = None
        clone_40: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_223: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_275);  clone_40 = _shape_param_275 = None
        view_224: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_81, _shape_param_276);  _shape_param_276 = None
        permute_112: "bf16[768, 768]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_61: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg171_1, view_224, permute_112);  arg171_1 = view_224 = permute_112 = None
        view_225: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_61, _shape_param_277);  addmm_61 = _shape_param_277 = None
        view_226: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_225, _shape_param_278);  view_225 = _shape_param_278 = None
        permute_113: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        permute_114: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_113, [0, 1, 3, 2]);  permute_113 = None
        mul_93: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_114, 0.3535533905932738);  permute_114 = None
        expand_43: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_93, _shape_param_279);  mul_93 = _shape_param_279 = None
        clone_41: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_227: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_41, _shape_param_280);  clone_41 = _shape_param_280 = None
        bmm_20: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_223, view_227);  view_223 = view_227 = None
        view_228: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_20, _shape_param_281);  bmm_20 = _shape_param_281 = None
        eq_10: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_228, -inf)
        logical_not_20: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        full_10: "bf16[32, 12, 512, 512]" = torch.ops.aten.full.default(_shape_param_282, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_282 = None
        convert_element_type_82: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_228, torch.float32);  view_228 = None
        amax_10: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_82, [-1], True)
        sub_31: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_82, amax_10);  convert_element_type_82 = amax_10 = None
        exp_10: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_83: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        where_10: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_21, full_10, convert_element_type_83);  logical_not_21 = full_10 = convert_element_type_83 = None
        expand_44: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_10, _shape_param_283);  where_10 = _shape_param_283 = None
        view_229: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_44, _shape_param_284);  expand_44 = _shape_param_284 = None
        view_230: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_81, _shape_param_285);  _shape_param_285 = None
        permute_115: "bf16[768, 768]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_62: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg173_1, view_230, permute_115);  arg173_1 = view_230 = permute_115 = None
        view_231: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_62, _shape_param_286);  addmm_62 = _shape_param_286 = None
        view_232: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_231, _shape_param_287);  view_231 = _shape_param_287 = None
        permute_116: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None
        expand_45: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_116, _shape_param_288);  permute_116 = _shape_param_288 = None
        clone_42: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_233: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_42, _shape_param_289);  clone_42 = _shape_param_289 = None
        bmm_21: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_229, view_233);  view_229 = view_233 = None
        view_234: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_21, _shape_param_290);  bmm_21 = _shape_param_290 = None
        permute_117: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_43: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_235: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_43, _shape_param_291);  clone_43 = _shape_param_291 = None
        view_236: "bf16[16384, 768]" = torch.ops.aten.view.default(view_235, _shape_param_292);  view_235 = _shape_param_292 = None
        permute_118: "bf16[768, 768]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        addmm_63: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg175_1, view_236, permute_118);  arg175_1 = view_236 = permute_118 = None
        view_237: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_63, _shape_param_293);  addmm_63 = _shape_param_293 = None
        add_74: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_237, convert_element_type_81);  view_237 = convert_element_type_81 = None
        convert_element_type_84: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_74, torch.float32);  add_74 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_84, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 512, 1]" = var_mean_21[0]
        getitem_43: "f32[32, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_32: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_84, getitem_43);  convert_element_type_84 = getitem_43 = None
        add_75: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_94: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_95: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_94, arg176_1);  mul_94 = arg176_1 = None
        add_76: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_95, arg177_1);  mul_95 = arg177_1 = None
        convert_element_type_85: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_76, torch.bfloat16);  add_76 = None
        view_238: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_85, _shape_param_294);  _shape_param_294 = None
        permute_119: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_64: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(arg179_1, view_238, permute_119);  arg179_1 = view_238 = permute_119 = None
        view_239: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_64, _shape_param_295);  addmm_64 = _shape_param_295 = None
        convert_element_type_86: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        mul_96: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_86, 0.5)
        mul_97: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_86, 0.7071067811865476);  convert_element_type_86 = None
        erf_10: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
        add_77: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_98: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_96, add_77);  mul_96 = add_77 = None
        convert_element_type_87: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_98, torch.bfloat16);  mul_98 = None
        view_240: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_87, _shape_param_296);  convert_element_type_87 = _shape_param_296 = None
        permute_120: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        addmm_65: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg181_1, view_240, permute_120);  arg181_1 = view_240 = permute_120 = None
        view_241: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_65, _shape_param_297);  addmm_65 = _shape_param_297 = None
        add_78: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_241, convert_element_type_85);  view_241 = convert_element_type_85 = None
        convert_element_type_88: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_78, torch.float32);  add_78 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_88, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 512, 1]" = var_mean_22[0]
        getitem_45: "f32[32, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_33: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_88, getitem_45);  convert_element_type_88 = getitem_45 = None
        add_79: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_99: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
        mul_100: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_99, arg182_1);  mul_99 = arg182_1 = None
        add_80: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_100, arg183_1);  mul_100 = arg183_1 = None
        convert_element_type_89: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16);  add_80 = None
        view_242: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_89, _shape_param_298);  _shape_param_298 = None
        permute_121: "bf16[768, 768]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_66: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg185_1, view_242, permute_121);  arg185_1 = view_242 = permute_121 = None
        view_243: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_66, _shape_param_299);  addmm_66 = _shape_param_299 = None
        view_244: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_243, _shape_param_300);  view_243 = _shape_param_300 = None
        permute_122: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
        mul_101: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_122, 0.3535533905932738);  permute_122 = None
        expand_46: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_101, _shape_param_301);  mul_101 = _shape_param_301 = None
        clone_44: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_245: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_44, _shape_param_302);  clone_44 = _shape_param_302 = None
        view_246: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_89, _shape_param_303);  _shape_param_303 = None
        permute_123: "bf16[768, 768]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        addmm_67: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg187_1, view_246, permute_123);  arg187_1 = view_246 = permute_123 = None
        view_247: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_67, _shape_param_304);  addmm_67 = _shape_param_304 = None
        view_248: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_247, _shape_param_305);  view_247 = _shape_param_305 = None
        permute_124: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
        permute_125: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_124, [0, 1, 3, 2]);  permute_124 = None
        mul_102: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_125, 0.3535533905932738);  permute_125 = None
        expand_47: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_102, _shape_param_306);  mul_102 = _shape_param_306 = None
        clone_45: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_249: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_45, _shape_param_307);  clone_45 = _shape_param_307 = None
        bmm_22: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_245, view_249);  view_245 = view_249 = None
        view_250: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_22, _shape_param_308);  bmm_22 = _shape_param_308 = None
        eq_11: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_250, -inf)
        logical_not_22: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        full_11: "bf16[32, 12, 512, 512]" = torch.ops.aten.full.default(_shape_param_309, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_309 = None
        convert_element_type_90: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_250, torch.float32);  view_250 = None
        amax_11: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_90, [-1], True)
        sub_34: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_90, amax_11);  convert_element_type_90 = amax_11 = None
        exp_11: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_91: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        where_11: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_23, full_11, convert_element_type_91);  logical_not_23 = full_11 = convert_element_type_91 = None
        expand_48: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(where_11, _shape_param_310);  where_11 = _shape_param_310 = None
        view_251: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_48, _shape_param_311);  expand_48 = _shape_param_311 = None
        view_252: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_89, _shape_param_312);  _shape_param_312 = None
        permute_126: "bf16[768, 768]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_68: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg189_1, view_252, permute_126);  arg189_1 = view_252 = permute_126 = None
        view_253: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_68, _shape_param_313);  addmm_68 = _shape_param_313 = None
        view_254: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_253, _shape_param_314);  view_253 = _shape_param_314 = None
        permute_127: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_254, [0, 2, 1, 3]);  view_254 = None
        expand_49: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_127, _shape_param_315);  permute_127 = _shape_param_315 = None
        clone_46: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_255: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_46, _shape_param_316);  clone_46 = _shape_param_316 = None
        bmm_23: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_251, view_255);  view_251 = view_255 = None
        view_256: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_317);  bmm_23 = _shape_param_317 = None
        permute_128: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        clone_47: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_257: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_47, _shape_param_318);  clone_47 = _shape_param_318 = None
        view_258: "bf16[16384, 768]" = torch.ops.aten.view.default(view_257, _shape_param_319);  view_257 = _shape_param_319 = None
        permute_129: "bf16[768, 768]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_69: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg191_1, view_258, permute_129);  arg191_1 = view_258 = permute_129 = None
        view_259: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_69, _shape_param_320);  addmm_69 = _shape_param_320 = None
        add_81: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_259, convert_element_type_89);  view_259 = convert_element_type_89 = None
        convert_element_type_92: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_81, torch.float32);  add_81 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_92, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 512, 1]" = var_mean_23[0]
        getitem_47: "f32[32, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_35: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_92, getitem_47);  convert_element_type_92 = getitem_47 = None
        add_82: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_103: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_104: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_103, arg192_1);  mul_103 = arg192_1 = None
        add_83: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_104, arg193_1);  mul_104 = arg193_1 = None
        convert_element_type_93: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None
        view_260: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_93, _shape_param_321);  _shape_param_321 = None
        permute_130: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        addmm_70: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(arg195_1, view_260, permute_130);  arg195_1 = view_260 = permute_130 = None
        view_261: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_70, _shape_param_322);  addmm_70 = _shape_param_322 = None
        convert_element_type_94: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        mul_105: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_94, 0.5)
        mul_106: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_94, 0.7071067811865476);  convert_element_type_94 = None
        erf_11: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_106);  mul_106 = None
        add_84: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_107: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_105, add_84);  mul_105 = add_84 = None
        convert_element_type_95: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_107, torch.bfloat16);  mul_107 = None
        view_262: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_95, _shape_param_323);  convert_element_type_95 = _shape_param_323 = None
        permute_131: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        addmm_71: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg197_1, view_262, permute_131);  arg197_1 = view_262 = permute_131 = None
        view_263: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_71, _shape_param_324);  addmm_71 = _shape_param_324 = None
        add_85: "bf16[32, 512, 768]" = torch.ops.aten.add.Tensor(view_263, convert_element_type_93);  view_263 = convert_element_type_93 = None
        convert_element_type_96: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_85, torch.float32);  add_85 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_96, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 512, 1]" = var_mean_24[0]
        getitem_49: "f32[32, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_36: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_96, getitem_49);  convert_element_type_96 = getitem_49 = None
        add_86: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_108: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_109: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_108, arg198_1);  mul_108 = arg198_1 = None
        add_87: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_109, arg199_1);  mul_109 = arg199_1 = None
        convert_element_type_97: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16);  add_87 = None
        view_264: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_97, _shape_param_325);  convert_element_type_97 = _shape_param_325 = None
        permute_132: "bf16[768, 768]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        addmm_72: "bf16[16384, 768]" = torch.ops.aten.addmm.default(arg201_1, view_264, permute_132);  arg201_1 = view_264 = permute_132 = None
        view_265: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_72, _shape_param_326);  addmm_72 = _shape_param_326 = None
        convert_element_type_98: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_265, torch.float32);  view_265 = None
        mul_110: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_98, 0.5)
        mul_111: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_98, 0.7071067811865476);  convert_element_type_98 = None
        erf_12: "f32[32, 512, 768]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_88: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_112: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_110, add_88);  mul_110 = add_88 = None
        convert_element_type_99: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None
        convert_element_type_100: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_99, torch.float32);  convert_element_type_99 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_100, [2], correction = 0, keepdim = True)
        getitem_50: "f32[32, 512, 1]" = var_mean_25[0]
        getitem_51: "f32[32, 512, 1]" = var_mean_25[1];  var_mean_25 = None
        view_266: "i64[16384]" = torch.ops.aten.view.default(arg202_1, [-1]);  arg202_1 = None
        ne: "b8[16384]" = torch.ops.aten.ne.Scalar(view_266, -100)
        full_12: "bf16[6]" = torch.ops.aten.full.default(_shape_param_327, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_327 = None
        cat: "bf16[30528]" = torch.ops.aten.cat.default([arg203_1, full_12]);  arg203_1 = full_12 = None
        sub_37: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_100, getitem_51);  convert_element_type_100 = getitem_51 = None
        add_89: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_113: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = rsqrt_25 = None
        mul_114: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_113, arg204_1);  mul_113 = arg204_1 = None
        add_90: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_114, arg205_1);  mul_114 = arg205_1 = None
        convert_element_type_101: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_90, torch.bfloat16);  add_90 = None
        view_267: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_101, _shape_param_328);  convert_element_type_101 = _shape_param_328 = None
        permute_133: "bf16[768, 30522]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        constant_pad_nd: "bf16[768, 30528]" = torch.ops.aten.constant_pad_nd.default(permute_133, _shape_param_329);  permute_133 = _shape_param_329 = None
        addmm_73: "bf16[16384, 30528]" = torch.ops.aten.addmm.default(cat, view_267, constant_pad_nd);  cat = view_267 = constant_pad_nd = None
        slice_1: "bf16[16384, 30522]" = torch.ops.aten.slice.Tensor(addmm_73, 1, 0, -6);  addmm_73 = None
        view_268: "bf16[32, 512, 30522]" = torch.ops.aten.view.default(slice_1, _shape_param_330);  slice_1 = _shape_param_330 = None
        view_269: "bf16[16384, 30522]" = torch.ops.aten.view.default(view_268, _shape_param_331);  _shape_param_331 = None
        convert_element_type_102: "f32[16384, 30522]" = torch.ops.prims.convert_element_type.default(view_269, torch.float32);  view_269 = None
        amax_12: "f32[16384, 1]" = torch.ops.aten.amax.default(convert_element_type_102, [1], True)
        sub_38: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(convert_element_type_102, amax_12);  convert_element_type_102 = amax_12 = None
        exp_12: "f32[16384, 30522]" = torch.ops.aten.exp.default(sub_38)
        sum_13: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log: "f32[16384, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_39: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = log = None
        convert_element_type_103: "bf16[16384, 30522]" = torch.ops.prims.convert_element_type.default(sub_39, torch.bfloat16);  sub_39 = None
        ne_1: "b8[16384]" = torch.ops.aten.ne.Scalar(view_266, -100)
        full_13: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "i64[16384]" = torch.ops.aten.where.self(ne_1, view_266, full_13);  ne_1 = full_13 = None
        unsqueeze: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather_1: "bf16[16384, 1]" = torch.ops.aten.gather.default(convert_element_type_103, 1, unsqueeze);  convert_element_type_103 = unsqueeze = None
        squeeze: "bf16[16384]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg: "bf16[16384]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_14: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "bf16[16384]" = torch.ops.aten.where.self(ne, neg, full_14);  ne = neg = full_14 = None
        sum_14: "bf16[]" = torch.ops.aten.sum.default(where_13);  where_13 = None
        ne_2: "b8[16384]" = torch.ops.aten.ne.Scalar(view_266, -100);  view_266 = None
        sum_15: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_104: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_15, torch.bfloat16);  sum_15 = None
        div_12: "bf16[]" = torch.ops.aten.div.Tensor(sum_14, convert_element_type_104);  sum_14 = convert_element_type_104 = None
        return (view_268, div_12)



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
