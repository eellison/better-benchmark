"""
Standalone repro captured via capture_hook.
Label: RobertaForCausalLM
Pattern hash: a96c298d305f
Shape hash: f6505928
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
    def forward(self, arg0_1: "i64[32, 512]", arg1_1: "i32[32, 512]", arg2_1: "i64[1, 512]", arg3_1: "f32[50265, 768]", arg4_1: "i64[32, 512]", arg5_1: "f32[2, 768]", arg6_1: "f32[512, 768]", arg7_1: "f32[768]", arg8_1: "f32[768]", arg9_1: "f32[768]", arg10_1: "f32[768, 768]", arg11_1: "f32[768]", arg12_1: "f32[768, 768]", arg13_1: "f32[768]", arg14_1: "f32[768, 768]", arg15_1: "f32[768]", arg16_1: "f32[768, 768]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[3072]", arg20_1: "f32[3072, 768]", arg21_1: "f32[768]", arg22_1: "f32[768, 3072]", arg23_1: "f32[768]", arg24_1: "f32[768]", arg25_1: "f32[768]", arg26_1: "f32[768, 768]", arg27_1: "f32[768]", arg28_1: "f32[768, 768]", arg29_1: "f32[768]", arg30_1: "f32[768, 768]", arg31_1: "f32[768]", arg32_1: "f32[768, 768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[3072]", arg36_1: "f32[3072, 768]", arg37_1: "f32[768]", arg38_1: "f32[768, 3072]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[768]", arg42_1: "f32[768, 768]", arg43_1: "f32[768]", arg44_1: "f32[768, 768]", arg45_1: "f32[768]", arg46_1: "f32[768, 768]", arg47_1: "f32[768]", arg48_1: "f32[768, 768]", arg49_1: "f32[768]", arg50_1: "f32[768]", arg51_1: "f32[3072]", arg52_1: "f32[3072, 768]", arg53_1: "f32[768]", arg54_1: "f32[768, 3072]", arg55_1: "f32[768]", arg56_1: "f32[768]", arg57_1: "f32[768]", arg58_1: "f32[768, 768]", arg59_1: "f32[768]", arg60_1: "f32[768, 768]", arg61_1: "f32[768]", arg62_1: "f32[768, 768]", arg63_1: "f32[768]", arg64_1: "f32[768, 768]", arg65_1: "f32[768]", arg66_1: "f32[768]", arg67_1: "f32[3072]", arg68_1: "f32[3072, 768]", arg69_1: "f32[768]", arg70_1: "f32[768, 3072]", arg71_1: "f32[768]", arg72_1: "f32[768]", arg73_1: "f32[768]", arg74_1: "f32[768, 768]", arg75_1: "f32[768]", arg76_1: "f32[768, 768]", arg77_1: "f32[768]", arg78_1: "f32[768, 768]", arg79_1: "f32[768]", arg80_1: "f32[768, 768]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[3072]", arg84_1: "f32[3072, 768]", arg85_1: "f32[768]", arg86_1: "f32[768, 3072]", arg87_1: "f32[768]", arg88_1: "f32[768]", arg89_1: "f32[768]", arg90_1: "f32[768, 768]", arg91_1: "f32[768]", arg92_1: "f32[768, 768]", arg93_1: "f32[768]", arg94_1: "f32[768, 768]", arg95_1: "f32[768]", arg96_1: "f32[768, 768]", arg97_1: "f32[768]", arg98_1: "f32[768]", arg99_1: "f32[3072]", arg100_1: "f32[3072, 768]", arg101_1: "f32[768]", arg102_1: "f32[768, 3072]", arg103_1: "f32[768]", arg104_1: "f32[768]", arg105_1: "f32[768]", arg106_1: "f32[768, 768]", arg107_1: "f32[768]", arg108_1: "f32[768, 768]", arg109_1: "f32[768]", arg110_1: "f32[768, 768]", arg111_1: "f32[768]", arg112_1: "f32[768, 768]", arg113_1: "f32[768]", arg114_1: "f32[768]", arg115_1: "f32[3072]", arg116_1: "f32[3072, 768]", arg117_1: "f32[768]", arg118_1: "f32[768, 3072]", arg119_1: "f32[768]", arg120_1: "f32[768]", arg121_1: "f32[768]", arg122_1: "f32[768, 768]", arg123_1: "f32[768]", arg124_1: "f32[768, 768]", arg125_1: "f32[768]", arg126_1: "f32[768, 768]", arg127_1: "f32[768]", arg128_1: "f32[768, 768]", arg129_1: "f32[768]", arg130_1: "f32[768]", arg131_1: "f32[3072]", arg132_1: "f32[3072, 768]", arg133_1: "f32[768]", arg134_1: "f32[768, 3072]", arg135_1: "f32[768]", arg136_1: "f32[768]", arg137_1: "f32[768]", arg138_1: "f32[768, 768]", arg139_1: "f32[768]", arg140_1: "f32[768, 768]", arg141_1: "f32[768]", arg142_1: "f32[768, 768]", arg143_1: "f32[768]", arg144_1: "f32[768, 768]", arg145_1: "f32[768]", arg146_1: "f32[768]", arg147_1: "f32[3072]", arg148_1: "f32[3072, 768]", arg149_1: "f32[768]", arg150_1: "f32[768, 3072]", arg151_1: "f32[768]", arg152_1: "f32[768]", arg153_1: "f32[768]", arg154_1: "f32[768, 768]", arg155_1: "f32[768]", arg156_1: "f32[768, 768]", arg157_1: "f32[768]", arg158_1: "f32[768, 768]", arg159_1: "f32[768]", arg160_1: "f32[768, 768]", arg161_1: "f32[768]", arg162_1: "f32[768]", arg163_1: "f32[3072]", arg164_1: "f32[3072, 768]", arg165_1: "f32[768]", arg166_1: "f32[768, 3072]", arg167_1: "f32[768]", arg168_1: "f32[768]", arg169_1: "f32[768]", arg170_1: "f32[768, 768]", arg171_1: "f32[768]", arg172_1: "f32[768, 768]", arg173_1: "f32[768]", arg174_1: "f32[768, 768]", arg175_1: "f32[768]", arg176_1: "f32[768, 768]", arg177_1: "f32[768]", arg178_1: "f32[768]", arg179_1: "f32[3072]", arg180_1: "f32[3072, 768]", arg181_1: "f32[768]", arg182_1: "f32[768, 3072]", arg183_1: "f32[768]", arg184_1: "f32[768]", arg185_1: "f32[768]", arg186_1: "f32[768, 768]", arg187_1: "f32[768]", arg188_1: "f32[768, 768]", arg189_1: "f32[768]", arg190_1: "f32[768, 768]", arg191_1: "f32[768]", arg192_1: "f32[768, 768]", arg193_1: "f32[768]", arg194_1: "f32[768]", arg195_1: "f32[3072]", arg196_1: "f32[3072, 768]", arg197_1: "f32[768]", arg198_1: "f32[768, 3072]", arg199_1: "f32[768]", arg200_1: "f32[768]", arg201_1: "f32[768]", arg202_1: "f32[768, 768]", arg203_1: "f32[768]", arg204_1: "f32[768]", arg205_1: "f32[50265]", arg206_1: "i64[32, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334):
        # No stacktrace found for following nodes
        convert_element_type: "i32[32, 512]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.int32);  arg0_1 = None
        add: "i32[32, 512]" = torch.ops.aten.add.Tensor(convert_element_type, 0);  convert_element_type = None
        mul: "i32[32, 512]" = torch.ops.aten.mul.Tensor(add, arg1_1);  add = arg1_1 = None
        convert_element_type_1: "i64[32, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.int64);  mul = None
        add_1: "i64[32, 512]" = torch.ops.aten.add.Tensor(convert_element_type_1, 0);  convert_element_type_1 = None
        expand: "i64[32, 512]" = torch.ops.aten.expand.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        gather: "i64[32, 512]" = torch.ops.aten.gather.default(expand, 1, add_1);  expand = None
        expand_1: "i64[32, 512]" = torch.ops.aten.expand.default(gather, _shape_param_1);  gather = _shape_param_1 = None
        embedding: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg3_1, arg4_1, 0);  arg4_1 = None
        embedding_1: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg5_1, expand_1);  arg5_1 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2: "f32[32, 512, 768]" = torch.ops.aten.embedding.default(arg6_1, add_1, 0);  arg6_1 = None
        add_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_2, embedding_2);  add_2 = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        add_4: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_1);  add_3 = getitem_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg7_1);  arg7_1 = None
        add_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_2, arg8_1);  mul_2 = arg8_1 = None
        inductor_seeds: "i64[37]" = torch.ops.prims.inductor_seeds.default(37, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_2, inductor_lookup_seed, 'rand');  _shape_param_2 = inductor_lookup_seed = None
        gt: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt, add_5);  add_5 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None
        convert_element_type_2: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        convert_element_type_3: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        convert_element_type_4: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16)
        view: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_4, _shape_param_3);  convert_element_type_4 = _shape_param_3 = None
        permute: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0]);  convert_element_type_3 = None
        addmm: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_2, view, permute);  convert_element_type_2 = None
        view_1: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm, _shape_param_4);  addmm = _shape_param_4 = None
        view_2: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_1, _shape_param_5);  view_1 = _shape_param_5 = None
        permute_1: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        convert_element_type_5: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.bfloat16);  arg11_1 = None
        convert_element_type_6: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.bfloat16);  arg12_1 = None
        permute_2: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_6, [1, 0]);  convert_element_type_6 = None
        addmm_1: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_5, view, permute_2);  convert_element_type_5 = None
        view_3: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_1, _shape_param_6);  addmm_1 = _shape_param_6 = None
        view_4: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_3, _shape_param_7);  view_3 = _shape_param_7 = None
        permute_3: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        convert_element_type_7: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        convert_element_type_8: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.bfloat16);  arg14_1 = None
        permute_4: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_8, [1, 0]);  convert_element_type_8 = None
        addmm_2: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_7, view, permute_4);  convert_element_type_7 = None
        view_5: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_2, _shape_param_8);  addmm_2 = _shape_param_8 = None
        view_6: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_5, _shape_param_9);  view_5 = _shape_param_9 = None
        permute_5: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        mul_5: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_1, 0.3535533905932738);  permute_1 = None
        permute_6: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        mul_6: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_6, 0.3535533905932738);  permute_6 = None
        expand_2: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_5, _shape_param_10);  mul_5 = _shape_param_10 = None
        clone: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_7: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_11);  clone = _shape_param_11 = None
        expand_3: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_6, _shape_param_12);  mul_6 = _shape_param_12 = None
        clone_1: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_8: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_1, _shape_param_13);  clone_1 = _shape_param_13 = None
        bmm: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_7, view_8)
        view_9: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_14);  _shape_param_14 = None
        convert_element_type_9: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_9, torch.float32)
        amax: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_9, [-1], True)
        sub_1: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_9, amax);  convert_element_type_9 = None
        exp: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_10: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        eq: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_9, -inf);  view_9 = None
        logical_not: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full: "bf16[32, 12, 512, 512]" = torch.ops.aten.full.default(_shape_param_15, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_15 = None
        where: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full, convert_element_type_10);  convert_element_type_10 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1)
        inductor_random_1: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_16, inductor_lookup_seed_1, 'rand');  _shape_param_16 = inductor_lookup_seed_1 = None
        convert_element_type_11: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_1, torch.bfloat16);  inductor_random_1 = None
        gt_1: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_11, 0.1);  convert_element_type_11 = None
        mul_7: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_1, where);  where = None
        mul_8: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112);  mul_7 = None
        expand_4: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_8, _shape_param_17);  mul_8 = _shape_param_17 = None
        view_10: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_4, _shape_param_18);  expand_4 = _shape_param_18 = None
        expand_5: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_5, _shape_param_19);  permute_5 = _shape_param_19 = None
        clone_2: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_11: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_20);  clone_2 = _shape_param_20 = None
        bmm_1: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_10, view_11)
        view_12: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_21);  bmm_1 = _shape_param_21 = None
        permute_7: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        clone_3: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_13: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_3, _shape_param_22);  clone_3 = _shape_param_22 = None
        convert_element_type_12: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg15_1, torch.bfloat16);  arg15_1 = None
        convert_element_type_13: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.bfloat16);  arg16_1 = None
        view_14: "bf16[16384, 768]" = torch.ops.aten.view.default(view_13, _shape_param_23);  view_13 = _shape_param_23 = None
        permute_8: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        addmm_3: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_12, view_14, permute_8);  convert_element_type_12 = None
        view_15: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_3, _shape_param_24);  addmm_3 = _shape_param_24 = None
        inductor_lookup_seed_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 2)
        inductor_random_2: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_25, inductor_lookup_seed_2, 'rand');  _shape_param_25 = inductor_lookup_seed_2 = None
        convert_element_type_14: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_2, torch.bfloat16);  inductor_random_2 = None
        gt_2: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_14, 0.1);  convert_element_type_14 = None
        mul_9: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_2, view_15);  view_15 = None
        mul_10: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, 1.1111111111111112);  mul_9 = None
        add_6: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_10, mul_4);  mul_10 = mul_4 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_7: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_6, getitem_3);  add_6 = getitem_3 = None
        mul_11: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_12: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_11, arg17_1);  arg17_1 = None
        add_8: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_12, arg18_1);  mul_12 = arg18_1 = None
        convert_element_type_15: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg19_1, torch.bfloat16);  arg19_1 = None
        convert_element_type_16: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.bfloat16);  arg20_1 = None
        convert_element_type_17: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16)
        view_16: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_17, _shape_param_26);  convert_element_type_17 = _shape_param_26 = None
        permute_9: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_16, [1, 0]);  convert_element_type_16 = None
        addmm_4: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(convert_element_type_15, view_16, permute_9);  convert_element_type_15 = None
        view_17: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_4, _shape_param_27);  _shape_param_27 = None
        convert_element_type_18: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None
        mul_13: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 0.5)
        mul_14: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 0.7071067811865476);  convert_element_type_18 = None
        erf: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_14);  mul_14 = None
        add_9: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_15: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_13, add_9);  mul_13 = add_9 = None
        convert_element_type_19: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16);  mul_15 = None
        convert_element_type_20: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg21_1, torch.bfloat16);  arg21_1 = None
        convert_element_type_21: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg22_1, torch.bfloat16);  arg22_1 = None
        view_18: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_19, _shape_param_28);  convert_element_type_19 = _shape_param_28 = None
        permute_10: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_21, [1, 0]);  convert_element_type_21 = None
        addmm_5: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_20, view_18, permute_10);  convert_element_type_20 = None
        view_19: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_5, _shape_param_29);  addmm_5 = _shape_param_29 = None
        inductor_lookup_seed_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 3)
        inductor_random_3: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_30, inductor_lookup_seed_3, 'rand');  _shape_param_30 = inductor_lookup_seed_3 = None
        convert_element_type_22: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_3, torch.bfloat16);  inductor_random_3 = None
        gt_3: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_22, 0.1);  convert_element_type_22 = None
        mul_16: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_3, view_19);  view_19 = None
        mul_17: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, 1.1111111111111112);  mul_16 = None
        add_10: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_17, add_8);  mul_17 = add_8 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[32, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_11: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_5);  add_10 = getitem_5 = None
        mul_18: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_19: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg23_1);  arg23_1 = None
        add_12: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_19, arg24_1);  mul_19 = arg24_1 = None
        convert_element_type_23: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg25_1, torch.bfloat16);  arg25_1 = None
        convert_element_type_24: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg26_1, torch.bfloat16);  arg26_1 = None
        convert_element_type_25: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16)
        view_20: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_25, _shape_param_31);  convert_element_type_25 = _shape_param_31 = None
        permute_11: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        addmm_6: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_23, view_20, permute_11);  convert_element_type_23 = None
        view_21: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_6, _shape_param_32);  addmm_6 = _shape_param_32 = None
        view_22: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_21, _shape_param_33);  view_21 = _shape_param_33 = None
        permute_12: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        convert_element_type_26: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg27_1, torch.bfloat16);  arg27_1 = None
        convert_element_type_27: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg28_1, torch.bfloat16);  arg28_1 = None
        permute_13: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_27, [1, 0]);  convert_element_type_27 = None
        addmm_7: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_26, view_20, permute_13);  convert_element_type_26 = None
        view_23: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_7, _shape_param_34);  addmm_7 = _shape_param_34 = None
        view_24: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_23, _shape_param_35);  view_23 = _shape_param_35 = None
        permute_14: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        convert_element_type_28: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg29_1, torch.bfloat16);  arg29_1 = None
        convert_element_type_29: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg30_1, torch.bfloat16);  arg30_1 = None
        permute_15: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        addmm_8: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_28, view_20, permute_15);  convert_element_type_28 = None
        view_25: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_8, _shape_param_36);  addmm_8 = _shape_param_36 = None
        view_26: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_25, _shape_param_37);  view_25 = _shape_param_37 = None
        permute_16: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        mul_20: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_12, 0.3535533905932738);  permute_12 = None
        permute_17: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
        mul_21: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_17, 0.3535533905932738);  permute_17 = None
        expand_6: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_20, _shape_param_38);  mul_20 = _shape_param_38 = None
        clone_4: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_27: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_39);  clone_4 = _shape_param_39 = None
        expand_7: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_21, _shape_param_40);  mul_21 = _shape_param_40 = None
        clone_5: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_28: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_5, _shape_param_41);  clone_5 = _shape_param_41 = None
        bmm_2: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_27, view_28)
        view_29: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_2, _shape_param_42);  bmm_2 = _shape_param_42 = None
        convert_element_type_30: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_29, torch.float32)
        amax_1: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_30, [-1], True)
        sub_4: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_30, amax_1);  convert_element_type_30 = amax_1 = None
        exp_1: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_31: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        eq_1: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_29, -inf);  view_29 = None
        logical_not_2: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        where_1: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_3, full, convert_element_type_31);  logical_not_3 = convert_element_type_31 = None
        inductor_lookup_seed_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 4)
        inductor_random_4: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_43, inductor_lookup_seed_4, 'rand');  _shape_param_43 = inductor_lookup_seed_4 = None
        convert_element_type_32: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_4, torch.bfloat16);  inductor_random_4 = None
        gt_4: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_32, 0.1);  convert_element_type_32 = None
        mul_22: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_4, where_1)
        mul_23: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_22, 1.1111111111111112);  mul_22 = None
        expand_8: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_23, _shape_param_44);  mul_23 = _shape_param_44 = None
        view_30: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_8, _shape_param_45);  expand_8 = _shape_param_45 = None
        expand_9: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_16, _shape_param_46);  permute_16 = _shape_param_46 = None
        clone_6: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_31: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_6, _shape_param_47);  clone_6 = _shape_param_47 = None
        bmm_3: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_30, view_31)
        view_32: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_48);  bmm_3 = _shape_param_48 = None
        permute_18: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        clone_7: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_33: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_7, _shape_param_49);  clone_7 = _shape_param_49 = None
        convert_element_type_33: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg31_1, torch.bfloat16);  arg31_1 = None
        convert_element_type_34: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg32_1, torch.bfloat16);  arg32_1 = None
        view_34: "bf16[16384, 768]" = torch.ops.aten.view.default(view_33, _shape_param_50);  view_33 = _shape_param_50 = None
        permute_19: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        addmm_9: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_33, view_34, permute_19);  convert_element_type_33 = None
        view_35: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_9, _shape_param_51);  addmm_9 = _shape_param_51 = None
        inductor_lookup_seed_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 5)
        inductor_random_5: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_52, inductor_lookup_seed_5, 'rand');  _shape_param_52 = inductor_lookup_seed_5 = None
        convert_element_type_35: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_5, torch.bfloat16);  inductor_random_5 = None
        gt_5: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_35, 0.1);  convert_element_type_35 = None
        mul_24: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_5, view_35);  view_35 = None
        mul_25: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, 1.1111111111111112);  mul_24 = None
        add_13: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_25, add_12);  mul_25 = add_12 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[32, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        add_14: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_5: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_13, getitem_7);  add_13 = getitem_7 = None
        mul_26: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_27: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_26, arg33_1);  arg33_1 = None
        add_15: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_27, arg34_1);  mul_27 = arg34_1 = None
        convert_element_type_36: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg35_1, torch.bfloat16);  arg35_1 = None
        convert_element_type_37: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg36_1, torch.bfloat16);  arg36_1 = None
        convert_element_type_38: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16)
        view_36: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_38, _shape_param_53);  convert_element_type_38 = _shape_param_53 = None
        permute_20: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_37, [1, 0]);  convert_element_type_37 = None
        addmm_10: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(convert_element_type_36, view_36, permute_20);  convert_element_type_36 = None
        view_37: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_10, _shape_param_54);  _shape_param_54 = None
        convert_element_type_39: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_37, torch.float32);  view_37 = None
        mul_28: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 0.5)
        mul_29: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 0.7071067811865476);  convert_element_type_39 = None
        erf_1: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_29);  mul_29 = None
        add_16: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_30: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_28, add_16);  mul_28 = add_16 = None
        convert_element_type_40: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        convert_element_type_41: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg37_1, torch.bfloat16);  arg37_1 = None
        convert_element_type_42: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg38_1, torch.bfloat16);  arg38_1 = None
        view_38: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_40, _shape_param_55);  convert_element_type_40 = _shape_param_55 = None
        permute_21: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_42, [1, 0]);  convert_element_type_42 = None
        addmm_11: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_41, view_38, permute_21);  convert_element_type_41 = None
        view_39: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_11, _shape_param_56);  addmm_11 = _shape_param_56 = None
        inductor_lookup_seed_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 6)
        inductor_random_6: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_57, inductor_lookup_seed_6, 'rand');  _shape_param_57 = inductor_lookup_seed_6 = None
        convert_element_type_43: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_6, torch.bfloat16);  inductor_random_6 = None
        gt_6: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_43, 0.1);  convert_element_type_43 = None
        mul_31: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_6, view_39);  view_39 = None
        mul_32: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_31, 1.1111111111111112);  mul_31 = None
        add_17: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_32, add_15);  mul_32 = add_15 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[32, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        add_18: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_6: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_9);  add_17 = getitem_9 = None
        mul_33: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_34: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, arg39_1);  arg39_1 = None
        add_19: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_34, arg40_1);  mul_34 = arg40_1 = None
        convert_element_type_44: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg41_1, torch.bfloat16);  arg41_1 = None
        convert_element_type_45: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg42_1, torch.bfloat16);  arg42_1 = None
        convert_element_type_46: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16)
        view_40: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_46, _shape_param_58);  convert_element_type_46 = _shape_param_58 = None
        permute_22: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_45, [1, 0]);  convert_element_type_45 = None
        addmm_12: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_44, view_40, permute_22);  convert_element_type_44 = None
        view_41: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_12, _shape_param_59);  addmm_12 = _shape_param_59 = None
        view_42: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_41, _shape_param_60);  view_41 = _shape_param_60 = None
        permute_23: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        convert_element_type_47: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg43_1, torch.bfloat16);  arg43_1 = None
        convert_element_type_48: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg44_1, torch.bfloat16);  arg44_1 = None
        permute_24: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_48, [1, 0]);  convert_element_type_48 = None
        addmm_13: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_47, view_40, permute_24);  convert_element_type_47 = None
        view_43: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_13, _shape_param_61);  addmm_13 = _shape_param_61 = None
        view_44: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_43, _shape_param_62);  view_43 = _shape_param_62 = None
        permute_25: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None
        convert_element_type_49: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg45_1, torch.bfloat16);  arg45_1 = None
        convert_element_type_50: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg46_1, torch.bfloat16);  arg46_1 = None
        permute_26: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_50, [1, 0]);  convert_element_type_50 = None
        addmm_14: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_49, view_40, permute_26);  convert_element_type_49 = None
        view_45: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_14, _shape_param_63);  addmm_14 = _shape_param_63 = None
        view_46: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_45, _shape_param_64);  view_45 = _shape_param_64 = None
        permute_27: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        mul_35: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_23, 0.3535533905932738);  permute_23 = None
        permute_28: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
        mul_36: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_28, 0.3535533905932738);  permute_28 = None
        expand_10: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_35, _shape_param_65);  mul_35 = _shape_param_65 = None
        clone_8: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_47: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_66);  clone_8 = _shape_param_66 = None
        expand_11: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_36, _shape_param_67);  mul_36 = _shape_param_67 = None
        clone_9: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_48: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_9, _shape_param_68);  clone_9 = _shape_param_68 = None
        bmm_4: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_47, view_48)
        view_49: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_4, _shape_param_69);  bmm_4 = _shape_param_69 = None
        convert_element_type_51: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_49, torch.float32)
        amax_2: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_51, [-1], True)
        sub_7: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_51, amax_2);  convert_element_type_51 = amax_2 = None
        exp_2: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_52: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        eq_2: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_49, -inf);  view_49 = None
        logical_not_4: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        where_2: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_5, full, convert_element_type_52);  logical_not_5 = convert_element_type_52 = None
        inductor_lookup_seed_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 7)
        inductor_random_7: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_70, inductor_lookup_seed_7, 'rand');  _shape_param_70 = inductor_lookup_seed_7 = None
        convert_element_type_53: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_7, torch.bfloat16);  inductor_random_7 = None
        gt_7: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_53, 0.1);  convert_element_type_53 = None
        mul_37: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_7, where_2)
        mul_38: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_37, 1.1111111111111112);  mul_37 = None
        expand_12: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_38, _shape_param_71);  mul_38 = _shape_param_71 = None
        view_50: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_12, _shape_param_72);  expand_12 = _shape_param_72 = None
        expand_13: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_27, _shape_param_73);  permute_27 = _shape_param_73 = None
        clone_10: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_51: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_10, _shape_param_74);  clone_10 = _shape_param_74 = None
        bmm_5: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_50, view_51)
        view_52: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_5, _shape_param_75);  bmm_5 = _shape_param_75 = None
        permute_29: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        clone_11: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_53: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_11, _shape_param_76);  clone_11 = _shape_param_76 = None
        convert_element_type_54: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg47_1, torch.bfloat16);  arg47_1 = None
        convert_element_type_55: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg48_1, torch.bfloat16);  arg48_1 = None
        view_54: "bf16[16384, 768]" = torch.ops.aten.view.default(view_53, _shape_param_77);  view_53 = _shape_param_77 = None
        permute_30: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_55, [1, 0]);  convert_element_type_55 = None
        addmm_15: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_54, view_54, permute_30);  convert_element_type_54 = None
        view_55: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_15, _shape_param_78);  addmm_15 = _shape_param_78 = None
        inductor_lookup_seed_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 8)
        inductor_random_8: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_79, inductor_lookup_seed_8, 'rand');  _shape_param_79 = inductor_lookup_seed_8 = None
        convert_element_type_56: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_8, torch.bfloat16);  inductor_random_8 = None
        gt_8: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_56, 0.1);  convert_element_type_56 = None
        mul_39: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_8, view_55);  view_55 = None
        mul_40: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_39, 1.1111111111111112);  mul_39 = None
        add_20: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_40, add_19);  mul_40 = add_19 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
        getitem_10: "f32[32, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[32, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        add_21: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_8: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_20, getitem_11);  add_20 = getitem_11 = None
        mul_41: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
        mul_42: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_41, arg49_1);  arg49_1 = None
        add_22: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_42, arg50_1);  mul_42 = arg50_1 = None
        convert_element_type_57: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg51_1, torch.bfloat16);  arg51_1 = None
        convert_element_type_58: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg52_1, torch.bfloat16);  arg52_1 = None
        convert_element_type_59: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16)
        view_56: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_59, _shape_param_80);  convert_element_type_59 = _shape_param_80 = None
        permute_31: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_58, [1, 0]);  convert_element_type_58 = None
        addmm_16: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(convert_element_type_57, view_56, permute_31);  convert_element_type_57 = None
        view_57: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_16, _shape_param_81);  _shape_param_81 = None
        convert_element_type_60: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_57, torch.float32);  view_57 = None
        mul_43: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 0.5)
        mul_44: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 0.7071067811865476);  convert_element_type_60 = None
        erf_2: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_44);  mul_44 = None
        add_23: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_45: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_43, add_23);  mul_43 = add_23 = None
        convert_element_type_61: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        convert_element_type_62: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg53_1, torch.bfloat16);  arg53_1 = None
        convert_element_type_63: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg54_1, torch.bfloat16);  arg54_1 = None
        view_58: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_61, _shape_param_82);  convert_element_type_61 = _shape_param_82 = None
        permute_32: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_63, [1, 0]);  convert_element_type_63 = None
        addmm_17: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_62, view_58, permute_32);  convert_element_type_62 = None
        view_59: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_17, _shape_param_83);  addmm_17 = _shape_param_83 = None
        inductor_lookup_seed_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 9)
        inductor_random_9: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_84, inductor_lookup_seed_9, 'rand');  _shape_param_84 = inductor_lookup_seed_9 = None
        convert_element_type_64: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_9, torch.bfloat16);  inductor_random_9 = None
        gt_9: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_64, 0.1);  convert_element_type_64 = None
        mul_46: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_9, view_59);  view_59 = None
        mul_47: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_46, 1.1111111111111112);  mul_46 = None
        add_24: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_47, add_22);  mul_47 = add_22 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_12: "f32[32, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[32, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        add_25: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_9: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_13);  add_24 = getitem_13 = None
        mul_48: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
        mul_49: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg55_1);  arg55_1 = None
        add_26: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_49, arg56_1);  mul_49 = arg56_1 = None
        convert_element_type_65: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg57_1, torch.bfloat16);  arg57_1 = None
        convert_element_type_66: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg58_1, torch.bfloat16);  arg58_1 = None
        convert_element_type_67: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_26, torch.bfloat16)
        view_60: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_67, _shape_param_85);  convert_element_type_67 = _shape_param_85 = None
        permute_33: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_66, [1, 0]);  convert_element_type_66 = None
        addmm_18: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_65, view_60, permute_33);  convert_element_type_65 = None
        view_61: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_18, _shape_param_86);  addmm_18 = _shape_param_86 = None
        view_62: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_61, _shape_param_87);  view_61 = _shape_param_87 = None
        permute_34: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_62, [0, 2, 1, 3]);  view_62 = None
        convert_element_type_68: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg59_1, torch.bfloat16);  arg59_1 = None
        convert_element_type_69: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg60_1, torch.bfloat16);  arg60_1 = None
        permute_35: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_69, [1, 0]);  convert_element_type_69 = None
        addmm_19: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_68, view_60, permute_35);  convert_element_type_68 = None
        view_63: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_19, _shape_param_88);  addmm_19 = _shape_param_88 = None
        view_64: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_63, _shape_param_89);  view_63 = _shape_param_89 = None
        permute_36: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        convert_element_type_70: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg61_1, torch.bfloat16);  arg61_1 = None
        convert_element_type_71: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg62_1, torch.bfloat16);  arg62_1 = None
        permute_37: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_71, [1, 0]);  convert_element_type_71 = None
        addmm_20: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_70, view_60, permute_37);  convert_element_type_70 = None
        view_65: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_20, _shape_param_90);  addmm_20 = _shape_param_90 = None
        view_66: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_65, _shape_param_91);  view_65 = _shape_param_91 = None
        permute_38: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        mul_50: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_34, 0.3535533905932738);  permute_34 = None
        permute_39: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
        mul_51: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_39, 0.3535533905932738);  permute_39 = None
        expand_14: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_50, _shape_param_92);  mul_50 = _shape_param_92 = None
        clone_12: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_67: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_93);  clone_12 = _shape_param_93 = None
        expand_15: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_51, _shape_param_94);  mul_51 = _shape_param_94 = None
        clone_13: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_68: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_13, _shape_param_95);  clone_13 = _shape_param_95 = None
        bmm_6: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_67, view_68)
        view_69: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_6, _shape_param_96);  bmm_6 = _shape_param_96 = None
        convert_element_type_72: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_69, torch.float32)
        amax_3: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_72, [-1], True)
        sub_10: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_72, amax_3);  convert_element_type_72 = amax_3 = None
        exp_3: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_73: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        eq_3: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_69, -inf);  view_69 = None
        logical_not_6: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        where_3: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_7, full, convert_element_type_73);  logical_not_7 = convert_element_type_73 = None
        inductor_lookup_seed_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 10)
        inductor_random_10: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_97, inductor_lookup_seed_10, 'rand');  _shape_param_97 = inductor_lookup_seed_10 = None
        convert_element_type_74: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_10, torch.bfloat16);  inductor_random_10 = None
        gt_10: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_74, 0.1);  convert_element_type_74 = None
        mul_52: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_10, where_3)
        mul_53: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_52, 1.1111111111111112);  mul_52 = None
        expand_16: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_53, _shape_param_98);  mul_53 = _shape_param_98 = None
        view_70: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_16, _shape_param_99);  expand_16 = _shape_param_99 = None
        expand_17: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_38, _shape_param_100);  permute_38 = _shape_param_100 = None
        clone_14: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_71: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_14, _shape_param_101);  clone_14 = _shape_param_101 = None
        bmm_7: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_70, view_71)
        view_72: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_102);  bmm_7 = _shape_param_102 = None
        permute_40: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        clone_15: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_73: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_15, _shape_param_103);  clone_15 = _shape_param_103 = None
        convert_element_type_75: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg63_1, torch.bfloat16);  arg63_1 = None
        convert_element_type_76: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg64_1, torch.bfloat16);  arg64_1 = None
        view_74: "bf16[16384, 768]" = torch.ops.aten.view.default(view_73, _shape_param_104);  view_73 = _shape_param_104 = None
        permute_41: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_76, [1, 0]);  convert_element_type_76 = None
        addmm_21: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_75, view_74, permute_41);  convert_element_type_75 = None
        view_75: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_21, _shape_param_105);  addmm_21 = _shape_param_105 = None
        inductor_lookup_seed_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 11)
        inductor_random_11: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_106, inductor_lookup_seed_11, 'rand');  _shape_param_106 = inductor_lookup_seed_11 = None
        convert_element_type_77: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_11, torch.bfloat16);  inductor_random_11 = None
        gt_11: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_77, 0.1);  convert_element_type_77 = None
        mul_54: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_11, view_75);  view_75 = None
        mul_55: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_54, 1.1111111111111112);  mul_54 = None
        add_27: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_55, add_26);  mul_55 = add_26 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[32, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        add_28: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_11: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_15);  add_27 = getitem_15 = None
        mul_56: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
        mul_57: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_56, arg65_1);  arg65_1 = None
        add_29: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_57, arg66_1);  mul_57 = arg66_1 = None
        convert_element_type_78: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg67_1, torch.bfloat16);  arg67_1 = None
        convert_element_type_79: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg68_1, torch.bfloat16);  arg68_1 = None
        convert_element_type_80: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16)
        view_76: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_80, _shape_param_107);  convert_element_type_80 = _shape_param_107 = None
        permute_42: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_79, [1, 0]);  convert_element_type_79 = None
        addmm_22: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(convert_element_type_78, view_76, permute_42);  convert_element_type_78 = None
        view_77: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_22, _shape_param_108);  _shape_param_108 = None
        convert_element_type_81: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        mul_58: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_81, 0.5)
        mul_59: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_81, 0.7071067811865476);  convert_element_type_81 = None
        erf_3: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_59);  mul_59 = None
        add_30: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_60: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_58, add_30);  mul_58 = add_30 = None
        convert_element_type_82: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_60, torch.bfloat16);  mul_60 = None
        convert_element_type_83: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg69_1, torch.bfloat16);  arg69_1 = None
        convert_element_type_84: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg70_1, torch.bfloat16);  arg70_1 = None
        view_78: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_82, _shape_param_109);  convert_element_type_82 = _shape_param_109 = None
        permute_43: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_84, [1, 0]);  convert_element_type_84 = None
        addmm_23: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_83, view_78, permute_43);  convert_element_type_83 = None
        view_79: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_23, _shape_param_110);  addmm_23 = _shape_param_110 = None
        inductor_lookup_seed_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 12)
        inductor_random_12: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_111, inductor_lookup_seed_12, 'rand');  _shape_param_111 = inductor_lookup_seed_12 = None
        convert_element_type_85: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_12, torch.bfloat16);  inductor_random_12 = None
        gt_12: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_85, 0.1);  convert_element_type_85 = None
        mul_61: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_12, view_79);  view_79 = None
        mul_62: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_61, 1.1111111111111112);  mul_61 = None
        add_31: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_62, add_29);  mul_62 = add_29 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[32, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        add_32: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_12: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_17);  add_31 = getitem_17 = None
        mul_63: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
        mul_64: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_63, arg71_1);  arg71_1 = None
        add_33: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_64, arg72_1);  mul_64 = arg72_1 = None
        convert_element_type_86: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg73_1, torch.bfloat16);  arg73_1 = None
        convert_element_type_87: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg74_1, torch.bfloat16);  arg74_1 = None
        convert_element_type_88: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16)
        view_80: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_88, _shape_param_112);  convert_element_type_88 = _shape_param_112 = None
        permute_44: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_87, [1, 0]);  convert_element_type_87 = None
        addmm_24: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_86, view_80, permute_44);  convert_element_type_86 = None
        view_81: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_24, _shape_param_113);  addmm_24 = _shape_param_113 = None
        view_82: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_81, _shape_param_114);  view_81 = _shape_param_114 = None
        permute_45: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        convert_element_type_89: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg75_1, torch.bfloat16);  arg75_1 = None
        convert_element_type_90: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg76_1, torch.bfloat16);  arg76_1 = None
        permute_46: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_90, [1, 0]);  convert_element_type_90 = None
        addmm_25: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_89, view_80, permute_46);  convert_element_type_89 = None
        view_83: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_25, _shape_param_115);  addmm_25 = _shape_param_115 = None
        view_84: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_83, _shape_param_116);  view_83 = _shape_param_116 = None
        permute_47: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        convert_element_type_91: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg77_1, torch.bfloat16);  arg77_1 = None
        convert_element_type_92: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg78_1, torch.bfloat16);  arg78_1 = None
        permute_48: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_92, [1, 0]);  convert_element_type_92 = None
        addmm_26: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_91, view_80, permute_48);  convert_element_type_91 = None
        view_85: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_26, _shape_param_117);  addmm_26 = _shape_param_117 = None
        view_86: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_85, _shape_param_118);  view_85 = _shape_param_118 = None
        permute_49: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        mul_65: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_45, 0.3535533905932738);  permute_45 = None
        permute_50: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
        mul_66: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_50, 0.3535533905932738);  permute_50 = None
        expand_18: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_65, _shape_param_119);  mul_65 = _shape_param_119 = None
        clone_16: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_87: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_120);  clone_16 = _shape_param_120 = None
        expand_19: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_66, _shape_param_121);  mul_66 = _shape_param_121 = None
        clone_17: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_88: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_17, _shape_param_122);  clone_17 = _shape_param_122 = None
        bmm_8: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_87, view_88)
        view_89: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_8, _shape_param_123);  bmm_8 = _shape_param_123 = None
        convert_element_type_93: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_89, torch.float32)
        amax_4: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_93, [-1], True)
        sub_13: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_93, amax_4);  convert_element_type_93 = amax_4 = None
        exp_4: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_94: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        eq_4: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_89, -inf);  view_89 = None
        logical_not_8: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        where_4: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_9, full, convert_element_type_94);  logical_not_9 = convert_element_type_94 = None
        inductor_lookup_seed_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 13)
        inductor_random_13: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_124, inductor_lookup_seed_13, 'rand');  _shape_param_124 = inductor_lookup_seed_13 = None
        convert_element_type_95: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_13, torch.bfloat16);  inductor_random_13 = None
        gt_13: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_95, 0.1);  convert_element_type_95 = None
        mul_67: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_13, where_4)
        mul_68: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_67, 1.1111111111111112);  mul_67 = None
        expand_20: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_68, _shape_param_125);  mul_68 = _shape_param_125 = None
        view_90: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_20, _shape_param_126);  expand_20 = _shape_param_126 = None
        expand_21: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_49, _shape_param_127);  permute_49 = _shape_param_127 = None
        clone_18: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_91: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_18, _shape_param_128);  clone_18 = _shape_param_128 = None
        bmm_9: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_90, view_91)
        view_92: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_9, _shape_param_129);  bmm_9 = _shape_param_129 = None
        permute_51: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
        clone_19: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_93: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_19, _shape_param_130);  clone_19 = _shape_param_130 = None
        convert_element_type_96: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg79_1, torch.bfloat16);  arg79_1 = None
        convert_element_type_97: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg80_1, torch.bfloat16);  arg80_1 = None
        view_94: "bf16[16384, 768]" = torch.ops.aten.view.default(view_93, _shape_param_131);  view_93 = _shape_param_131 = None
        permute_52: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_97, [1, 0]);  convert_element_type_97 = None
        addmm_27: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_96, view_94, permute_52);  convert_element_type_96 = None
        view_95: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_27, _shape_param_132);  addmm_27 = _shape_param_132 = None
        inductor_lookup_seed_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 14)
        inductor_random_14: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_133, inductor_lookup_seed_14, 'rand');  _shape_param_133 = inductor_lookup_seed_14 = None
        convert_element_type_98: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_14, torch.bfloat16);  inductor_random_14 = None
        gt_14: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_98, 0.1);  convert_element_type_98 = None
        mul_69: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_14, view_95);  view_95 = None
        mul_70: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_69, 1.1111111111111112);  mul_69 = None
        add_34: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_70, add_33);  mul_70 = add_33 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_34, [2], correction = 0, keepdim = True)
        getitem_18: "f32[32, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[32, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        add_35: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        sub_14: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_34, getitem_19);  add_34 = getitem_19 = None
        mul_71: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
        mul_72: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_71, arg81_1);  arg81_1 = None
        add_36: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_72, arg82_1);  mul_72 = arg82_1 = None
        convert_element_type_99: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg83_1, torch.bfloat16);  arg83_1 = None
        convert_element_type_100: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg84_1, torch.bfloat16);  arg84_1 = None
        convert_element_type_101: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_36, torch.bfloat16)
        view_96: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_101, _shape_param_134);  convert_element_type_101 = _shape_param_134 = None
        permute_53: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_100, [1, 0]);  convert_element_type_100 = None
        addmm_28: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(convert_element_type_99, view_96, permute_53);  convert_element_type_99 = None
        view_97: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_28, _shape_param_135);  _shape_param_135 = None
        convert_element_type_102: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_97, torch.float32);  view_97 = None
        mul_73: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_102, 0.5)
        mul_74: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_102, 0.7071067811865476);  convert_element_type_102 = None
        erf_4: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_74);  mul_74 = None
        add_37: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_75: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_73, add_37);  mul_73 = add_37 = None
        convert_element_type_103: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_75, torch.bfloat16);  mul_75 = None
        convert_element_type_104: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg85_1, torch.bfloat16);  arg85_1 = None
        convert_element_type_105: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg86_1, torch.bfloat16);  arg86_1 = None
        view_98: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_103, _shape_param_136);  convert_element_type_103 = _shape_param_136 = None
        permute_54: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_105, [1, 0]);  convert_element_type_105 = None
        addmm_29: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_104, view_98, permute_54);  convert_element_type_104 = None
        view_99: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_29, _shape_param_137);  addmm_29 = _shape_param_137 = None
        inductor_lookup_seed_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 15)
        inductor_random_15: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_138, inductor_lookup_seed_15, 'rand');  _shape_param_138 = inductor_lookup_seed_15 = None
        convert_element_type_106: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_15, torch.bfloat16);  inductor_random_15 = None
        gt_15: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_106, 0.1);  convert_element_type_106 = None
        mul_76: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_15, view_99);  view_99 = None
        mul_77: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_76, 1.1111111111111112);  mul_76 = None
        add_38: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_77, add_36);  mul_77 = add_36 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[32, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        add_39: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_15: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_21);  add_38 = getitem_21 = None
        mul_78: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
        mul_79: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_78, arg87_1);  arg87_1 = None
        add_40: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_79, arg88_1);  mul_79 = arg88_1 = None
        convert_element_type_107: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg89_1, torch.bfloat16);  arg89_1 = None
        convert_element_type_108: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg90_1, torch.bfloat16);  arg90_1 = None
        convert_element_type_109: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16)
        view_100: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_109, _shape_param_139);  convert_element_type_109 = _shape_param_139 = None
        permute_55: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_108, [1, 0]);  convert_element_type_108 = None
        addmm_30: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_107, view_100, permute_55);  convert_element_type_107 = None
        view_101: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_30, _shape_param_140);  addmm_30 = _shape_param_140 = None
        view_102: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_101, _shape_param_141);  view_101 = _shape_param_141 = None
        permute_56: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        convert_element_type_110: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg91_1, torch.bfloat16);  arg91_1 = None
        convert_element_type_111: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg92_1, torch.bfloat16);  arg92_1 = None
        permute_57: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_111, [1, 0]);  convert_element_type_111 = None
        addmm_31: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_110, view_100, permute_57);  convert_element_type_110 = None
        view_103: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_31, _shape_param_142);  addmm_31 = _shape_param_142 = None
        view_104: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_103, _shape_param_143);  view_103 = _shape_param_143 = None
        permute_58: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        convert_element_type_112: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg93_1, torch.bfloat16);  arg93_1 = None
        convert_element_type_113: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg94_1, torch.bfloat16);  arg94_1 = None
        permute_59: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_113, [1, 0]);  convert_element_type_113 = None
        addmm_32: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_112, view_100, permute_59);  convert_element_type_112 = None
        view_105: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_32, _shape_param_144);  addmm_32 = _shape_param_144 = None
        view_106: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_105, _shape_param_145);  view_105 = _shape_param_145 = None
        permute_60: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        mul_80: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_56, 0.3535533905932738);  permute_56 = None
        permute_61: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
        mul_81: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_61, 0.3535533905932738);  permute_61 = None
        expand_22: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_80, _shape_param_146);  mul_80 = _shape_param_146 = None
        clone_20: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_107: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_147);  clone_20 = _shape_param_147 = None
        expand_23: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_81, _shape_param_148);  mul_81 = _shape_param_148 = None
        clone_21: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_108: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_21, _shape_param_149);  clone_21 = _shape_param_149 = None
        bmm_10: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_107, view_108)
        view_109: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_10, _shape_param_150);  bmm_10 = _shape_param_150 = None
        convert_element_type_114: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_109, torch.float32)
        amax_5: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_114, [-1], True)
        sub_16: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_114, amax_5);  convert_element_type_114 = amax_5 = None
        exp_5: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_115: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        eq_5: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_109, -inf);  view_109 = None
        logical_not_10: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        where_5: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_11, full, convert_element_type_115);  logical_not_11 = convert_element_type_115 = None
        inductor_lookup_seed_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 16)
        inductor_random_16: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_151, inductor_lookup_seed_16, 'rand');  _shape_param_151 = inductor_lookup_seed_16 = None
        convert_element_type_116: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_16, torch.bfloat16);  inductor_random_16 = None
        gt_16: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_116, 0.1);  convert_element_type_116 = None
        mul_82: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_16, where_5)
        mul_83: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_82, 1.1111111111111112);  mul_82 = None
        expand_24: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_83, _shape_param_152);  mul_83 = _shape_param_152 = None
        view_110: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_24, _shape_param_153);  expand_24 = _shape_param_153 = None
        expand_25: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_60, _shape_param_154);  permute_60 = _shape_param_154 = None
        clone_22: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_111: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_22, _shape_param_155);  clone_22 = _shape_param_155 = None
        bmm_11: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_110, view_111)
        view_112: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_156);  bmm_11 = _shape_param_156 = None
        permute_62: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        clone_23: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_113: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_23, _shape_param_157);  clone_23 = _shape_param_157 = None
        convert_element_type_117: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg95_1, torch.bfloat16);  arg95_1 = None
        convert_element_type_118: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg96_1, torch.bfloat16);  arg96_1 = None
        view_114: "bf16[16384, 768]" = torch.ops.aten.view.default(view_113, _shape_param_158);  view_113 = _shape_param_158 = None
        permute_63: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_118, [1, 0]);  convert_element_type_118 = None
        addmm_33: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_117, view_114, permute_63);  convert_element_type_117 = None
        view_115: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_33, _shape_param_159);  addmm_33 = _shape_param_159 = None
        inductor_lookup_seed_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 17)
        inductor_random_17: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_160, inductor_lookup_seed_17, 'rand');  _shape_param_160 = inductor_lookup_seed_17 = None
        convert_element_type_119: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_17, torch.bfloat16);  inductor_random_17 = None
        gt_17: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_119, 0.1);  convert_element_type_119 = None
        mul_84: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_17, view_115);  view_115 = None
        mul_85: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_84, 1.1111111111111112);  mul_84 = None
        add_41: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_85, add_40);  mul_85 = add_40 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[32, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        add_42: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_17: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_23);  add_41 = getitem_23 = None
        mul_86: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
        mul_87: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_86, arg97_1);  arg97_1 = None
        add_43: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_87, arg98_1);  mul_87 = arg98_1 = None
        convert_element_type_120: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg99_1, torch.bfloat16);  arg99_1 = None
        convert_element_type_121: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg100_1, torch.bfloat16);  arg100_1 = None
        convert_element_type_122: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16)
        view_116: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_122, _shape_param_161);  convert_element_type_122 = _shape_param_161 = None
        permute_64: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_121, [1, 0]);  convert_element_type_121 = None
        addmm_34: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(convert_element_type_120, view_116, permute_64);  convert_element_type_120 = None
        view_117: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_34, _shape_param_162);  _shape_param_162 = None
        convert_element_type_123: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_117, torch.float32);  view_117 = None
        mul_88: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_123, 0.5)
        mul_89: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_123, 0.7071067811865476);  convert_element_type_123 = None
        erf_5: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
        add_44: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_90: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_88, add_44);  mul_88 = add_44 = None
        convert_element_type_124: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16);  mul_90 = None
        convert_element_type_125: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg101_1, torch.bfloat16);  arg101_1 = None
        convert_element_type_126: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg102_1, torch.bfloat16);  arg102_1 = None
        view_118: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_124, _shape_param_163);  convert_element_type_124 = _shape_param_163 = None
        permute_65: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_126, [1, 0]);  convert_element_type_126 = None
        addmm_35: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_125, view_118, permute_65);  convert_element_type_125 = None
        view_119: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_35, _shape_param_164);  addmm_35 = _shape_param_164 = None
        inductor_lookup_seed_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 18)
        inductor_random_18: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_165, inductor_lookup_seed_18, 'rand');  _shape_param_165 = inductor_lookup_seed_18 = None
        convert_element_type_127: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_18, torch.bfloat16);  inductor_random_18 = None
        gt_18: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_127, 0.1);  convert_element_type_127 = None
        mul_91: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_18, view_119);  view_119 = None
        mul_92: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_91, 1.1111111111111112);  mul_91 = None
        add_45: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_92, add_43);  mul_92 = add_43 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[32, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        add_46: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_18: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_25);  add_45 = getitem_25 = None
        mul_93: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
        mul_94: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_93, arg103_1);  arg103_1 = None
        add_47: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_94, arg104_1);  mul_94 = arg104_1 = None
        convert_element_type_128: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg105_1, torch.bfloat16);  arg105_1 = None
        convert_element_type_129: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg106_1, torch.bfloat16);  arg106_1 = None
        convert_element_type_130: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16)
        view_120: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_130, _shape_param_166);  convert_element_type_130 = _shape_param_166 = None
        permute_66: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_129, [1, 0]);  convert_element_type_129 = None
        addmm_36: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_128, view_120, permute_66);  convert_element_type_128 = None
        view_121: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_36, _shape_param_167);  addmm_36 = _shape_param_167 = None
        view_122: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_121, _shape_param_168);  view_121 = _shape_param_168 = None
        permute_67: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        convert_element_type_131: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg107_1, torch.bfloat16);  arg107_1 = None
        convert_element_type_132: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg108_1, torch.bfloat16);  arg108_1 = None
        permute_68: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_132, [1, 0]);  convert_element_type_132 = None
        addmm_37: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_131, view_120, permute_68);  convert_element_type_131 = None
        view_123: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_37, _shape_param_169);  addmm_37 = _shape_param_169 = None
        view_124: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_123, _shape_param_170);  view_123 = _shape_param_170 = None
        permute_69: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        convert_element_type_133: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg109_1, torch.bfloat16);  arg109_1 = None
        convert_element_type_134: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg110_1, torch.bfloat16);  arg110_1 = None
        permute_70: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_134, [1, 0]);  convert_element_type_134 = None
        addmm_38: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_133, view_120, permute_70);  convert_element_type_133 = None
        view_125: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_38, _shape_param_171);  addmm_38 = _shape_param_171 = None
        view_126: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_125, _shape_param_172);  view_125 = _shape_param_172 = None
        permute_71: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        mul_95: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_67, 0.3535533905932738);  permute_67 = None
        permute_72: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_69, [0, 1, 3, 2]);  permute_69 = None
        mul_96: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_72, 0.3535533905932738);  permute_72 = None
        expand_26: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_95, _shape_param_173);  mul_95 = _shape_param_173 = None
        clone_24: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_127: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_174);  clone_24 = _shape_param_174 = None
        expand_27: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_96, _shape_param_175);  mul_96 = _shape_param_175 = None
        clone_25: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_128: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_25, _shape_param_176);  clone_25 = _shape_param_176 = None
        bmm_12: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_127, view_128)
        view_129: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_12, _shape_param_177);  bmm_12 = _shape_param_177 = None
        convert_element_type_135: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_129, torch.float32)
        amax_6: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_135, [-1], True)
        sub_19: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_135, amax_6);  convert_element_type_135 = amax_6 = None
        exp_6: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_136: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        eq_6: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_129, -inf);  view_129 = None
        logical_not_12: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        where_6: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_13, full, convert_element_type_136);  logical_not_13 = convert_element_type_136 = None
        inductor_lookup_seed_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 19)
        inductor_random_19: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_178, inductor_lookup_seed_19, 'rand');  _shape_param_178 = inductor_lookup_seed_19 = None
        convert_element_type_137: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_19, torch.bfloat16);  inductor_random_19 = None
        gt_19: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_137, 0.1);  convert_element_type_137 = None
        mul_97: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_19, where_6)
        mul_98: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_97, 1.1111111111111112);  mul_97 = None
        expand_28: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_98, _shape_param_179);  mul_98 = _shape_param_179 = None
        view_130: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_28, _shape_param_180);  expand_28 = _shape_param_180 = None
        expand_29: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_71, _shape_param_181);  permute_71 = _shape_param_181 = None
        clone_26: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_131: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_26, _shape_param_182);  clone_26 = _shape_param_182 = None
        bmm_13: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_130, view_131)
        view_132: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_13, _shape_param_183);  bmm_13 = _shape_param_183 = None
        permute_73: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
        clone_27: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_133: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_27, _shape_param_184);  clone_27 = _shape_param_184 = None
        convert_element_type_138: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg111_1, torch.bfloat16);  arg111_1 = None
        convert_element_type_139: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg112_1, torch.bfloat16);  arg112_1 = None
        view_134: "bf16[16384, 768]" = torch.ops.aten.view.default(view_133, _shape_param_185);  view_133 = _shape_param_185 = None
        permute_74: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_139, [1, 0]);  convert_element_type_139 = None
        addmm_39: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_138, view_134, permute_74);  convert_element_type_138 = None
        view_135: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_39, _shape_param_186);  addmm_39 = _shape_param_186 = None
        inductor_lookup_seed_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 20)
        inductor_random_20: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_187, inductor_lookup_seed_20, 'rand');  _shape_param_187 = inductor_lookup_seed_20 = None
        convert_element_type_140: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_20, torch.bfloat16);  inductor_random_20 = None
        gt_20: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_140, 0.1);  convert_element_type_140 = None
        mul_99: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_20, view_135);  view_135 = None
        mul_100: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_99, 1.1111111111111112);  mul_99 = None
        add_48: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_100, add_47);  mul_100 = add_47 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[32, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        add_49: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_20: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_27);  add_48 = getitem_27 = None
        mul_101: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
        mul_102: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_101, arg113_1);  arg113_1 = None
        add_50: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_102, arg114_1);  mul_102 = arg114_1 = None
        convert_element_type_141: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg115_1, torch.bfloat16);  arg115_1 = None
        convert_element_type_142: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg116_1, torch.bfloat16);  arg116_1 = None
        convert_element_type_143: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16)
        view_136: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_143, _shape_param_188);  convert_element_type_143 = _shape_param_188 = None
        permute_75: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_142, [1, 0]);  convert_element_type_142 = None
        addmm_40: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(convert_element_type_141, view_136, permute_75);  convert_element_type_141 = None
        view_137: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_40, _shape_param_189);  _shape_param_189 = None
        convert_element_type_144: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_137, torch.float32);  view_137 = None
        mul_103: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_144, 0.5)
        mul_104: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_144, 0.7071067811865476);  convert_element_type_144 = None
        erf_6: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_104);  mul_104 = None
        add_51: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_105: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_103, add_51);  mul_103 = add_51 = None
        convert_element_type_145: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_105, torch.bfloat16);  mul_105 = None
        convert_element_type_146: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg117_1, torch.bfloat16);  arg117_1 = None
        convert_element_type_147: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg118_1, torch.bfloat16);  arg118_1 = None
        view_138: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_145, _shape_param_190);  convert_element_type_145 = _shape_param_190 = None
        permute_76: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_147, [1, 0]);  convert_element_type_147 = None
        addmm_41: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_146, view_138, permute_76);  convert_element_type_146 = None
        view_139: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_41, _shape_param_191);  addmm_41 = _shape_param_191 = None
        inductor_lookup_seed_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 21)
        inductor_random_21: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_192, inductor_lookup_seed_21, 'rand');  _shape_param_192 = inductor_lookup_seed_21 = None
        convert_element_type_148: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_21, torch.bfloat16);  inductor_random_21 = None
        gt_21: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_148, 0.1);  convert_element_type_148 = None
        mul_106: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_21, view_139);  view_139 = None
        mul_107: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_106, 1.1111111111111112);  mul_106 = None
        add_52: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_107, add_50);  mul_107 = add_50 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 512, 1]" = var_mean_14[0]
        getitem_29: "f32[32, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        add_53: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_21: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_29);  add_52 = getitem_29 = None
        mul_108: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
        mul_109: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_108, arg119_1);  arg119_1 = None
        add_54: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_109, arg120_1);  mul_109 = arg120_1 = None
        convert_element_type_149: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg121_1, torch.bfloat16);  arg121_1 = None
        convert_element_type_150: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg122_1, torch.bfloat16);  arg122_1 = None
        convert_element_type_151: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16)
        view_140: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_151, _shape_param_193);  convert_element_type_151 = _shape_param_193 = None
        permute_77: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_150, [1, 0]);  convert_element_type_150 = None
        addmm_42: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_149, view_140, permute_77);  convert_element_type_149 = None
        view_141: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_42, _shape_param_194);  addmm_42 = _shape_param_194 = None
        view_142: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_141, _shape_param_195);  view_141 = _shape_param_195 = None
        permute_78: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
        convert_element_type_152: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg123_1, torch.bfloat16);  arg123_1 = None
        convert_element_type_153: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg124_1, torch.bfloat16);  arg124_1 = None
        permute_79: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_153, [1, 0]);  convert_element_type_153 = None
        addmm_43: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_152, view_140, permute_79);  convert_element_type_152 = None
        view_143: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_43, _shape_param_196);  addmm_43 = _shape_param_196 = None
        view_144: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_143, _shape_param_197);  view_143 = _shape_param_197 = None
        permute_80: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        convert_element_type_154: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg125_1, torch.bfloat16);  arg125_1 = None
        convert_element_type_155: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg126_1, torch.bfloat16);  arg126_1 = None
        permute_81: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_155, [1, 0]);  convert_element_type_155 = None
        addmm_44: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_154, view_140, permute_81);  convert_element_type_154 = None
        view_145: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_44, _shape_param_198);  addmm_44 = _shape_param_198 = None
        view_146: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_145, _shape_param_199);  view_145 = _shape_param_199 = None
        permute_82: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        mul_110: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_78, 0.3535533905932738);  permute_78 = None
        permute_83: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2]);  permute_80 = None
        mul_111: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_83, 0.3535533905932738);  permute_83 = None
        expand_30: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_110, _shape_param_200);  mul_110 = _shape_param_200 = None
        clone_28: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_147: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_201);  clone_28 = _shape_param_201 = None
        expand_31: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_111, _shape_param_202);  mul_111 = _shape_param_202 = None
        clone_29: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_148: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_29, _shape_param_203);  clone_29 = _shape_param_203 = None
        bmm_14: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_147, view_148)
        view_149: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_14, _shape_param_204);  bmm_14 = _shape_param_204 = None
        convert_element_type_156: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_149, torch.float32)
        amax_7: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_156, [-1], True)
        sub_22: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_156, amax_7);  convert_element_type_156 = amax_7 = None
        exp_7: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_157: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        eq_7: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_149, -inf);  view_149 = None
        logical_not_14: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        where_7: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_15, full, convert_element_type_157);  logical_not_15 = convert_element_type_157 = None
        inductor_lookup_seed_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 22)
        inductor_random_22: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_205, inductor_lookup_seed_22, 'rand');  _shape_param_205 = inductor_lookup_seed_22 = None
        convert_element_type_158: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_22, torch.bfloat16);  inductor_random_22 = None
        gt_22: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_158, 0.1);  convert_element_type_158 = None
        mul_112: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_22, where_7)
        mul_113: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_112, 1.1111111111111112);  mul_112 = None
        expand_32: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_113, _shape_param_206);  mul_113 = _shape_param_206 = None
        view_150: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_32, _shape_param_207);  expand_32 = _shape_param_207 = None
        expand_33: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_82, _shape_param_208);  permute_82 = _shape_param_208 = None
        clone_30: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_151: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_30, _shape_param_209);  clone_30 = _shape_param_209 = None
        bmm_15: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_150, view_151)
        view_152: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_210);  bmm_15 = _shape_param_210 = None
        permute_84: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        clone_31: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_153: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_31, _shape_param_211);  clone_31 = _shape_param_211 = None
        convert_element_type_159: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg127_1, torch.bfloat16);  arg127_1 = None
        convert_element_type_160: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg128_1, torch.bfloat16);  arg128_1 = None
        view_154: "bf16[16384, 768]" = torch.ops.aten.view.default(view_153, _shape_param_212);  view_153 = _shape_param_212 = None
        permute_85: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_160, [1, 0]);  convert_element_type_160 = None
        addmm_45: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_159, view_154, permute_85);  convert_element_type_159 = None
        view_155: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_45, _shape_param_213);  addmm_45 = _shape_param_213 = None
        inductor_lookup_seed_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 23)
        inductor_random_23: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_214, inductor_lookup_seed_23, 'rand');  _shape_param_214 = inductor_lookup_seed_23 = None
        convert_element_type_161: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_23, torch.bfloat16);  inductor_random_23 = None
        gt_23: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_161, 0.1);  convert_element_type_161 = None
        mul_114: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_23, view_155);  view_155 = None
        mul_115: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_114, 1.1111111111111112);  mul_114 = None
        add_55: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_115, add_54);  mul_115 = add_54 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 512, 1]" = var_mean_15[0]
        getitem_31: "f32[32, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        add_56: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_23: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_31);  add_55 = getitem_31 = None
        mul_116: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
        mul_117: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_116, arg129_1);  arg129_1 = None
        add_57: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_117, arg130_1);  mul_117 = arg130_1 = None
        convert_element_type_162: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg131_1, torch.bfloat16);  arg131_1 = None
        convert_element_type_163: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg132_1, torch.bfloat16);  arg132_1 = None
        convert_element_type_164: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16)
        view_156: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_164, _shape_param_215);  convert_element_type_164 = _shape_param_215 = None
        permute_86: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_163, [1, 0]);  convert_element_type_163 = None
        addmm_46: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(convert_element_type_162, view_156, permute_86);  convert_element_type_162 = None
        view_157: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_46, _shape_param_216);  _shape_param_216 = None
        convert_element_type_165: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_157, torch.float32);  view_157 = None
        mul_118: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_165, 0.5)
        mul_119: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_165, 0.7071067811865476);  convert_element_type_165 = None
        erf_7: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_119);  mul_119 = None
        add_58: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_120: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_118, add_58);  mul_118 = add_58 = None
        convert_element_type_166: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None
        convert_element_type_167: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg133_1, torch.bfloat16);  arg133_1 = None
        convert_element_type_168: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg134_1, torch.bfloat16);  arg134_1 = None
        view_158: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_166, _shape_param_217);  convert_element_type_166 = _shape_param_217 = None
        permute_87: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_168, [1, 0]);  convert_element_type_168 = None
        addmm_47: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_167, view_158, permute_87);  convert_element_type_167 = None
        view_159: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_47, _shape_param_218);  addmm_47 = _shape_param_218 = None
        inductor_lookup_seed_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 24)
        inductor_random_24: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_219, inductor_lookup_seed_24, 'rand');  _shape_param_219 = inductor_lookup_seed_24 = None
        convert_element_type_169: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_24, torch.bfloat16);  inductor_random_24 = None
        gt_24: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_169, 0.1);  convert_element_type_169 = None
        mul_121: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_24, view_159);  view_159 = None
        mul_122: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_121, 1.1111111111111112);  mul_121 = None
        add_59: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_122, add_57);  mul_122 = add_57 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 512, 1]" = var_mean_16[0]
        getitem_33: "f32[32, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        add_60: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_24: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_33);  add_59 = getitem_33 = None
        mul_123: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
        mul_124: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_123, arg135_1);  arg135_1 = None
        add_61: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_124, arg136_1);  mul_124 = arg136_1 = None
        convert_element_type_170: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg137_1, torch.bfloat16);  arg137_1 = None
        convert_element_type_171: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg138_1, torch.bfloat16);  arg138_1 = None
        convert_element_type_172: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16)
        view_160: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_172, _shape_param_220);  convert_element_type_172 = _shape_param_220 = None
        permute_88: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_171, [1, 0]);  convert_element_type_171 = None
        addmm_48: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_170, view_160, permute_88);  convert_element_type_170 = None
        view_161: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_48, _shape_param_221);  addmm_48 = _shape_param_221 = None
        view_162: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_161, _shape_param_222);  view_161 = _shape_param_222 = None
        permute_89: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        convert_element_type_173: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg139_1, torch.bfloat16);  arg139_1 = None
        convert_element_type_174: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg140_1, torch.bfloat16);  arg140_1 = None
        permute_90: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_174, [1, 0]);  convert_element_type_174 = None
        addmm_49: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_173, view_160, permute_90);  convert_element_type_173 = None
        view_163: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_49, _shape_param_223);  addmm_49 = _shape_param_223 = None
        view_164: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_163, _shape_param_224);  view_163 = _shape_param_224 = None
        permute_91: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        convert_element_type_175: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg141_1, torch.bfloat16);  arg141_1 = None
        convert_element_type_176: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg142_1, torch.bfloat16);  arg142_1 = None
        permute_92: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_176, [1, 0]);  convert_element_type_176 = None
        addmm_50: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_175, view_160, permute_92);  convert_element_type_175 = None
        view_165: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_50, _shape_param_225);  addmm_50 = _shape_param_225 = None
        view_166: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_165, _shape_param_226);  view_165 = _shape_param_226 = None
        permute_93: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        mul_125: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_89, 0.3535533905932738);  permute_89 = None
        permute_94: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
        mul_126: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_94, 0.3535533905932738);  permute_94 = None
        expand_34: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_125, _shape_param_227);  mul_125 = _shape_param_227 = None
        clone_32: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_167: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_32, _shape_param_228);  clone_32 = _shape_param_228 = None
        expand_35: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_126, _shape_param_229);  mul_126 = _shape_param_229 = None
        clone_33: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_168: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_33, _shape_param_230);  clone_33 = _shape_param_230 = None
        bmm_16: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_167, view_168)
        view_169: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_16, _shape_param_231);  bmm_16 = _shape_param_231 = None
        convert_element_type_177: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_169, torch.float32)
        amax_8: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_177, [-1], True)
        sub_25: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_177, amax_8);  convert_element_type_177 = amax_8 = None
        exp_8: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_178: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        eq_8: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_169, -inf);  view_169 = None
        logical_not_16: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        where_8: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_17, full, convert_element_type_178);  logical_not_17 = convert_element_type_178 = None
        inductor_lookup_seed_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 25)
        inductor_random_25: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_232, inductor_lookup_seed_25, 'rand');  _shape_param_232 = inductor_lookup_seed_25 = None
        convert_element_type_179: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_25, torch.bfloat16);  inductor_random_25 = None
        gt_25: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_179, 0.1);  convert_element_type_179 = None
        mul_127: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_25, where_8)
        mul_128: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_127, 1.1111111111111112);  mul_127 = None
        expand_36: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_128, _shape_param_233);  mul_128 = _shape_param_233 = None
        view_170: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_36, _shape_param_234);  expand_36 = _shape_param_234 = None
        expand_37: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_93, _shape_param_235);  permute_93 = _shape_param_235 = None
        clone_34: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_171: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_34, _shape_param_236);  clone_34 = _shape_param_236 = None
        bmm_17: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_170, view_171)
        view_172: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_17, _shape_param_237);  bmm_17 = _shape_param_237 = None
        permute_95: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
        clone_35: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_173: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_35, _shape_param_238);  clone_35 = _shape_param_238 = None
        convert_element_type_180: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg143_1, torch.bfloat16);  arg143_1 = None
        convert_element_type_181: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg144_1, torch.bfloat16);  arg144_1 = None
        view_174: "bf16[16384, 768]" = torch.ops.aten.view.default(view_173, _shape_param_239);  view_173 = _shape_param_239 = None
        permute_96: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_181, [1, 0]);  convert_element_type_181 = None
        addmm_51: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_180, view_174, permute_96);  convert_element_type_180 = None
        view_175: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_51, _shape_param_240);  addmm_51 = _shape_param_240 = None
        inductor_lookup_seed_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 26)
        inductor_random_26: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_241, inductor_lookup_seed_26, 'rand');  _shape_param_241 = inductor_lookup_seed_26 = None
        convert_element_type_182: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_26, torch.bfloat16);  inductor_random_26 = None
        gt_26: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_182, 0.1);  convert_element_type_182 = None
        mul_129: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_26, view_175);  view_175 = None
        mul_130: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_129, 1.1111111111111112);  mul_129 = None
        add_62: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_130, add_61);  mul_130 = add_61 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
        getitem_34: "f32[32, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[32, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        add_63: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        sub_26: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_62, getitem_35);  add_62 = getitem_35 = None
        mul_131: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
        mul_132: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_131, arg145_1);  arg145_1 = None
        add_64: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_132, arg146_1);  mul_132 = arg146_1 = None
        convert_element_type_183: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg147_1, torch.bfloat16);  arg147_1 = None
        convert_element_type_184: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg148_1, torch.bfloat16);  arg148_1 = None
        convert_element_type_185: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16)
        view_176: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_185, _shape_param_242);  convert_element_type_185 = _shape_param_242 = None
        permute_97: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_184, [1, 0]);  convert_element_type_184 = None
        addmm_52: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(convert_element_type_183, view_176, permute_97);  convert_element_type_183 = None
        view_177: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_52, _shape_param_243);  _shape_param_243 = None
        convert_element_type_186: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_177, torch.float32);  view_177 = None
        mul_133: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_186, 0.5)
        mul_134: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_186, 0.7071067811865476);  convert_element_type_186 = None
        erf_8: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_134);  mul_134 = None
        add_65: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_135: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_133, add_65);  mul_133 = add_65 = None
        convert_element_type_187: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_135, torch.bfloat16);  mul_135 = None
        convert_element_type_188: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg149_1, torch.bfloat16);  arg149_1 = None
        convert_element_type_189: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg150_1, torch.bfloat16);  arg150_1 = None
        view_178: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_187, _shape_param_244);  convert_element_type_187 = _shape_param_244 = None
        permute_98: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_189, [1, 0]);  convert_element_type_189 = None
        addmm_53: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_188, view_178, permute_98);  convert_element_type_188 = None
        view_179: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_53, _shape_param_245);  addmm_53 = _shape_param_245 = None
        inductor_lookup_seed_27: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 27)
        inductor_random_27: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_246, inductor_lookup_seed_27, 'rand');  _shape_param_246 = inductor_lookup_seed_27 = None
        convert_element_type_190: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_27, torch.bfloat16);  inductor_random_27 = None
        gt_27: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_190, 0.1);  convert_element_type_190 = None
        mul_136: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_27, view_179);  view_179 = None
        mul_137: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_136, 1.1111111111111112);  mul_136 = None
        add_66: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_137, add_64);  mul_137 = add_64 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
        getitem_36: "f32[32, 512, 1]" = var_mean_18[0]
        getitem_37: "f32[32, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        add_67: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        sub_27: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_66, getitem_37);  add_66 = getitem_37 = None
        mul_138: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
        mul_139: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_138, arg151_1);  arg151_1 = None
        add_68: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_139, arg152_1);  mul_139 = arg152_1 = None
        convert_element_type_191: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg153_1, torch.bfloat16);  arg153_1 = None
        convert_element_type_192: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg154_1, torch.bfloat16);  arg154_1 = None
        convert_element_type_193: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16)
        view_180: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_193, _shape_param_247);  convert_element_type_193 = _shape_param_247 = None
        permute_99: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_192, [1, 0]);  convert_element_type_192 = None
        addmm_54: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_191, view_180, permute_99);  convert_element_type_191 = None
        view_181: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_54, _shape_param_248);  addmm_54 = _shape_param_248 = None
        view_182: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_181, _shape_param_249);  view_181 = _shape_param_249 = None
        permute_100: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        convert_element_type_194: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg155_1, torch.bfloat16);  arg155_1 = None
        convert_element_type_195: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg156_1, torch.bfloat16);  arg156_1 = None
        permute_101: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_195, [1, 0]);  convert_element_type_195 = None
        addmm_55: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_194, view_180, permute_101);  convert_element_type_194 = None
        view_183: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_55, _shape_param_250);  addmm_55 = _shape_param_250 = None
        view_184: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_183, _shape_param_251);  view_183 = _shape_param_251 = None
        permute_102: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        convert_element_type_196: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg157_1, torch.bfloat16);  arg157_1 = None
        convert_element_type_197: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg158_1, torch.bfloat16);  arg158_1 = None
        permute_103: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_197, [1, 0]);  convert_element_type_197 = None
        addmm_56: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_196, view_180, permute_103);  convert_element_type_196 = None
        view_185: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_56, _shape_param_252);  addmm_56 = _shape_param_252 = None
        view_186: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_185, _shape_param_253);  view_185 = _shape_param_253 = None
        permute_104: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        mul_140: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_100, 0.3535533905932738);  permute_100 = None
        permute_105: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_102, [0, 1, 3, 2]);  permute_102 = None
        mul_141: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_105, 0.3535533905932738);  permute_105 = None
        expand_38: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_140, _shape_param_254);  mul_140 = _shape_param_254 = None
        clone_36: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_187: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_255);  clone_36 = _shape_param_255 = None
        expand_39: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_141, _shape_param_256);  mul_141 = _shape_param_256 = None
        clone_37: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_188: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_37, _shape_param_257);  clone_37 = _shape_param_257 = None
        bmm_18: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_187, view_188)
        view_189: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_18, _shape_param_258);  bmm_18 = _shape_param_258 = None
        convert_element_type_198: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_189, torch.float32)
        amax_9: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_198, [-1], True)
        sub_28: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_198, amax_9);  convert_element_type_198 = amax_9 = None
        exp_9: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_199: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        eq_9: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_189, -inf);  view_189 = None
        logical_not_18: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        where_9: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_19, full, convert_element_type_199);  logical_not_19 = convert_element_type_199 = None
        inductor_lookup_seed_28: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 28)
        inductor_random_28: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_259, inductor_lookup_seed_28, 'rand');  _shape_param_259 = inductor_lookup_seed_28 = None
        convert_element_type_200: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_28, torch.bfloat16);  inductor_random_28 = None
        gt_28: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_200, 0.1);  convert_element_type_200 = None
        mul_142: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_28, where_9)
        mul_143: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_142, 1.1111111111111112);  mul_142 = None
        expand_40: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_143, _shape_param_260);  mul_143 = _shape_param_260 = None
        view_190: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_40, _shape_param_261);  expand_40 = _shape_param_261 = None
        expand_41: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_104, _shape_param_262);  permute_104 = _shape_param_262 = None
        clone_38: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_191: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_38, _shape_param_263);  clone_38 = _shape_param_263 = None
        bmm_19: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_190, view_191)
        view_192: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_264);  bmm_19 = _shape_param_264 = None
        permute_106: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        clone_39: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_193: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_39, _shape_param_265);  clone_39 = _shape_param_265 = None
        convert_element_type_201: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg159_1, torch.bfloat16);  arg159_1 = None
        convert_element_type_202: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg160_1, torch.bfloat16);  arg160_1 = None
        view_194: "bf16[16384, 768]" = torch.ops.aten.view.default(view_193, _shape_param_266);  view_193 = _shape_param_266 = None
        permute_107: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_202, [1, 0]);  convert_element_type_202 = None
        addmm_57: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_201, view_194, permute_107);  convert_element_type_201 = None
        view_195: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_57, _shape_param_267);  addmm_57 = _shape_param_267 = None
        inductor_lookup_seed_29: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 29)
        inductor_random_29: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_268, inductor_lookup_seed_29, 'rand');  _shape_param_268 = inductor_lookup_seed_29 = None
        convert_element_type_203: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_29, torch.bfloat16);  inductor_random_29 = None
        gt_29: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_203, 0.1);  convert_element_type_203 = None
        mul_144: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_29, view_195);  view_195 = None
        mul_145: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_144, 1.1111111111111112);  mul_144 = None
        add_69: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_145, add_68);  mul_145 = add_68 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
        getitem_38: "f32[32, 512, 1]" = var_mean_19[0]
        getitem_39: "f32[32, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        add_70: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_29: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_69, getitem_39);  add_69 = getitem_39 = None
        mul_146: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
        mul_147: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_146, arg161_1);  arg161_1 = None
        add_71: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_147, arg162_1);  mul_147 = arg162_1 = None
        convert_element_type_204: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg163_1, torch.bfloat16);  arg163_1 = None
        convert_element_type_205: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg164_1, torch.bfloat16);  arg164_1 = None
        convert_element_type_206: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16)
        view_196: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_206, _shape_param_269);  convert_element_type_206 = _shape_param_269 = None
        permute_108: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_205, [1, 0]);  convert_element_type_205 = None
        addmm_58: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(convert_element_type_204, view_196, permute_108);  convert_element_type_204 = None
        view_197: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_58, _shape_param_270);  _shape_param_270 = None
        convert_element_type_207: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_197, torch.float32);  view_197 = None
        mul_148: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_207, 0.5)
        mul_149: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_207, 0.7071067811865476);  convert_element_type_207 = None
        erf_9: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_149);  mul_149 = None
        add_72: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_150: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_148, add_72);  mul_148 = add_72 = None
        convert_element_type_208: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_150, torch.bfloat16);  mul_150 = None
        convert_element_type_209: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg165_1, torch.bfloat16);  arg165_1 = None
        convert_element_type_210: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg166_1, torch.bfloat16);  arg166_1 = None
        view_198: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_208, _shape_param_271);  convert_element_type_208 = _shape_param_271 = None
        permute_109: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_210, [1, 0]);  convert_element_type_210 = None
        addmm_59: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_209, view_198, permute_109);  convert_element_type_209 = None
        view_199: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_59, _shape_param_272);  addmm_59 = _shape_param_272 = None
        inductor_lookup_seed_30: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 30)
        inductor_random_30: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_273, inductor_lookup_seed_30, 'rand');  _shape_param_273 = inductor_lookup_seed_30 = None
        convert_element_type_211: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_30, torch.bfloat16);  inductor_random_30 = None
        gt_30: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_211, 0.1);  convert_element_type_211 = None
        mul_151: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_30, view_199);  view_199 = None
        mul_152: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_151, 1.1111111111111112);  mul_151 = None
        add_73: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_152, add_71);  mul_152 = add_71 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_40: "f32[32, 512, 1]" = var_mean_20[0]
        getitem_41: "f32[32, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        add_74: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_30: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_41);  add_73 = getitem_41 = None
        mul_153: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
        mul_154: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_153, arg167_1);  arg167_1 = None
        add_75: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_154, arg168_1);  mul_154 = arg168_1 = None
        convert_element_type_212: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg169_1, torch.bfloat16);  arg169_1 = None
        convert_element_type_213: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg170_1, torch.bfloat16);  arg170_1 = None
        convert_element_type_214: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_75, torch.bfloat16)
        view_200: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_214, _shape_param_274);  convert_element_type_214 = _shape_param_274 = None
        permute_110: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_213, [1, 0]);  convert_element_type_213 = None
        addmm_60: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_212, view_200, permute_110);  convert_element_type_212 = None
        view_201: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_60, _shape_param_275);  addmm_60 = _shape_param_275 = None
        view_202: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_201, _shape_param_276);  view_201 = _shape_param_276 = None
        permute_111: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
        convert_element_type_215: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg171_1, torch.bfloat16);  arg171_1 = None
        convert_element_type_216: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg172_1, torch.bfloat16);  arg172_1 = None
        permute_112: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_216, [1, 0]);  convert_element_type_216 = None
        addmm_61: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_215, view_200, permute_112);  convert_element_type_215 = None
        view_203: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_61, _shape_param_277);  addmm_61 = _shape_param_277 = None
        view_204: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_203, _shape_param_278);  view_203 = _shape_param_278 = None
        permute_113: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
        convert_element_type_217: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg173_1, torch.bfloat16);  arg173_1 = None
        convert_element_type_218: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg174_1, torch.bfloat16);  arg174_1 = None
        permute_114: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_218, [1, 0]);  convert_element_type_218 = None
        addmm_62: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_217, view_200, permute_114);  convert_element_type_217 = None
        view_205: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_62, _shape_param_279);  addmm_62 = _shape_param_279 = None
        view_206: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_205, _shape_param_280);  view_205 = _shape_param_280 = None
        permute_115: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        mul_155: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_111, 0.3535533905932738);  permute_111 = None
        permute_116: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_113, [0, 1, 3, 2]);  permute_113 = None
        mul_156: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_116, 0.3535533905932738);  permute_116 = None
        expand_42: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_155, _shape_param_281);  mul_155 = _shape_param_281 = None
        clone_40: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_207: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_282);  clone_40 = _shape_param_282 = None
        expand_43: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_156, _shape_param_283);  mul_156 = _shape_param_283 = None
        clone_41: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_208: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_41, _shape_param_284);  clone_41 = _shape_param_284 = None
        bmm_20: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_207, view_208)
        view_209: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_20, _shape_param_285);  bmm_20 = _shape_param_285 = None
        convert_element_type_219: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_209, torch.float32)
        amax_10: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_219, [-1], True)
        sub_31: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_219, amax_10);  convert_element_type_219 = amax_10 = None
        exp_10: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_220: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        eq_10: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_209, -inf);  view_209 = None
        logical_not_20: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        where_10: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_21, full, convert_element_type_220);  logical_not_21 = convert_element_type_220 = None
        inductor_lookup_seed_31: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 31)
        inductor_random_31: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_286, inductor_lookup_seed_31, 'rand');  _shape_param_286 = inductor_lookup_seed_31 = None
        convert_element_type_221: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_31, torch.bfloat16);  inductor_random_31 = None
        gt_31: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_221, 0.1);  convert_element_type_221 = None
        mul_157: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_31, where_10)
        mul_158: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_157, 1.1111111111111112);  mul_157 = None
        expand_44: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_158, _shape_param_287);  mul_158 = _shape_param_287 = None
        view_210: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_44, _shape_param_288);  expand_44 = _shape_param_288 = None
        expand_45: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_115, _shape_param_289);  permute_115 = _shape_param_289 = None
        clone_42: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_211: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_42, _shape_param_290);  clone_42 = _shape_param_290 = None
        bmm_21: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_210, view_211)
        view_212: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_21, _shape_param_291);  bmm_21 = _shape_param_291 = None
        permute_117: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        clone_43: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_213: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_43, _shape_param_292);  clone_43 = _shape_param_292 = None
        convert_element_type_222: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg175_1, torch.bfloat16);  arg175_1 = None
        convert_element_type_223: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg176_1, torch.bfloat16);  arg176_1 = None
        view_214: "bf16[16384, 768]" = torch.ops.aten.view.default(view_213, _shape_param_293);  view_213 = _shape_param_293 = None
        permute_118: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_223, [1, 0]);  convert_element_type_223 = None
        addmm_63: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_222, view_214, permute_118);  convert_element_type_222 = None
        view_215: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_63, _shape_param_294);  addmm_63 = _shape_param_294 = None
        inductor_lookup_seed_32: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 32)
        inductor_random_32: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_295, inductor_lookup_seed_32, 'rand');  _shape_param_295 = inductor_lookup_seed_32 = None
        convert_element_type_224: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_32, torch.bfloat16);  inductor_random_32 = None
        gt_32: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_224, 0.1);  convert_element_type_224 = None
        mul_159: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_32, view_215);  view_215 = None
        mul_160: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_159, 1.1111111111111112);  mul_159 = None
        add_76: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_160, add_75);  mul_160 = add_75 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 512, 1]" = var_mean_21[0]
        getitem_43: "f32[32, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        add_77: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_32: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_76, getitem_43);  add_76 = getitem_43 = None
        mul_161: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
        mul_162: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_161, arg177_1);  arg177_1 = None
        add_78: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_162, arg178_1);  mul_162 = arg178_1 = None
        convert_element_type_225: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg179_1, torch.bfloat16);  arg179_1 = None
        convert_element_type_226: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg180_1, torch.bfloat16);  arg180_1 = None
        convert_element_type_227: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16)
        view_216: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_227, _shape_param_296);  convert_element_type_227 = _shape_param_296 = None
        permute_119: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_226, [1, 0]);  convert_element_type_226 = None
        addmm_64: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(convert_element_type_225, view_216, permute_119);  convert_element_type_225 = None
        view_217: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_64, _shape_param_297);  _shape_param_297 = None
        convert_element_type_228: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_163: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_228, 0.5)
        mul_164: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_228, 0.7071067811865476);  convert_element_type_228 = None
        erf_10: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_164);  mul_164 = None
        add_79: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_165: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_163, add_79);  mul_163 = add_79 = None
        convert_element_type_229: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_165, torch.bfloat16);  mul_165 = None
        convert_element_type_230: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg181_1, torch.bfloat16);  arg181_1 = None
        convert_element_type_231: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg182_1, torch.bfloat16);  arg182_1 = None
        view_218: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_229, _shape_param_298);  convert_element_type_229 = _shape_param_298 = None
        permute_120: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_231, [1, 0]);  convert_element_type_231 = None
        addmm_65: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_230, view_218, permute_120);  convert_element_type_230 = None
        view_219: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_65, _shape_param_299);  addmm_65 = _shape_param_299 = None
        inductor_lookup_seed_33: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 33)
        inductor_random_33: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_300, inductor_lookup_seed_33, 'rand');  _shape_param_300 = inductor_lookup_seed_33 = None
        convert_element_type_232: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_33, torch.bfloat16);  inductor_random_33 = None
        gt_33: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_232, 0.1);  convert_element_type_232 = None
        mul_166: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_33, view_219);  view_219 = None
        mul_167: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_166, 1.1111111111111112);  mul_166 = None
        add_80: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_167, add_78);  mul_167 = add_78 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 512, 1]" = var_mean_22[0]
        getitem_45: "f32[32, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        add_81: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_33: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_45);  add_80 = getitem_45 = None
        mul_168: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
        mul_169: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_168, arg183_1);  arg183_1 = None
        add_82: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_169, arg184_1);  mul_169 = arg184_1 = None
        convert_element_type_233: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg185_1, torch.bfloat16);  arg185_1 = None
        convert_element_type_234: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg186_1, torch.bfloat16);  arg186_1 = None
        convert_element_type_235: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_82, torch.bfloat16)
        view_220: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_235, _shape_param_301);  convert_element_type_235 = _shape_param_301 = None
        permute_121: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_234, [1, 0]);  convert_element_type_234 = None
        addmm_66: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_233, view_220, permute_121);  convert_element_type_233 = None
        view_221: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_66, _shape_param_302);  addmm_66 = _shape_param_302 = None
        view_222: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_221, _shape_param_303);  view_221 = _shape_param_303 = None
        permute_122: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
        convert_element_type_236: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg187_1, torch.bfloat16);  arg187_1 = None
        convert_element_type_237: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg188_1, torch.bfloat16);  arg188_1 = None
        permute_123: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_237, [1, 0]);  convert_element_type_237 = None
        addmm_67: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_236, view_220, permute_123);  convert_element_type_236 = None
        view_223: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_67, _shape_param_304);  addmm_67 = _shape_param_304 = None
        view_224: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_223, _shape_param_305);  view_223 = _shape_param_305 = None
        permute_124: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
        convert_element_type_238: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg189_1, torch.bfloat16);  arg189_1 = None
        convert_element_type_239: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg190_1, torch.bfloat16);  arg190_1 = None
        permute_125: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_239, [1, 0]);  convert_element_type_239 = None
        addmm_68: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_238, view_220, permute_125);  convert_element_type_238 = None
        view_225: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_68, _shape_param_306);  addmm_68 = _shape_param_306 = None
        view_226: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_225, _shape_param_307);  view_225 = _shape_param_307 = None
        permute_126: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        mul_170: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_122, 0.3535533905932738);  permute_122 = None
        permute_127: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_124, [0, 1, 3, 2]);  permute_124 = None
        mul_171: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_127, 0.3535533905932738);  permute_127 = None
        expand_46: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(mul_170, _shape_param_308);  mul_170 = _shape_param_308 = None
        clone_44: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_227: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_44, _shape_param_309);  clone_44 = _shape_param_309 = None
        expand_47: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(mul_171, _shape_param_310);  mul_171 = _shape_param_310 = None
        clone_45: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_228: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_45, _shape_param_311);  clone_45 = _shape_param_311 = None
        bmm_22: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_227, view_228)
        view_229: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_22, _shape_param_312);  bmm_22 = _shape_param_312 = None
        convert_element_type_240: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_229, torch.float32)
        amax_11: "f32[32, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_240, [-1], True)
        sub_34: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_240, amax_11);  convert_element_type_240 = amax_11 = None
        exp_11: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_241: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        eq_11: "b8[32, 12, 512, 512]" = torch.ops.aten.eq.Scalar(view_229, -inf);  view_229 = None
        logical_not_22: "b8[32, 12, 512, 512]" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[32, 12, 512, 1]" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[32, 12, 512, 1]" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        where_11: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(logical_not_23, full, convert_element_type_241);  logical_not_23 = full = convert_element_type_241 = None
        inductor_lookup_seed_34: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 34)
        inductor_random_34: "f32[32, 12, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_313, inductor_lookup_seed_34, 'rand');  _shape_param_313 = inductor_lookup_seed_34 = None
        convert_element_type_242: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_34, torch.bfloat16);  inductor_random_34 = None
        gt_34: "b8[32, 12, 512, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_242, 0.1);  convert_element_type_242 = None
        mul_172: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(gt_34, where_11)
        mul_173: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_172, 1.1111111111111112);  mul_172 = None
        expand_48: "bf16[32, 12, 512, 512]" = torch.ops.aten.expand.default(mul_173, _shape_param_314);  mul_173 = _shape_param_314 = None
        view_230: "bf16[384, 512, 512]" = torch.ops.aten.view.default(expand_48, _shape_param_315);  expand_48 = _shape_param_315 = None
        expand_49: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_126, _shape_param_316);  permute_126 = _shape_param_316 = None
        clone_46: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_231: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_46, _shape_param_317);  clone_46 = _shape_param_317 = None
        bmm_23: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_230, view_231)
        view_232: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_318);  bmm_23 = _shape_param_318 = None
        permute_128: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None
        clone_47: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_233: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_47, _shape_param_319);  clone_47 = _shape_param_319 = None
        convert_element_type_243: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg191_1, torch.bfloat16);  arg191_1 = None
        convert_element_type_244: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg192_1, torch.bfloat16);  arg192_1 = None
        view_234: "bf16[16384, 768]" = torch.ops.aten.view.default(view_233, _shape_param_320);  view_233 = _shape_param_320 = None
        permute_129: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_244, [1, 0]);  convert_element_type_244 = None
        addmm_69: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_243, view_234, permute_129);  convert_element_type_243 = None
        view_235: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_69, _shape_param_321);  addmm_69 = _shape_param_321 = None
        inductor_lookup_seed_35: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 35)
        inductor_random_35: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_322, inductor_lookup_seed_35, 'rand');  _shape_param_322 = inductor_lookup_seed_35 = None
        convert_element_type_245: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_35, torch.bfloat16);  inductor_random_35 = None
        gt_35: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_245, 0.1);  convert_element_type_245 = None
        mul_174: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_35, view_235);  view_235 = None
        mul_175: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_174, 1.1111111111111112);  mul_174 = None
        add_83: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_175, add_82);  mul_175 = add_82 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 512, 1]" = var_mean_23[0]
        getitem_47: "f32[32, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        add_84: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_35: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_47);  add_83 = getitem_47 = None
        mul_176: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
        mul_177: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_176, arg193_1);  arg193_1 = None
        add_85: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_177, arg194_1);  mul_177 = arg194_1 = None
        convert_element_type_246: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg195_1, torch.bfloat16);  arg195_1 = None
        convert_element_type_247: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg196_1, torch.bfloat16);  arg196_1 = None
        convert_element_type_248: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16)
        view_236: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_248, _shape_param_323);  convert_element_type_248 = _shape_param_323 = None
        permute_130: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_247, [1, 0]);  convert_element_type_247 = None
        addmm_70: "bf16[16384, 3072]" = torch.ops.aten.addmm.default(convert_element_type_246, view_236, permute_130);  convert_element_type_246 = None
        view_237: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(addmm_70, _shape_param_324);  _shape_param_324 = None
        convert_element_type_249: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_237, torch.float32);  view_237 = None
        mul_178: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_249, 0.5)
        mul_179: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_249, 0.7071067811865476);  convert_element_type_249 = None
        erf_11: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_179);  mul_179 = None
        add_86: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_180: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_178, add_86);  mul_178 = add_86 = None
        convert_element_type_250: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_180, torch.bfloat16);  mul_180 = None
        convert_element_type_251: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg197_1, torch.bfloat16);  arg197_1 = None
        convert_element_type_252: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg198_1, torch.bfloat16);  arg198_1 = None
        view_238: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_250, _shape_param_325);  convert_element_type_250 = _shape_param_325 = None
        permute_131: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_252, [1, 0]);  convert_element_type_252 = None
        addmm_71: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_251, view_238, permute_131);  convert_element_type_251 = None
        view_239: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_71, _shape_param_326);  addmm_71 = _shape_param_326 = None
        inductor_lookup_seed_36: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 36);  inductor_seeds = None
        inductor_random_36: "f32[32, 512, 768]" = torch.ops.prims.inductor_random.default(_shape_param_327, inductor_lookup_seed_36, 'rand');  _shape_param_327 = inductor_lookup_seed_36 = None
        convert_element_type_253: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_36, torch.bfloat16);  inductor_random_36 = None
        gt_36: "b8[32, 512, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_253, 0.1);  convert_element_type_253 = None
        mul_181: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(gt_36, view_239);  view_239 = None
        mul_182: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_181, 1.1111111111111112);  mul_181 = None
        add_87: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_182, add_85);  mul_182 = add_85 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 512, 1]" = var_mean_24[0]
        getitem_49: "f32[32, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        add_88: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_36: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_49);  add_87 = getitem_49 = None
        mul_183: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
        mul_184: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_183, arg199_1);  arg199_1 = None
        add_89: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_184, arg200_1);  mul_184 = arg200_1 = None
        convert_element_type_254: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg201_1, torch.bfloat16);  arg201_1 = None
        convert_element_type_255: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg202_1, torch.bfloat16);  arg202_1 = None
        convert_element_type_256: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16);  add_89 = None
        view_240: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_256, _shape_param_328);  convert_element_type_256 = _shape_param_328 = None
        permute_132: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_255, [1, 0]);  convert_element_type_255 = None
        addmm_72: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_254, view_240, permute_132);  convert_element_type_254 = None
        view_241: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_72, _shape_param_329);  _shape_param_329 = None
        convert_element_type_257: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_241, torch.float32);  view_241 = None
        mul_185: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_257, 0.5)
        mul_186: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_257, 0.7071067811865476);  convert_element_type_257 = None
        erf_12: "f32[32, 512, 768]" = torch.ops.aten.erf.default(mul_186);  mul_186 = None
        add_90: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_187: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_185, add_90);  mul_185 = add_90 = None
        convert_element_type_258: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_187, torch.bfloat16);  mul_187 = None
        convert_element_type_259: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_258, torch.float32);  convert_element_type_258 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_259, [2], correction = 0, keepdim = True)
        getitem_50: "f32[32, 512, 1]" = var_mean_25[0]
        getitem_51: "f32[32, 512, 1]" = var_mean_25[1];  var_mean_25 = None
        add_91: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        sub_37: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_259, getitem_51);  convert_element_type_259 = None
        mul_188: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = None
        mul_189: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_188, arg203_1);  mul_188 = arg203_1 = None
        add_92: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_189, arg204_1);  mul_189 = arg204_1 = None
        convert_element_type_260: "bf16[50265]" = torch.ops.prims.convert_element_type.default(arg205_1, torch.bfloat16);  arg205_1 = None
        convert_element_type_261: "bf16[50265, 768]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_262: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(add_92, torch.bfloat16);  add_92 = None
        view_242: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_262, _shape_param_330);  convert_element_type_262 = _shape_param_330 = None
        permute_133: "bf16[768, 50265]" = torch.ops.aten.permute.default(convert_element_type_261, [1, 0]);  convert_element_type_261 = None
        constant_pad_nd: "bf16[768, 50272]" = torch.ops.aten.constant_pad_nd.default(permute_133, _shape_param_331);  _shape_param_331 = None
        full_1: "bf16[7]" = torch.ops.aten.full.default(_shape_param_332, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_332 = None
        cat: "bf16[50272]" = torch.ops.aten.cat.default([convert_element_type_260, full_1]);  convert_element_type_260 = full_1 = None
        addmm_73: "bf16[16384, 50272]" = torch.ops.aten.addmm.default(cat, view_242, constant_pad_nd);  cat = constant_pad_nd = None
        slice_1: "bf16[16384, 50265]" = torch.ops.aten.slice.Tensor(addmm_73, 1, 0, -7);  addmm_73 = None
        view_243: "bf16[32, 512, 50265]" = torch.ops.aten.view.default(slice_1, _shape_param_333);  slice_1 = _shape_param_333 = None
        convert_element_type_263: "f32[32, 512, 50265]" = torch.ops.prims.convert_element_type.default(view_243, torch.float32)
        constant_pad_nd_1: "i64[32, 513]" = torch.ops.aten.constant_pad_nd.default(arg206_1, [0, 1], -100.0);  arg206_1 = None
        slice_2: "i64[32, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd_1, 1, 1, 9223372036854775807)
        clone_48: "i64[32, 512]" = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format);  slice_2 = None
        view_244: "f32[16384, 50265]" = torch.ops.aten.view.default(convert_element_type_263, _shape_param_334);  convert_element_type_263 = _shape_param_334 = None
        view_245: "i64[16384]" = torch.ops.aten.view.default(clone_48, [-1]);  clone_48 = None
        amax_12: "f32[16384, 1]" = torch.ops.aten.amax.default(view_244, [1], True)
        sub_38: "f32[16384, 50265]" = torch.ops.aten.sub.Tensor(view_244, amax_12);  view_244 = None
        exp_12: "f32[16384, 50265]" = torch.ops.aten.exp.default(sub_38)
        sum_13: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log: "f32[16384, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_39: "f32[16384, 50265]" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = None
        ne: "b8[16384]" = torch.ops.aten.ne.Scalar(view_245, -100)
        full_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "i64[16384]" = torch.ops.aten.where.self(ne, view_245, full_2);  view_245 = full_2 = None
        unsqueeze: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(where_12, 1);  where_12 = None
        gather_1: "f32[16384, 1]" = torch.ops.aten.gather.default(sub_39, 1, unsqueeze);  sub_39 = unsqueeze = None
        squeeze: "f32[16384]" = torch.ops.aten.squeeze.dim(gather_1, 1);  gather_1 = None
        neg: "f32[16384]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "f32[16384]" = torch.ops.aten.where.self(ne, neg, full_3);  neg = full_3 = None
        sum_14: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_264: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        sum_15: "f32[]" = torch.ops.aten.sum.default(where_13);  where_13 = None
        div_12: "f32[]" = torch.ops.aten.div.Tensor(sum_15, convert_element_type_264);  sum_15 = None
        permute_134: "bf16[50265, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        permute_135: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        div_13: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
        permute_136: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        permute_137: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        div_14: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        permute_138: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        permute_139: "bf16[384, 512, 512]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
        permute_140: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
        permute_141: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_227, [0, 2, 1]);  view_227 = None
        permute_142: "bf16[384, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None
        permute_143: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        permute_144: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        permute_145: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        div_15: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        permute_146: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        permute_147: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        div_16: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        permute_148: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        permute_149: "bf16[384, 512, 512]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        permute_150: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
        permute_151: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        permute_152: "bf16[384, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
        permute_153: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
        permute_154: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        permute_155: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        div_17: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        permute_156: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        permute_157: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        div_18: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        permute_158: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        permute_159: "bf16[384, 512, 512]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
        permute_160: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
        permute_161: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
        permute_162: "bf16[384, 512, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        permute_163: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        permute_164: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        permute_165: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        div_19: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        permute_166: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        permute_167: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        div_20: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        permute_168: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        permute_169: "bf16[384, 512, 512]" = torch.ops.aten.permute.default(view_170, [0, 2, 1]);  view_170 = None
        permute_170: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
        permute_171: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
        permute_172: "bf16[384, 512, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
        permute_173: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        permute_174: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        permute_175: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        div_21: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        permute_176: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        permute_177: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        div_22: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        permute_178: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        permute_179: "bf16[384, 512, 512]" = torch.ops.aten.permute.default(view_150, [0, 2, 1]);  view_150 = None
        permute_180: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_151, [0, 2, 1]);  view_151 = None
        permute_181: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
        permute_182: "bf16[384, 512, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1]);  view_148 = None
        permute_183: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        permute_184: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        permute_185: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        div_23: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        permute_186: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        permute_187: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        div_24: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        permute_188: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        permute_189: "bf16[384, 512, 512]" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
        permute_190: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None
        permute_191: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_127, [0, 2, 1]);  view_127 = None
        permute_192: "bf16[384, 512, 64]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None
        permute_193: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        permute_194: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        permute_195: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        div_25: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        permute_196: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        permute_197: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        div_26: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        permute_198: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        permute_199: "bf16[384, 512, 512]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
        permute_200: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
        permute_201: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
        permute_202: "bf16[384, 512, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1]);  view_108 = None
        permute_203: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        permute_204: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        permute_205: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        div_27: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        permute_206: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        permute_207: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        div_28: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        permute_208: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        permute_209: "bf16[384, 512, 512]" = torch.ops.aten.permute.default(view_90, [0, 2, 1]);  view_90 = None
        permute_210: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_211: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
        permute_212: "bf16[384, 512, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1]);  view_88 = None
        permute_213: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        permute_214: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        permute_215: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        div_29: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        permute_216: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        permute_217: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        div_30: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        permute_218: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        permute_219: "bf16[384, 512, 512]" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None
        permute_220: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None
        permute_221: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_67, [0, 2, 1]);  view_67 = None
        permute_222: "bf16[384, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
        permute_223: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        permute_224: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        permute_225: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        div_31: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        permute_226: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        permute_227: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        div_32: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        permute_228: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        permute_229: "bf16[384, 512, 512]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
        permute_230: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
        permute_231: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_47, [0, 2, 1]);  view_47 = None
        permute_232: "bf16[384, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1]);  view_48 = None
        permute_233: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        permute_234: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        permute_235: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        div_33: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        permute_236: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        permute_237: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        div_34: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        permute_238: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        permute_239: "bf16[384, 512, 512]" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
        permute_240: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_241: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
        permute_242: "bf16[384, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
        permute_243: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        permute_244: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        permute_245: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        div_35: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        permute_246: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_247: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        div_36: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        permute_248: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_249: "bf16[384, 512, 512]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_250: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        permute_251: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        permute_252: "bf16[384, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
        permute_253: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        permute_254: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_255: "bf16[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        div_37: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (add_1, expand_1, mul_1, gt, view, bmm, amax, sum_1, logical_not_1, gt_1, view_14, gt_2, mul_11, view_16, addmm_4, view_18, gt_3, mul_18, view_20, where_1, gt_4, view_34, gt_5, mul_26, view_36, addmm_10, view_38, gt_6, mul_33, view_40, where_2, gt_7, view_54, gt_8, mul_41, view_56, addmm_16, view_58, gt_9, mul_48, view_60, where_3, gt_10, view_74, gt_11, mul_56, view_76, addmm_22, view_78, gt_12, mul_63, view_80, where_4, gt_13, view_94, gt_14, mul_71, view_96, addmm_28, view_98, gt_15, mul_78, view_100, where_5, gt_16, view_114, gt_17, mul_86, view_116, addmm_34, view_118, gt_18, mul_93, view_120, where_6, gt_19, view_134, gt_20, mul_101, view_136, addmm_40, view_138, gt_21, mul_108, view_140, where_7, gt_22, view_154, gt_23, mul_116, view_156, addmm_46, view_158, gt_24, mul_123, view_160, where_8, gt_25, view_174, gt_26, mul_131, view_176, addmm_52, view_178, gt_27, mul_138, view_180, where_9, gt_28, view_194, gt_29, mul_146, view_196, addmm_58, view_198, gt_30, mul_153, view_200, where_10, gt_31, view_214, gt_32, mul_161, view_216, addmm_64, view_218, gt_33, mul_168, view_220, where_11, gt_34, view_234, gt_35, mul_176, view_236, addmm_70, view_238, gt_36, mul_183, view_240, addmm_72, getitem_51, rsqrt_25, view_242, view_243, constant_pad_nd_1, amax_12, log, convert_element_type_264, div_12, permute_134, permute_135, div_13, permute_136, permute_137, div_14, permute_138, permute_139, permute_140, permute_141, permute_142, permute_143, permute_144, permute_145, div_15, permute_146, permute_147, div_16, permute_148, permute_149, permute_150, permute_151, permute_152, permute_153, permute_154, permute_155, div_17, permute_156, permute_157, div_18, permute_158, permute_159, permute_160, permute_161, permute_162, permute_163, permute_164, permute_165, div_19, permute_166, permute_167, div_20, permute_168, permute_169, permute_170, permute_171, permute_172, permute_173, permute_174, permute_175, div_21, permute_176, permute_177, div_22, permute_178, permute_179, permute_180, permute_181, permute_182, permute_183, permute_184, permute_185, div_23, permute_186, permute_187, div_24, permute_188, permute_189, permute_190, permute_191, permute_192, permute_193, permute_194, permute_195, div_25, permute_196, permute_197, div_26, permute_198, permute_199, permute_200, permute_201, permute_202, permute_203, permute_204, permute_205, div_27, permute_206, permute_207, div_28, permute_208, permute_209, permute_210, permute_211, permute_212, permute_213, permute_214, permute_215, div_29, permute_216, permute_217, div_30, permute_218, permute_219, permute_220, permute_221, permute_222, permute_223, permute_224, permute_225, div_31, permute_226, permute_227, div_32, permute_228, permute_229, permute_230, permute_231, permute_232, permute_233, permute_234, permute_235, div_33, permute_236, permute_237, div_34, permute_238, permute_239, permute_240, permute_241, permute_242, permute_243, permute_244, permute_245, div_35, permute_246, permute_247, div_36, permute_248, permute_249, permute_250, permute_251, permute_252, permute_253, permute_254, permute_255, div_37)



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
