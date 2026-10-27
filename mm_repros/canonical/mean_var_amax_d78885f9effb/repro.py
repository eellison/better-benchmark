"""
Standalone repro captured via capture_hook.
Label: BERT_pytorch
Pattern hash: d78885f9effb
Shape hash: 789c2563
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
    def forward(self, arg0_1: "i64[16, 128]", arg1_1: "bf16[20005, 768]", arg2_1: "bf16[1, 512, 768]", arg3_1: "bf16[3, 768]", arg4_1: "i64[16, 128]", arg5_1: "bf16[768]", arg6_1: "bf16[768]", arg7_1: "bf16[768, 768]", arg8_1: "bf16[768]", arg9_1: "bf16[768, 768]", arg10_1: "bf16[768]", arg11_1: "bf16[768, 768]", arg12_1: "bf16[768]", arg13_1: "bf16[768, 768]", arg14_1: "bf16[768]", arg15_1: "bf16[768]", arg16_1: "bf16[768]", arg17_1: "bf16[3072, 768]", arg18_1: "bf16[3072]", arg19_1: "bf16[768, 3072]", arg20_1: "bf16[768]", arg21_1: "bf16[768]", arg22_1: "bf16[768]", arg23_1: "bf16[768, 768]", arg24_1: "bf16[768]", arg25_1: "bf16[768, 768]", arg26_1: "bf16[768]", arg27_1: "bf16[768, 768]", arg28_1: "bf16[768]", arg29_1: "bf16[768, 768]", arg30_1: "bf16[768]", arg31_1: "bf16[768]", arg32_1: "bf16[768]", arg33_1: "bf16[3072, 768]", arg34_1: "bf16[3072]", arg35_1: "bf16[768, 3072]", arg36_1: "bf16[768]", arg37_1: "bf16[768]", arg38_1: "bf16[768]", arg39_1: "bf16[768, 768]", arg40_1: "bf16[768]", arg41_1: "bf16[768, 768]", arg42_1: "bf16[768]", arg43_1: "bf16[768, 768]", arg44_1: "bf16[768]", arg45_1: "bf16[768, 768]", arg46_1: "bf16[768]", arg47_1: "bf16[768]", arg48_1: "bf16[768]", arg49_1: "bf16[3072, 768]", arg50_1: "bf16[3072]", arg51_1: "bf16[768, 3072]", arg52_1: "bf16[768]", arg53_1: "bf16[768]", arg54_1: "bf16[768]", arg55_1: "bf16[768, 768]", arg56_1: "bf16[768]", arg57_1: "bf16[768, 768]", arg58_1: "bf16[768]", arg59_1: "bf16[768, 768]", arg60_1: "bf16[768]", arg61_1: "bf16[768, 768]", arg62_1: "bf16[768]", arg63_1: "bf16[768]", arg64_1: "bf16[768]", arg65_1: "bf16[3072, 768]", arg66_1: "bf16[3072]", arg67_1: "bf16[768, 3072]", arg68_1: "bf16[768]", arg69_1: "bf16[768]", arg70_1: "bf16[768]", arg71_1: "bf16[768, 768]", arg72_1: "bf16[768]", arg73_1: "bf16[768, 768]", arg74_1: "bf16[768]", arg75_1: "bf16[768, 768]", arg76_1: "bf16[768]", arg77_1: "bf16[768, 768]", arg78_1: "bf16[768]", arg79_1: "bf16[768]", arg80_1: "bf16[768]", arg81_1: "bf16[3072, 768]", arg82_1: "bf16[3072]", arg83_1: "bf16[768, 3072]", arg84_1: "bf16[768]", arg85_1: "bf16[768]", arg86_1: "bf16[768]", arg87_1: "bf16[768, 768]", arg88_1: "bf16[768]", arg89_1: "bf16[768, 768]", arg90_1: "bf16[768]", arg91_1: "bf16[768, 768]", arg92_1: "bf16[768]", arg93_1: "bf16[768, 768]", arg94_1: "bf16[768]", arg95_1: "bf16[768]", arg96_1: "bf16[768]", arg97_1: "bf16[3072, 768]", arg98_1: "bf16[3072]", arg99_1: "bf16[768, 3072]", arg100_1: "bf16[768]", arg101_1: "bf16[768]", arg102_1: "bf16[768]", arg103_1: "bf16[768, 768]", arg104_1: "bf16[768]", arg105_1: "bf16[768, 768]", arg106_1: "bf16[768]", arg107_1: "bf16[768, 768]", arg108_1: "bf16[768]", arg109_1: "bf16[768, 768]", arg110_1: "bf16[768]", arg111_1: "bf16[768]", arg112_1: "bf16[768]", arg113_1: "bf16[3072, 768]", arg114_1: "bf16[3072]", arg115_1: "bf16[768, 3072]", arg116_1: "bf16[768]", arg117_1: "bf16[768]", arg118_1: "bf16[768]", arg119_1: "bf16[768, 768]", arg120_1: "bf16[768]", arg121_1: "bf16[768, 768]", arg122_1: "bf16[768]", arg123_1: "bf16[768, 768]", arg124_1: "bf16[768]", arg125_1: "bf16[768, 768]", arg126_1: "bf16[768]", arg127_1: "bf16[768]", arg128_1: "bf16[768]", arg129_1: "bf16[3072, 768]", arg130_1: "bf16[3072]", arg131_1: "bf16[768, 3072]", arg132_1: "bf16[768]", arg133_1: "bf16[768]", arg134_1: "bf16[768]", arg135_1: "bf16[768, 768]", arg136_1: "bf16[768]", arg137_1: "bf16[768, 768]", arg138_1: "bf16[768]", arg139_1: "bf16[768, 768]", arg140_1: "bf16[768]", arg141_1: "bf16[768, 768]", arg142_1: "bf16[768]", arg143_1: "bf16[768]", arg144_1: "bf16[768]", arg145_1: "bf16[3072, 768]", arg146_1: "bf16[3072]", arg147_1: "bf16[768, 3072]", arg148_1: "bf16[768]", arg149_1: "bf16[768]", arg150_1: "bf16[768]", arg151_1: "bf16[768, 768]", arg152_1: "bf16[768]", arg153_1: "bf16[768, 768]", arg154_1: "bf16[768]", arg155_1: "bf16[768, 768]", arg156_1: "bf16[768]", arg157_1: "bf16[768, 768]", arg158_1: "bf16[768]", arg159_1: "bf16[768]", arg160_1: "bf16[768]", arg161_1: "bf16[3072, 768]", arg162_1: "bf16[3072]", arg163_1: "bf16[768, 3072]", arg164_1: "bf16[768]", arg165_1: "bf16[768]", arg166_1: "bf16[768]", arg167_1: "bf16[768, 768]", arg168_1: "bf16[768]", arg169_1: "bf16[768, 768]", arg170_1: "bf16[768]", arg171_1: "bf16[768, 768]", arg172_1: "bf16[768]", arg173_1: "bf16[768, 768]", arg174_1: "bf16[768]", arg175_1: "bf16[768]", arg176_1: "bf16[768]", arg177_1: "bf16[3072, 768]", arg178_1: "bf16[3072]", arg179_1: "bf16[768, 3072]", arg180_1: "bf16[768]", arg181_1: "bf16[768]", arg182_1: "bf16[768]", arg183_1: "bf16[768, 768]", arg184_1: "bf16[768]", arg185_1: "bf16[768, 768]", arg186_1: "bf16[768]", arg187_1: "bf16[768, 768]", arg188_1: "bf16[768]", arg189_1: "bf16[768, 768]", arg190_1: "bf16[768]", arg191_1: "bf16[768]", arg192_1: "bf16[768]", arg193_1: "bf16[3072, 768]", arg194_1: "bf16[3072]", arg195_1: "bf16[768, 3072]", arg196_1: "bf16[768]", arg197_1: "bf16[2, 768]", arg198_1: "bf16[2]", arg199_1: "bf16[20005]", arg200_1: "bf16[20005, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316):
        # No stacktrace found for following nodes
        gt: "b8[16, 128]" = torch.ops.aten.gt.Scalar(arg0_1, 0)
        unsqueeze: "b8[16, 1, 128]" = torch.ops.aten.unsqueeze.default(gt, 1);  gt = None
        repeat: "b8[16, 128, 128]" = torch.ops.aten.repeat.default(unsqueeze, _shape_param_0);  unsqueeze = _shape_param_0 = None
        unsqueeze_1: "b8[16, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(repeat, 1);  repeat = None
        eq: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
        full: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        eq_1: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
        full_1: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        eq_2: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
        full_2: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        eq_3: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
        full_3: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        eq_4: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
        full_4: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        eq_5: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
        full_5: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        eq_6: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
        full_6: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        eq_7: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
        full_7: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        eq_8: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
        full_8: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        eq_9: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
        full_9: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        eq_10: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
        full_10: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        eq_11: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
        full_11: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        embedding: "bf16[16, 128, 768]" = torch.ops.aten.embedding.default(arg1_1, arg0_1, 0);  arg1_1 = arg0_1 = None
        slice_1: "bf16[1, 128, 768]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 0, 128);  arg2_1 = None
        add: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(embedding, slice_1);  embedding = slice_1 = None
        embedding_1: "bf16[16, 128, 768]" = torch.ops.aten.embedding.default(arg3_1, arg4_1, 0);  arg3_1 = arg4_1 = None
        add_1: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add, embedding_1);  add = embedding_1 = None
        mean: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_1, [-1], True)
        sub: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_1, mean);  mean = None
        mul: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg5_1, sub);  arg5_1 = sub = None
        convert_element_type: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32)
        var: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type, [-1], correction = 1.0, keepdim = True);  convert_element_type = None
        sqrt: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var);  var = None
        convert_element_type_1: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt, torch.bfloat16);  sqrt = None
        add_2: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-06);  convert_element_type_1 = None
        div: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul, add_2);  mul = add_2 = None
        add_3: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div, arg6_1);  div = arg6_1 = None
        view: "bf16[2048, 768]" = torch.ops.aten.view.default(add_3, _shape_param_1);  _shape_param_1 = None
        permute: "bf16[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg8_1, view, permute);  arg8_1 = view = permute = None
        view_1: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        view_2: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_1, _shape_param_3);  view_1 = _shape_param_3 = None
        permute_1: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        expand: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_1, _shape_param_4);  permute_1 = _shape_param_4 = None
        clone: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_3: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        view_4: "bf16[2048, 768]" = torch.ops.aten.view.default(add_3, _shape_param_6);  _shape_param_6 = None
        permute_2: "bf16[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_1: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg10_1, view_4, permute_2);  arg10_1 = view_4 = permute_2 = None
        view_5: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_1, _shape_param_7);  addmm_1 = _shape_param_7 = None
        view_6: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_5, _shape_param_8);  view_5 = _shape_param_8 = None
        permute_3: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_4: "bf16[16, 12, 64, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand_1: "bf16[16, 12, 64, 128]" = torch.ops.aten.expand.default(permute_4, _shape_param_9);  permute_4 = _shape_param_9 = None
        clone_1: "bf16[16, 12, 64, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_7: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_10);  clone_1 = _shape_param_10 = None
        bmm: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_3, view_7);  view_3 = view_7 = None
        view_8: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_11);  bmm = _shape_param_11 = None
        div_1: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_8, 8.0);  view_8 = None
        where: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_11, full_11, div_1);  eq_11 = full_11 = div_1 = None
        convert_element_type_2: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        amax: "f32[16, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
        sub_1: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        expand_2: "bf16[16, 12, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_3, _shape_param_12);  convert_element_type_3 = _shape_param_12 = None
        view_9: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_2, _shape_param_13);  expand_2 = _shape_param_13 = None
        view_10: "bf16[2048, 768]" = torch.ops.aten.view.default(add_3, _shape_param_14);  add_3 = _shape_param_14 = None
        permute_5: "bf16[768, 768]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_2: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg12_1, view_10, permute_5);  arg12_1 = view_10 = permute_5 = None
        view_11: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_2, _shape_param_15);  addmm_2 = _shape_param_15 = None
        view_12: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_11, _shape_param_16);  view_11 = _shape_param_16 = None
        permute_6: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        expand_3: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_6, _shape_param_17);  permute_6 = _shape_param_17 = None
        clone_2: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_13: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_18);  clone_2 = _shape_param_18 = None
        bmm_1: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_9, view_13);  view_9 = view_13 = None
        view_14: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_19);  bmm_1 = _shape_param_19 = None
        permute_7: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_3: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_15: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_3, _shape_param_20);  clone_3 = _shape_param_20 = None
        view_16: "bf16[2048, 768]" = torch.ops.aten.view.default(view_15, _shape_param_21);  view_15 = _shape_param_21 = None
        permute_8: "bf16[768, 768]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_3: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg14_1, view_16, permute_8);  arg14_1 = view_16 = permute_8 = None
        view_17: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_3, _shape_param_22);  addmm_3 = _shape_param_22 = None
        add_4: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_1, view_17);  add_1 = view_17 = None
        mean_1: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_4, [-1], True)
        sub_2: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_4, mean_1);  mean_1 = None
        mul_1: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg15_1, sub_2);  arg15_1 = sub_2 = None
        convert_element_type_4: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_4, torch.float32)
        var_1: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_4, [-1], correction = 1.0, keepdim = True);  convert_element_type_4 = None
        sqrt_1: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_1);  var_1 = None
        convert_element_type_5: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_1, torch.bfloat16);  sqrt_1 = None
        add_5: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-06);  convert_element_type_5 = None
        div_3: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_1, add_5);  mul_1 = add_5 = None
        add_6: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_3, arg16_1);  div_3 = arg16_1 = None
        view_18: "bf16[2048, 768]" = torch.ops.aten.view.default(add_6, _shape_param_23);  add_6 = _shape_param_23 = None
        permute_9: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_4: "bf16[2048, 3072]" = torch.ops.aten.addmm.default(arg18_1, view_18, permute_9);  arg18_1 = view_18 = permute_9 = None
        view_19: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(addmm_4, _shape_param_24);  addmm_4 = _shape_param_24 = None
        convert_element_type_6: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_2: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.5)
        mul_3: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476);  convert_element_type_6 = None
        erf: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add_7: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_4: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_2, add_7);  mul_2 = add_7 = None
        convert_element_type_7: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        view_20: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_25);  convert_element_type_7 = _shape_param_25 = None
        permute_10: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_5: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg20_1, view_20, permute_10);  arg20_1 = view_20 = permute_10 = None
        view_21: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_5, _shape_param_26);  addmm_5 = _shape_param_26 = None
        add_8: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_4, view_21);  add_4 = view_21 = None
        mean_2: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_8, [-1], True)
        sub_3: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_8, mean_2);  mean_2 = None
        mul_5: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg21_1, sub_3);  arg21_1 = sub_3 = None
        convert_element_type_8: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_8, torch.float32)
        var_2: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_8, [-1], correction = 1.0, keepdim = True);  convert_element_type_8 = None
        sqrt_2: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_2);  var_2 = None
        convert_element_type_9: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_2, torch.bfloat16);  sqrt_2 = None
        add_9: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-06);  convert_element_type_9 = None
        div_4: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_5, add_9);  mul_5 = add_9 = None
        add_10: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_4, arg22_1);  div_4 = arg22_1 = None
        view_22: "bf16[2048, 768]" = torch.ops.aten.view.default(add_10, _shape_param_27);  _shape_param_27 = None
        permute_11: "bf16[768, 768]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_6: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg24_1, view_22, permute_11);  arg24_1 = view_22 = permute_11 = None
        view_23: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_6, _shape_param_28);  addmm_6 = _shape_param_28 = None
        view_24: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_23, _shape_param_29);  view_23 = _shape_param_29 = None
        permute_12: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        expand_4: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_12, _shape_param_30);  permute_12 = _shape_param_30 = None
        clone_4: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_25: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_31);  clone_4 = _shape_param_31 = None
        view_26: "bf16[2048, 768]" = torch.ops.aten.view.default(add_10, _shape_param_32);  _shape_param_32 = None
        permute_13: "bf16[768, 768]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_7: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg26_1, view_26, permute_13);  arg26_1 = view_26 = permute_13 = None
        view_27: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_7, _shape_param_33);  addmm_7 = _shape_param_33 = None
        view_28: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_27, _shape_param_34);  view_27 = _shape_param_34 = None
        permute_14: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        permute_15: "bf16[16, 12, 64, 128]" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
        expand_5: "bf16[16, 12, 64, 128]" = torch.ops.aten.expand.default(permute_15, _shape_param_35);  permute_15 = _shape_param_35 = None
        clone_5: "bf16[16, 12, 64, 128]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_29: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_5, _shape_param_36);  clone_5 = _shape_param_36 = None
        bmm_2: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_25, view_29);  view_25 = view_29 = None
        view_30: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_37);  bmm_2 = _shape_param_37 = None
        div_5: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_30, 8.0);  view_30 = None
        where_1: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_10, full_10, div_5);  eq_10 = full_10 = div_5 = None
        convert_element_type_10: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        amax_1: "f32[16, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_10, [-1], True)
        sub_4: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_10, amax_1);  convert_element_type_10 = amax_1 = None
        exp_1: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_6: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_11: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        expand_6: "bf16[16, 12, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_11, _shape_param_38);  convert_element_type_11 = _shape_param_38 = None
        view_31: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_6, _shape_param_39);  expand_6 = _shape_param_39 = None
        view_32: "bf16[2048, 768]" = torch.ops.aten.view.default(add_10, _shape_param_40);  add_10 = _shape_param_40 = None
        permute_16: "bf16[768, 768]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_8: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg28_1, view_32, permute_16);  arg28_1 = view_32 = permute_16 = None
        view_33: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_8, _shape_param_41);  addmm_8 = _shape_param_41 = None
        view_34: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_33, _shape_param_42);  view_33 = _shape_param_42 = None
        permute_17: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        expand_7: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_17, _shape_param_43);  permute_17 = _shape_param_43 = None
        clone_6: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_35: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_6, _shape_param_44);  clone_6 = _shape_param_44 = None
        bmm_3: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_31, view_35);  view_31 = view_35 = None
        view_36: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_45);  bmm_3 = _shape_param_45 = None
        permute_18: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_7: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_37: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_7, _shape_param_46);  clone_7 = _shape_param_46 = None
        view_38: "bf16[2048, 768]" = torch.ops.aten.view.default(view_37, _shape_param_47);  view_37 = _shape_param_47 = None
        permute_19: "bf16[768, 768]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_9: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg30_1, view_38, permute_19);  arg30_1 = view_38 = permute_19 = None
        view_39: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_9, _shape_param_48);  addmm_9 = _shape_param_48 = None
        add_11: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_8, view_39);  add_8 = view_39 = None
        mean_3: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_11, [-1], True)
        sub_5: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_11, mean_3);  mean_3 = None
        mul_6: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg31_1, sub_5);  arg31_1 = sub_5 = None
        convert_element_type_12: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_11, torch.float32)
        var_3: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_12, [-1], correction = 1.0, keepdim = True);  convert_element_type_12 = None
        sqrt_3: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_3);  var_3 = None
        convert_element_type_13: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_3, torch.bfloat16);  sqrt_3 = None
        add_12: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-06);  convert_element_type_13 = None
        div_7: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_6, add_12);  mul_6 = add_12 = None
        add_13: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_7, arg32_1);  div_7 = arg32_1 = None
        view_40: "bf16[2048, 768]" = torch.ops.aten.view.default(add_13, _shape_param_49);  add_13 = _shape_param_49 = None
        permute_20: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_10: "bf16[2048, 3072]" = torch.ops.aten.addmm.default(arg34_1, view_40, permute_20);  arg34_1 = view_40 = permute_20 = None
        view_41: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(addmm_10, _shape_param_50);  addmm_10 = _shape_param_50 = None
        convert_element_type_14: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_7: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.5)
        mul_8: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.7071067811865476);  convert_element_type_14 = None
        erf_1: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_14: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_9: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_7, add_14);  mul_7 = add_14 = None
        convert_element_type_15: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        view_42: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_15, _shape_param_51);  convert_element_type_15 = _shape_param_51 = None
        permute_21: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_11: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg36_1, view_42, permute_21);  arg36_1 = view_42 = permute_21 = None
        view_43: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_11, _shape_param_52);  addmm_11 = _shape_param_52 = None
        add_15: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_11, view_43);  add_11 = view_43 = None
        mean_4: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_15, [-1], True)
        sub_6: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_15, mean_4);  mean_4 = None
        mul_10: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg37_1, sub_6);  arg37_1 = sub_6 = None
        convert_element_type_16: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_15, torch.float32)
        var_4: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_16, [-1], correction = 1.0, keepdim = True);  convert_element_type_16 = None
        sqrt_4: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_4);  var_4 = None
        convert_element_type_17: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_4, torch.bfloat16);  sqrt_4 = None
        add_16: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-06);  convert_element_type_17 = None
        div_8: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_10, add_16);  mul_10 = add_16 = None
        add_17: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_8, arg38_1);  div_8 = arg38_1 = None
        view_44: "bf16[2048, 768]" = torch.ops.aten.view.default(add_17, _shape_param_53);  _shape_param_53 = None
        permute_22: "bf16[768, 768]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_12: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg40_1, view_44, permute_22);  arg40_1 = view_44 = permute_22 = None
        view_45: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_12, _shape_param_54);  addmm_12 = _shape_param_54 = None
        view_46: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_45, _shape_param_55);  view_45 = _shape_param_55 = None
        permute_23: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        expand_8: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_23, _shape_param_56);  permute_23 = _shape_param_56 = None
        clone_8: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_47: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_57);  clone_8 = _shape_param_57 = None
        view_48: "bf16[2048, 768]" = torch.ops.aten.view.default(add_17, _shape_param_58);  _shape_param_58 = None
        permute_24: "bf16[768, 768]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_13: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg42_1, view_48, permute_24);  arg42_1 = view_48 = permute_24 = None
        view_49: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_13, _shape_param_59);  addmm_13 = _shape_param_59 = None
        view_50: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_49, _shape_param_60);  view_49 = _shape_param_60 = None
        permute_25: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        permute_26: "bf16[16, 12, 64, 128]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
        expand_9: "bf16[16, 12, 64, 128]" = torch.ops.aten.expand.default(permute_26, _shape_param_61);  permute_26 = _shape_param_61 = None
        clone_9: "bf16[16, 12, 64, 128]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_51: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_9, _shape_param_62);  clone_9 = _shape_param_62 = None
        bmm_4: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_47, view_51);  view_47 = view_51 = None
        view_52: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_4, _shape_param_63);  bmm_4 = _shape_param_63 = None
        div_9: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_52, 8.0);  view_52 = None
        where_2: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_9, full_9, div_9);  eq_9 = full_9 = div_9 = None
        convert_element_type_18: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        amax_2: "f32[16, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_18, [-1], True)
        sub_7: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_18, amax_2);  convert_element_type_18 = amax_2 = None
        exp_2: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_10: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_19: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        expand_10: "bf16[16, 12, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_19, _shape_param_64);  convert_element_type_19 = _shape_param_64 = None
        view_53: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_10, _shape_param_65);  expand_10 = _shape_param_65 = None
        view_54: "bf16[2048, 768]" = torch.ops.aten.view.default(add_17, _shape_param_66);  add_17 = _shape_param_66 = None
        permute_27: "bf16[768, 768]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_14: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg44_1, view_54, permute_27);  arg44_1 = view_54 = permute_27 = None
        view_55: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_14, _shape_param_67);  addmm_14 = _shape_param_67 = None
        view_56: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_55, _shape_param_68);  view_55 = _shape_param_68 = None
        permute_28: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        expand_11: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_28, _shape_param_69);  permute_28 = _shape_param_69 = None
        clone_10: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_57: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_10, _shape_param_70);  clone_10 = _shape_param_70 = None
        bmm_5: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_53, view_57);  view_53 = view_57 = None
        view_58: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_5, _shape_param_71);  bmm_5 = _shape_param_71 = None
        permute_29: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_11: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_59: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_11, _shape_param_72);  clone_11 = _shape_param_72 = None
        view_60: "bf16[2048, 768]" = torch.ops.aten.view.default(view_59, _shape_param_73);  view_59 = _shape_param_73 = None
        permute_30: "bf16[768, 768]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_15: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg46_1, view_60, permute_30);  arg46_1 = view_60 = permute_30 = None
        view_61: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_15, _shape_param_74);  addmm_15 = _shape_param_74 = None
        add_18: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_15, view_61);  add_15 = view_61 = None
        mean_5: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_18, [-1], True)
        sub_8: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_18, mean_5);  mean_5 = None
        mul_11: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg47_1, sub_8);  arg47_1 = sub_8 = None
        convert_element_type_20: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_18, torch.float32)
        var_5: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_20, [-1], correction = 1.0, keepdim = True);  convert_element_type_20 = None
        sqrt_5: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_5);  var_5 = None
        convert_element_type_21: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_5, torch.bfloat16);  sqrt_5 = None
        add_19: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-06);  convert_element_type_21 = None
        div_11: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_11, add_19);  mul_11 = add_19 = None
        add_20: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_11, arg48_1);  div_11 = arg48_1 = None
        view_62: "bf16[2048, 768]" = torch.ops.aten.view.default(add_20, _shape_param_75);  add_20 = _shape_param_75 = None
        permute_31: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_16: "bf16[2048, 3072]" = torch.ops.aten.addmm.default(arg50_1, view_62, permute_31);  arg50_1 = view_62 = permute_31 = None
        view_63: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(addmm_16, _shape_param_76);  addmm_16 = _shape_param_76 = None
        convert_element_type_22: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        mul_12: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.5)
        mul_13: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.7071067811865476);  convert_element_type_22 = None
        erf_2: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_21: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_14: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_21);  mul_12 = add_21 = None
        convert_element_type_23: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        view_64: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_23, _shape_param_77);  convert_element_type_23 = _shape_param_77 = None
        permute_32: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_17: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg52_1, view_64, permute_32);  arg52_1 = view_64 = permute_32 = None
        view_65: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_17, _shape_param_78);  addmm_17 = _shape_param_78 = None
        add_22: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_18, view_65);  add_18 = view_65 = None
        mean_6: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_22, [-1], True)
        sub_9: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_22, mean_6);  mean_6 = None
        mul_15: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg53_1, sub_9);  arg53_1 = sub_9 = None
        convert_element_type_24: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_22, torch.float32)
        var_6: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_24, [-1], correction = 1.0, keepdim = True);  convert_element_type_24 = None
        sqrt_6: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_6);  var_6 = None
        convert_element_type_25: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_6, torch.bfloat16);  sqrt_6 = None
        add_23: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-06);  convert_element_type_25 = None
        div_12: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_15, add_23);  mul_15 = add_23 = None
        add_24: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_12, arg54_1);  div_12 = arg54_1 = None
        view_66: "bf16[2048, 768]" = torch.ops.aten.view.default(add_24, _shape_param_79);  _shape_param_79 = None
        permute_33: "bf16[768, 768]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_18: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg56_1, view_66, permute_33);  arg56_1 = view_66 = permute_33 = None
        view_67: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_18, _shape_param_80);  addmm_18 = _shape_param_80 = None
        view_68: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_67, _shape_param_81);  view_67 = _shape_param_81 = None
        permute_34: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        expand_12: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_34, _shape_param_82);  permute_34 = _shape_param_82 = None
        clone_12: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_69: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_83);  clone_12 = _shape_param_83 = None
        view_70: "bf16[2048, 768]" = torch.ops.aten.view.default(add_24, _shape_param_84);  _shape_param_84 = None
        permute_35: "bf16[768, 768]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_19: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg58_1, view_70, permute_35);  arg58_1 = view_70 = permute_35 = None
        view_71: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_19, _shape_param_85);  addmm_19 = _shape_param_85 = None
        view_72: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_71, _shape_param_86);  view_71 = _shape_param_86 = None
        permute_36: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        permute_37: "bf16[16, 12, 64, 128]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
        expand_13: "bf16[16, 12, 64, 128]" = torch.ops.aten.expand.default(permute_37, _shape_param_87);  permute_37 = _shape_param_87 = None
        clone_13: "bf16[16, 12, 64, 128]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_73: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_13, _shape_param_88);  clone_13 = _shape_param_88 = None
        bmm_6: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_69, view_73);  view_69 = view_73 = None
        view_74: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_89);  bmm_6 = _shape_param_89 = None
        div_13: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_74, 8.0);  view_74 = None
        where_3: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_8, full_8, div_13);  eq_8 = full_8 = div_13 = None
        convert_element_type_26: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        amax_3: "f32[16, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_26, [-1], True)
        sub_10: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_26, amax_3);  convert_element_type_26 = amax_3 = None
        exp_3: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_14: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_27: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None
        expand_14: "bf16[16, 12, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_27, _shape_param_90);  convert_element_type_27 = _shape_param_90 = None
        view_75: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_14, _shape_param_91);  expand_14 = _shape_param_91 = None
        view_76: "bf16[2048, 768]" = torch.ops.aten.view.default(add_24, _shape_param_92);  add_24 = _shape_param_92 = None
        permute_38: "bf16[768, 768]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_20: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg60_1, view_76, permute_38);  arg60_1 = view_76 = permute_38 = None
        view_77: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_20, _shape_param_93);  addmm_20 = _shape_param_93 = None
        view_78: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_77, _shape_param_94);  view_77 = _shape_param_94 = None
        permute_39: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
        expand_15: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_39, _shape_param_95);  permute_39 = _shape_param_95 = None
        clone_14: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_79: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_14, _shape_param_96);  clone_14 = _shape_param_96 = None
        bmm_7: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_75, view_79);  view_75 = view_79 = None
        view_80: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_97);  bmm_7 = _shape_param_97 = None
        permute_40: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        clone_15: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_81: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_15, _shape_param_98);  clone_15 = _shape_param_98 = None
        view_82: "bf16[2048, 768]" = torch.ops.aten.view.default(view_81, _shape_param_99);  view_81 = _shape_param_99 = None
        permute_41: "bf16[768, 768]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_21: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg62_1, view_82, permute_41);  arg62_1 = view_82 = permute_41 = None
        view_83: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_21, _shape_param_100);  addmm_21 = _shape_param_100 = None
        add_25: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_22, view_83);  add_22 = view_83 = None
        mean_7: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_25, [-1], True)
        sub_11: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_25, mean_7);  mean_7 = None
        mul_16: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg63_1, sub_11);  arg63_1 = sub_11 = None
        convert_element_type_28: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_25, torch.float32)
        var_7: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_28, [-1], correction = 1.0, keepdim = True);  convert_element_type_28 = None
        sqrt_7: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_7);  var_7 = None
        convert_element_type_29: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_7, torch.bfloat16);  sqrt_7 = None
        add_26: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-06);  convert_element_type_29 = None
        div_15: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_16, add_26);  mul_16 = add_26 = None
        add_27: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_15, arg64_1);  div_15 = arg64_1 = None
        view_84: "bf16[2048, 768]" = torch.ops.aten.view.default(add_27, _shape_param_101);  add_27 = _shape_param_101 = None
        permute_42: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_22: "bf16[2048, 3072]" = torch.ops.aten.addmm.default(arg66_1, view_84, permute_42);  arg66_1 = view_84 = permute_42 = None
        view_85: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(addmm_22, _shape_param_102);  addmm_22 = _shape_param_102 = None
        convert_element_type_30: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_17: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 0.5)
        mul_18: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 0.7071067811865476);  convert_element_type_30 = None
        erf_3: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
        add_28: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_19: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_17, add_28);  mul_17 = add_28 = None
        convert_element_type_31: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16);  mul_19 = None
        view_86: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_31, _shape_param_103);  convert_element_type_31 = _shape_param_103 = None
        permute_43: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_23: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg68_1, view_86, permute_43);  arg68_1 = view_86 = permute_43 = None
        view_87: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_23, _shape_param_104);  addmm_23 = _shape_param_104 = None
        add_29: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_25, view_87);  add_25 = view_87 = None
        mean_8: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_29, [-1], True)
        sub_12: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_29, mean_8);  mean_8 = None
        mul_20: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg69_1, sub_12);  arg69_1 = sub_12 = None
        convert_element_type_32: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_29, torch.float32)
        var_8: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_32, [-1], correction = 1.0, keepdim = True);  convert_element_type_32 = None
        sqrt_8: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_8);  var_8 = None
        convert_element_type_33: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_8, torch.bfloat16);  sqrt_8 = None
        add_30: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-06);  convert_element_type_33 = None
        div_16: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_20, add_30);  mul_20 = add_30 = None
        add_31: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_16, arg70_1);  div_16 = arg70_1 = None
        view_88: "bf16[2048, 768]" = torch.ops.aten.view.default(add_31, _shape_param_105);  _shape_param_105 = None
        permute_44: "bf16[768, 768]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_24: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg72_1, view_88, permute_44);  arg72_1 = view_88 = permute_44 = None
        view_89: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_24, _shape_param_106);  addmm_24 = _shape_param_106 = None
        view_90: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_89, _shape_param_107);  view_89 = _shape_param_107 = None
        permute_45: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        expand_16: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_45, _shape_param_108);  permute_45 = _shape_param_108 = None
        clone_16: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_91: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_109);  clone_16 = _shape_param_109 = None
        view_92: "bf16[2048, 768]" = torch.ops.aten.view.default(add_31, _shape_param_110);  _shape_param_110 = None
        permute_46: "bf16[768, 768]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_25: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg74_1, view_92, permute_46);  arg74_1 = view_92 = permute_46 = None
        view_93: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_25, _shape_param_111);  addmm_25 = _shape_param_111 = None
        view_94: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_93, _shape_param_112);  view_93 = _shape_param_112 = None
        permute_47: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        permute_48: "bf16[16, 12, 64, 128]" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
        expand_17: "bf16[16, 12, 64, 128]" = torch.ops.aten.expand.default(permute_48, _shape_param_113);  permute_48 = _shape_param_113 = None
        clone_17: "bf16[16, 12, 64, 128]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_95: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_17, _shape_param_114);  clone_17 = _shape_param_114 = None
        bmm_8: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_91, view_95);  view_91 = view_95 = None
        view_96: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_8, _shape_param_115);  bmm_8 = _shape_param_115 = None
        div_17: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_96, 8.0);  view_96 = None
        where_4: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_7, full_7, div_17);  eq_7 = full_7 = div_17 = None
        convert_element_type_34: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where_4, torch.float32);  where_4 = None
        amax_4: "f32[16, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_34, [-1], True)
        sub_13: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_34, amax_4);  convert_element_type_34 = amax_4 = None
        exp_4: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_18: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_35: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None
        expand_18: "bf16[16, 12, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_35, _shape_param_116);  convert_element_type_35 = _shape_param_116 = None
        view_97: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_18, _shape_param_117);  expand_18 = _shape_param_117 = None
        view_98: "bf16[2048, 768]" = torch.ops.aten.view.default(add_31, _shape_param_118);  add_31 = _shape_param_118 = None
        permute_49: "bf16[768, 768]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_26: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg76_1, view_98, permute_49);  arg76_1 = view_98 = permute_49 = None
        view_99: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_26, _shape_param_119);  addmm_26 = _shape_param_119 = None
        view_100: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_99, _shape_param_120);  view_99 = _shape_param_120 = None
        permute_50: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        expand_19: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_50, _shape_param_121);  permute_50 = _shape_param_121 = None
        clone_18: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_101: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_18, _shape_param_122);  clone_18 = _shape_param_122 = None
        bmm_9: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_97, view_101);  view_97 = view_101 = None
        view_102: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_9, _shape_param_123);  bmm_9 = _shape_param_123 = None
        permute_51: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_19: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_103: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_19, _shape_param_124);  clone_19 = _shape_param_124 = None
        view_104: "bf16[2048, 768]" = torch.ops.aten.view.default(view_103, _shape_param_125);  view_103 = _shape_param_125 = None
        permute_52: "bf16[768, 768]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_27: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg78_1, view_104, permute_52);  arg78_1 = view_104 = permute_52 = None
        view_105: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_27, _shape_param_126);  addmm_27 = _shape_param_126 = None
        add_32: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_29, view_105);  add_29 = view_105 = None
        mean_9: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_32, [-1], True)
        sub_14: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_32, mean_9);  mean_9 = None
        mul_21: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg79_1, sub_14);  arg79_1 = sub_14 = None
        convert_element_type_36: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_32, torch.float32)
        var_9: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_36, [-1], correction = 1.0, keepdim = True);  convert_element_type_36 = None
        sqrt_9: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_9);  var_9 = None
        convert_element_type_37: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_9, torch.bfloat16);  sqrt_9 = None
        add_33: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-06);  convert_element_type_37 = None
        div_19: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_21, add_33);  mul_21 = add_33 = None
        add_34: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_19, arg80_1);  div_19 = arg80_1 = None
        view_106: "bf16[2048, 768]" = torch.ops.aten.view.default(add_34, _shape_param_127);  add_34 = _shape_param_127 = None
        permute_53: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_28: "bf16[2048, 3072]" = torch.ops.aten.addmm.default(arg82_1, view_106, permute_53);  arg82_1 = view_106 = permute_53 = None
        view_107: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(addmm_28, _shape_param_128);  addmm_28 = _shape_param_128 = None
        convert_element_type_38: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        mul_22: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.5)
        mul_23: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.7071067811865476);  convert_element_type_38 = None
        erf_4: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
        add_35: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_24: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_22, add_35);  mul_22 = add_35 = None
        convert_element_type_39: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_24, torch.bfloat16);  mul_24 = None
        view_108: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_39, _shape_param_129);  convert_element_type_39 = _shape_param_129 = None
        permute_54: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_29: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg84_1, view_108, permute_54);  arg84_1 = view_108 = permute_54 = None
        view_109: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_29, _shape_param_130);  addmm_29 = _shape_param_130 = None
        add_36: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_32, view_109);  add_32 = view_109 = None
        mean_10: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_36, [-1], True)
        sub_15: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_36, mean_10);  mean_10 = None
        mul_25: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg85_1, sub_15);  arg85_1 = sub_15 = None
        convert_element_type_40: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_36, torch.float32)
        var_10: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_40, [-1], correction = 1.0, keepdim = True);  convert_element_type_40 = None
        sqrt_10: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_10);  var_10 = None
        convert_element_type_41: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_10, torch.bfloat16);  sqrt_10 = None
        add_37: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-06);  convert_element_type_41 = None
        div_20: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_25, add_37);  mul_25 = add_37 = None
        add_38: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_20, arg86_1);  div_20 = arg86_1 = None
        view_110: "bf16[2048, 768]" = torch.ops.aten.view.default(add_38, _shape_param_131);  _shape_param_131 = None
        permute_55: "bf16[768, 768]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_30: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg88_1, view_110, permute_55);  arg88_1 = view_110 = permute_55 = None
        view_111: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_30, _shape_param_132);  addmm_30 = _shape_param_132 = None
        view_112: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_111, _shape_param_133);  view_111 = _shape_param_133 = None
        permute_56: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        expand_20: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_56, _shape_param_134);  permute_56 = _shape_param_134 = None
        clone_20: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_113: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_135);  clone_20 = _shape_param_135 = None
        view_114: "bf16[2048, 768]" = torch.ops.aten.view.default(add_38, _shape_param_136);  _shape_param_136 = None
        permute_57: "bf16[768, 768]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_31: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg90_1, view_114, permute_57);  arg90_1 = view_114 = permute_57 = None
        view_115: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_31, _shape_param_137);  addmm_31 = _shape_param_137 = None
        view_116: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_115, _shape_param_138);  view_115 = _shape_param_138 = None
        permute_58: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        permute_59: "bf16[16, 12, 64, 128]" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
        expand_21: "bf16[16, 12, 64, 128]" = torch.ops.aten.expand.default(permute_59, _shape_param_139);  permute_59 = _shape_param_139 = None
        clone_21: "bf16[16, 12, 64, 128]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_117: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_21, _shape_param_140);  clone_21 = _shape_param_140 = None
        bmm_10: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_113, view_117);  view_113 = view_117 = None
        view_118: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_141);  bmm_10 = _shape_param_141 = None
        div_21: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_118, 8.0);  view_118 = None
        where_5: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_6, full_6, div_21);  eq_6 = full_6 = div_21 = None
        convert_element_type_42: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where_5, torch.float32);  where_5 = None
        amax_5: "f32[16, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_42, [-1], True)
        sub_16: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_42, amax_5);  convert_element_type_42 = amax_5 = None
        exp_5: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_22: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_43: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None
        expand_22: "bf16[16, 12, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_43, _shape_param_142);  convert_element_type_43 = _shape_param_142 = None
        view_119: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_22, _shape_param_143);  expand_22 = _shape_param_143 = None
        view_120: "bf16[2048, 768]" = torch.ops.aten.view.default(add_38, _shape_param_144);  add_38 = _shape_param_144 = None
        permute_60: "bf16[768, 768]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_32: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg92_1, view_120, permute_60);  arg92_1 = view_120 = permute_60 = None
        view_121: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_32, _shape_param_145);  addmm_32 = _shape_param_145 = None
        view_122: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_121, _shape_param_146);  view_121 = _shape_param_146 = None
        permute_61: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        expand_23: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_61, _shape_param_147);  permute_61 = _shape_param_147 = None
        clone_22: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_123: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_22, _shape_param_148);  clone_22 = _shape_param_148 = None
        bmm_11: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_119, view_123);  view_119 = view_123 = None
        view_124: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_149);  bmm_11 = _shape_param_149 = None
        permute_62: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        clone_23: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_125: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_23, _shape_param_150);  clone_23 = _shape_param_150 = None
        view_126: "bf16[2048, 768]" = torch.ops.aten.view.default(view_125, _shape_param_151);  view_125 = _shape_param_151 = None
        permute_63: "bf16[768, 768]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_33: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg94_1, view_126, permute_63);  arg94_1 = view_126 = permute_63 = None
        view_127: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_33, _shape_param_152);  addmm_33 = _shape_param_152 = None
        add_39: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_36, view_127);  add_36 = view_127 = None
        mean_11: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_39, [-1], True)
        sub_17: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_39, mean_11);  mean_11 = None
        mul_26: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg95_1, sub_17);  arg95_1 = sub_17 = None
        convert_element_type_44: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_39, torch.float32)
        var_11: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_44, [-1], correction = 1.0, keepdim = True);  convert_element_type_44 = None
        sqrt_11: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_11);  var_11 = None
        convert_element_type_45: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_11, torch.bfloat16);  sqrt_11 = None
        add_40: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-06);  convert_element_type_45 = None
        div_23: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_26, add_40);  mul_26 = add_40 = None
        add_41: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_23, arg96_1);  div_23 = arg96_1 = None
        view_128: "bf16[2048, 768]" = torch.ops.aten.view.default(add_41, _shape_param_153);  add_41 = _shape_param_153 = None
        permute_64: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_34: "bf16[2048, 3072]" = torch.ops.aten.addmm.default(arg98_1, view_128, permute_64);  arg98_1 = view_128 = permute_64 = None
        view_129: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(addmm_34, _shape_param_154);  addmm_34 = _shape_param_154 = None
        convert_element_type_46: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_27: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 0.5)
        mul_28: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 0.7071067811865476);  convert_element_type_46 = None
        erf_5: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_42: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_29: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_27, add_42);  mul_27 = add_42 = None
        convert_element_type_47: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        view_130: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_47, _shape_param_155);  convert_element_type_47 = _shape_param_155 = None
        permute_65: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_35: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg100_1, view_130, permute_65);  arg100_1 = view_130 = permute_65 = None
        view_131: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_35, _shape_param_156);  addmm_35 = _shape_param_156 = None
        add_43: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_39, view_131);  add_39 = view_131 = None
        mean_12: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_43, [-1], True)
        sub_18: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_43, mean_12);  mean_12 = None
        mul_30: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg101_1, sub_18);  arg101_1 = sub_18 = None
        convert_element_type_48: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_43, torch.float32)
        var_12: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_48, [-1], correction = 1.0, keepdim = True);  convert_element_type_48 = None
        sqrt_12: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_12);  var_12 = None
        convert_element_type_49: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_12, torch.bfloat16);  sqrt_12 = None
        add_44: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-06);  convert_element_type_49 = None
        div_24: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_30, add_44);  mul_30 = add_44 = None
        add_45: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_24, arg102_1);  div_24 = arg102_1 = None
        view_132: "bf16[2048, 768]" = torch.ops.aten.view.default(add_45, _shape_param_157);  _shape_param_157 = None
        permute_66: "bf16[768, 768]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_36: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg104_1, view_132, permute_66);  arg104_1 = view_132 = permute_66 = None
        view_133: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_36, _shape_param_158);  addmm_36 = _shape_param_158 = None
        view_134: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_133, _shape_param_159);  view_133 = _shape_param_159 = None
        permute_67: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        expand_24: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_67, _shape_param_160);  permute_67 = _shape_param_160 = None
        clone_24: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_135: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_161);  clone_24 = _shape_param_161 = None
        view_136: "bf16[2048, 768]" = torch.ops.aten.view.default(add_45, _shape_param_162);  _shape_param_162 = None
        permute_68: "bf16[768, 768]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_37: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg106_1, view_136, permute_68);  arg106_1 = view_136 = permute_68 = None
        view_137: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_37, _shape_param_163);  addmm_37 = _shape_param_163 = None
        view_138: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_137, _shape_param_164);  view_137 = _shape_param_164 = None
        permute_69: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        permute_70: "bf16[16, 12, 64, 128]" = torch.ops.aten.permute.default(permute_69, [0, 1, 3, 2]);  permute_69 = None
        expand_25: "bf16[16, 12, 64, 128]" = torch.ops.aten.expand.default(permute_70, _shape_param_165);  permute_70 = _shape_param_165 = None
        clone_25: "bf16[16, 12, 64, 128]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_139: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_25, _shape_param_166);  clone_25 = _shape_param_166 = None
        bmm_12: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_135, view_139);  view_135 = view_139 = None
        view_140: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_12, _shape_param_167);  bmm_12 = _shape_param_167 = None
        div_25: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_140, 8.0);  view_140 = None
        where_6: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_5, full_5, div_25);  eq_5 = full_5 = div_25 = None
        convert_element_type_50: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where_6, torch.float32);  where_6 = None
        amax_6: "f32[16, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_50, [-1], True)
        sub_19: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_50, amax_6);  convert_element_type_50 = amax_6 = None
        exp_6: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_26: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_51: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_26, torch.bfloat16);  div_26 = None
        expand_26: "bf16[16, 12, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_51, _shape_param_168);  convert_element_type_51 = _shape_param_168 = None
        view_141: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_26, _shape_param_169);  expand_26 = _shape_param_169 = None
        view_142: "bf16[2048, 768]" = torch.ops.aten.view.default(add_45, _shape_param_170);  add_45 = _shape_param_170 = None
        permute_71: "bf16[768, 768]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_38: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg108_1, view_142, permute_71);  arg108_1 = view_142 = permute_71 = None
        view_143: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_38, _shape_param_171);  addmm_38 = _shape_param_171 = None
        view_144: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_143, _shape_param_172);  view_143 = _shape_param_172 = None
        permute_72: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        expand_27: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_72, _shape_param_173);  permute_72 = _shape_param_173 = None
        clone_26: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_145: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_26, _shape_param_174);  clone_26 = _shape_param_174 = None
        bmm_13: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_141, view_145);  view_141 = view_145 = None
        view_146: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_13, _shape_param_175);  bmm_13 = _shape_param_175 = None
        permute_73: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_27: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_147: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_27, _shape_param_176);  clone_27 = _shape_param_176 = None
        view_148: "bf16[2048, 768]" = torch.ops.aten.view.default(view_147, _shape_param_177);  view_147 = _shape_param_177 = None
        permute_74: "bf16[768, 768]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_39: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg110_1, view_148, permute_74);  arg110_1 = view_148 = permute_74 = None
        view_149: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_39, _shape_param_178);  addmm_39 = _shape_param_178 = None
        add_46: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_43, view_149);  add_43 = view_149 = None
        mean_13: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_46, [-1], True)
        sub_20: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_46, mean_13);  mean_13 = None
        mul_31: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg111_1, sub_20);  arg111_1 = sub_20 = None
        convert_element_type_52: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_46, torch.float32)
        var_13: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_52, [-1], correction = 1.0, keepdim = True);  convert_element_type_52 = None
        sqrt_13: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_13);  var_13 = None
        convert_element_type_53: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_13, torch.bfloat16);  sqrt_13 = None
        add_47: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-06);  convert_element_type_53 = None
        div_27: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_31, add_47);  mul_31 = add_47 = None
        add_48: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_27, arg112_1);  div_27 = arg112_1 = None
        view_150: "bf16[2048, 768]" = torch.ops.aten.view.default(add_48, _shape_param_179);  add_48 = _shape_param_179 = None
        permute_75: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_40: "bf16[2048, 3072]" = torch.ops.aten.addmm.default(arg114_1, view_150, permute_75);  arg114_1 = view_150 = permute_75 = None
        view_151: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(addmm_40, _shape_param_180);  addmm_40 = _shape_param_180 = None
        convert_element_type_54: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        mul_32: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 0.5)
        mul_33: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 0.7071067811865476);  convert_element_type_54 = None
        erf_6: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_49: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_34: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_49);  mul_32 = add_49 = None
        convert_element_type_55: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_34, torch.bfloat16);  mul_34 = None
        view_152: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_55, _shape_param_181);  convert_element_type_55 = _shape_param_181 = None
        permute_76: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_41: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg116_1, view_152, permute_76);  arg116_1 = view_152 = permute_76 = None
        view_153: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_41, _shape_param_182);  addmm_41 = _shape_param_182 = None
        add_50: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_46, view_153);  add_46 = view_153 = None
        mean_14: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_50, [-1], True)
        sub_21: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_50, mean_14);  mean_14 = None
        mul_35: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg117_1, sub_21);  arg117_1 = sub_21 = None
        convert_element_type_56: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_50, torch.float32)
        var_14: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_56, [-1], correction = 1.0, keepdim = True);  convert_element_type_56 = None
        sqrt_14: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_14);  var_14 = None
        convert_element_type_57: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_14, torch.bfloat16);  sqrt_14 = None
        add_51: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_57, 1e-06);  convert_element_type_57 = None
        div_28: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_35, add_51);  mul_35 = add_51 = None
        add_52: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_28, arg118_1);  div_28 = arg118_1 = None
        view_154: "bf16[2048, 768]" = torch.ops.aten.view.default(add_52, _shape_param_183);  _shape_param_183 = None
        permute_77: "bf16[768, 768]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_42: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg120_1, view_154, permute_77);  arg120_1 = view_154 = permute_77 = None
        view_155: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_42, _shape_param_184);  addmm_42 = _shape_param_184 = None
        view_156: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_155, _shape_param_185);  view_155 = _shape_param_185 = None
        permute_78: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
        expand_28: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_78, _shape_param_186);  permute_78 = _shape_param_186 = None
        clone_28: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_157: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_187);  clone_28 = _shape_param_187 = None
        view_158: "bf16[2048, 768]" = torch.ops.aten.view.default(add_52, _shape_param_188);  _shape_param_188 = None
        permute_79: "bf16[768, 768]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_43: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg122_1, view_158, permute_79);  arg122_1 = view_158 = permute_79 = None
        view_159: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_43, _shape_param_189);  addmm_43 = _shape_param_189 = None
        view_160: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_159, _shape_param_190);  view_159 = _shape_param_190 = None
        permute_80: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
        permute_81: "bf16[16, 12, 64, 128]" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2]);  permute_80 = None
        expand_29: "bf16[16, 12, 64, 128]" = torch.ops.aten.expand.default(permute_81, _shape_param_191);  permute_81 = _shape_param_191 = None
        clone_29: "bf16[16, 12, 64, 128]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_161: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_29, _shape_param_192);  clone_29 = _shape_param_192 = None
        bmm_14: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_157, view_161);  view_157 = view_161 = None
        view_162: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_14, _shape_param_193);  bmm_14 = _shape_param_193 = None
        div_29: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_162, 8.0);  view_162 = None
        where_7: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_4, full_4, div_29);  eq_4 = full_4 = div_29 = None
        convert_element_type_58: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where_7, torch.float32);  where_7 = None
        amax_7: "f32[16, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_58, [-1], True)
        sub_22: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_58, amax_7);  convert_element_type_58 = amax_7 = None
        exp_7: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_30: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_59: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_30, torch.bfloat16);  div_30 = None
        expand_30: "bf16[16, 12, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_59, _shape_param_194);  convert_element_type_59 = _shape_param_194 = None
        view_163: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_30, _shape_param_195);  expand_30 = _shape_param_195 = None
        view_164: "bf16[2048, 768]" = torch.ops.aten.view.default(add_52, _shape_param_196);  add_52 = _shape_param_196 = None
        permute_82: "bf16[768, 768]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_44: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg124_1, view_164, permute_82);  arg124_1 = view_164 = permute_82 = None
        view_165: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_44, _shape_param_197);  addmm_44 = _shape_param_197 = None
        view_166: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_165, _shape_param_198);  view_165 = _shape_param_198 = None
        permute_83: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        expand_31: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_83, _shape_param_199);  permute_83 = _shape_param_199 = None
        clone_30: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_167: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_30, _shape_param_200);  clone_30 = _shape_param_200 = None
        bmm_15: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_163, view_167);  view_163 = view_167 = None
        view_168: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_201);  bmm_15 = _shape_param_201 = None
        permute_84: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_31: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_169: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_31, _shape_param_202);  clone_31 = _shape_param_202 = None
        view_170: "bf16[2048, 768]" = torch.ops.aten.view.default(view_169, _shape_param_203);  view_169 = _shape_param_203 = None
        permute_85: "bf16[768, 768]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_45: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg126_1, view_170, permute_85);  arg126_1 = view_170 = permute_85 = None
        view_171: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_45, _shape_param_204);  addmm_45 = _shape_param_204 = None
        add_53: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_50, view_171);  add_50 = view_171 = None
        mean_15: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_53, [-1], True)
        sub_23: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_53, mean_15);  mean_15 = None
        mul_36: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg127_1, sub_23);  arg127_1 = sub_23 = None
        convert_element_type_60: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_53, torch.float32)
        var_15: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_60, [-1], correction = 1.0, keepdim = True);  convert_element_type_60 = None
        sqrt_15: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_15);  var_15 = None
        convert_element_type_61: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_15, torch.bfloat16);  sqrt_15 = None
        add_54: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-06);  convert_element_type_61 = None
        div_31: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_36, add_54);  mul_36 = add_54 = None
        add_55: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_31, arg128_1);  div_31 = arg128_1 = None
        view_172: "bf16[2048, 768]" = torch.ops.aten.view.default(add_55, _shape_param_205);  add_55 = _shape_param_205 = None
        permute_86: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_46: "bf16[2048, 3072]" = torch.ops.aten.addmm.default(arg130_1, view_172, permute_86);  arg130_1 = view_172 = permute_86 = None
        view_173: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(addmm_46, _shape_param_206);  addmm_46 = _shape_param_206 = None
        convert_element_type_62: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_37: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 0.5)
        mul_38: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 0.7071067811865476);  convert_element_type_62 = None
        erf_7: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
        add_56: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_39: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_37, add_56);  mul_37 = add_56 = None
        convert_element_type_63: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None
        view_174: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_63, _shape_param_207);  convert_element_type_63 = _shape_param_207 = None
        permute_87: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_47: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg132_1, view_174, permute_87);  arg132_1 = view_174 = permute_87 = None
        view_175: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_47, _shape_param_208);  addmm_47 = _shape_param_208 = None
        add_57: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_53, view_175);  add_53 = view_175 = None
        mean_16: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_57, [-1], True)
        sub_24: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_57, mean_16);  mean_16 = None
        mul_40: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg133_1, sub_24);  arg133_1 = sub_24 = None
        convert_element_type_64: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_57, torch.float32)
        var_16: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_64, [-1], correction = 1.0, keepdim = True);  convert_element_type_64 = None
        sqrt_16: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_16);  var_16 = None
        convert_element_type_65: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_16, torch.bfloat16);  sqrt_16 = None
        add_58: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_65, 1e-06);  convert_element_type_65 = None
        div_32: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_40, add_58);  mul_40 = add_58 = None
        add_59: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_32, arg134_1);  div_32 = arg134_1 = None
        view_176: "bf16[2048, 768]" = torch.ops.aten.view.default(add_59, _shape_param_209);  _shape_param_209 = None
        permute_88: "bf16[768, 768]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_48: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg136_1, view_176, permute_88);  arg136_1 = view_176 = permute_88 = None
        view_177: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_48, _shape_param_210);  addmm_48 = _shape_param_210 = None
        view_178: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_177, _shape_param_211);  view_177 = _shape_param_211 = None
        permute_89: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        expand_32: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_89, _shape_param_212);  permute_89 = _shape_param_212 = None
        clone_32: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_179: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_32, _shape_param_213);  clone_32 = _shape_param_213 = None
        view_180: "bf16[2048, 768]" = torch.ops.aten.view.default(add_59, _shape_param_214);  _shape_param_214 = None
        permute_90: "bf16[768, 768]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_49: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg138_1, view_180, permute_90);  arg138_1 = view_180 = permute_90 = None
        view_181: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_49, _shape_param_215);  addmm_49 = _shape_param_215 = None
        view_182: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_181, _shape_param_216);  view_181 = _shape_param_216 = None
        permute_91: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        permute_92: "bf16[16, 12, 64, 128]" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
        expand_33: "bf16[16, 12, 64, 128]" = torch.ops.aten.expand.default(permute_92, _shape_param_217);  permute_92 = _shape_param_217 = None
        clone_33: "bf16[16, 12, 64, 128]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_183: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_33, _shape_param_218);  clone_33 = _shape_param_218 = None
        bmm_16: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_179, view_183);  view_179 = view_183 = None
        view_184: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_16, _shape_param_219);  bmm_16 = _shape_param_219 = None
        div_33: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_184, 8.0);  view_184 = None
        where_8: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_3, full_3, div_33);  eq_3 = full_3 = div_33 = None
        convert_element_type_66: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where_8, torch.float32);  where_8 = None
        amax_8: "f32[16, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_66, [-1], True)
        sub_25: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_66, amax_8);  convert_element_type_66 = amax_8 = None
        exp_8: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_34: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_67: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_34, torch.bfloat16);  div_34 = None
        expand_34: "bf16[16, 12, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_67, _shape_param_220);  convert_element_type_67 = _shape_param_220 = None
        view_185: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_34, _shape_param_221);  expand_34 = _shape_param_221 = None
        view_186: "bf16[2048, 768]" = torch.ops.aten.view.default(add_59, _shape_param_222);  add_59 = _shape_param_222 = None
        permute_93: "bf16[768, 768]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_50: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg140_1, view_186, permute_93);  arg140_1 = view_186 = permute_93 = None
        view_187: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_50, _shape_param_223);  addmm_50 = _shape_param_223 = None
        view_188: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_187, _shape_param_224);  view_187 = _shape_param_224 = None
        permute_94: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
        expand_35: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_94, _shape_param_225);  permute_94 = _shape_param_225 = None
        clone_34: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_189: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_34, _shape_param_226);  clone_34 = _shape_param_226 = None
        bmm_17: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_185, view_189);  view_185 = view_189 = None
        view_190: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_17, _shape_param_227);  bmm_17 = _shape_param_227 = None
        permute_95: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_35: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_191: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_35, _shape_param_228);  clone_35 = _shape_param_228 = None
        view_192: "bf16[2048, 768]" = torch.ops.aten.view.default(view_191, _shape_param_229);  view_191 = _shape_param_229 = None
        permute_96: "bf16[768, 768]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_51: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg142_1, view_192, permute_96);  arg142_1 = view_192 = permute_96 = None
        view_193: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_51, _shape_param_230);  addmm_51 = _shape_param_230 = None
        add_60: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_57, view_193);  add_57 = view_193 = None
        mean_17: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_60, [-1], True)
        sub_26: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_60, mean_17);  mean_17 = None
        mul_41: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg143_1, sub_26);  arg143_1 = sub_26 = None
        convert_element_type_68: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_60, torch.float32)
        var_17: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_68, [-1], correction = 1.0, keepdim = True);  convert_element_type_68 = None
        sqrt_17: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_17);  var_17 = None
        convert_element_type_69: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_17, torch.bfloat16);  sqrt_17 = None
        add_61: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_69, 1e-06);  convert_element_type_69 = None
        div_35: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_41, add_61);  mul_41 = add_61 = None
        add_62: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_35, arg144_1);  div_35 = arg144_1 = None
        view_194: "bf16[2048, 768]" = torch.ops.aten.view.default(add_62, _shape_param_231);  add_62 = _shape_param_231 = None
        permute_97: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_52: "bf16[2048, 3072]" = torch.ops.aten.addmm.default(arg146_1, view_194, permute_97);  arg146_1 = view_194 = permute_97 = None
        view_195: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(addmm_52, _shape_param_232);  addmm_52 = _shape_param_232 = None
        convert_element_type_70: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        mul_42: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 0.5)
        mul_43: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 0.7071067811865476);  convert_element_type_70 = None
        erf_8: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_63: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_44: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_42, add_63);  mul_42 = add_63 = None
        convert_element_type_71: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_44, torch.bfloat16);  mul_44 = None
        view_196: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_71, _shape_param_233);  convert_element_type_71 = _shape_param_233 = None
        permute_98: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_53: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg148_1, view_196, permute_98);  arg148_1 = view_196 = permute_98 = None
        view_197: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_53, _shape_param_234);  addmm_53 = _shape_param_234 = None
        add_64: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_60, view_197);  add_60 = view_197 = None
        mean_18: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_64, [-1], True)
        sub_27: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_64, mean_18);  mean_18 = None
        mul_45: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg149_1, sub_27);  arg149_1 = sub_27 = None
        convert_element_type_72: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_64, torch.float32)
        var_18: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_72, [-1], correction = 1.0, keepdim = True);  convert_element_type_72 = None
        sqrt_18: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_18);  var_18 = None
        convert_element_type_73: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_18, torch.bfloat16);  sqrt_18 = None
        add_65: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_73, 1e-06);  convert_element_type_73 = None
        div_36: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_45, add_65);  mul_45 = add_65 = None
        add_66: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_36, arg150_1);  div_36 = arg150_1 = None
        view_198: "bf16[2048, 768]" = torch.ops.aten.view.default(add_66, _shape_param_235);  _shape_param_235 = None
        permute_99: "bf16[768, 768]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_54: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg152_1, view_198, permute_99);  arg152_1 = view_198 = permute_99 = None
        view_199: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_54, _shape_param_236);  addmm_54 = _shape_param_236 = None
        view_200: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_199, _shape_param_237);  view_199 = _shape_param_237 = None
        permute_100: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
        expand_36: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_100, _shape_param_238);  permute_100 = _shape_param_238 = None
        clone_36: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_201: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_239);  clone_36 = _shape_param_239 = None
        view_202: "bf16[2048, 768]" = torch.ops.aten.view.default(add_66, _shape_param_240);  _shape_param_240 = None
        permute_101: "bf16[768, 768]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_55: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg154_1, view_202, permute_101);  arg154_1 = view_202 = permute_101 = None
        view_203: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_55, _shape_param_241);  addmm_55 = _shape_param_241 = None
        view_204: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_203, _shape_param_242);  view_203 = _shape_param_242 = None
        permute_102: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
        permute_103: "bf16[16, 12, 64, 128]" = torch.ops.aten.permute.default(permute_102, [0, 1, 3, 2]);  permute_102 = None
        expand_37: "bf16[16, 12, 64, 128]" = torch.ops.aten.expand.default(permute_103, _shape_param_243);  permute_103 = _shape_param_243 = None
        clone_37: "bf16[16, 12, 64, 128]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_205: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_37, _shape_param_244);  clone_37 = _shape_param_244 = None
        bmm_18: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_201, view_205);  view_201 = view_205 = None
        view_206: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_18, _shape_param_245);  bmm_18 = _shape_param_245 = None
        div_37: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_206, 8.0);  view_206 = None
        where_9: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_2, full_2, div_37);  eq_2 = full_2 = div_37 = None
        convert_element_type_74: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where_9, torch.float32);  where_9 = None
        amax_9: "f32[16, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_74, [-1], True)
        sub_28: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_74, amax_9);  convert_element_type_74 = amax_9 = None
        exp_9: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_38: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_75: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_38, torch.bfloat16);  div_38 = None
        expand_38: "bf16[16, 12, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_75, _shape_param_246);  convert_element_type_75 = _shape_param_246 = None
        view_207: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_38, _shape_param_247);  expand_38 = _shape_param_247 = None
        view_208: "bf16[2048, 768]" = torch.ops.aten.view.default(add_66, _shape_param_248);  add_66 = _shape_param_248 = None
        permute_104: "bf16[768, 768]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_56: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg156_1, view_208, permute_104);  arg156_1 = view_208 = permute_104 = None
        view_209: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_56, _shape_param_249);  addmm_56 = _shape_param_249 = None
        view_210: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_209, _shape_param_250);  view_209 = _shape_param_250 = None
        permute_105: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
        expand_39: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_105, _shape_param_251);  permute_105 = _shape_param_251 = None
        clone_38: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_211: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_38, _shape_param_252);  clone_38 = _shape_param_252 = None
        bmm_19: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_207, view_211);  view_207 = view_211 = None
        view_212: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_253);  bmm_19 = _shape_param_253 = None
        permute_106: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        clone_39: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_213: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_39, _shape_param_254);  clone_39 = _shape_param_254 = None
        view_214: "bf16[2048, 768]" = torch.ops.aten.view.default(view_213, _shape_param_255);  view_213 = _shape_param_255 = None
        permute_107: "bf16[768, 768]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_57: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg158_1, view_214, permute_107);  arg158_1 = view_214 = permute_107 = None
        view_215: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_57, _shape_param_256);  addmm_57 = _shape_param_256 = None
        add_67: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_64, view_215);  add_64 = view_215 = None
        mean_19: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_67, [-1], True)
        sub_29: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_67, mean_19);  mean_19 = None
        mul_46: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg159_1, sub_29);  arg159_1 = sub_29 = None
        convert_element_type_76: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_67, torch.float32)
        var_19: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_76, [-1], correction = 1.0, keepdim = True);  convert_element_type_76 = None
        sqrt_19: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_19);  var_19 = None
        convert_element_type_77: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_19, torch.bfloat16);  sqrt_19 = None
        add_68: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_77, 1e-06);  convert_element_type_77 = None
        div_39: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_46, add_68);  mul_46 = add_68 = None
        add_69: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_39, arg160_1);  div_39 = arg160_1 = None
        view_216: "bf16[2048, 768]" = torch.ops.aten.view.default(add_69, _shape_param_257);  add_69 = _shape_param_257 = None
        permute_108: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_58: "bf16[2048, 3072]" = torch.ops.aten.addmm.default(arg162_1, view_216, permute_108);  arg162_1 = view_216 = permute_108 = None
        view_217: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(addmm_58, _shape_param_258);  addmm_58 = _shape_param_258 = None
        convert_element_type_78: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_47: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_78, 0.5)
        mul_48: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_78, 0.7071067811865476);  convert_element_type_78 = None
        erf_9: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
        add_70: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_49: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_47, add_70);  mul_47 = add_70 = None
        convert_element_type_79: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        view_218: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_79, _shape_param_259);  convert_element_type_79 = _shape_param_259 = None
        permute_109: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_59: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg164_1, view_218, permute_109);  arg164_1 = view_218 = permute_109 = None
        view_219: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_59, _shape_param_260);  addmm_59 = _shape_param_260 = None
        add_71: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_67, view_219);  add_67 = view_219 = None
        mean_20: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_71, [-1], True)
        sub_30: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_71, mean_20);  mean_20 = None
        mul_50: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg165_1, sub_30);  arg165_1 = sub_30 = None
        convert_element_type_80: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_71, torch.float32)
        var_20: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_80, [-1], correction = 1.0, keepdim = True);  convert_element_type_80 = None
        sqrt_20: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_20);  var_20 = None
        convert_element_type_81: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_20, torch.bfloat16);  sqrt_20 = None
        add_72: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_81, 1e-06);  convert_element_type_81 = None
        div_40: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_50, add_72);  mul_50 = add_72 = None
        add_73: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_40, arg166_1);  div_40 = arg166_1 = None
        view_220: "bf16[2048, 768]" = torch.ops.aten.view.default(add_73, _shape_param_261);  _shape_param_261 = None
        permute_110: "bf16[768, 768]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_60: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg168_1, view_220, permute_110);  arg168_1 = view_220 = permute_110 = None
        view_221: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_60, _shape_param_262);  addmm_60 = _shape_param_262 = None
        view_222: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_221, _shape_param_263);  view_221 = _shape_param_263 = None
        permute_111: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
        expand_40: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_111, _shape_param_264);  permute_111 = _shape_param_264 = None
        clone_40: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_223: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_265);  clone_40 = _shape_param_265 = None
        view_224: "bf16[2048, 768]" = torch.ops.aten.view.default(add_73, _shape_param_266);  _shape_param_266 = None
        permute_112: "bf16[768, 768]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_61: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg170_1, view_224, permute_112);  arg170_1 = view_224 = permute_112 = None
        view_225: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_61, _shape_param_267);  addmm_61 = _shape_param_267 = None
        view_226: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_225, _shape_param_268);  view_225 = _shape_param_268 = None
        permute_113: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        permute_114: "bf16[16, 12, 64, 128]" = torch.ops.aten.permute.default(permute_113, [0, 1, 3, 2]);  permute_113 = None
        expand_41: "bf16[16, 12, 64, 128]" = torch.ops.aten.expand.default(permute_114, _shape_param_269);  permute_114 = _shape_param_269 = None
        clone_41: "bf16[16, 12, 64, 128]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_227: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_41, _shape_param_270);  clone_41 = _shape_param_270 = None
        bmm_20: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_223, view_227);  view_223 = view_227 = None
        view_228: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_20, _shape_param_271);  bmm_20 = _shape_param_271 = None
        div_41: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_228, 8.0);  view_228 = None
        where_10: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, full_1, div_41);  eq_1 = full_1 = div_41 = None
        convert_element_type_82: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where_10, torch.float32);  where_10 = None
        amax_10: "f32[16, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_82, [-1], True)
        sub_31: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_82, amax_10);  convert_element_type_82 = amax_10 = None
        exp_10: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_42: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_83: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_42, torch.bfloat16);  div_42 = None
        expand_42: "bf16[16, 12, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_83, _shape_param_272);  convert_element_type_83 = _shape_param_272 = None
        view_229: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_42, _shape_param_273);  expand_42 = _shape_param_273 = None
        view_230: "bf16[2048, 768]" = torch.ops.aten.view.default(add_73, _shape_param_274);  add_73 = _shape_param_274 = None
        permute_115: "bf16[768, 768]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_62: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg172_1, view_230, permute_115);  arg172_1 = view_230 = permute_115 = None
        view_231: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_62, _shape_param_275);  addmm_62 = _shape_param_275 = None
        view_232: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_231, _shape_param_276);  view_231 = _shape_param_276 = None
        permute_116: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None
        expand_43: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_116, _shape_param_277);  permute_116 = _shape_param_277 = None
        clone_42: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_233: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_42, _shape_param_278);  clone_42 = _shape_param_278 = None
        bmm_21: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_229, view_233);  view_229 = view_233 = None
        view_234: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_21, _shape_param_279);  bmm_21 = _shape_param_279 = None
        permute_117: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_43: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_235: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_43, _shape_param_280);  clone_43 = _shape_param_280 = None
        view_236: "bf16[2048, 768]" = torch.ops.aten.view.default(view_235, _shape_param_281);  view_235 = _shape_param_281 = None
        permute_118: "bf16[768, 768]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_63: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg174_1, view_236, permute_118);  arg174_1 = view_236 = permute_118 = None
        view_237: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_63, _shape_param_282);  addmm_63 = _shape_param_282 = None
        add_74: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_71, view_237);  add_71 = view_237 = None
        mean_21: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_74, [-1], True)
        sub_32: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_74, mean_21);  mean_21 = None
        mul_51: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg175_1, sub_32);  arg175_1 = sub_32 = None
        convert_element_type_84: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_74, torch.float32)
        var_21: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_84, [-1], correction = 1.0, keepdim = True);  convert_element_type_84 = None
        sqrt_21: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_21);  var_21 = None
        convert_element_type_85: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_21, torch.bfloat16);  sqrt_21 = None
        add_75: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_85, 1e-06);  convert_element_type_85 = None
        div_43: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_51, add_75);  mul_51 = add_75 = None
        add_76: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_43, arg176_1);  div_43 = arg176_1 = None
        view_238: "bf16[2048, 768]" = torch.ops.aten.view.default(add_76, _shape_param_283);  add_76 = _shape_param_283 = None
        permute_119: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_64: "bf16[2048, 3072]" = torch.ops.aten.addmm.default(arg178_1, view_238, permute_119);  arg178_1 = view_238 = permute_119 = None
        view_239: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(addmm_64, _shape_param_284);  addmm_64 = _shape_param_284 = None
        convert_element_type_86: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        mul_52: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_86, 0.5)
        mul_53: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_86, 0.7071067811865476);  convert_element_type_86 = None
        erf_10: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_53);  mul_53 = None
        add_77: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_54: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_77);  mul_52 = add_77 = None
        convert_element_type_87: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None
        view_240: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_87, _shape_param_285);  convert_element_type_87 = _shape_param_285 = None
        permute_120: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_65: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg180_1, view_240, permute_120);  arg180_1 = view_240 = permute_120 = None
        view_241: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_65, _shape_param_286);  addmm_65 = _shape_param_286 = None
        add_78: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_74, view_241);  add_74 = view_241 = None
        mean_22: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_78, [-1], True)
        sub_33: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_78, mean_22);  mean_22 = None
        mul_55: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg181_1, sub_33);  arg181_1 = sub_33 = None
        convert_element_type_88: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_78, torch.float32)
        var_22: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_88, [-1], correction = 1.0, keepdim = True);  convert_element_type_88 = None
        sqrt_22: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_22);  var_22 = None
        convert_element_type_89: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_22, torch.bfloat16);  sqrt_22 = None
        add_79: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_89, 1e-06);  convert_element_type_89 = None
        div_44: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_55, add_79);  mul_55 = add_79 = None
        add_80: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_44, arg182_1);  div_44 = arg182_1 = None
        view_242: "bf16[2048, 768]" = torch.ops.aten.view.default(add_80, _shape_param_287);  _shape_param_287 = None
        permute_121: "bf16[768, 768]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_66: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg184_1, view_242, permute_121);  arg184_1 = view_242 = permute_121 = None
        view_243: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_66, _shape_param_288);  addmm_66 = _shape_param_288 = None
        view_244: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_243, _shape_param_289);  view_243 = _shape_param_289 = None
        permute_122: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
        expand_44: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_122, _shape_param_290);  permute_122 = _shape_param_290 = None
        clone_44: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_245: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_44, _shape_param_291);  clone_44 = _shape_param_291 = None
        view_246: "bf16[2048, 768]" = torch.ops.aten.view.default(add_80, _shape_param_292);  _shape_param_292 = None
        permute_123: "bf16[768, 768]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_67: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg186_1, view_246, permute_123);  arg186_1 = view_246 = permute_123 = None
        view_247: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_67, _shape_param_293);  addmm_67 = _shape_param_293 = None
        view_248: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_247, _shape_param_294);  view_247 = _shape_param_294 = None
        permute_124: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
        permute_125: "bf16[16, 12, 64, 128]" = torch.ops.aten.permute.default(permute_124, [0, 1, 3, 2]);  permute_124 = None
        expand_45: "bf16[16, 12, 64, 128]" = torch.ops.aten.expand.default(permute_125, _shape_param_295);  permute_125 = _shape_param_295 = None
        clone_45: "bf16[16, 12, 64, 128]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_249: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_45, _shape_param_296);  clone_45 = _shape_param_296 = None
        bmm_22: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_245, view_249);  view_245 = view_249 = None
        view_250: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_297);  bmm_22 = _shape_param_297 = None
        div_45: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_250, 8.0);  view_250 = None
        where_11: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq, full, div_45);  eq = full = div_45 = None
        convert_element_type_90: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where_11, torch.float32);  where_11 = None
        amax_11: "f32[16, 12, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_90, [-1], True)
        sub_34: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_90, amax_11);  convert_element_type_90 = amax_11 = None
        exp_11: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_46: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_91: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(div_46, torch.bfloat16);  div_46 = None
        expand_46: "bf16[16, 12, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_91, _shape_param_298);  convert_element_type_91 = _shape_param_298 = None
        view_251: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_46, _shape_param_299);  expand_46 = _shape_param_299 = None
        view_252: "bf16[2048, 768]" = torch.ops.aten.view.default(add_80, _shape_param_300);  add_80 = _shape_param_300 = None
        permute_126: "bf16[768, 768]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_68: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg188_1, view_252, permute_126);  arg188_1 = view_252 = permute_126 = None
        view_253: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_68, _shape_param_301);  addmm_68 = _shape_param_301 = None
        view_254: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_253, _shape_param_302);  view_253 = _shape_param_302 = None
        permute_127: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_254, [0, 2, 1, 3]);  view_254 = None
        expand_47: "bf16[16, 12, 128, 64]" = torch.ops.aten.expand.default(permute_127, _shape_param_303);  permute_127 = _shape_param_303 = None
        clone_46: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_255: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_46, _shape_param_304);  clone_46 = _shape_param_304 = None
        bmm_23: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_251, view_255);  view_251 = view_255 = None
        view_256: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_305);  bmm_23 = _shape_param_305 = None
        permute_128: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        clone_47: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_257: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_47, _shape_param_306);  clone_47 = _shape_param_306 = None
        view_258: "bf16[2048, 768]" = torch.ops.aten.view.default(view_257, _shape_param_307);  view_257 = _shape_param_307 = None
        permute_129: "bf16[768, 768]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_69: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg190_1, view_258, permute_129);  arg190_1 = view_258 = permute_129 = None
        view_259: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_69, _shape_param_308);  addmm_69 = _shape_param_308 = None
        add_81: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_78, view_259);  add_78 = view_259 = None
        mean_23: "bf16[16, 128, 1]" = torch.ops.aten.mean.dim(add_81, [-1], True)
        sub_35: "bf16[16, 128, 768]" = torch.ops.aten.sub.Tensor(add_81, mean_23);  mean_23 = None
        mul_56: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg191_1, sub_35);  arg191_1 = sub_35 = None
        convert_element_type_92: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_81, torch.float32)
        var_23: "f32[16, 128, 1]" = torch.ops.aten.var.correction(convert_element_type_92, [-1], correction = 1.0, keepdim = True);  convert_element_type_92 = None
        sqrt_23: "f32[16, 128, 1]" = torch.ops.aten.sqrt.default(var_23);  var_23 = None
        convert_element_type_93: "bf16[16, 128, 1]" = torch.ops.prims.convert_element_type.default(sqrt_23, torch.bfloat16);  sqrt_23 = None
        add_82: "bf16[16, 128, 1]" = torch.ops.aten.add.Tensor(convert_element_type_93, 1e-06);  convert_element_type_93 = None
        div_47: "bf16[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_56, add_82);  mul_56 = add_82 = None
        add_83: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(div_47, arg192_1);  div_47 = arg192_1 = None
        view_260: "bf16[2048, 768]" = torch.ops.aten.view.default(add_83, _shape_param_309);  add_83 = _shape_param_309 = None
        permute_130: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_70: "bf16[2048, 3072]" = torch.ops.aten.addmm.default(arg194_1, view_260, permute_130);  arg194_1 = view_260 = permute_130 = None
        view_261: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(addmm_70, _shape_param_310);  addmm_70 = _shape_param_310 = None
        convert_element_type_94: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        mul_57: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_94, 0.5)
        mul_58: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_94, 0.7071067811865476);  convert_element_type_94 = None
        erf_11: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
        add_84: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_59: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_57, add_84);  mul_57 = add_84 = None
        convert_element_type_95: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None
        view_262: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_95, _shape_param_311);  convert_element_type_95 = _shape_param_311 = None
        permute_131: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_71: "bf16[2048, 768]" = torch.ops.aten.addmm.default(arg196_1, view_262, permute_131);  arg196_1 = view_262 = permute_131 = None
        view_263: "bf16[16, 128, 768]" = torch.ops.aten.view.default(addmm_71, _shape_param_312);  addmm_71 = _shape_param_312 = None
        add_85: "bf16[16, 128, 768]" = torch.ops.aten.add.Tensor(add_81, view_263);  add_81 = view_263 = None
        select: "bf16[16, 768]" = torch.ops.aten.select.int(add_85, 1, 0)
        permute_132: "bf16[768, 2]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_72: "bf16[16, 2]" = torch.ops.aten.addmm.default(arg198_1, select, permute_132);  arg198_1 = select = permute_132 = None
        convert_element_type_96: "f32[16, 2]" = torch.ops.prims.convert_element_type.default(addmm_72, torch.float32);  addmm_72 = None
        amax_12: "f32[16, 1]" = torch.ops.aten.amax.default(convert_element_type_96, [-1], True)
        sub_36: "f32[16, 2]" = torch.ops.aten.sub.Tensor(convert_element_type_96, amax_12);  convert_element_type_96 = amax_12 = None
        exp_12: "f32[16, 2]" = torch.ops.aten.exp.default(sub_36)
        sum_13: "f32[16, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True);  exp_12 = None
        log: "f32[16, 1]" = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_37: "f32[16, 2]" = torch.ops.aten.sub.Tensor(sub_36, log);  sub_36 = log = None
        convert_element_type_97: "bf16[16, 2]" = torch.ops.prims.convert_element_type.default(sub_37, torch.bfloat16);  sub_37 = None
        full_12: "bf16[3]" = torch.ops.aten.full.default(_shape_param_313, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_313 = None
        cat: "bf16[20008]" = torch.ops.aten.cat.default([arg199_1, full_12]);  arg199_1 = full_12 = None
        view_264: "bf16[2048, 768]" = torch.ops.aten.view.default(add_85, _shape_param_314);  add_85 = _shape_param_314 = None
        permute_133: "bf16[768, 20005]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        constant_pad_nd: "bf16[768, 20008]" = torch.ops.aten.constant_pad_nd.default(permute_133, _shape_param_315);  permute_133 = _shape_param_315 = None
        addmm_73: "bf16[2048, 20008]" = torch.ops.aten.addmm.default(cat, view_264, constant_pad_nd);  cat = view_264 = constant_pad_nd = None
        slice_2: "bf16[2048, 20005]" = torch.ops.aten.slice.Tensor(addmm_73, 1, 0, -3);  addmm_73 = None
        view_265: "bf16[16, 128, 20005]" = torch.ops.aten.view.default(slice_2, _shape_param_316);  slice_2 = _shape_param_316 = None
        convert_element_type_98: "f32[16, 128, 20005]" = torch.ops.prims.convert_element_type.default(view_265, torch.float32);  view_265 = None
        amax_13: "f32[16, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_98, [-1], True)
        sub_38: "f32[16, 128, 20005]" = torch.ops.aten.sub.Tensor(convert_element_type_98, amax_13);  convert_element_type_98 = amax_13 = None
        exp_13: "f32[16, 128, 20005]" = torch.ops.aten.exp.default(sub_38)
        sum_14: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True);  exp_13 = None
        log_1: "f32[16, 128, 1]" = torch.ops.aten.log.default(sum_14);  sum_14 = None
        sub_39: "f32[16, 128, 20005]" = torch.ops.aten.sub.Tensor(sub_38, log_1);  sub_38 = log_1 = None
        convert_element_type_99: "bf16[16, 128, 20005]" = torch.ops.prims.convert_element_type.default(sub_39, torch.bfloat16);  sub_39 = None
        return (unsqueeze_1, convert_element_type_97, convert_element_type_99)



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
