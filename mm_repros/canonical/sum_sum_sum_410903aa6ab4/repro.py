"""
Standalone repro captured via capture_hook.
Label: AlbertForMaskedLM
Pattern hash: 410903aa6ab4
Shape hash: 2d21179d
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
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "i64[8, 512]", arg3_1: "bf16[8, 512, 30000]", arg4_1: "f32[4096, 1]", arg5_1: "f32[4096, 1]", arg6_1: "bf16[8, 512, 30000]", arg7_1: "bf16[30000, 128]", arg8_1: "bf16[4096, 128]", arg9_1: "f32[128]", arg10_1: "bf16[4096, 128]", arg11_1: "f32[8, 512, 1]", arg12_1: "f32[8, 512, 1]", arg13_1: "bf16[128, 4096]", arg14_1: "bf16[4096, 4096]", arg15_1: "f32[4096]", arg16_1: "f32[8, 512, 4096]", arg17_1: "f32[8, 512, 1]", arg18_1: "bf16[4096, 16384]", arg19_1: "bf16[4096, 16384]", arg20_1: "bf16[4096, 16384]", arg21_1: "bf16[16384, 4096]", arg22_1: "bf16[4096, 4096]", arg23_1: "f32[4096]", arg24_1: "f32[8, 512, 4096]", arg25_1: "f32[8, 512, 1]", arg26_1: "bf16[4096, 4096]", arg27_1: "bf16[4096, 4096]", arg28_1: "bf16[8, 64, 512, 512]", arg29_1: "bf16[512, 64, 512]", arg30_1: "bf16[512, 64, 512]", arg31_1: "bf16[512, 512, 64]", arg32_1: "bf16[4096, 4096]", arg33_1: "bf16[4096, 4096]", arg34_1: "bf16[4096, 4096]", arg35_1: "bf16[4096, 4096]", arg36_1: "f32[8, 512, 4096]", arg37_1: "f32[8, 512, 1]", arg38_1: "bf16[4096, 16384]", arg39_1: "bf16[4096, 16384]", arg40_1: "bf16[4096, 4096]", arg41_1: "f32[8, 512, 4096]", arg42_1: "f32[8, 512, 1]", arg43_1: "bf16[4096, 4096]", arg44_1: "bf16[8, 64, 512, 512]", arg45_1: "bf16[512, 64, 512]", arg46_1: "bf16[512, 64, 512]", arg47_1: "bf16[512, 512, 64]", arg48_1: "bf16[4096, 4096]", arg49_1: "f32[8, 512, 4096]", arg50_1: "f32[8, 512, 1]", arg51_1: "bf16[4096, 16384]", arg52_1: "bf16[4096, 16384]", arg53_1: "bf16[4096, 4096]", arg54_1: "f32[8, 512, 4096]", arg55_1: "f32[8, 512, 1]", arg56_1: "bf16[4096, 4096]", arg57_1: "bf16[8, 64, 512, 512]", arg58_1: "bf16[512, 64, 512]", arg59_1: "bf16[512, 64, 512]", arg60_1: "bf16[512, 512, 64]", arg61_1: "bf16[4096, 4096]", arg62_1: "f32[8, 512, 4096]", arg63_1: "f32[8, 512, 1]", arg64_1: "bf16[4096, 16384]", arg65_1: "bf16[4096, 16384]", arg66_1: "bf16[4096, 4096]", arg67_1: "f32[8, 512, 4096]", arg68_1: "f32[8, 512, 1]", arg69_1: "bf16[4096, 4096]", arg70_1: "bf16[8, 64, 512, 512]", arg71_1: "bf16[512, 64, 512]", arg72_1: "bf16[512, 64, 512]", arg73_1: "bf16[512, 512, 64]", arg74_1: "bf16[4096, 4096]", arg75_1: "f32[8, 512, 4096]", arg76_1: "f32[8, 512, 1]", arg77_1: "bf16[4096, 16384]", arg78_1: "bf16[4096, 16384]", arg79_1: "bf16[4096, 4096]", arg80_1: "f32[8, 512, 4096]", arg81_1: "f32[8, 512, 1]", arg82_1: "bf16[4096, 4096]", arg83_1: "bf16[8, 64, 512, 512]", arg84_1: "bf16[512, 64, 512]", arg85_1: "bf16[512, 64, 512]", arg86_1: "bf16[512, 512, 64]", arg87_1: "bf16[4096, 4096]", arg88_1: "f32[8, 512, 4096]", arg89_1: "f32[8, 512, 1]", arg90_1: "bf16[4096, 16384]", arg91_1: "bf16[4096, 16384]", arg92_1: "bf16[4096, 4096]", arg93_1: "f32[8, 512, 4096]", arg94_1: "f32[8, 512, 1]", arg95_1: "bf16[4096, 4096]", arg96_1: "bf16[8, 64, 512, 512]", arg97_1: "bf16[512, 64, 512]", arg98_1: "bf16[512, 64, 512]", arg99_1: "bf16[512, 512, 64]", arg100_1: "bf16[4096, 4096]", arg101_1: "f32[8, 512, 4096]", arg102_1: "f32[8, 512, 1]", arg103_1: "bf16[4096, 16384]", arg104_1: "bf16[4096, 16384]", arg105_1: "bf16[4096, 4096]", arg106_1: "f32[8, 512, 4096]", arg107_1: "f32[8, 512, 1]", arg108_1: "bf16[4096, 4096]", arg109_1: "bf16[8, 64, 512, 512]", arg110_1: "bf16[512, 64, 512]", arg111_1: "bf16[512, 64, 512]", arg112_1: "bf16[512, 512, 64]", arg113_1: "bf16[4096, 4096]", arg114_1: "f32[8, 512, 4096]", arg115_1: "f32[8, 512, 1]", arg116_1: "bf16[4096, 16384]", arg117_1: "bf16[4096, 16384]", arg118_1: "bf16[4096, 4096]", arg119_1: "f32[8, 512, 4096]", arg120_1: "f32[8, 512, 1]", arg121_1: "bf16[4096, 4096]", arg122_1: "bf16[8, 64, 512, 512]", arg123_1: "bf16[512, 64, 512]", arg124_1: "bf16[512, 64, 512]", arg125_1: "bf16[512, 512, 64]", arg126_1: "bf16[4096, 4096]", arg127_1: "f32[8, 512, 4096]", arg128_1: "f32[8, 512, 1]", arg129_1: "bf16[4096, 16384]", arg130_1: "bf16[4096, 16384]", arg131_1: "bf16[4096, 4096]", arg132_1: "f32[8, 512, 4096]", arg133_1: "f32[8, 512, 1]", arg134_1: "bf16[4096, 4096]", arg135_1: "bf16[8, 64, 512, 512]", arg136_1: "bf16[512, 64, 512]", arg137_1: "bf16[512, 64, 512]", arg138_1: "bf16[512, 512, 64]", arg139_1: "bf16[4096, 4096]", arg140_1: "f32[8, 512, 4096]", arg141_1: "f32[8, 512, 1]", arg142_1: "bf16[4096, 16384]", arg143_1: "bf16[4096, 16384]", arg144_1: "bf16[4096, 4096]", arg145_1: "f32[8, 512, 4096]", arg146_1: "f32[8, 512, 1]", arg147_1: "bf16[4096, 4096]", arg148_1: "bf16[8, 64, 512, 512]", arg149_1: "bf16[512, 64, 512]", arg150_1: "bf16[512, 64, 512]", arg151_1: "bf16[512, 512, 64]", arg152_1: "bf16[4096, 4096]", arg153_1: "f32[8, 512, 4096]", arg154_1: "f32[8, 512, 1]", arg155_1: "bf16[4096, 16384]", arg156_1: "bf16[4096, 16384]", arg157_1: "bf16[4096, 4096]", arg158_1: "f32[8, 512, 4096]", arg159_1: "f32[8, 512, 1]", arg160_1: "bf16[4096, 4096]", arg161_1: "bf16[8, 64, 512, 512]", arg162_1: "bf16[512, 64, 512]", arg163_1: "bf16[512, 64, 512]", arg164_1: "bf16[512, 512, 64]", arg165_1: "bf16[4096, 4096]", arg166_1: "f32[8, 512, 4096]", arg167_1: "f32[8, 512, 1]", arg168_1: "bf16[4096, 16384]", arg169_1: "bf16[4096, 16384]", arg170_1: "bf16[4096, 4096]", arg171_1: "bf16[8, 512, 4096]", arg172_1: "f32[8, 512, 1]", arg173_1: "f32[8, 512, 1]", arg174_1: "bf16[4096, 4096]", arg175_1: "bf16[8, 64, 512, 512]", arg176_1: "bf16[512, 64, 512]", arg177_1: "bf16[512, 64, 512]", arg178_1: "bf16[512, 512, 64]", arg179_1: "bf16[4096, 4096]", arg180_1: "bf16[4096, 128]", arg181_1: "bf16[4096, 128]", arg182_1: "f32[128]", arg183_1: "f32[8, 512, 128]", arg184_1: "f32[8, 512, 1]", arg185_1: "i64[1, 512]", arg186_1: "i64[1, 512]", arg187_1: "i64[8, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "i64[4096]" = torch.ops.aten.view.default(arg2_1, [-1]);  arg2_1 = None
        unsqueeze: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(view, 1);  view = None
        ne: "b8[4096, 1]" = torch.ops.aten.ne.Scalar(unsqueeze, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[4096, 1]" = torch.ops.aten.where.self(ne, unsqueeze, full);  unsqueeze = full = None
        iota: "i64[30000]" = torch.ops.prims.iota.default(30000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_1: "i64[1, 30000]" = torch.ops.aten.view.default(iota, _shape_param_0);  iota = _shape_param_0 = None
        expand: "i64[4096, 30000]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        eq: "b8[4096, 30000]" = torch.ops.aten.eq.Tensor(expand, view_1);  expand = view_1 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[4096, 30000]" = torch.ops.aten.where.self(eq, scalar_tensor_1, scalar_tensor);  eq = scalar_tensor_1 = scalar_tensor = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[4096, 1]" = torch.ops.aten.where.self(ne, div, full_1);  ne = div = full_1 = None
        mul: "f32[4096, 30000]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        convert_element_type: "bf16[4096, 30000]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_1: "f32[4096, 30000]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        view_2: "bf16[4096, 30000]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_2: "f32[4096, 30000]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        sub: "f32[4096, 30000]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg4_1);  convert_element_type_2 = arg4_1 = None
        sub_1: "f32[4096, 30000]" = torch.ops.aten.sub.Tensor(sub, arg5_1);  sub = arg5_1 = None
        convert_element_type_3: "bf16[4096, 30000]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_4: "f32[4096, 30000]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        exp: "f32[4096, 30000]" = torch.ops.aten.exp.default(convert_element_type_4);  convert_element_type_4 = None
        sum_1: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [1], True)
        mul_1: "f32[4096, 30000]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub_2: "f32[4096, 30000]" = torch.ops.aten.sub.Tensor(convert_element_type_1, mul_1);  convert_element_type_1 = mul_1 = None
        convert_element_type_5: "bf16[4096, 30000]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        view_3: "bf16[8, 512, 30000]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_3);  convert_element_type_5 = _shape_param_3 = None
        add: "bf16[8, 512, 30000]" = torch.ops.aten.add.Tensor(arg6_1, view_3);  arg6_1 = view_3 = None
        view_4: "bf16[4096, 30000]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        mm: "bf16[4096, 128]" = torch.ops.aten.mm.default(view_4, arg7_1);  arg7_1 = None
        permute: "bf16[30000, 4096]" = torch.ops.aten.permute.default(view_4, [1, 0])
        mm_1: "bf16[30000, 128]" = torch.ops.aten.mm.default(permute, arg8_1);  permute = arg8_1 = None
        sum_2: "f32[1, 30000]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32);  view_4 = None
        view_5: "f32[30000]" = torch.ops.aten.view.default(sum_2, _shape_param_5);  sum_2 = _shape_param_5 = None
        convert_element_type_6: "bf16[30000]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[8, 512, 128]" = torch.ops.aten.view.default(mm, _shape_param_6);  mm = _shape_param_6 = None
        convert_element_type_7: "f32[8, 512, 128]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_8: "f32[30000, 128]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_9: "f32[30000]" = torch.ops.prims.convert_element_type.default(convert_element_type_6, torch.float32);  convert_element_type_6 = None
        mul_2: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_7, arg9_1);  arg9_1 = None
        mul_3: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_2, 128)
        sum_3: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        view_7: "bf16[8, 512, 128]" = torch.ops.aten.view.default(arg10_1, _shape_param_7);  arg10_1 = _shape_param_7 = None
        mul_4: "bf16[8, 512, 128]" = torch.ops.aten.mul.Tensor(view_7, 0.5)
        convert_element_type_10: "f32[8, 512, 128]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32)
        pow_1: "f32[8, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_10, 3.0)
        mul_5: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_1: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(view_7, mul_5);  view_7 = mul_5 = None
        mul_6: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(add_1, 0.7978845608028654);  add_1 = None
        tanh: "f32[8, 512, 128]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_2: "f32[8, 512, 128]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_7: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_4, add_2)
        sub_3: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_7, arg11_1);  mul_7 = arg11_1 = None
        mul_8: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(sub_3, arg12_1);  sub_3 = None
        mul_9: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_2, mul_8);  mul_2 = None
        sum_4: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_9, [2], True);  mul_9 = None
        mul_10: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_8, sum_4);  sum_4 = None
        sub_4: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_3, sum_3);  mul_3 = sum_3 = None
        sub_5: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(sub_4, mul_10);  sub_4 = mul_10 = None
        div_1: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(arg12_1, 128);  arg12_1 = None
        mul_11: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_12: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_7, mul_8);  mul_8 = None
        sum_5: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_12, [0, 1]);  mul_12 = None
        sum_6: "f32[128]" = torch.ops.aten.sum.dim_IntList(convert_element_type_7, [0, 1]);  convert_element_type_7 = None
        mul_13: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_11, mul_4);  mul_4 = None
        mul_14: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_11, add_2);  mul_11 = add_2 = None
        convert_element_type_11: "bf16[8, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        mul_15: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_6: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(1, mul_15);  mul_15 = None
        mul_16: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_13, sub_6);  mul_13 = sub_6 = None
        mul_17: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_16, 0.7978845608028654);  mul_16 = None
        convert_element_type_12: "bf16[8, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16)
        mul_18: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_17, 0.044715);  mul_17 = None
        pow_2: "f32[8, 512, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_10, 2.0);  convert_element_type_10 = None
        mul_19: "f32[8, 512, 128]" = torch.ops.aten.mul.Scalar(pow_2, 3.0);  pow_2 = None
        mul_20: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        convert_element_type_13: "bf16[8, 512, 128]" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None
        add_3: "bf16[8, 512, 128]" = torch.ops.aten.add.Tensor(convert_element_type_12, convert_element_type_13);  convert_element_type_12 = convert_element_type_13 = None
        mul_21: "bf16[8, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 0.5);  convert_element_type_11 = None
        add_4: "bf16[8, 512, 128]" = torch.ops.aten.add.Tensor(add_3, mul_21);  add_3 = mul_21 = None
        view_8: "bf16[4096, 128]" = torch.ops.aten.view.default(add_4, _shape_param_8);  add_4 = _shape_param_8 = None
        mm_2: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_8, arg13_1);  arg13_1 = None
        permute_1: "bf16[128, 4096]" = torch.ops.aten.permute.default(view_8, [1, 0])
        mm_3: "bf16[128, 4096]" = torch.ops.aten.mm.default(permute_1, arg14_1);  permute_1 = arg14_1 = None
        sum_7: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True, dtype = torch.float32);  view_8 = None
        view_9: "f32[128]" = torch.ops.aten.view.default(sum_7, _shape_param_9);  sum_7 = _shape_param_9 = None
        convert_element_type_14: "bf16[128]" = torch.ops.prims.convert_element_type.default(view_9, torch.bfloat16);  view_9 = None
        view_10: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_2, _shape_param_10);  mm_2 = _shape_param_10 = None
        convert_element_type_15: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        convert_element_type_16: "f32[128, 4096]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_17: "f32[128]" = torch.ops.prims.convert_element_type.default(convert_element_type_14, torch.float32);  convert_element_type_14 = None
        mul_22: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_15, arg15_1)
        mul_23: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_22, 4096)
        sum_8: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_22, [2], True)
        mul_24: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_22, arg16_1);  mul_22 = None
        sum_9: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_24, [2], True);  mul_24 = None
        mul_25: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg16_1, sum_9);  sum_9 = None
        sub_7: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_23, sum_8);  mul_23 = sum_8 = None
        sub_8: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_7, mul_25);  sub_7 = mul_25 = None
        mul_26: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg17_1, sub_8);  arg17_1 = sub_8 = None
        mul_27: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_15, arg16_1);  arg16_1 = None
        sum_10: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_27, [0, 1]);  mul_27 = None
        sum_11: "f32[4096]" = torch.ops.aten.sum.dim_IntList(convert_element_type_15, [0, 1]);  convert_element_type_15 = None
        convert_element_type_18: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_26, torch.bfloat16)
        view_11: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_18, _shape_param_11);  convert_element_type_18 = _shape_param_11 = None
        mm_4: "bf16[4096, 16384]" = torch.ops.aten.mm.default(view_11, arg18_1)
        permute_2: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_11, [1, 0])
        mm_5: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_2, arg19_1);  permute_2 = arg19_1 = None
        sum_12: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_11, [0], True, dtype = torch.float32);  view_11 = None
        view_12: "f32[4096]" = torch.ops.aten.view.default(sum_12, _shape_param_12);  sum_12 = _shape_param_12 = None
        convert_element_type_19: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_12, torch.bfloat16);  view_12 = None
        view_13: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(mm_4, _shape_param_13);  mm_4 = _shape_param_13 = None
        convert_element_type_20: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        convert_element_type_21: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_22: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_19, torch.float32);  convert_element_type_19 = None
        view_14: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(arg20_1, _shape_param_14);  arg20_1 = _shape_param_14 = None
        mul_28: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_14, 0.5)
        mul_29: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_20, mul_28);  mul_28 = None
        convert_element_type_23: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_14, torch.float32)
        pow_3: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_23, 3.0)
        mul_30: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_5: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_14, mul_30);  view_14 = mul_30 = None
        mul_31: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_5, 0.7978845608028654);  add_5 = None
        tanh_1: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_31);  mul_31 = None
        add_6: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_32: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_20, add_6);  convert_element_type_20 = add_6 = None
        convert_element_type_24: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None
        mul_33: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_9: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_33);  mul_33 = None
        mul_34: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_29, sub_9);  mul_29 = sub_9 = None
        mul_35: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_34, 0.7978845608028654);  mul_34 = None
        convert_element_type_25: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16)
        mul_36: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_35, 0.044715);  mul_35 = None
        pow_4: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_23, 2.0);  convert_element_type_23 = None
        mul_37: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_4, 3.0);  pow_4 = None
        mul_38: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        convert_element_type_26: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_38, torch.bfloat16);  mul_38 = None
        add_7: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_25, convert_element_type_26);  convert_element_type_25 = convert_element_type_26 = None
        mul_39: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.5);  convert_element_type_24 = None
        add_8: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_7, mul_39);  add_7 = mul_39 = None
        view_15: "bf16[4096, 16384]" = torch.ops.aten.view.default(add_8, _shape_param_15);  add_8 = _shape_param_15 = None
        mm_6: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_15, arg21_1)
        permute_3: "bf16[16384, 4096]" = torch.ops.aten.permute.default(view_15, [1, 0])
        mm_7: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_3, arg22_1);  permute_3 = arg22_1 = None
        sum_13: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_15, [0], True, dtype = torch.float32);  view_15 = None
        view_16: "f32[16384]" = torch.ops.aten.view.default(sum_13, _shape_param_16);  sum_13 = _shape_param_16 = None
        convert_element_type_27: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_16, torch.bfloat16);  view_16 = None
        view_17: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_6, _shape_param_17);  mm_6 = _shape_param_17 = None
        convert_element_type_28: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None
        add_9: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_26, convert_element_type_28);  mul_26 = convert_element_type_28 = None
        convert_element_type_29: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_30: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_27, torch.float32);  convert_element_type_27 = None
        mul_40: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_9, arg23_1)
        mul_41: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_40, 4096)
        sum_14: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_40, [2], True)
        mul_42: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_40, arg24_1);  mul_40 = None
        sum_15: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_42, [2], True);  mul_42 = None
        mul_43: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg24_1, sum_15);  sum_15 = None
        sub_10: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_41, sum_14);  mul_41 = sum_14 = None
        sub_11: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_10, mul_43);  sub_10 = mul_43 = None
        mul_44: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg25_1, sub_11);  arg25_1 = sub_11 = None
        mul_45: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_9, arg24_1);  arg24_1 = None
        sum_16: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_45, [0, 1]);  mul_45 = None
        sum_17: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_9, [0, 1]);  add_9 = None
        convert_element_type_31: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_44, torch.bfloat16)
        view_18: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_31, _shape_param_18);  convert_element_type_31 = _shape_param_18 = None
        mm_8: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_18, arg26_1)
        permute_4: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_18, [1, 0])
        mm_9: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_4, arg27_1);  permute_4 = arg27_1 = None
        sum_18: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_18, [0], True, dtype = torch.float32);  view_18 = None
        view_19: "f32[4096]" = torch.ops.aten.view.default(sum_18, _shape_param_19);  sum_18 = _shape_param_19 = None
        convert_element_type_32: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_19, torch.bfloat16);  view_19 = None
        view_20: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_8, _shape_param_20);  mm_8 = _shape_param_20 = None
        convert_element_type_33: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_34: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_32, torch.float32);  convert_element_type_32 = None
        view_21: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_20, _shape_param_21);  view_20 = _shape_param_21 = None
        permute_5: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        clone: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_22: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_22);  clone = _shape_param_22 = None
        expand_1: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(arg28_1, _shape_param_23);  _shape_param_23 = None
        view_23: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_1, _shape_param_24);  expand_1 = _shape_param_24 = None
        permute_6: "bf16[512, 512, 512]" = torch.ops.aten.permute.default(view_23, [0, 2, 1]);  view_23 = None
        bmm: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(permute_6, view_22);  permute_6 = None
        bmm_1: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_22, arg29_1);  view_22 = arg29_1 = None
        view_24: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm, _shape_param_25);  bmm = _shape_param_25 = None
        view_25: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_1, _shape_param_26);  bmm_1 = _shape_param_26 = None
        convert_element_type_35: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_25, torch.float32);  view_25 = None
        convert_element_type_36: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(arg28_1, torch.float32);  arg28_1 = None
        mul_46: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_35, convert_element_type_36);  convert_element_type_35 = None
        sum_19: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_46, [-1], True)
        neg: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_36);  convert_element_type_36 = None
        fma: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg, sum_19, mul_46);  neg = sum_19 = mul_46 = None
        convert_element_type_37: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_26: "bf16[512, 512, 512]" = torch.ops.aten.view.default(convert_element_type_37, _shape_param_27);  convert_element_type_37 = _shape_param_27 = None
        bmm_2: "bf16[512, 64, 512]" = torch.ops.aten.bmm.default(arg30_1, view_26);  arg30_1 = None
        bmm_3: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_26, arg31_1);  view_26 = arg31_1 = None
        view_27: "bf16[8, 64, 64, 512]" = torch.ops.aten.view.default(bmm_2, _shape_param_28);  bmm_2 = _shape_param_28 = None
        view_28: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_29);  bmm_3 = _shape_param_29 = None
        mul_47: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_27, 0.3535533905932738);  view_27 = None
        permute_7: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_47, [0, 1, 3, 2]);  mul_47 = None
        mul_48: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_28, 0.3535533905932738);  view_28 = None
        permute_8: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        clone_1: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_29: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_1, _shape_param_30);  clone_1 = _shape_param_30 = None
        view_30: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_29, _shape_param_31);  view_29 = _shape_param_31 = None
        mm_10: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_30, arg32_1)
        permute_9: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_30, [1, 0])
        mm_11: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_9, arg33_1);  permute_9 = None
        sum_20: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_30, [0], True, dtype = torch.float32);  view_30 = None
        view_31: "f32[4096]" = torch.ops.aten.view.default(sum_20, _shape_param_32);  sum_20 = _shape_param_32 = None
        convert_element_type_38: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_31, torch.bfloat16);  view_31 = None
        view_32: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_10, _shape_param_33);  mm_10 = _shape_param_33 = None
        convert_element_type_39: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_32, torch.float32);  view_32 = None
        add_10: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_44, convert_element_type_39);  mul_44 = convert_element_type_39 = None
        convert_element_type_40: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_41: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_38, torch.float32);  convert_element_type_38 = None
        permute_10: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_7, [0, 2, 1, 3]);  permute_7 = None
        view_33: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(permute_10, _shape_param_34);  permute_10 = _shape_param_34 = None
        clone_2: "bf16[8, 512, 4096]" = torch.ops.aten.clone.default(view_33, memory_format = torch.contiguous_format);  view_33 = None
        view_34: "bf16[4096, 4096]" = torch.ops.aten.view.default(clone_2, _shape_param_35);  clone_2 = _shape_param_35 = None
        mm_12: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_34, arg34_1)
        permute_11: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_34, [1, 0])
        mm_13: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_11, arg33_1);  permute_11 = None
        sum_21: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_34, [0], True, dtype = torch.float32);  view_34 = None
        view_35: "f32[4096]" = torch.ops.aten.view.default(sum_21, _shape_param_36);  sum_21 = _shape_param_36 = None
        convert_element_type_42: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_35, torch.bfloat16);  view_35 = None
        view_36: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_12, _shape_param_37);  mm_12 = _shape_param_37 = None
        convert_element_type_43: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_36, torch.float32);  view_36 = None
        add_11: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_10, convert_element_type_43);  add_10 = convert_element_type_43 = None
        convert_element_type_44: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_45: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_42, torch.float32);  convert_element_type_42 = None
        permute_12: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_48, [0, 2, 1, 3]);  mul_48 = None
        clone_3: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
        view_37: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_3, _shape_param_38);  clone_3 = _shape_param_38 = None
        view_38: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_37, _shape_param_39);  view_37 = _shape_param_39 = None
        mm_14: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_38, arg35_1)
        permute_13: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_38, [1, 0])
        mm_15: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_13, arg33_1);  permute_13 = arg33_1 = None
        sum_22: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_38, [0], True, dtype = torch.float32);  view_38 = None
        view_39: "f32[4096]" = torch.ops.aten.view.default(sum_22, _shape_param_40);  sum_22 = _shape_param_40 = None
        convert_element_type_46: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_39, torch.bfloat16);  view_39 = None
        view_40: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_14, _shape_param_41);  mm_14 = _shape_param_41 = None
        convert_element_type_47: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_40, torch.float32);  view_40 = None
        add_12: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_11, convert_element_type_47);  add_11 = convert_element_type_47 = None
        convert_element_type_48: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_49: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_46, torch.float32);  convert_element_type_46 = None
        mul_49: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_12, arg15_1)
        mul_50: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_49, 4096)
        sum_23: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_49, [2], True)
        mul_51: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_49, arg36_1);  mul_49 = None
        sum_24: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_51, [2], True);  mul_51 = None
        mul_52: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg36_1, sum_24);  sum_24 = None
        sub_12: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_50, sum_23);  mul_50 = sum_23 = None
        sub_13: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_12, mul_52);  sub_12 = mul_52 = None
        mul_53: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg37_1, sub_13);  arg37_1 = sub_13 = None
        mul_54: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_12, arg36_1);  arg36_1 = None
        sum_25: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_54, [0, 1]);  mul_54 = None
        sum_26: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_12, [0, 1]);  add_12 = None
        add_13: "f32[4096]" = torch.ops.aten.add.Tensor(sum_10, sum_25);  sum_10 = sum_25 = None
        add_14: "f32[4096]" = torch.ops.aten.add.Tensor(sum_11, sum_26);  sum_11 = sum_26 = None
        convert_element_type_50: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16)
        view_41: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_50, _shape_param_42);  convert_element_type_50 = _shape_param_42 = None
        mm_16: "bf16[4096, 16384]" = torch.ops.aten.mm.default(view_41, arg18_1)
        permute_14: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_41, [1, 0])
        mm_17: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_14, arg38_1);  permute_14 = arg38_1 = None
        sum_27: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_41, [0], True, dtype = torch.float32);  view_41 = None
        view_42: "f32[4096]" = torch.ops.aten.view.default(sum_27, _shape_param_43);  sum_27 = _shape_param_43 = None
        convert_element_type_51: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_42, torch.bfloat16);  view_42 = None
        view_43: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(mm_16, _shape_param_44);  mm_16 = _shape_param_44 = None
        convert_element_type_52: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_43, torch.float32);  view_43 = None
        convert_element_type_53: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        add_15: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_21, convert_element_type_53);  convert_element_type_21 = convert_element_type_53 = None
        convert_element_type_54: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_51, torch.float32);  convert_element_type_51 = None
        add_16: "f32[4096]" = torch.ops.aten.add.Tensor(convert_element_type_22, convert_element_type_54);  convert_element_type_22 = convert_element_type_54 = None
        view_44: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(arg39_1, _shape_param_45);  arg39_1 = _shape_param_45 = None
        mul_55: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_44, 0.5)
        mul_56: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_52, mul_55);  mul_55 = None
        convert_element_type_55: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_44, torch.float32)
        pow_5: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_55, 3.0)
        mul_57: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_17: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_44, mul_57);  view_44 = mul_57 = None
        mul_58: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_2: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_18: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_59: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_52, add_18);  convert_element_type_52 = add_18 = None
        convert_element_type_56: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None
        mul_60: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_14: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_60);  mul_60 = None
        mul_61: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_56, sub_14);  mul_56 = sub_14 = None
        mul_62: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_61, 0.7978845608028654);  mul_61 = None
        convert_element_type_57: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16)
        mul_63: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_62, 0.044715);  mul_62 = None
        pow_6: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_55, 2.0);  convert_element_type_55 = None
        mul_64: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_6, 3.0);  pow_6 = None
        mul_65: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None
        convert_element_type_58: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None
        add_19: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_57, convert_element_type_58);  convert_element_type_57 = convert_element_type_58 = None
        mul_66: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_56, 0.5);  convert_element_type_56 = None
        add_20: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_19, mul_66);  add_19 = mul_66 = None
        view_45: "bf16[4096, 16384]" = torch.ops.aten.view.default(add_20, _shape_param_46);  add_20 = _shape_param_46 = None
        mm_18: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_45, arg21_1)
        permute_15: "bf16[16384, 4096]" = torch.ops.aten.permute.default(view_45, [1, 0])
        mm_19: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_15, arg40_1);  permute_15 = arg40_1 = None
        sum_28: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_45, [0], True, dtype = torch.float32);  view_45 = None
        view_46: "f32[16384]" = torch.ops.aten.view.default(sum_28, _shape_param_47);  sum_28 = _shape_param_47 = None
        convert_element_type_59: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_46, torch.bfloat16);  view_46 = None
        view_47: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_18, _shape_param_48);  mm_18 = _shape_param_48 = None
        convert_element_type_60: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_47, torch.float32);  view_47 = None
        add_21: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_53, convert_element_type_60);  mul_53 = convert_element_type_60 = None
        convert_element_type_61: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        add_22: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_29, convert_element_type_61);  convert_element_type_29 = convert_element_type_61 = None
        convert_element_type_62: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_59, torch.float32);  convert_element_type_59 = None
        add_23: "f32[16384]" = torch.ops.aten.add.Tensor(convert_element_type_30, convert_element_type_62);  convert_element_type_30 = convert_element_type_62 = None
        mul_67: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_21, arg23_1)
        mul_68: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_67, 4096)
        sum_29: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_67, [2], True)
        mul_69: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_67, arg41_1);  mul_67 = None
        sum_30: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_69, [2], True);  mul_69 = None
        mul_70: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg41_1, sum_30);  sum_30 = None
        sub_15: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_68, sum_29);  mul_68 = sum_29 = None
        sub_16: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_15, mul_70);  sub_15 = mul_70 = None
        mul_71: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg42_1, sub_16);  arg42_1 = sub_16 = None
        mul_72: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_21, arg41_1);  arg41_1 = None
        sum_31: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_72, [0, 1]);  mul_72 = None
        sum_32: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_21, [0, 1]);  add_21 = None
        add_24: "f32[4096]" = torch.ops.aten.add.Tensor(sum_16, sum_31);  sum_16 = sum_31 = None
        add_25: "f32[4096]" = torch.ops.aten.add.Tensor(sum_17, sum_32);  sum_17 = sum_32 = None
        convert_element_type_63: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_71, torch.bfloat16)
        view_48: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_63, _shape_param_49);  convert_element_type_63 = _shape_param_49 = None
        mm_20: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_48, arg26_1)
        permute_16: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_48, [1, 0])
        mm_21: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_16, arg43_1);  permute_16 = arg43_1 = None
        sum_33: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_48, [0], True, dtype = torch.float32);  view_48 = None
        view_49: "f32[4096]" = torch.ops.aten.view.default(sum_33, _shape_param_50);  sum_33 = _shape_param_50 = None
        convert_element_type_64: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_49, torch.bfloat16);  view_49 = None
        view_50: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_20, _shape_param_51);  mm_20 = _shape_param_51 = None
        convert_element_type_65: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        add_26: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_33, convert_element_type_65);  convert_element_type_33 = convert_element_type_65 = None
        convert_element_type_66: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_64, torch.float32);  convert_element_type_64 = None
        add_27: "f32[4096]" = torch.ops.aten.add.Tensor(convert_element_type_34, convert_element_type_66);  convert_element_type_34 = convert_element_type_66 = None
        view_51: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_50, _shape_param_52);  view_50 = _shape_param_52 = None
        permute_17: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        clone_4: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
        view_52: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_53);  clone_4 = _shape_param_53 = None
        expand_2: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(arg44_1, _shape_param_54);  _shape_param_54 = None
        view_53: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_2, _shape_param_55);  expand_2 = _shape_param_55 = None
        permute_18: "bf16[512, 512, 512]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        bmm_4: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(permute_18, view_52);  permute_18 = None
        bmm_5: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_52, arg45_1);  view_52 = arg45_1 = None
        view_54: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_4, _shape_param_56);  bmm_4 = _shape_param_56 = None
        view_55: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_5, _shape_param_57);  bmm_5 = _shape_param_57 = None
        convert_element_type_67: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_55, torch.float32);  view_55 = None
        convert_element_type_68: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(arg44_1, torch.float32);  arg44_1 = None
        mul_73: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_67, convert_element_type_68);  convert_element_type_67 = None
        sum_34: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_73, [-1], True)
        neg_1: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_68);  convert_element_type_68 = None
        fma_1: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_1, sum_34, mul_73);  neg_1 = sum_34 = mul_73 = None
        convert_element_type_69: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None
        view_56: "bf16[512, 512, 512]" = torch.ops.aten.view.default(convert_element_type_69, _shape_param_58);  convert_element_type_69 = _shape_param_58 = None
        bmm_6: "bf16[512, 64, 512]" = torch.ops.aten.bmm.default(arg46_1, view_56);  arg46_1 = None
        bmm_7: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_56, arg47_1);  view_56 = arg47_1 = None
        view_57: "bf16[8, 64, 64, 512]" = torch.ops.aten.view.default(bmm_6, _shape_param_59);  bmm_6 = _shape_param_59 = None
        view_58: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_60);  bmm_7 = _shape_param_60 = None
        mul_74: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_57, 0.3535533905932738);  view_57 = None
        permute_19: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_74, [0, 1, 3, 2]);  mul_74 = None
        mul_75: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_58, 0.3535533905932738);  view_58 = None
        permute_20: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        clone_5: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
        view_59: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_5, _shape_param_61);  clone_5 = _shape_param_61 = None
        view_60: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_59, _shape_param_62);  view_59 = _shape_param_62 = None
        mm_22: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_60, arg32_1)
        permute_21: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_60, [1, 0])
        mm_23: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_21, arg48_1);  permute_21 = None
        sum_35: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_60, [0], True, dtype = torch.float32);  view_60 = None
        view_61: "f32[4096]" = torch.ops.aten.view.default(sum_35, _shape_param_63);  sum_35 = _shape_param_63 = None
        convert_element_type_70: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_61, torch.bfloat16);  view_61 = None
        view_62: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_22, _shape_param_64);  mm_22 = _shape_param_64 = None
        convert_element_type_71: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_62, torch.float32);  view_62 = None
        add_28: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_71, convert_element_type_71);  mul_71 = convert_element_type_71 = None
        convert_element_type_72: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        add_29: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_40, convert_element_type_72);  convert_element_type_40 = convert_element_type_72 = None
        convert_element_type_73: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_70, torch.float32);  convert_element_type_70 = None
        add_30: "f32[4096]" = torch.ops.aten.add.Tensor(convert_element_type_41, convert_element_type_73);  convert_element_type_41 = convert_element_type_73 = None
        permute_22: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_19, [0, 2, 1, 3]);  permute_19 = None
        view_63: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(permute_22, _shape_param_65);  permute_22 = _shape_param_65 = None
        clone_6: "bf16[8, 512, 4096]" = torch.ops.aten.clone.default(view_63, memory_format = torch.contiguous_format);  view_63 = None
        view_64: "bf16[4096, 4096]" = torch.ops.aten.view.default(clone_6, _shape_param_66);  clone_6 = _shape_param_66 = None
        mm_24: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_64, arg34_1)
        permute_23: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_64, [1, 0])
        mm_25: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_23, arg48_1);  permute_23 = None
        sum_36: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_64, [0], True, dtype = torch.float32);  view_64 = None
        view_65: "f32[4096]" = torch.ops.aten.view.default(sum_36, _shape_param_67);  sum_36 = _shape_param_67 = None
        convert_element_type_74: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_65, torch.bfloat16);  view_65 = None
        view_66: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_24, _shape_param_68);  mm_24 = _shape_param_68 = None
        convert_element_type_75: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_66, torch.float32);  view_66 = None
        add_31: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_28, convert_element_type_75);  add_28 = convert_element_type_75 = None
        convert_element_type_76: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        add_32: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_44, convert_element_type_76);  convert_element_type_44 = convert_element_type_76 = None
        convert_element_type_77: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_74, torch.float32);  convert_element_type_74 = None
        add_33: "f32[4096]" = torch.ops.aten.add.Tensor(convert_element_type_45, convert_element_type_77);  convert_element_type_45 = convert_element_type_77 = None
        permute_24: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_75, [0, 2, 1, 3]);  mul_75 = None
        clone_7: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        view_67: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_7, _shape_param_69);  clone_7 = _shape_param_69 = None
        view_68: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_67, _shape_param_70);  view_67 = _shape_param_70 = None
        mm_26: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_68, arg35_1)
        permute_25: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_68, [1, 0])
        mm_27: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_25, arg48_1);  permute_25 = arg48_1 = None
        sum_37: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_68, [0], True, dtype = torch.float32);  view_68 = None
        view_69: "f32[4096]" = torch.ops.aten.view.default(sum_37, _shape_param_71);  sum_37 = _shape_param_71 = None
        convert_element_type_78: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_69, torch.bfloat16);  view_69 = None
        view_70: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_26, _shape_param_72);  mm_26 = _shape_param_72 = None
        convert_element_type_79: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_70, torch.float32);  view_70 = None
        add_34: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_31, convert_element_type_79);  add_31 = convert_element_type_79 = None
        convert_element_type_80: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        add_35: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_48, convert_element_type_80);  convert_element_type_48 = convert_element_type_80 = None
        convert_element_type_81: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_78, torch.float32);  convert_element_type_78 = None
        add_36: "f32[4096]" = torch.ops.aten.add.Tensor(convert_element_type_49, convert_element_type_81);  convert_element_type_49 = convert_element_type_81 = None
        mul_76: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_34, arg15_1)
        mul_77: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_76, 4096)
        sum_38: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_76, [2], True)
        mul_78: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_76, arg49_1);  mul_76 = None
        sum_39: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_78, [2], True);  mul_78 = None
        mul_79: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg49_1, sum_39);  sum_39 = None
        sub_17: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_77, sum_38);  mul_77 = sum_38 = None
        sub_18: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_17, mul_79);  sub_17 = mul_79 = None
        mul_80: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg50_1, sub_18);  arg50_1 = sub_18 = None
        mul_81: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_34, arg49_1);  arg49_1 = None
        sum_40: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_81, [0, 1]);  mul_81 = None
        sum_41: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_34, [0, 1]);  add_34 = None
        add_37: "f32[4096]" = torch.ops.aten.add.Tensor(add_13, sum_40);  add_13 = sum_40 = None
        add_38: "f32[4096]" = torch.ops.aten.add.Tensor(add_14, sum_41);  add_14 = sum_41 = None
        convert_element_type_82: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_80, torch.bfloat16)
        view_71: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_82, _shape_param_73);  convert_element_type_82 = _shape_param_73 = None
        mm_28: "bf16[4096, 16384]" = torch.ops.aten.mm.default(view_71, arg18_1)
        permute_26: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_71, [1, 0])
        mm_29: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_26, arg51_1);  permute_26 = arg51_1 = None
        sum_42: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_71, [0], True, dtype = torch.float32);  view_71 = None
        view_72: "f32[4096]" = torch.ops.aten.view.default(sum_42, _shape_param_74);  sum_42 = _shape_param_74 = None
        convert_element_type_83: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_72, torch.bfloat16);  view_72 = None
        view_73: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(mm_28, _shape_param_75);  mm_28 = _shape_param_75 = None
        convert_element_type_84: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_73, torch.float32);  view_73 = None
        convert_element_type_85: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        add_39: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_15, convert_element_type_85);  add_15 = convert_element_type_85 = None
        convert_element_type_86: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_83, torch.float32);  convert_element_type_83 = None
        add_40: "f32[4096]" = torch.ops.aten.add.Tensor(add_16, convert_element_type_86);  add_16 = convert_element_type_86 = None
        view_74: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(arg52_1, _shape_param_76);  arg52_1 = _shape_param_76 = None
        mul_82: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_74, 0.5)
        mul_83: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_84, mul_82);  mul_82 = None
        convert_element_type_87: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_74, torch.float32)
        pow_7: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_87, 3.0)
        mul_84: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_41: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_74, mul_84);  view_74 = mul_84 = None
        mul_85: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_41, 0.7978845608028654);  add_41 = None
        tanh_3: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_85);  mul_85 = None
        add_42: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_86: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_84, add_42);  convert_element_type_84 = add_42 = None
        convert_element_type_88: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_86, torch.bfloat16);  mul_86 = None
        mul_87: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_19: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_87);  mul_87 = None
        mul_88: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_83, sub_19);  mul_83 = sub_19 = None
        mul_89: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_88, 0.7978845608028654);  mul_88 = None
        convert_element_type_89: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16)
        mul_90: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_89, 0.044715);  mul_89 = None
        pow_8: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_87, 2.0);  convert_element_type_87 = None
        mul_91: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_8, 3.0);  pow_8 = None
        mul_92: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None
        convert_element_type_90: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_92, torch.bfloat16);  mul_92 = None
        add_43: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_89, convert_element_type_90);  convert_element_type_89 = convert_element_type_90 = None
        mul_93: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_88, 0.5);  convert_element_type_88 = None
        add_44: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_43, mul_93);  add_43 = mul_93 = None
        view_75: "bf16[4096, 16384]" = torch.ops.aten.view.default(add_44, _shape_param_77);  add_44 = _shape_param_77 = None
        mm_30: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_75, arg21_1)
        permute_27: "bf16[16384, 4096]" = torch.ops.aten.permute.default(view_75, [1, 0])
        mm_31: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_27, arg53_1);  permute_27 = arg53_1 = None
        sum_43: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_75, [0], True, dtype = torch.float32);  view_75 = None
        view_76: "f32[16384]" = torch.ops.aten.view.default(sum_43, _shape_param_78);  sum_43 = _shape_param_78 = None
        convert_element_type_91: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_76, torch.bfloat16);  view_76 = None
        view_77: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_30, _shape_param_79);  mm_30 = _shape_param_79 = None
        convert_element_type_92: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        add_45: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_80, convert_element_type_92);  mul_80 = convert_element_type_92 = None
        convert_element_type_93: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        add_46: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_22, convert_element_type_93);  add_22 = convert_element_type_93 = None
        convert_element_type_94: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_91, torch.float32);  convert_element_type_91 = None
        add_47: "f32[16384]" = torch.ops.aten.add.Tensor(add_23, convert_element_type_94);  add_23 = convert_element_type_94 = None
        mul_94: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_45, arg23_1)
        mul_95: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_94, 4096)
        sum_44: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_94, [2], True)
        mul_96: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_94, arg54_1);  mul_94 = None
        sum_45: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_96, [2], True);  mul_96 = None
        mul_97: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg54_1, sum_45);  sum_45 = None
        sub_20: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_95, sum_44);  mul_95 = sum_44 = None
        sub_21: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_20, mul_97);  sub_20 = mul_97 = None
        mul_98: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg55_1, sub_21);  arg55_1 = sub_21 = None
        mul_99: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_45, arg54_1);  arg54_1 = None
        sum_46: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_99, [0, 1]);  mul_99 = None
        sum_47: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_45, [0, 1]);  add_45 = None
        add_48: "f32[4096]" = torch.ops.aten.add.Tensor(add_24, sum_46);  add_24 = sum_46 = None
        add_49: "f32[4096]" = torch.ops.aten.add.Tensor(add_25, sum_47);  add_25 = sum_47 = None
        convert_element_type_95: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_98, torch.bfloat16)
        view_78: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_95, _shape_param_80);  convert_element_type_95 = _shape_param_80 = None
        mm_32: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_78, arg26_1)
        permute_28: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_78, [1, 0])
        mm_33: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_28, arg56_1);  permute_28 = arg56_1 = None
        sum_48: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_78, [0], True, dtype = torch.float32);  view_78 = None
        view_79: "f32[4096]" = torch.ops.aten.view.default(sum_48, _shape_param_81);  sum_48 = _shape_param_81 = None
        convert_element_type_96: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_79, torch.bfloat16);  view_79 = None
        view_80: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_32, _shape_param_82);  mm_32 = _shape_param_82 = None
        convert_element_type_97: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        add_50: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_26, convert_element_type_97);  add_26 = convert_element_type_97 = None
        convert_element_type_98: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_96, torch.float32);  convert_element_type_96 = None
        add_51: "f32[4096]" = torch.ops.aten.add.Tensor(add_27, convert_element_type_98);  add_27 = convert_element_type_98 = None
        view_81: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_80, _shape_param_83);  view_80 = _shape_param_83 = None
        permute_29: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None
        clone_8: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_82: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_84);  clone_8 = _shape_param_84 = None
        expand_3: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(arg57_1, _shape_param_85);  _shape_param_85 = None
        view_83: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_3, _shape_param_86);  expand_3 = _shape_param_86 = None
        permute_30: "bf16[512, 512, 512]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
        bmm_8: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(permute_30, view_82);  permute_30 = None
        bmm_9: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_82, arg58_1);  view_82 = arg58_1 = None
        view_84: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_8, _shape_param_87);  bmm_8 = _shape_param_87 = None
        view_85: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_9, _shape_param_88);  bmm_9 = _shape_param_88 = None
        convert_element_type_99: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        convert_element_type_100: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(arg57_1, torch.float32);  arg57_1 = None
        mul_100: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_99, convert_element_type_100);  convert_element_type_99 = None
        sum_49: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_100, [-1], True)
        neg_2: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_100);  convert_element_type_100 = None
        fma_2: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_2, sum_49, mul_100);  neg_2 = sum_49 = mul_100 = None
        convert_element_type_101: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None
        view_86: "bf16[512, 512, 512]" = torch.ops.aten.view.default(convert_element_type_101, _shape_param_89);  convert_element_type_101 = _shape_param_89 = None
        bmm_10: "bf16[512, 64, 512]" = torch.ops.aten.bmm.default(arg59_1, view_86);  arg59_1 = None
        bmm_11: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_86, arg60_1);  view_86 = arg60_1 = None
        view_87: "bf16[8, 64, 64, 512]" = torch.ops.aten.view.default(bmm_10, _shape_param_90);  bmm_10 = _shape_param_90 = None
        view_88: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_91);  bmm_11 = _shape_param_91 = None
        mul_101: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_87, 0.3535533905932738);  view_87 = None
        permute_31: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_101, [0, 1, 3, 2]);  mul_101 = None
        mul_102: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_88, 0.3535533905932738);  view_88 = None
        permute_32: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        clone_9: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_89: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_9, _shape_param_92);  clone_9 = _shape_param_92 = None
        view_90: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_89, _shape_param_93);  view_89 = _shape_param_93 = None
        mm_34: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_90, arg32_1)
        permute_33: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_90, [1, 0])
        mm_35: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_33, arg61_1);  permute_33 = None
        sum_50: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_90, [0], True, dtype = torch.float32);  view_90 = None
        view_91: "f32[4096]" = torch.ops.aten.view.default(sum_50, _shape_param_94);  sum_50 = _shape_param_94 = None
        convert_element_type_102: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_91, torch.bfloat16);  view_91 = None
        view_92: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_34, _shape_param_95);  mm_34 = _shape_param_95 = None
        convert_element_type_103: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_92, torch.float32);  view_92 = None
        add_52: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_98, convert_element_type_103);  mul_98 = convert_element_type_103 = None
        convert_element_type_104: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        add_53: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_29, convert_element_type_104);  add_29 = convert_element_type_104 = None
        convert_element_type_105: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_102, torch.float32);  convert_element_type_102 = None
        add_54: "f32[4096]" = torch.ops.aten.add.Tensor(add_30, convert_element_type_105);  add_30 = convert_element_type_105 = None
        permute_34: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_31, [0, 2, 1, 3]);  permute_31 = None
        view_93: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(permute_34, _shape_param_96);  permute_34 = _shape_param_96 = None
        clone_10: "bf16[8, 512, 4096]" = torch.ops.aten.clone.default(view_93, memory_format = torch.contiguous_format);  view_93 = None
        view_94: "bf16[4096, 4096]" = torch.ops.aten.view.default(clone_10, _shape_param_97);  clone_10 = _shape_param_97 = None
        mm_36: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_94, arg34_1)
        permute_35: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_94, [1, 0])
        mm_37: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_35, arg61_1);  permute_35 = None
        sum_51: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_94, [0], True, dtype = torch.float32);  view_94 = None
        view_95: "f32[4096]" = torch.ops.aten.view.default(sum_51, _shape_param_98);  sum_51 = _shape_param_98 = None
        convert_element_type_106: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_95, torch.bfloat16);  view_95 = None
        view_96: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_36, _shape_param_99);  mm_36 = _shape_param_99 = None
        convert_element_type_107: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_96, torch.float32);  view_96 = None
        add_55: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_52, convert_element_type_107);  add_52 = convert_element_type_107 = None
        convert_element_type_108: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        add_56: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_32, convert_element_type_108);  add_32 = convert_element_type_108 = None
        convert_element_type_109: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_106, torch.float32);  convert_element_type_106 = None
        add_57: "f32[4096]" = torch.ops.aten.add.Tensor(add_33, convert_element_type_109);  add_33 = convert_element_type_109 = None
        permute_36: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_102, [0, 2, 1, 3]);  mul_102 = None
        clone_11: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        view_97: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_11, _shape_param_100);  clone_11 = _shape_param_100 = None
        view_98: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_97, _shape_param_101);  view_97 = _shape_param_101 = None
        mm_38: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_98, arg35_1)
        permute_37: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_98, [1, 0])
        mm_39: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_37, arg61_1);  permute_37 = arg61_1 = None
        sum_52: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_98, [0], True, dtype = torch.float32);  view_98 = None
        view_99: "f32[4096]" = torch.ops.aten.view.default(sum_52, _shape_param_102);  sum_52 = _shape_param_102 = None
        convert_element_type_110: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_99, torch.bfloat16);  view_99 = None
        view_100: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_38, _shape_param_103);  mm_38 = _shape_param_103 = None
        convert_element_type_111: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_100, torch.float32);  view_100 = None
        add_58: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_55, convert_element_type_111);  add_55 = convert_element_type_111 = None
        convert_element_type_112: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        add_59: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_35, convert_element_type_112);  add_35 = convert_element_type_112 = None
        convert_element_type_113: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_110, torch.float32);  convert_element_type_110 = None
        add_60: "f32[4096]" = torch.ops.aten.add.Tensor(add_36, convert_element_type_113);  add_36 = convert_element_type_113 = None
        mul_103: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_58, arg15_1)
        mul_104: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_103, 4096)
        sum_53: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_103, [2], True)
        mul_105: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_103, arg62_1);  mul_103 = None
        sum_54: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_105, [2], True);  mul_105 = None
        mul_106: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg62_1, sum_54);  sum_54 = None
        sub_22: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_104, sum_53);  mul_104 = sum_53 = None
        sub_23: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_22, mul_106);  sub_22 = mul_106 = None
        mul_107: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg63_1, sub_23);  arg63_1 = sub_23 = None
        mul_108: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_58, arg62_1);  arg62_1 = None
        sum_55: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_108, [0, 1]);  mul_108 = None
        sum_56: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_58, [0, 1]);  add_58 = None
        add_61: "f32[4096]" = torch.ops.aten.add.Tensor(add_37, sum_55);  add_37 = sum_55 = None
        add_62: "f32[4096]" = torch.ops.aten.add.Tensor(add_38, sum_56);  add_38 = sum_56 = None
        convert_element_type_114: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_107, torch.bfloat16)
        view_101: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_114, _shape_param_104);  convert_element_type_114 = _shape_param_104 = None
        mm_40: "bf16[4096, 16384]" = torch.ops.aten.mm.default(view_101, arg18_1)
        permute_38: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_101, [1, 0])
        mm_41: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_38, arg64_1);  permute_38 = arg64_1 = None
        sum_57: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_101, [0], True, dtype = torch.float32);  view_101 = None
        view_102: "f32[4096]" = torch.ops.aten.view.default(sum_57, _shape_param_105);  sum_57 = _shape_param_105 = None
        convert_element_type_115: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_102, torch.bfloat16);  view_102 = None
        view_103: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(mm_40, _shape_param_106);  mm_40 = _shape_param_106 = None
        convert_element_type_116: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_103, torch.float32);  view_103 = None
        convert_element_type_117: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        add_63: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_39, convert_element_type_117);  add_39 = convert_element_type_117 = None
        convert_element_type_118: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_115, torch.float32);  convert_element_type_115 = None
        add_64: "f32[4096]" = torch.ops.aten.add.Tensor(add_40, convert_element_type_118);  add_40 = convert_element_type_118 = None
        view_104: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(arg65_1, _shape_param_107);  arg65_1 = _shape_param_107 = None
        mul_109: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_104, 0.5)
        mul_110: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_116, mul_109);  mul_109 = None
        convert_element_type_119: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_104, torch.float32)
        pow_9: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_119, 3.0)
        mul_111: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_65: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_104, mul_111);  view_104 = mul_111 = None
        mul_112: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_65, 0.7978845608028654);  add_65 = None
        tanh_4: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_112);  mul_112 = None
        add_66: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_113: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_116, add_66);  convert_element_type_116 = add_66 = None
        convert_element_type_120: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_113, torch.bfloat16);  mul_113 = None
        mul_114: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_24: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_114);  mul_114 = None
        mul_115: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_110, sub_24);  mul_110 = sub_24 = None
        mul_116: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_115, 0.7978845608028654);  mul_115 = None
        convert_element_type_121: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_116, torch.bfloat16)
        mul_117: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_116, 0.044715);  mul_116 = None
        pow_10: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_119, 2.0);  convert_element_type_119 = None
        mul_118: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_10, 3.0);  pow_10 = None
        mul_119: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
        convert_element_type_122: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_119, torch.bfloat16);  mul_119 = None
        add_67: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_121, convert_element_type_122);  convert_element_type_121 = convert_element_type_122 = None
        mul_120: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_120, 0.5);  convert_element_type_120 = None
        add_68: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_67, mul_120);  add_67 = mul_120 = None
        view_105: "bf16[4096, 16384]" = torch.ops.aten.view.default(add_68, _shape_param_108);  add_68 = _shape_param_108 = None
        mm_42: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_105, arg21_1)
        permute_39: "bf16[16384, 4096]" = torch.ops.aten.permute.default(view_105, [1, 0])
        mm_43: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_39, arg66_1);  permute_39 = arg66_1 = None
        sum_58: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_105, [0], True, dtype = torch.float32);  view_105 = None
        view_106: "f32[16384]" = torch.ops.aten.view.default(sum_58, _shape_param_109);  sum_58 = _shape_param_109 = None
        convert_element_type_123: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_106, torch.bfloat16);  view_106 = None
        view_107: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_42, _shape_param_110);  mm_42 = _shape_param_110 = None
        convert_element_type_124: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        add_69: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_107, convert_element_type_124);  mul_107 = convert_element_type_124 = None
        convert_element_type_125: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        add_70: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_46, convert_element_type_125);  add_46 = convert_element_type_125 = None
        convert_element_type_126: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_123, torch.float32);  convert_element_type_123 = None
        add_71: "f32[16384]" = torch.ops.aten.add.Tensor(add_47, convert_element_type_126);  add_47 = convert_element_type_126 = None
        mul_121: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_69, arg23_1)
        mul_122: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_121, 4096)
        sum_59: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [2], True)
        mul_123: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_121, arg67_1);  mul_121 = None
        sum_60: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True);  mul_123 = None
        mul_124: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg67_1, sum_60);  sum_60 = None
        sub_25: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_122, sum_59);  mul_122 = sum_59 = None
        sub_26: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_25, mul_124);  sub_25 = mul_124 = None
        mul_125: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg68_1, sub_26);  arg68_1 = sub_26 = None
        mul_126: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_69, arg67_1);  arg67_1 = None
        sum_61: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_126, [0, 1]);  mul_126 = None
        sum_62: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_69, [0, 1]);  add_69 = None
        add_72: "f32[4096]" = torch.ops.aten.add.Tensor(add_48, sum_61);  add_48 = sum_61 = None
        add_73: "f32[4096]" = torch.ops.aten.add.Tensor(add_49, sum_62);  add_49 = sum_62 = None
        convert_element_type_127: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_125, torch.bfloat16)
        view_108: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_127, _shape_param_111);  convert_element_type_127 = _shape_param_111 = None
        mm_44: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_108, arg26_1)
        permute_40: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_108, [1, 0])
        mm_45: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_40, arg69_1);  permute_40 = arg69_1 = None
        sum_63: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_108, [0], True, dtype = torch.float32);  view_108 = None
        view_109: "f32[4096]" = torch.ops.aten.view.default(sum_63, _shape_param_112);  sum_63 = _shape_param_112 = None
        convert_element_type_128: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_109, torch.bfloat16);  view_109 = None
        view_110: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_44, _shape_param_113);  mm_44 = _shape_param_113 = None
        convert_element_type_129: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        add_74: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_50, convert_element_type_129);  add_50 = convert_element_type_129 = None
        convert_element_type_130: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_128, torch.float32);  convert_element_type_128 = None
        add_75: "f32[4096]" = torch.ops.aten.add.Tensor(add_51, convert_element_type_130);  add_51 = convert_element_type_130 = None
        view_111: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_110, _shape_param_114);  view_110 = _shape_param_114 = None
        permute_41: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
        clone_12: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
        view_112: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_115);  clone_12 = _shape_param_115 = None
        expand_4: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(arg70_1, _shape_param_116);  _shape_param_116 = None
        view_113: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_4, _shape_param_117);  expand_4 = _shape_param_117 = None
        permute_42: "bf16[512, 512, 512]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
        bmm_12: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(permute_42, view_112);  permute_42 = None
        bmm_13: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_112, arg71_1);  view_112 = arg71_1 = None
        view_114: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_12, _shape_param_118);  bmm_12 = _shape_param_118 = None
        view_115: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_13, _shape_param_119);  bmm_13 = _shape_param_119 = None
        convert_element_type_131: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_115, torch.float32);  view_115 = None
        convert_element_type_132: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(arg70_1, torch.float32);  arg70_1 = None
        mul_127: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_131, convert_element_type_132);  convert_element_type_131 = None
        sum_64: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_127, [-1], True)
        neg_3: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_132);  convert_element_type_132 = None
        fma_3: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_3, sum_64, mul_127);  neg_3 = sum_64 = mul_127 = None
        convert_element_type_133: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None
        view_116: "bf16[512, 512, 512]" = torch.ops.aten.view.default(convert_element_type_133, _shape_param_120);  convert_element_type_133 = _shape_param_120 = None
        bmm_14: "bf16[512, 64, 512]" = torch.ops.aten.bmm.default(arg72_1, view_116);  arg72_1 = None
        bmm_15: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_116, arg73_1);  view_116 = arg73_1 = None
        view_117: "bf16[8, 64, 64, 512]" = torch.ops.aten.view.default(bmm_14, _shape_param_121);  bmm_14 = _shape_param_121 = None
        view_118: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_122);  bmm_15 = _shape_param_122 = None
        mul_128: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_117, 0.3535533905932738);  view_117 = None
        permute_43: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_128, [0, 1, 3, 2]);  mul_128 = None
        mul_129: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_118, 0.3535533905932738);  view_118 = None
        permute_44: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        clone_13: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_119: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_13, _shape_param_123);  clone_13 = _shape_param_123 = None
        view_120: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_119, _shape_param_124);  view_119 = _shape_param_124 = None
        mm_46: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_120, arg32_1)
        permute_45: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_120, [1, 0])
        mm_47: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_45, arg74_1);  permute_45 = None
        sum_65: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_120, [0], True, dtype = torch.float32);  view_120 = None
        view_121: "f32[4096]" = torch.ops.aten.view.default(sum_65, _shape_param_125);  sum_65 = _shape_param_125 = None
        convert_element_type_134: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_121, torch.bfloat16);  view_121 = None
        view_122: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_46, _shape_param_126);  mm_46 = _shape_param_126 = None
        convert_element_type_135: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_122, torch.float32);  view_122 = None
        add_76: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_125, convert_element_type_135);  mul_125 = convert_element_type_135 = None
        convert_element_type_136: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        add_77: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_53, convert_element_type_136);  add_53 = convert_element_type_136 = None
        convert_element_type_137: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_134, torch.float32);  convert_element_type_134 = None
        add_78: "f32[4096]" = torch.ops.aten.add.Tensor(add_54, convert_element_type_137);  add_54 = convert_element_type_137 = None
        permute_46: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_43, [0, 2, 1, 3]);  permute_43 = None
        view_123: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(permute_46, _shape_param_127);  permute_46 = _shape_param_127 = None
        clone_14: "bf16[8, 512, 4096]" = torch.ops.aten.clone.default(view_123, memory_format = torch.contiguous_format);  view_123 = None
        view_124: "bf16[4096, 4096]" = torch.ops.aten.view.default(clone_14, _shape_param_128);  clone_14 = _shape_param_128 = None
        mm_48: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_124, arg34_1)
        permute_47: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_124, [1, 0])
        mm_49: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_47, arg74_1);  permute_47 = None
        sum_66: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_124, [0], True, dtype = torch.float32);  view_124 = None
        view_125: "f32[4096]" = torch.ops.aten.view.default(sum_66, _shape_param_129);  sum_66 = _shape_param_129 = None
        convert_element_type_138: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_125, torch.bfloat16);  view_125 = None
        view_126: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_48, _shape_param_130);  mm_48 = _shape_param_130 = None
        convert_element_type_139: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_126, torch.float32);  view_126 = None
        add_79: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_76, convert_element_type_139);  add_76 = convert_element_type_139 = None
        convert_element_type_140: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        add_80: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_56, convert_element_type_140);  add_56 = convert_element_type_140 = None
        convert_element_type_141: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_138, torch.float32);  convert_element_type_138 = None
        add_81: "f32[4096]" = torch.ops.aten.add.Tensor(add_57, convert_element_type_141);  add_57 = convert_element_type_141 = None
        permute_48: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_129, [0, 2, 1, 3]);  mul_129 = None
        clone_15: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_127: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_15, _shape_param_131);  clone_15 = _shape_param_131 = None
        view_128: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_127, _shape_param_132);  view_127 = _shape_param_132 = None
        mm_50: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_128, arg35_1)
        permute_49: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_128, [1, 0])
        mm_51: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_49, arg74_1);  permute_49 = arg74_1 = None
        sum_67: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_128, [0], True, dtype = torch.float32);  view_128 = None
        view_129: "f32[4096]" = torch.ops.aten.view.default(sum_67, _shape_param_133);  sum_67 = _shape_param_133 = None
        convert_element_type_142: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_129, torch.bfloat16);  view_129 = None
        view_130: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_50, _shape_param_134);  mm_50 = _shape_param_134 = None
        convert_element_type_143: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_130, torch.float32);  view_130 = None
        add_82: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_79, convert_element_type_143);  add_79 = convert_element_type_143 = None
        convert_element_type_144: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        add_83: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_59, convert_element_type_144);  add_59 = convert_element_type_144 = None
        convert_element_type_145: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_142, torch.float32);  convert_element_type_142 = None
        add_84: "f32[4096]" = torch.ops.aten.add.Tensor(add_60, convert_element_type_145);  add_60 = convert_element_type_145 = None
        mul_130: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_82, arg15_1)
        mul_131: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_130, 4096)
        sum_68: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_130, [2], True)
        mul_132: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_130, arg75_1);  mul_130 = None
        sum_69: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_132, [2], True);  mul_132 = None
        mul_133: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg75_1, sum_69);  sum_69 = None
        sub_27: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_131, sum_68);  mul_131 = sum_68 = None
        sub_28: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_27, mul_133);  sub_27 = mul_133 = None
        mul_134: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg76_1, sub_28);  arg76_1 = sub_28 = None
        mul_135: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_82, arg75_1);  arg75_1 = None
        sum_70: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_135, [0, 1]);  mul_135 = None
        sum_71: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_82, [0, 1]);  add_82 = None
        add_85: "f32[4096]" = torch.ops.aten.add.Tensor(add_61, sum_70);  add_61 = sum_70 = None
        add_86: "f32[4096]" = torch.ops.aten.add.Tensor(add_62, sum_71);  add_62 = sum_71 = None
        convert_element_type_146: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_134, torch.bfloat16)
        view_131: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_146, _shape_param_135);  convert_element_type_146 = _shape_param_135 = None
        mm_52: "bf16[4096, 16384]" = torch.ops.aten.mm.default(view_131, arg18_1)
        permute_50: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_131, [1, 0])
        mm_53: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_50, arg77_1);  permute_50 = arg77_1 = None
        sum_72: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_131, [0], True, dtype = torch.float32);  view_131 = None
        view_132: "f32[4096]" = torch.ops.aten.view.default(sum_72, _shape_param_136);  sum_72 = _shape_param_136 = None
        convert_element_type_147: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_132, torch.bfloat16);  view_132 = None
        view_133: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(mm_52, _shape_param_137);  mm_52 = _shape_param_137 = None
        convert_element_type_148: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_133, torch.float32);  view_133 = None
        convert_element_type_149: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        add_87: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_63, convert_element_type_149);  add_63 = convert_element_type_149 = None
        convert_element_type_150: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_147, torch.float32);  convert_element_type_147 = None
        add_88: "f32[4096]" = torch.ops.aten.add.Tensor(add_64, convert_element_type_150);  add_64 = convert_element_type_150 = None
        view_134: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(arg78_1, _shape_param_138);  arg78_1 = _shape_param_138 = None
        mul_136: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_134, 0.5)
        mul_137: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_148, mul_136);  mul_136 = None
        convert_element_type_151: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_134, torch.float32)
        pow_11: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_151, 3.0)
        mul_138: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_89: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_134, mul_138);  view_134 = mul_138 = None
        mul_139: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
        tanh_5: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_139);  mul_139 = None
        add_90: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_140: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_148, add_90);  convert_element_type_148 = add_90 = None
        convert_element_type_152: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_140, torch.bfloat16);  mul_140 = None
        mul_141: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_29: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_141);  mul_141 = None
        mul_142: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_137, sub_29);  mul_137 = sub_29 = None
        mul_143: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_142, 0.7978845608028654);  mul_142 = None
        convert_element_type_153: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_143, torch.bfloat16)
        mul_144: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_143, 0.044715);  mul_143 = None
        pow_12: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_151, 2.0);  convert_element_type_151 = None
        mul_145: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_12, 3.0);  pow_12 = None
        mul_146: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        convert_element_type_154: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_146, torch.bfloat16);  mul_146 = None
        add_91: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_153, convert_element_type_154);  convert_element_type_153 = convert_element_type_154 = None
        mul_147: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_152, 0.5);  convert_element_type_152 = None
        add_92: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_91, mul_147);  add_91 = mul_147 = None
        view_135: "bf16[4096, 16384]" = torch.ops.aten.view.default(add_92, _shape_param_139);  add_92 = _shape_param_139 = None
        mm_54: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_135, arg21_1)
        permute_51: "bf16[16384, 4096]" = torch.ops.aten.permute.default(view_135, [1, 0])
        mm_55: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_51, arg79_1);  permute_51 = arg79_1 = None
        sum_73: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_135, [0], True, dtype = torch.float32);  view_135 = None
        view_136: "f32[16384]" = torch.ops.aten.view.default(sum_73, _shape_param_140);  sum_73 = _shape_param_140 = None
        convert_element_type_155: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_136, torch.bfloat16);  view_136 = None
        view_137: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_54, _shape_param_141);  mm_54 = _shape_param_141 = None
        convert_element_type_156: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_137, torch.float32);  view_137 = None
        add_93: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_134, convert_element_type_156);  mul_134 = convert_element_type_156 = None
        convert_element_type_157: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        add_94: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_70, convert_element_type_157);  add_70 = convert_element_type_157 = None
        convert_element_type_158: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_155, torch.float32);  convert_element_type_155 = None
        add_95: "f32[16384]" = torch.ops.aten.add.Tensor(add_71, convert_element_type_158);  add_71 = convert_element_type_158 = None
        mul_148: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_93, arg23_1)
        mul_149: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_148, 4096)
        sum_74: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_148, [2], True)
        mul_150: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_148, arg80_1);  mul_148 = None
        sum_75: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True);  mul_150 = None
        mul_151: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg80_1, sum_75);  sum_75 = None
        sub_30: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_149, sum_74);  mul_149 = sum_74 = None
        sub_31: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_30, mul_151);  sub_30 = mul_151 = None
        mul_152: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg81_1, sub_31);  arg81_1 = sub_31 = None
        mul_153: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_93, arg80_1);  arg80_1 = None
        sum_76: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 1]);  mul_153 = None
        sum_77: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_93, [0, 1]);  add_93 = None
        add_96: "f32[4096]" = torch.ops.aten.add.Tensor(add_72, sum_76);  add_72 = sum_76 = None
        add_97: "f32[4096]" = torch.ops.aten.add.Tensor(add_73, sum_77);  add_73 = sum_77 = None
        convert_element_type_159: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16)
        view_138: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_159, _shape_param_142);  convert_element_type_159 = _shape_param_142 = None
        mm_56: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_138, arg26_1)
        permute_52: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_138, [1, 0])
        mm_57: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_52, arg82_1);  permute_52 = arg82_1 = None
        sum_78: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_138, [0], True, dtype = torch.float32);  view_138 = None
        view_139: "f32[4096]" = torch.ops.aten.view.default(sum_78, _shape_param_143);  sum_78 = _shape_param_143 = None
        convert_element_type_160: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_139, torch.bfloat16);  view_139 = None
        view_140: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_56, _shape_param_144);  mm_56 = _shape_param_144 = None
        convert_element_type_161: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        add_98: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_74, convert_element_type_161);  add_74 = convert_element_type_161 = None
        convert_element_type_162: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_160, torch.float32);  convert_element_type_160 = None
        add_99: "f32[4096]" = torch.ops.aten.add.Tensor(add_75, convert_element_type_162);  add_75 = convert_element_type_162 = None
        view_141: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_140, _shape_param_145);  view_140 = _shape_param_145 = None
        permute_53: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
        clone_16: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
        view_142: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_146);  clone_16 = _shape_param_146 = None
        expand_5: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(arg83_1, _shape_param_147);  _shape_param_147 = None
        view_143: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_5, _shape_param_148);  expand_5 = _shape_param_148 = None
        permute_54: "bf16[512, 512, 512]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
        bmm_16: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(permute_54, view_142);  permute_54 = None
        bmm_17: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_142, arg84_1);  view_142 = arg84_1 = None
        view_144: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_16, _shape_param_149);  bmm_16 = _shape_param_149 = None
        view_145: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_17, _shape_param_150);  bmm_17 = _shape_param_150 = None
        convert_element_type_163: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_145, torch.float32);  view_145 = None
        convert_element_type_164: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(arg83_1, torch.float32);  arg83_1 = None
        mul_154: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_163, convert_element_type_164);  convert_element_type_163 = None
        sum_79: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_154, [-1], True)
        neg_4: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_164);  convert_element_type_164 = None
        fma_4: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_4, sum_79, mul_154);  neg_4 = sum_79 = mul_154 = None
        convert_element_type_165: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None
        view_146: "bf16[512, 512, 512]" = torch.ops.aten.view.default(convert_element_type_165, _shape_param_151);  convert_element_type_165 = _shape_param_151 = None
        bmm_18: "bf16[512, 64, 512]" = torch.ops.aten.bmm.default(arg85_1, view_146);  arg85_1 = None
        bmm_19: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_146, arg86_1);  view_146 = arg86_1 = None
        view_147: "bf16[8, 64, 64, 512]" = torch.ops.aten.view.default(bmm_18, _shape_param_152);  bmm_18 = _shape_param_152 = None
        view_148: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_153);  bmm_19 = _shape_param_153 = None
        mul_155: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_147, 0.3535533905932738);  view_147 = None
        permute_55: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_155, [0, 1, 3, 2]);  mul_155 = None
        mul_156: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_148, 0.3535533905932738);  view_148 = None
        permute_56: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        clone_17: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_149: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_17, _shape_param_154);  clone_17 = _shape_param_154 = None
        view_150: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_149, _shape_param_155);  view_149 = _shape_param_155 = None
        mm_58: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_150, arg32_1)
        permute_57: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_150, [1, 0])
        mm_59: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_57, arg87_1);  permute_57 = None
        sum_80: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_150, [0], True, dtype = torch.float32);  view_150 = None
        view_151: "f32[4096]" = torch.ops.aten.view.default(sum_80, _shape_param_156);  sum_80 = _shape_param_156 = None
        convert_element_type_166: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_151, torch.bfloat16);  view_151 = None
        view_152: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_58, _shape_param_157);  mm_58 = _shape_param_157 = None
        convert_element_type_167: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_152, torch.float32);  view_152 = None
        add_100: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_152, convert_element_type_167);  mul_152 = convert_element_type_167 = None
        convert_element_type_168: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        add_101: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_77, convert_element_type_168);  add_77 = convert_element_type_168 = None
        convert_element_type_169: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_166, torch.float32);  convert_element_type_166 = None
        add_102: "f32[4096]" = torch.ops.aten.add.Tensor(add_78, convert_element_type_169);  add_78 = convert_element_type_169 = None
        permute_58: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_55, [0, 2, 1, 3]);  permute_55 = None
        view_153: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(permute_58, _shape_param_158);  permute_58 = _shape_param_158 = None
        clone_18: "bf16[8, 512, 4096]" = torch.ops.aten.clone.default(view_153, memory_format = torch.contiguous_format);  view_153 = None
        view_154: "bf16[4096, 4096]" = torch.ops.aten.view.default(clone_18, _shape_param_159);  clone_18 = _shape_param_159 = None
        mm_60: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_154, arg34_1)
        permute_59: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_154, [1, 0])
        mm_61: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_59, arg87_1);  permute_59 = None
        sum_81: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_154, [0], True, dtype = torch.float32);  view_154 = None
        view_155: "f32[4096]" = torch.ops.aten.view.default(sum_81, _shape_param_160);  sum_81 = _shape_param_160 = None
        convert_element_type_170: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_155, torch.bfloat16);  view_155 = None
        view_156: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_60, _shape_param_161);  mm_60 = _shape_param_161 = None
        convert_element_type_171: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_156, torch.float32);  view_156 = None
        add_103: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_100, convert_element_type_171);  add_100 = convert_element_type_171 = None
        convert_element_type_172: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        add_104: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_80, convert_element_type_172);  add_80 = convert_element_type_172 = None
        convert_element_type_173: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_170, torch.float32);  convert_element_type_170 = None
        add_105: "f32[4096]" = torch.ops.aten.add.Tensor(add_81, convert_element_type_173);  add_81 = convert_element_type_173 = None
        permute_60: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_156, [0, 2, 1, 3]);  mul_156 = None
        clone_19: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_157: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_19, _shape_param_162);  clone_19 = _shape_param_162 = None
        view_158: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_157, _shape_param_163);  view_157 = _shape_param_163 = None
        mm_62: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_158, arg35_1)
        permute_61: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_158, [1, 0])
        mm_63: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_61, arg87_1);  permute_61 = arg87_1 = None
        sum_82: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_158, [0], True, dtype = torch.float32);  view_158 = None
        view_159: "f32[4096]" = torch.ops.aten.view.default(sum_82, _shape_param_164);  sum_82 = _shape_param_164 = None
        convert_element_type_174: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_159, torch.bfloat16);  view_159 = None
        view_160: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_62, _shape_param_165);  mm_62 = _shape_param_165 = None
        convert_element_type_175: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_160, torch.float32);  view_160 = None
        add_106: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_103, convert_element_type_175);  add_103 = convert_element_type_175 = None
        convert_element_type_176: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        add_107: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_83, convert_element_type_176);  add_83 = convert_element_type_176 = None
        convert_element_type_177: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_174, torch.float32);  convert_element_type_174 = None
        add_108: "f32[4096]" = torch.ops.aten.add.Tensor(add_84, convert_element_type_177);  add_84 = convert_element_type_177 = None
        mul_157: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_106, arg15_1)
        mul_158: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_157, 4096)
        sum_83: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_157, [2], True)
        mul_159: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_157, arg88_1);  mul_157 = None
        sum_84: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_159, [2], True);  mul_159 = None
        mul_160: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg88_1, sum_84);  sum_84 = None
        sub_32: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_158, sum_83);  mul_158 = sum_83 = None
        sub_33: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_32, mul_160);  sub_32 = mul_160 = None
        mul_161: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg89_1, sub_33);  arg89_1 = sub_33 = None
        mul_162: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_106, arg88_1);  arg88_1 = None
        sum_85: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_162, [0, 1]);  mul_162 = None
        sum_86: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_106, [0, 1]);  add_106 = None
        add_109: "f32[4096]" = torch.ops.aten.add.Tensor(add_85, sum_85);  add_85 = sum_85 = None
        add_110: "f32[4096]" = torch.ops.aten.add.Tensor(add_86, sum_86);  add_86 = sum_86 = None
        convert_element_type_178: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_161, torch.bfloat16)
        view_161: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_178, _shape_param_166);  convert_element_type_178 = _shape_param_166 = None
        mm_64: "bf16[4096, 16384]" = torch.ops.aten.mm.default(view_161, arg18_1)
        permute_62: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_161, [1, 0])
        mm_65: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_62, arg90_1);  permute_62 = arg90_1 = None
        sum_87: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_161, [0], True, dtype = torch.float32);  view_161 = None
        view_162: "f32[4096]" = torch.ops.aten.view.default(sum_87, _shape_param_167);  sum_87 = _shape_param_167 = None
        convert_element_type_179: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_162, torch.bfloat16);  view_162 = None
        view_163: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(mm_64, _shape_param_168);  mm_64 = _shape_param_168 = None
        convert_element_type_180: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_163, torch.float32);  view_163 = None
        convert_element_type_181: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        add_111: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_87, convert_element_type_181);  add_87 = convert_element_type_181 = None
        convert_element_type_182: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_179, torch.float32);  convert_element_type_179 = None
        add_112: "f32[4096]" = torch.ops.aten.add.Tensor(add_88, convert_element_type_182);  add_88 = convert_element_type_182 = None
        view_164: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(arg91_1, _shape_param_169);  arg91_1 = _shape_param_169 = None
        mul_163: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_164, 0.5)
        mul_164: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_180, mul_163);  mul_163 = None
        convert_element_type_183: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_164, torch.float32)
        pow_13: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_183, 3.0)
        mul_165: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_113: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_164, mul_165);  view_164 = mul_165 = None
        mul_166: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_113, 0.7978845608028654);  add_113 = None
        tanh_6: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_166);  mul_166 = None
        add_114: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_167: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_180, add_114);  convert_element_type_180 = add_114 = None
        convert_element_type_184: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_167, torch.bfloat16);  mul_167 = None
        mul_168: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_34: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_168);  mul_168 = None
        mul_169: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_164, sub_34);  mul_164 = sub_34 = None
        mul_170: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_169, 0.7978845608028654);  mul_169 = None
        convert_element_type_185: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_170, torch.bfloat16)
        mul_171: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_170, 0.044715);  mul_170 = None
        pow_14: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_183, 2.0);  convert_element_type_183 = None
        mul_172: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
        mul_173: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_171, mul_172);  mul_171 = mul_172 = None
        convert_element_type_186: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_173, torch.bfloat16);  mul_173 = None
        add_115: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_185, convert_element_type_186);  convert_element_type_185 = convert_element_type_186 = None
        mul_174: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_184, 0.5);  convert_element_type_184 = None
        add_116: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_115, mul_174);  add_115 = mul_174 = None
        view_165: "bf16[4096, 16384]" = torch.ops.aten.view.default(add_116, _shape_param_170);  add_116 = _shape_param_170 = None
        mm_66: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_165, arg21_1)
        permute_63: "bf16[16384, 4096]" = torch.ops.aten.permute.default(view_165, [1, 0])
        mm_67: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_63, arg92_1);  permute_63 = arg92_1 = None
        sum_88: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_165, [0], True, dtype = torch.float32);  view_165 = None
        view_166: "f32[16384]" = torch.ops.aten.view.default(sum_88, _shape_param_171);  sum_88 = _shape_param_171 = None
        convert_element_type_187: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_166, torch.bfloat16);  view_166 = None
        view_167: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_66, _shape_param_172);  mm_66 = _shape_param_172 = None
        convert_element_type_188: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_167, torch.float32);  view_167 = None
        add_117: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_161, convert_element_type_188);  mul_161 = convert_element_type_188 = None
        convert_element_type_189: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        add_118: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_94, convert_element_type_189);  add_94 = convert_element_type_189 = None
        convert_element_type_190: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_187, torch.float32);  convert_element_type_187 = None
        add_119: "f32[16384]" = torch.ops.aten.add.Tensor(add_95, convert_element_type_190);  add_95 = convert_element_type_190 = None
        mul_175: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_117, arg23_1)
        mul_176: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_175, 4096)
        sum_89: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_175, [2], True)
        mul_177: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_175, arg93_1);  mul_175 = None
        sum_90: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_177, [2], True);  mul_177 = None
        mul_178: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg93_1, sum_90);  sum_90 = None
        sub_35: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_176, sum_89);  mul_176 = sum_89 = None
        sub_36: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_35, mul_178);  sub_35 = mul_178 = None
        mul_179: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg94_1, sub_36);  arg94_1 = sub_36 = None
        mul_180: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_117, arg93_1);  arg93_1 = None
        sum_91: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_180, [0, 1]);  mul_180 = None
        sum_92: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_117, [0, 1]);  add_117 = None
        add_120: "f32[4096]" = torch.ops.aten.add.Tensor(add_96, sum_91);  add_96 = sum_91 = None
        add_121: "f32[4096]" = torch.ops.aten.add.Tensor(add_97, sum_92);  add_97 = sum_92 = None
        convert_element_type_191: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_179, torch.bfloat16)
        view_168: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_191, _shape_param_173);  convert_element_type_191 = _shape_param_173 = None
        mm_68: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_168, arg26_1)
        permute_64: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_168, [1, 0])
        mm_69: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_64, arg95_1);  permute_64 = arg95_1 = None
        sum_93: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_168, [0], True, dtype = torch.float32);  view_168 = None
        view_169: "f32[4096]" = torch.ops.aten.view.default(sum_93, _shape_param_174);  sum_93 = _shape_param_174 = None
        convert_element_type_192: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_169, torch.bfloat16);  view_169 = None
        view_170: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_68, _shape_param_175);  mm_68 = _shape_param_175 = None
        convert_element_type_193: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        add_122: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_98, convert_element_type_193);  add_98 = convert_element_type_193 = None
        convert_element_type_194: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_192, torch.float32);  convert_element_type_192 = None
        add_123: "f32[4096]" = torch.ops.aten.add.Tensor(add_99, convert_element_type_194);  add_99 = convert_element_type_194 = None
        view_171: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_170, _shape_param_176);  view_170 = _shape_param_176 = None
        permute_65: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
        clone_20: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
        view_172: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_177);  clone_20 = _shape_param_177 = None
        expand_6: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(arg96_1, _shape_param_178);  _shape_param_178 = None
        view_173: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_6, _shape_param_179);  expand_6 = _shape_param_179 = None
        permute_66: "bf16[512, 512, 512]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
        bmm_20: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(permute_66, view_172);  permute_66 = None
        bmm_21: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_172, arg97_1);  view_172 = arg97_1 = None
        view_174: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_20, _shape_param_180);  bmm_20 = _shape_param_180 = None
        view_175: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_21, _shape_param_181);  bmm_21 = _shape_param_181 = None
        convert_element_type_195: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_175, torch.float32);  view_175 = None
        convert_element_type_196: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(arg96_1, torch.float32);  arg96_1 = None
        mul_181: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_195, convert_element_type_196);  convert_element_type_195 = None
        sum_94: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [-1], True)
        neg_5: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_196);  convert_element_type_196 = None
        fma_5: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_5, sum_94, mul_181);  neg_5 = sum_94 = mul_181 = None
        convert_element_type_197: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None
        view_176: "bf16[512, 512, 512]" = torch.ops.aten.view.default(convert_element_type_197, _shape_param_182);  convert_element_type_197 = _shape_param_182 = None
        bmm_22: "bf16[512, 64, 512]" = torch.ops.aten.bmm.default(arg98_1, view_176);  arg98_1 = None
        bmm_23: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_176, arg99_1);  view_176 = arg99_1 = None
        view_177: "bf16[8, 64, 64, 512]" = torch.ops.aten.view.default(bmm_22, _shape_param_183);  bmm_22 = _shape_param_183 = None
        view_178: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_184);  bmm_23 = _shape_param_184 = None
        mul_182: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_177, 0.3535533905932738);  view_177 = None
        permute_67: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_182, [0, 1, 3, 2]);  mul_182 = None
        mul_183: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_178, 0.3535533905932738);  view_178 = None
        permute_68: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
        clone_21: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_179: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_21, _shape_param_185);  clone_21 = _shape_param_185 = None
        view_180: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_179, _shape_param_186);  view_179 = _shape_param_186 = None
        mm_70: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_180, arg32_1)
        permute_69: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_180, [1, 0])
        mm_71: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_69, arg100_1);  permute_69 = None
        sum_95: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_180, [0], True, dtype = torch.float32);  view_180 = None
        view_181: "f32[4096]" = torch.ops.aten.view.default(sum_95, _shape_param_187);  sum_95 = _shape_param_187 = None
        convert_element_type_198: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_181, torch.bfloat16);  view_181 = None
        view_182: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_70, _shape_param_188);  mm_70 = _shape_param_188 = None
        convert_element_type_199: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_182, torch.float32);  view_182 = None
        add_124: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_179, convert_element_type_199);  mul_179 = convert_element_type_199 = None
        convert_element_type_200: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        add_125: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_101, convert_element_type_200);  add_101 = convert_element_type_200 = None
        convert_element_type_201: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_198, torch.float32);  convert_element_type_198 = None
        add_126: "f32[4096]" = torch.ops.aten.add.Tensor(add_102, convert_element_type_201);  add_102 = convert_element_type_201 = None
        permute_70: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_67, [0, 2, 1, 3]);  permute_67 = None
        view_183: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(permute_70, _shape_param_189);  permute_70 = _shape_param_189 = None
        clone_22: "bf16[8, 512, 4096]" = torch.ops.aten.clone.default(view_183, memory_format = torch.contiguous_format);  view_183 = None
        view_184: "bf16[4096, 4096]" = torch.ops.aten.view.default(clone_22, _shape_param_190);  clone_22 = _shape_param_190 = None
        mm_72: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_184, arg34_1)
        permute_71: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_184, [1, 0])
        mm_73: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_71, arg100_1);  permute_71 = None
        sum_96: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_184, [0], True, dtype = torch.float32);  view_184 = None
        view_185: "f32[4096]" = torch.ops.aten.view.default(sum_96, _shape_param_191);  sum_96 = _shape_param_191 = None
        convert_element_type_202: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_185, torch.bfloat16);  view_185 = None
        view_186: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_72, _shape_param_192);  mm_72 = _shape_param_192 = None
        convert_element_type_203: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_186, torch.float32);  view_186 = None
        add_127: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_124, convert_element_type_203);  add_124 = convert_element_type_203 = None
        convert_element_type_204: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        add_128: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_104, convert_element_type_204);  add_104 = convert_element_type_204 = None
        convert_element_type_205: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_202, torch.float32);  convert_element_type_202 = None
        add_129: "f32[4096]" = torch.ops.aten.add.Tensor(add_105, convert_element_type_205);  add_105 = convert_element_type_205 = None
        permute_72: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_183, [0, 2, 1, 3]);  mul_183 = None
        clone_23: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
        view_187: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_23, _shape_param_193);  clone_23 = _shape_param_193 = None
        view_188: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_187, _shape_param_194);  view_187 = _shape_param_194 = None
        mm_74: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_188, arg35_1)
        permute_73: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_188, [1, 0])
        mm_75: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_73, arg100_1);  permute_73 = arg100_1 = None
        sum_97: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True, dtype = torch.float32);  view_188 = None
        view_189: "f32[4096]" = torch.ops.aten.view.default(sum_97, _shape_param_195);  sum_97 = _shape_param_195 = None
        convert_element_type_206: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_189, torch.bfloat16);  view_189 = None
        view_190: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_74, _shape_param_196);  mm_74 = _shape_param_196 = None
        convert_element_type_207: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_190, torch.float32);  view_190 = None
        add_130: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_127, convert_element_type_207);  add_127 = convert_element_type_207 = None
        convert_element_type_208: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        add_131: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_107, convert_element_type_208);  add_107 = convert_element_type_208 = None
        convert_element_type_209: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_206, torch.float32);  convert_element_type_206 = None
        add_132: "f32[4096]" = torch.ops.aten.add.Tensor(add_108, convert_element_type_209);  add_108 = convert_element_type_209 = None
        mul_184: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_130, arg15_1)
        mul_185: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_184, 4096)
        sum_98: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_184, [2], True)
        mul_186: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_184, arg101_1);  mul_184 = None
        sum_99: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_186, [2], True);  mul_186 = None
        mul_187: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg101_1, sum_99);  sum_99 = None
        sub_37: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_185, sum_98);  mul_185 = sum_98 = None
        sub_38: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_37, mul_187);  sub_37 = mul_187 = None
        mul_188: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg102_1, sub_38);  arg102_1 = sub_38 = None
        mul_189: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_130, arg101_1);  arg101_1 = None
        sum_100: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_189, [0, 1]);  mul_189 = None
        sum_101: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_130, [0, 1]);  add_130 = None
        add_133: "f32[4096]" = torch.ops.aten.add.Tensor(add_109, sum_100);  add_109 = sum_100 = None
        add_134: "f32[4096]" = torch.ops.aten.add.Tensor(add_110, sum_101);  add_110 = sum_101 = None
        convert_element_type_210: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_188, torch.bfloat16)
        view_191: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_210, _shape_param_197);  convert_element_type_210 = _shape_param_197 = None
        mm_76: "bf16[4096, 16384]" = torch.ops.aten.mm.default(view_191, arg18_1)
        permute_74: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_191, [1, 0])
        mm_77: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_74, arg103_1);  permute_74 = arg103_1 = None
        sum_102: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_191, [0], True, dtype = torch.float32);  view_191 = None
        view_192: "f32[4096]" = torch.ops.aten.view.default(sum_102, _shape_param_198);  sum_102 = _shape_param_198 = None
        convert_element_type_211: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_192, torch.bfloat16);  view_192 = None
        view_193: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(mm_76, _shape_param_199);  mm_76 = _shape_param_199 = None
        convert_element_type_212: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_193, torch.float32);  view_193 = None
        convert_element_type_213: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        add_135: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_111, convert_element_type_213);  add_111 = convert_element_type_213 = None
        convert_element_type_214: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_211, torch.float32);  convert_element_type_211 = None
        add_136: "f32[4096]" = torch.ops.aten.add.Tensor(add_112, convert_element_type_214);  add_112 = convert_element_type_214 = None
        view_194: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(arg104_1, _shape_param_200);  arg104_1 = _shape_param_200 = None
        mul_190: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_194, 0.5)
        mul_191: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_212, mul_190);  mul_190 = None
        convert_element_type_215: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_194, torch.float32)
        pow_15: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_215, 3.0)
        mul_192: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_137: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_194, mul_192);  view_194 = mul_192 = None
        mul_193: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_137, 0.7978845608028654);  add_137 = None
        tanh_7: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_193);  mul_193 = None
        add_138: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_194: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_212, add_138);  convert_element_type_212 = add_138 = None
        convert_element_type_216: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_194, torch.bfloat16);  mul_194 = None
        mul_195: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_39: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_195);  mul_195 = None
        mul_196: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_191, sub_39);  mul_191 = sub_39 = None
        mul_197: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_196, 0.7978845608028654);  mul_196 = None
        convert_element_type_217: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_197, torch.bfloat16)
        mul_198: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_197, 0.044715);  mul_197 = None
        pow_16: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_215, 2.0);  convert_element_type_215 = None
        mul_199: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
        mul_200: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_198, mul_199);  mul_198 = mul_199 = None
        convert_element_type_218: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_200, torch.bfloat16);  mul_200 = None
        add_139: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_217, convert_element_type_218);  convert_element_type_217 = convert_element_type_218 = None
        mul_201: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_216, 0.5);  convert_element_type_216 = None
        add_140: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_139, mul_201);  add_139 = mul_201 = None
        view_195: "bf16[4096, 16384]" = torch.ops.aten.view.default(add_140, _shape_param_201);  add_140 = _shape_param_201 = None
        mm_78: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_195, arg21_1)
        permute_75: "bf16[16384, 4096]" = torch.ops.aten.permute.default(view_195, [1, 0])
        mm_79: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_75, arg105_1);  permute_75 = arg105_1 = None
        sum_103: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_195, [0], True, dtype = torch.float32);  view_195 = None
        view_196: "f32[16384]" = torch.ops.aten.view.default(sum_103, _shape_param_202);  sum_103 = _shape_param_202 = None
        convert_element_type_219: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_196, torch.bfloat16);  view_196 = None
        view_197: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_78, _shape_param_203);  mm_78 = _shape_param_203 = None
        convert_element_type_220: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_197, torch.float32);  view_197 = None
        add_141: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_188, convert_element_type_220);  mul_188 = convert_element_type_220 = None
        convert_element_type_221: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        add_142: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_118, convert_element_type_221);  add_118 = convert_element_type_221 = None
        convert_element_type_222: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_219, torch.float32);  convert_element_type_219 = None
        add_143: "f32[16384]" = torch.ops.aten.add.Tensor(add_119, convert_element_type_222);  add_119 = convert_element_type_222 = None
        mul_202: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_141, arg23_1)
        mul_203: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_202, 4096)
        sum_104: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_202, [2], True)
        mul_204: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_202, arg106_1);  mul_202 = None
        sum_105: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_204, [2], True);  mul_204 = None
        mul_205: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg106_1, sum_105);  sum_105 = None
        sub_40: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_203, sum_104);  mul_203 = sum_104 = None
        sub_41: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_40, mul_205);  sub_40 = mul_205 = None
        mul_206: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg107_1, sub_41);  arg107_1 = sub_41 = None
        mul_207: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_141, arg106_1);  arg106_1 = None
        sum_106: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_207, [0, 1]);  mul_207 = None
        sum_107: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_141, [0, 1]);  add_141 = None
        add_144: "f32[4096]" = torch.ops.aten.add.Tensor(add_120, sum_106);  add_120 = sum_106 = None
        add_145: "f32[4096]" = torch.ops.aten.add.Tensor(add_121, sum_107);  add_121 = sum_107 = None
        convert_element_type_223: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_206, torch.bfloat16)
        view_198: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_223, _shape_param_204);  convert_element_type_223 = _shape_param_204 = None
        mm_80: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_198, arg26_1)
        permute_76: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_198, [1, 0])
        mm_81: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_76, arg108_1);  permute_76 = arg108_1 = None
        sum_108: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_198, [0], True, dtype = torch.float32);  view_198 = None
        view_199: "f32[4096]" = torch.ops.aten.view.default(sum_108, _shape_param_205);  sum_108 = _shape_param_205 = None
        convert_element_type_224: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_199, torch.bfloat16);  view_199 = None
        view_200: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_80, _shape_param_206);  mm_80 = _shape_param_206 = None
        convert_element_type_225: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        add_146: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_122, convert_element_type_225);  add_122 = convert_element_type_225 = None
        convert_element_type_226: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_224, torch.float32);  convert_element_type_224 = None
        add_147: "f32[4096]" = torch.ops.aten.add.Tensor(add_123, convert_element_type_226);  add_123 = convert_element_type_226 = None
        view_201: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_200, _shape_param_207);  view_200 = _shape_param_207 = None
        permute_77: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None
        clone_24: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
        view_202: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_208);  clone_24 = _shape_param_208 = None
        expand_7: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(arg109_1, _shape_param_209);  _shape_param_209 = None
        view_203: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_7, _shape_param_210);  expand_7 = _shape_param_210 = None
        permute_78: "bf16[512, 512, 512]" = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None
        bmm_24: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(permute_78, view_202);  permute_78 = None
        bmm_25: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_202, arg110_1);  view_202 = arg110_1 = None
        view_204: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_24, _shape_param_211);  bmm_24 = _shape_param_211 = None
        view_205: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_25, _shape_param_212);  bmm_25 = _shape_param_212 = None
        convert_element_type_227: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_205, torch.float32);  view_205 = None
        convert_element_type_228: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(arg109_1, torch.float32);  arg109_1 = None
        mul_208: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_227, convert_element_type_228);  convert_element_type_227 = None
        sum_109: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [-1], True)
        neg_6: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_228);  convert_element_type_228 = None
        fma_6: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_6, sum_109, mul_208);  neg_6 = sum_109 = mul_208 = None
        convert_element_type_229: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None
        view_206: "bf16[512, 512, 512]" = torch.ops.aten.view.default(convert_element_type_229, _shape_param_213);  convert_element_type_229 = _shape_param_213 = None
        bmm_26: "bf16[512, 64, 512]" = torch.ops.aten.bmm.default(arg111_1, view_206);  arg111_1 = None
        bmm_27: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_206, arg112_1);  view_206 = arg112_1 = None
        view_207: "bf16[8, 64, 64, 512]" = torch.ops.aten.view.default(bmm_26, _shape_param_214);  bmm_26 = _shape_param_214 = None
        view_208: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_27, _shape_param_215);  bmm_27 = _shape_param_215 = None
        mul_209: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_207, 0.3535533905932738);  view_207 = None
        permute_79: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_209, [0, 1, 3, 2]);  mul_209 = None
        mul_210: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_208, 0.3535533905932738);  view_208 = None
        permute_80: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
        clone_25: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_209: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_25, _shape_param_216);  clone_25 = _shape_param_216 = None
        view_210: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_209, _shape_param_217);  view_209 = _shape_param_217 = None
        mm_82: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_210, arg32_1)
        permute_81: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_210, [1, 0])
        mm_83: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_81, arg113_1);  permute_81 = None
        sum_110: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_210, [0], True, dtype = torch.float32);  view_210 = None
        view_211: "f32[4096]" = torch.ops.aten.view.default(sum_110, _shape_param_218);  sum_110 = _shape_param_218 = None
        convert_element_type_230: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_211, torch.bfloat16);  view_211 = None
        view_212: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_82, _shape_param_219);  mm_82 = _shape_param_219 = None
        convert_element_type_231: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_212, torch.float32);  view_212 = None
        add_148: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_206, convert_element_type_231);  mul_206 = convert_element_type_231 = None
        convert_element_type_232: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        add_149: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_125, convert_element_type_232);  add_125 = convert_element_type_232 = None
        convert_element_type_233: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_230, torch.float32);  convert_element_type_230 = None
        add_150: "f32[4096]" = torch.ops.aten.add.Tensor(add_126, convert_element_type_233);  add_126 = convert_element_type_233 = None
        permute_82: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_79, [0, 2, 1, 3]);  permute_79 = None
        view_213: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(permute_82, _shape_param_220);  permute_82 = _shape_param_220 = None
        clone_26: "bf16[8, 512, 4096]" = torch.ops.aten.clone.default(view_213, memory_format = torch.contiguous_format);  view_213 = None
        view_214: "bf16[4096, 4096]" = torch.ops.aten.view.default(clone_26, _shape_param_221);  clone_26 = _shape_param_221 = None
        mm_84: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_214, arg34_1)
        permute_83: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_214, [1, 0])
        mm_85: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_83, arg113_1);  permute_83 = None
        sum_111: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_214, [0], True, dtype = torch.float32);  view_214 = None
        view_215: "f32[4096]" = torch.ops.aten.view.default(sum_111, _shape_param_222);  sum_111 = _shape_param_222 = None
        convert_element_type_234: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_215, torch.bfloat16);  view_215 = None
        view_216: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_84, _shape_param_223);  mm_84 = _shape_param_223 = None
        convert_element_type_235: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_216, torch.float32);  view_216 = None
        add_151: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_148, convert_element_type_235);  add_148 = convert_element_type_235 = None
        convert_element_type_236: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        add_152: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_128, convert_element_type_236);  add_128 = convert_element_type_236 = None
        convert_element_type_237: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_234, torch.float32);  convert_element_type_234 = None
        add_153: "f32[4096]" = torch.ops.aten.add.Tensor(add_129, convert_element_type_237);  add_129 = convert_element_type_237 = None
        permute_84: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_210, [0, 2, 1, 3]);  mul_210 = None
        clone_27: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_217: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_27, _shape_param_224);  clone_27 = _shape_param_224 = None
        view_218: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_217, _shape_param_225);  view_217 = _shape_param_225 = None
        mm_86: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_218, arg35_1)
        permute_85: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_218, [1, 0])
        mm_87: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_85, arg113_1);  permute_85 = arg113_1 = None
        sum_112: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True, dtype = torch.float32);  view_218 = None
        view_219: "f32[4096]" = torch.ops.aten.view.default(sum_112, _shape_param_226);  sum_112 = _shape_param_226 = None
        convert_element_type_238: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_219, torch.bfloat16);  view_219 = None
        view_220: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_86, _shape_param_227);  mm_86 = _shape_param_227 = None
        convert_element_type_239: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_220, torch.float32);  view_220 = None
        add_154: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_151, convert_element_type_239);  add_151 = convert_element_type_239 = None
        convert_element_type_240: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        add_155: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_131, convert_element_type_240);  add_131 = convert_element_type_240 = None
        convert_element_type_241: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_238, torch.float32);  convert_element_type_238 = None
        add_156: "f32[4096]" = torch.ops.aten.add.Tensor(add_132, convert_element_type_241);  add_132 = convert_element_type_241 = None
        mul_211: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_154, arg15_1)
        mul_212: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_211, 4096)
        sum_113: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True)
        mul_213: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_211, arg114_1);  mul_211 = None
        sum_114: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True);  mul_213 = None
        mul_214: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg114_1, sum_114);  sum_114 = None
        sub_42: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_212, sum_113);  mul_212 = sum_113 = None
        sub_43: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_42, mul_214);  sub_42 = mul_214 = None
        mul_215: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg115_1, sub_43);  arg115_1 = sub_43 = None
        mul_216: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_154, arg114_1);  arg114_1 = None
        sum_115: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 1]);  mul_216 = None
        sum_116: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_154, [0, 1]);  add_154 = None
        add_157: "f32[4096]" = torch.ops.aten.add.Tensor(add_133, sum_115);  add_133 = sum_115 = None
        add_158: "f32[4096]" = torch.ops.aten.add.Tensor(add_134, sum_116);  add_134 = sum_116 = None
        convert_element_type_242: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_215, torch.bfloat16)
        view_221: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_242, _shape_param_228);  convert_element_type_242 = _shape_param_228 = None
        mm_88: "bf16[4096, 16384]" = torch.ops.aten.mm.default(view_221, arg18_1)
        permute_86: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_221, [1, 0])
        mm_89: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_86, arg116_1);  permute_86 = arg116_1 = None
        sum_117: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_221, [0], True, dtype = torch.float32);  view_221 = None
        view_222: "f32[4096]" = torch.ops.aten.view.default(sum_117, _shape_param_229);  sum_117 = _shape_param_229 = None
        convert_element_type_243: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_222, torch.bfloat16);  view_222 = None
        view_223: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(mm_88, _shape_param_230);  mm_88 = _shape_param_230 = None
        convert_element_type_244: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_223, torch.float32);  view_223 = None
        convert_element_type_245: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        add_159: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_135, convert_element_type_245);  add_135 = convert_element_type_245 = None
        convert_element_type_246: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_243, torch.float32);  convert_element_type_243 = None
        add_160: "f32[4096]" = torch.ops.aten.add.Tensor(add_136, convert_element_type_246);  add_136 = convert_element_type_246 = None
        view_224: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(arg117_1, _shape_param_231);  arg117_1 = _shape_param_231 = None
        mul_217: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_224, 0.5)
        mul_218: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_244, mul_217);  mul_217 = None
        convert_element_type_247: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_224, torch.float32)
        pow_17: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_247, 3.0)
        mul_219: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_161: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_224, mul_219);  view_224 = mul_219 = None
        mul_220: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_161, 0.7978845608028654);  add_161 = None
        tanh_8: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_220);  mul_220 = None
        add_162: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_221: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_244, add_162);  convert_element_type_244 = add_162 = None
        convert_element_type_248: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_221, torch.bfloat16);  mul_221 = None
        mul_222: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_44: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_222);  mul_222 = None
        mul_223: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_218, sub_44);  mul_218 = sub_44 = None
        mul_224: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_223, 0.7978845608028654);  mul_223 = None
        convert_element_type_249: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_224, torch.bfloat16)
        mul_225: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_224, 0.044715);  mul_224 = None
        pow_18: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_247, 2.0);  convert_element_type_247 = None
        mul_226: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
        mul_227: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
        convert_element_type_250: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_227, torch.bfloat16);  mul_227 = None
        add_163: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_249, convert_element_type_250);  convert_element_type_249 = convert_element_type_250 = None
        mul_228: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_248, 0.5);  convert_element_type_248 = None
        add_164: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_163, mul_228);  add_163 = mul_228 = None
        view_225: "bf16[4096, 16384]" = torch.ops.aten.view.default(add_164, _shape_param_232);  add_164 = _shape_param_232 = None
        mm_90: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_225, arg21_1)
        permute_87: "bf16[16384, 4096]" = torch.ops.aten.permute.default(view_225, [1, 0])
        mm_91: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_87, arg118_1);  permute_87 = arg118_1 = None
        sum_118: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_225, [0], True, dtype = torch.float32);  view_225 = None
        view_226: "f32[16384]" = torch.ops.aten.view.default(sum_118, _shape_param_233);  sum_118 = _shape_param_233 = None
        convert_element_type_251: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_226, torch.bfloat16);  view_226 = None
        view_227: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_90, _shape_param_234);  mm_90 = _shape_param_234 = None
        convert_element_type_252: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_227, torch.float32);  view_227 = None
        add_165: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_215, convert_element_type_252);  mul_215 = convert_element_type_252 = None
        convert_element_type_253: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        add_166: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_142, convert_element_type_253);  add_142 = convert_element_type_253 = None
        convert_element_type_254: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_251, torch.float32);  convert_element_type_251 = None
        add_167: "f32[16384]" = torch.ops.aten.add.Tensor(add_143, convert_element_type_254);  add_143 = convert_element_type_254 = None
        mul_229: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_165, arg23_1)
        mul_230: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_229, 4096)
        sum_119: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True)
        mul_231: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_229, arg119_1);  mul_229 = None
        sum_120: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_231, [2], True);  mul_231 = None
        mul_232: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg119_1, sum_120);  sum_120 = None
        sub_45: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_230, sum_119);  mul_230 = sum_119 = None
        sub_46: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_45, mul_232);  sub_45 = mul_232 = None
        mul_233: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg120_1, sub_46);  arg120_1 = sub_46 = None
        mul_234: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_165, arg119_1);  arg119_1 = None
        sum_121: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_234, [0, 1]);  mul_234 = None
        sum_122: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_165, [0, 1]);  add_165 = None
        add_168: "f32[4096]" = torch.ops.aten.add.Tensor(add_144, sum_121);  add_144 = sum_121 = None
        add_169: "f32[4096]" = torch.ops.aten.add.Tensor(add_145, sum_122);  add_145 = sum_122 = None
        convert_element_type_255: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_233, torch.bfloat16)
        view_228: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_255, _shape_param_235);  convert_element_type_255 = _shape_param_235 = None
        mm_92: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_228, arg26_1)
        permute_88: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_228, [1, 0])
        mm_93: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_88, arg121_1);  permute_88 = arg121_1 = None
        sum_123: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_228, [0], True, dtype = torch.float32);  view_228 = None
        view_229: "f32[4096]" = torch.ops.aten.view.default(sum_123, _shape_param_236);  sum_123 = _shape_param_236 = None
        convert_element_type_256: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_229, torch.bfloat16);  view_229 = None
        view_230: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_92, _shape_param_237);  mm_92 = _shape_param_237 = None
        convert_element_type_257: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        add_170: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_146, convert_element_type_257);  add_146 = convert_element_type_257 = None
        convert_element_type_258: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_256, torch.float32);  convert_element_type_256 = None
        add_171: "f32[4096]" = torch.ops.aten.add.Tensor(add_147, convert_element_type_258);  add_147 = convert_element_type_258 = None
        view_231: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_230, _shape_param_238);  view_230 = _shape_param_238 = None
        permute_89: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_231, [0, 2, 1, 3]);  view_231 = None
        clone_28: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
        view_232: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_239);  clone_28 = _shape_param_239 = None
        expand_8: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(arg122_1, _shape_param_240);  _shape_param_240 = None
        view_233: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_8, _shape_param_241);  expand_8 = _shape_param_241 = None
        permute_90: "bf16[512, 512, 512]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
        bmm_28: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(permute_90, view_232);  permute_90 = None
        bmm_29: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_232, arg123_1);  view_232 = arg123_1 = None
        view_234: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_28, _shape_param_242);  bmm_28 = _shape_param_242 = None
        view_235: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_29, _shape_param_243);  bmm_29 = _shape_param_243 = None
        convert_element_type_259: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_235, torch.float32);  view_235 = None
        convert_element_type_260: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(arg122_1, torch.float32);  arg122_1 = None
        mul_235: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_259, convert_element_type_260);  convert_element_type_259 = None
        sum_124: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_235, [-1], True)
        neg_7: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_260);  convert_element_type_260 = None
        fma_7: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_7, sum_124, mul_235);  neg_7 = sum_124 = mul_235 = None
        convert_element_type_261: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None
        view_236: "bf16[512, 512, 512]" = torch.ops.aten.view.default(convert_element_type_261, _shape_param_244);  convert_element_type_261 = _shape_param_244 = None
        bmm_30: "bf16[512, 64, 512]" = torch.ops.aten.bmm.default(arg124_1, view_236);  arg124_1 = None
        bmm_31: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_236, arg125_1);  view_236 = arg125_1 = None
        view_237: "bf16[8, 64, 64, 512]" = torch.ops.aten.view.default(bmm_30, _shape_param_245);  bmm_30 = _shape_param_245 = None
        view_238: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_31, _shape_param_246);  bmm_31 = _shape_param_246 = None
        mul_236: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_237, 0.3535533905932738);  view_237 = None
        permute_91: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_236, [0, 1, 3, 2]);  mul_236 = None
        mul_237: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_238, 0.3535533905932738);  view_238 = None
        permute_92: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_29: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_239: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_29, _shape_param_247);  clone_29 = _shape_param_247 = None
        view_240: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_239, _shape_param_248);  view_239 = _shape_param_248 = None
        mm_94: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_240, arg32_1)
        permute_93: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_240, [1, 0])
        mm_95: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_93, arg126_1);  permute_93 = None
        sum_125: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_240, [0], True, dtype = torch.float32);  view_240 = None
        view_241: "f32[4096]" = torch.ops.aten.view.default(sum_125, _shape_param_249);  sum_125 = _shape_param_249 = None
        convert_element_type_262: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_241, torch.bfloat16);  view_241 = None
        view_242: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_94, _shape_param_250);  mm_94 = _shape_param_250 = None
        convert_element_type_263: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_242, torch.float32);  view_242 = None
        add_172: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_233, convert_element_type_263);  mul_233 = convert_element_type_263 = None
        convert_element_type_264: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        add_173: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_149, convert_element_type_264);  add_149 = convert_element_type_264 = None
        convert_element_type_265: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_262, torch.float32);  convert_element_type_262 = None
        add_174: "f32[4096]" = torch.ops.aten.add.Tensor(add_150, convert_element_type_265);  add_150 = convert_element_type_265 = None
        permute_94: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_91, [0, 2, 1, 3]);  permute_91 = None
        view_243: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(permute_94, _shape_param_251);  permute_94 = _shape_param_251 = None
        clone_30: "bf16[8, 512, 4096]" = torch.ops.aten.clone.default(view_243, memory_format = torch.contiguous_format);  view_243 = None
        view_244: "bf16[4096, 4096]" = torch.ops.aten.view.default(clone_30, _shape_param_252);  clone_30 = _shape_param_252 = None
        mm_96: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_244, arg34_1)
        permute_95: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_244, [1, 0])
        mm_97: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_95, arg126_1);  permute_95 = None
        sum_126: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_244, [0], True, dtype = torch.float32);  view_244 = None
        view_245: "f32[4096]" = torch.ops.aten.view.default(sum_126, _shape_param_253);  sum_126 = _shape_param_253 = None
        convert_element_type_266: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_245, torch.bfloat16);  view_245 = None
        view_246: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_96, _shape_param_254);  mm_96 = _shape_param_254 = None
        convert_element_type_267: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_246, torch.float32);  view_246 = None
        add_175: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_172, convert_element_type_267);  add_172 = convert_element_type_267 = None
        convert_element_type_268: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        add_176: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_152, convert_element_type_268);  add_152 = convert_element_type_268 = None
        convert_element_type_269: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_266, torch.float32);  convert_element_type_266 = None
        add_177: "f32[4096]" = torch.ops.aten.add.Tensor(add_153, convert_element_type_269);  add_153 = convert_element_type_269 = None
        permute_96: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_237, [0, 2, 1, 3]);  mul_237 = None
        clone_31: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
        view_247: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_31, _shape_param_255);  clone_31 = _shape_param_255 = None
        view_248: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_247, _shape_param_256);  view_247 = _shape_param_256 = None
        mm_98: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_248, arg35_1)
        permute_97: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_248, [1, 0])
        mm_99: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_97, arg126_1);  permute_97 = arg126_1 = None
        sum_127: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_248, [0], True, dtype = torch.float32);  view_248 = None
        view_249: "f32[4096]" = torch.ops.aten.view.default(sum_127, _shape_param_257);  sum_127 = _shape_param_257 = None
        convert_element_type_270: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_249, torch.bfloat16);  view_249 = None
        view_250: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_98, _shape_param_258);  mm_98 = _shape_param_258 = None
        convert_element_type_271: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_250, torch.float32);  view_250 = None
        add_178: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_175, convert_element_type_271);  add_175 = convert_element_type_271 = None
        convert_element_type_272: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        add_179: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_155, convert_element_type_272);  add_155 = convert_element_type_272 = None
        convert_element_type_273: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_270, torch.float32);  convert_element_type_270 = None
        add_180: "f32[4096]" = torch.ops.aten.add.Tensor(add_156, convert_element_type_273);  add_156 = convert_element_type_273 = None
        mul_238: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_178, arg15_1)
        mul_239: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_238, 4096)
        sum_128: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True)
        mul_240: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_238, arg127_1);  mul_238 = None
        sum_129: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True);  mul_240 = None
        mul_241: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg127_1, sum_129);  sum_129 = None
        sub_47: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_239, sum_128);  mul_239 = sum_128 = None
        sub_48: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_47, mul_241);  sub_47 = mul_241 = None
        mul_242: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg128_1, sub_48);  arg128_1 = sub_48 = None
        mul_243: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_178, arg127_1);  arg127_1 = None
        sum_130: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 1]);  mul_243 = None
        sum_131: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_178, [0, 1]);  add_178 = None
        add_181: "f32[4096]" = torch.ops.aten.add.Tensor(add_157, sum_130);  add_157 = sum_130 = None
        add_182: "f32[4096]" = torch.ops.aten.add.Tensor(add_158, sum_131);  add_158 = sum_131 = None
        convert_element_type_274: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_242, torch.bfloat16)
        view_251: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_274, _shape_param_259);  convert_element_type_274 = _shape_param_259 = None
        mm_100: "bf16[4096, 16384]" = torch.ops.aten.mm.default(view_251, arg18_1)
        permute_98: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_251, [1, 0])
        mm_101: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_98, arg129_1);  permute_98 = arg129_1 = None
        sum_132: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_251, [0], True, dtype = torch.float32);  view_251 = None
        view_252: "f32[4096]" = torch.ops.aten.view.default(sum_132, _shape_param_260);  sum_132 = _shape_param_260 = None
        convert_element_type_275: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_252, torch.bfloat16);  view_252 = None
        view_253: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(mm_100, _shape_param_261);  mm_100 = _shape_param_261 = None
        convert_element_type_276: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_253, torch.float32);  view_253 = None
        convert_element_type_277: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        add_183: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_159, convert_element_type_277);  add_159 = convert_element_type_277 = None
        convert_element_type_278: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_275, torch.float32);  convert_element_type_275 = None
        add_184: "f32[4096]" = torch.ops.aten.add.Tensor(add_160, convert_element_type_278);  add_160 = convert_element_type_278 = None
        view_254: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(arg130_1, _shape_param_262);  arg130_1 = _shape_param_262 = None
        mul_244: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_254, 0.5)
        mul_245: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_276, mul_244);  mul_244 = None
        convert_element_type_279: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_254, torch.float32)
        pow_19: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_279, 3.0)
        mul_246: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_185: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_254, mul_246);  view_254 = mul_246 = None
        mul_247: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_185, 0.7978845608028654);  add_185 = None
        tanh_9: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_247);  mul_247 = None
        add_186: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_248: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_276, add_186);  convert_element_type_276 = add_186 = None
        convert_element_type_280: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_248, torch.bfloat16);  mul_248 = None
        mul_249: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_49: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_249);  mul_249 = None
        mul_250: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_245, sub_49);  mul_245 = sub_49 = None
        mul_251: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_250, 0.7978845608028654);  mul_250 = None
        convert_element_type_281: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_251, torch.bfloat16)
        mul_252: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_251, 0.044715);  mul_251 = None
        pow_20: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_279, 2.0);  convert_element_type_279 = None
        mul_253: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
        mul_254: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
        convert_element_type_282: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_254, torch.bfloat16);  mul_254 = None
        add_187: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_281, convert_element_type_282);  convert_element_type_281 = convert_element_type_282 = None
        mul_255: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_280, 0.5);  convert_element_type_280 = None
        add_188: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_187, mul_255);  add_187 = mul_255 = None
        view_255: "bf16[4096, 16384]" = torch.ops.aten.view.default(add_188, _shape_param_263);  add_188 = _shape_param_263 = None
        mm_102: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_255, arg21_1)
        permute_99: "bf16[16384, 4096]" = torch.ops.aten.permute.default(view_255, [1, 0])
        mm_103: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_99, arg131_1);  permute_99 = arg131_1 = None
        sum_133: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_255, [0], True, dtype = torch.float32);  view_255 = None
        view_256: "f32[16384]" = torch.ops.aten.view.default(sum_133, _shape_param_264);  sum_133 = _shape_param_264 = None
        convert_element_type_283: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_256, torch.bfloat16);  view_256 = None
        view_257: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_102, _shape_param_265);  mm_102 = _shape_param_265 = None
        convert_element_type_284: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_257, torch.float32);  view_257 = None
        add_189: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_242, convert_element_type_284);  mul_242 = convert_element_type_284 = None
        convert_element_type_285: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        add_190: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_166, convert_element_type_285);  add_166 = convert_element_type_285 = None
        convert_element_type_286: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_283, torch.float32);  convert_element_type_283 = None
        add_191: "f32[16384]" = torch.ops.aten.add.Tensor(add_167, convert_element_type_286);  add_167 = convert_element_type_286 = None
        mul_256: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_189, arg23_1)
        mul_257: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_256, 4096)
        sum_134: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_256, [2], True)
        mul_258: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_256, arg132_1);  mul_256 = None
        sum_135: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_258, [2], True);  mul_258 = None
        mul_259: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg132_1, sum_135);  sum_135 = None
        sub_50: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_257, sum_134);  mul_257 = sum_134 = None
        sub_51: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_50, mul_259);  sub_50 = mul_259 = None
        mul_260: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg133_1, sub_51);  arg133_1 = sub_51 = None
        mul_261: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_189, arg132_1);  arg132_1 = None
        sum_136: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_261, [0, 1]);  mul_261 = None
        sum_137: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_189, [0, 1]);  add_189 = None
        add_192: "f32[4096]" = torch.ops.aten.add.Tensor(add_168, sum_136);  add_168 = sum_136 = None
        add_193: "f32[4096]" = torch.ops.aten.add.Tensor(add_169, sum_137);  add_169 = sum_137 = None
        convert_element_type_287: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_260, torch.bfloat16)
        view_258: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_287, _shape_param_266);  convert_element_type_287 = _shape_param_266 = None
        mm_104: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_258, arg26_1)
        permute_100: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_258, [1, 0])
        mm_105: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_100, arg134_1);  permute_100 = arg134_1 = None
        sum_138: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_258, [0], True, dtype = torch.float32);  view_258 = None
        view_259: "f32[4096]" = torch.ops.aten.view.default(sum_138, _shape_param_267);  sum_138 = _shape_param_267 = None
        convert_element_type_288: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_259, torch.bfloat16);  view_259 = None
        view_260: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_104, _shape_param_268);  mm_104 = _shape_param_268 = None
        convert_element_type_289: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None
        add_194: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_170, convert_element_type_289);  add_170 = convert_element_type_289 = None
        convert_element_type_290: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_288, torch.float32);  convert_element_type_288 = None
        add_195: "f32[4096]" = torch.ops.aten.add.Tensor(add_171, convert_element_type_290);  add_171 = convert_element_type_290 = None
        view_261: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_260, _shape_param_269);  view_260 = _shape_param_269 = None
        permute_101: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
        clone_32: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
        view_262: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_32, _shape_param_270);  clone_32 = _shape_param_270 = None
        expand_9: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(arg135_1, _shape_param_271);  _shape_param_271 = None
        view_263: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_9, _shape_param_272);  expand_9 = _shape_param_272 = None
        permute_102: "bf16[512, 512, 512]" = torch.ops.aten.permute.default(view_263, [0, 2, 1]);  view_263 = None
        bmm_32: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(permute_102, view_262);  permute_102 = None
        bmm_33: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_262, arg136_1);  view_262 = arg136_1 = None
        view_264: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_32, _shape_param_273);  bmm_32 = _shape_param_273 = None
        view_265: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_33, _shape_param_274);  bmm_33 = _shape_param_274 = None
        convert_element_type_291: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_265, torch.float32);  view_265 = None
        convert_element_type_292: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(arg135_1, torch.float32);  arg135_1 = None
        mul_262: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_291, convert_element_type_292);  convert_element_type_291 = None
        sum_139: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_262, [-1], True)
        neg_8: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_292);  convert_element_type_292 = None
        fma_8: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_8, sum_139, mul_262);  neg_8 = sum_139 = mul_262 = None
        convert_element_type_293: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None
        view_266: "bf16[512, 512, 512]" = torch.ops.aten.view.default(convert_element_type_293, _shape_param_275);  convert_element_type_293 = _shape_param_275 = None
        bmm_34: "bf16[512, 64, 512]" = torch.ops.aten.bmm.default(arg137_1, view_266);  arg137_1 = None
        bmm_35: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_266, arg138_1);  view_266 = arg138_1 = None
        view_267: "bf16[8, 64, 64, 512]" = torch.ops.aten.view.default(bmm_34, _shape_param_276);  bmm_34 = _shape_param_276 = None
        view_268: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_35, _shape_param_277);  bmm_35 = _shape_param_277 = None
        mul_263: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_267, 0.3535533905932738);  view_267 = None
        permute_103: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_263, [0, 1, 3, 2]);  mul_263 = None
        mul_264: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_268, 0.3535533905932738);  view_268 = None
        permute_104: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None
        clone_33: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_269: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_33, _shape_param_278);  clone_33 = _shape_param_278 = None
        view_270: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_269, _shape_param_279);  view_269 = _shape_param_279 = None
        mm_106: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_270, arg32_1)
        permute_105: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_270, [1, 0])
        mm_107: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_105, arg139_1);  permute_105 = None
        sum_140: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_270, [0], True, dtype = torch.float32);  view_270 = None
        view_271: "f32[4096]" = torch.ops.aten.view.default(sum_140, _shape_param_280);  sum_140 = _shape_param_280 = None
        convert_element_type_294: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_271, torch.bfloat16);  view_271 = None
        view_272: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_106, _shape_param_281);  mm_106 = _shape_param_281 = None
        convert_element_type_295: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_272, torch.float32);  view_272 = None
        add_196: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_260, convert_element_type_295);  mul_260 = convert_element_type_295 = None
        convert_element_type_296: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None
        add_197: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_173, convert_element_type_296);  add_173 = convert_element_type_296 = None
        convert_element_type_297: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_294, torch.float32);  convert_element_type_294 = None
        add_198: "f32[4096]" = torch.ops.aten.add.Tensor(add_174, convert_element_type_297);  add_174 = convert_element_type_297 = None
        permute_106: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_103, [0, 2, 1, 3]);  permute_103 = None
        view_273: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(permute_106, _shape_param_282);  permute_106 = _shape_param_282 = None
        clone_34: "bf16[8, 512, 4096]" = torch.ops.aten.clone.default(view_273, memory_format = torch.contiguous_format);  view_273 = None
        view_274: "bf16[4096, 4096]" = torch.ops.aten.view.default(clone_34, _shape_param_283);  clone_34 = _shape_param_283 = None
        mm_108: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_274, arg34_1)
        permute_107: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_274, [1, 0])
        mm_109: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_107, arg139_1);  permute_107 = None
        sum_141: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True, dtype = torch.float32);  view_274 = None
        view_275: "f32[4096]" = torch.ops.aten.view.default(sum_141, _shape_param_284);  sum_141 = _shape_param_284 = None
        convert_element_type_298: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_275, torch.bfloat16);  view_275 = None
        view_276: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_108, _shape_param_285);  mm_108 = _shape_param_285 = None
        convert_element_type_299: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_276, torch.float32);  view_276 = None
        add_199: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_196, convert_element_type_299);  add_196 = convert_element_type_299 = None
        convert_element_type_300: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        add_200: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_176, convert_element_type_300);  add_176 = convert_element_type_300 = None
        convert_element_type_301: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_298, torch.float32);  convert_element_type_298 = None
        add_201: "f32[4096]" = torch.ops.aten.add.Tensor(add_177, convert_element_type_301);  add_177 = convert_element_type_301 = None
        permute_108: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_264, [0, 2, 1, 3]);  mul_264 = None
        clone_35: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_108, memory_format = torch.contiguous_format);  permute_108 = None
        view_277: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_35, _shape_param_286);  clone_35 = _shape_param_286 = None
        view_278: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_277, _shape_param_287);  view_277 = _shape_param_287 = None
        mm_110: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_278, arg35_1)
        permute_109: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_278, [1, 0])
        mm_111: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_109, arg139_1);  permute_109 = arg139_1 = None
        sum_142: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True, dtype = torch.float32);  view_278 = None
        view_279: "f32[4096]" = torch.ops.aten.view.default(sum_142, _shape_param_288);  sum_142 = _shape_param_288 = None
        convert_element_type_302: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_279, torch.bfloat16);  view_279 = None
        view_280: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_110, _shape_param_289);  mm_110 = _shape_param_289 = None
        convert_element_type_303: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_280, torch.float32);  view_280 = None
        add_202: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_199, convert_element_type_303);  add_199 = convert_element_type_303 = None
        convert_element_type_304: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        add_203: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_179, convert_element_type_304);  add_179 = convert_element_type_304 = None
        convert_element_type_305: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_302, torch.float32);  convert_element_type_302 = None
        add_204: "f32[4096]" = torch.ops.aten.add.Tensor(add_180, convert_element_type_305);  add_180 = convert_element_type_305 = None
        mul_265: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_202, arg15_1)
        mul_266: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_265, 4096)
        sum_143: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_265, [2], True)
        mul_267: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_265, arg140_1);  mul_265 = None
        sum_144: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [2], True);  mul_267 = None
        mul_268: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg140_1, sum_144);  sum_144 = None
        sub_52: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_266, sum_143);  mul_266 = sum_143 = None
        sub_53: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_52, mul_268);  sub_52 = mul_268 = None
        mul_269: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg141_1, sub_53);  arg141_1 = sub_53 = None
        mul_270: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_202, arg140_1);  arg140_1 = None
        sum_145: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_270, [0, 1]);  mul_270 = None
        sum_146: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_202, [0, 1]);  add_202 = None
        add_205: "f32[4096]" = torch.ops.aten.add.Tensor(add_181, sum_145);  add_181 = sum_145 = None
        add_206: "f32[4096]" = torch.ops.aten.add.Tensor(add_182, sum_146);  add_182 = sum_146 = None
        convert_element_type_306: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_269, torch.bfloat16)
        view_281: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_306, _shape_param_290);  convert_element_type_306 = _shape_param_290 = None
        mm_112: "bf16[4096, 16384]" = torch.ops.aten.mm.default(view_281, arg18_1)
        permute_110: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_281, [1, 0])
        mm_113: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_110, arg142_1);  permute_110 = arg142_1 = None
        sum_147: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_281, [0], True, dtype = torch.float32);  view_281 = None
        view_282: "f32[4096]" = torch.ops.aten.view.default(sum_147, _shape_param_291);  sum_147 = _shape_param_291 = None
        convert_element_type_307: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_282, torch.bfloat16);  view_282 = None
        view_283: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(mm_112, _shape_param_292);  mm_112 = _shape_param_292 = None
        convert_element_type_308: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_283, torch.float32);  view_283 = None
        convert_element_type_309: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        add_207: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_183, convert_element_type_309);  add_183 = convert_element_type_309 = None
        convert_element_type_310: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_307, torch.float32);  convert_element_type_307 = None
        add_208: "f32[4096]" = torch.ops.aten.add.Tensor(add_184, convert_element_type_310);  add_184 = convert_element_type_310 = None
        view_284: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(arg143_1, _shape_param_293);  arg143_1 = _shape_param_293 = None
        mul_271: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_284, 0.5)
        mul_272: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_308, mul_271);  mul_271 = None
        convert_element_type_311: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_284, torch.float32)
        pow_21: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_311, 3.0)
        mul_273: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_209: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_284, mul_273);  view_284 = mul_273 = None
        mul_274: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_209, 0.7978845608028654);  add_209 = None
        tanh_10: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_274);  mul_274 = None
        add_210: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_275: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_308, add_210);  convert_element_type_308 = add_210 = None
        convert_element_type_312: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_275, torch.bfloat16);  mul_275 = None
        mul_276: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_54: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_276);  mul_276 = None
        mul_277: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_272, sub_54);  mul_272 = sub_54 = None
        mul_278: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_277, 0.7978845608028654);  mul_277 = None
        convert_element_type_313: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_278, torch.bfloat16)
        mul_279: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_278, 0.044715);  mul_278 = None
        pow_22: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_311, 2.0);  convert_element_type_311 = None
        mul_280: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
        mul_281: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_279, mul_280);  mul_279 = mul_280 = None
        convert_element_type_314: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_281, torch.bfloat16);  mul_281 = None
        add_211: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_313, convert_element_type_314);  convert_element_type_313 = convert_element_type_314 = None
        mul_282: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_312, 0.5);  convert_element_type_312 = None
        add_212: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_211, mul_282);  add_211 = mul_282 = None
        view_285: "bf16[4096, 16384]" = torch.ops.aten.view.default(add_212, _shape_param_294);  add_212 = _shape_param_294 = None
        mm_114: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_285, arg21_1)
        permute_111: "bf16[16384, 4096]" = torch.ops.aten.permute.default(view_285, [1, 0])
        mm_115: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_111, arg144_1);  permute_111 = arg144_1 = None
        sum_148: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True, dtype = torch.float32);  view_285 = None
        view_286: "f32[16384]" = torch.ops.aten.view.default(sum_148, _shape_param_295);  sum_148 = _shape_param_295 = None
        convert_element_type_315: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_286, torch.bfloat16);  view_286 = None
        view_287: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_114, _shape_param_296);  mm_114 = _shape_param_296 = None
        convert_element_type_316: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_287, torch.float32);  view_287 = None
        add_213: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_269, convert_element_type_316);  mul_269 = convert_element_type_316 = None
        convert_element_type_317: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        add_214: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_190, convert_element_type_317);  add_190 = convert_element_type_317 = None
        convert_element_type_318: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_315, torch.float32);  convert_element_type_315 = None
        add_215: "f32[16384]" = torch.ops.aten.add.Tensor(add_191, convert_element_type_318);  add_191 = convert_element_type_318 = None
        mul_283: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_213, arg23_1)
        mul_284: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_283, 4096)
        sum_149: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_283, [2], True)
        mul_285: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_283, arg145_1);  mul_283 = None
        sum_150: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_285, [2], True);  mul_285 = None
        mul_286: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg145_1, sum_150);  sum_150 = None
        sub_55: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_284, sum_149);  mul_284 = sum_149 = None
        sub_56: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_55, mul_286);  sub_55 = mul_286 = None
        mul_287: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg146_1, sub_56);  arg146_1 = sub_56 = None
        mul_288: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_213, arg145_1);  arg145_1 = None
        sum_151: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_288, [0, 1]);  mul_288 = None
        sum_152: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_213, [0, 1]);  add_213 = None
        add_216: "f32[4096]" = torch.ops.aten.add.Tensor(add_192, sum_151);  add_192 = sum_151 = None
        add_217: "f32[4096]" = torch.ops.aten.add.Tensor(add_193, sum_152);  add_193 = sum_152 = None
        convert_element_type_319: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_287, torch.bfloat16)
        view_288: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_319, _shape_param_297);  convert_element_type_319 = _shape_param_297 = None
        mm_116: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_288, arg26_1)
        permute_112: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_288, [1, 0])
        mm_117: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_112, arg147_1);  permute_112 = arg147_1 = None
        sum_153: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True, dtype = torch.float32);  view_288 = None
        view_289: "f32[4096]" = torch.ops.aten.view.default(sum_153, _shape_param_298);  sum_153 = _shape_param_298 = None
        convert_element_type_320: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_289, torch.bfloat16);  view_289 = None
        view_290: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_116, _shape_param_299);  mm_116 = _shape_param_299 = None
        convert_element_type_321: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        add_218: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_194, convert_element_type_321);  add_194 = convert_element_type_321 = None
        convert_element_type_322: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_320, torch.float32);  convert_element_type_320 = None
        add_219: "f32[4096]" = torch.ops.aten.add.Tensor(add_195, convert_element_type_322);  add_195 = convert_element_type_322 = None
        view_291: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_290, _shape_param_300);  view_290 = _shape_param_300 = None
        permute_113: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        clone_36: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
        view_292: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_301);  clone_36 = _shape_param_301 = None
        expand_10: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(arg148_1, _shape_param_302);  _shape_param_302 = None
        view_293: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_10, _shape_param_303);  expand_10 = _shape_param_303 = None
        permute_114: "bf16[512, 512, 512]" = torch.ops.aten.permute.default(view_293, [0, 2, 1]);  view_293 = None
        bmm_36: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(permute_114, view_292);  permute_114 = None
        bmm_37: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_292, arg149_1);  view_292 = arg149_1 = None
        view_294: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_36, _shape_param_304);  bmm_36 = _shape_param_304 = None
        view_295: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_37, _shape_param_305);  bmm_37 = _shape_param_305 = None
        convert_element_type_323: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_295, torch.float32);  view_295 = None
        convert_element_type_324: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(arg148_1, torch.float32);  arg148_1 = None
        mul_289: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_323, convert_element_type_324);  convert_element_type_323 = None
        sum_154: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_289, [-1], True)
        neg_9: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_324);  convert_element_type_324 = None
        fma_9: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_9, sum_154, mul_289);  neg_9 = sum_154 = mul_289 = None
        convert_element_type_325: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None
        view_296: "bf16[512, 512, 512]" = torch.ops.aten.view.default(convert_element_type_325, _shape_param_306);  convert_element_type_325 = _shape_param_306 = None
        bmm_38: "bf16[512, 64, 512]" = torch.ops.aten.bmm.default(arg150_1, view_296);  arg150_1 = None
        bmm_39: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_296, arg151_1);  view_296 = arg151_1 = None
        view_297: "bf16[8, 64, 64, 512]" = torch.ops.aten.view.default(bmm_38, _shape_param_307);  bmm_38 = _shape_param_307 = None
        view_298: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_39, _shape_param_308);  bmm_39 = _shape_param_308 = None
        mul_290: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_297, 0.3535533905932738);  view_297 = None
        permute_115: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_290, [0, 1, 3, 2]);  mul_290 = None
        mul_291: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_298, 0.3535533905932738);  view_298 = None
        permute_116: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        clone_37: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
        view_299: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_37, _shape_param_309);  clone_37 = _shape_param_309 = None
        view_300: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_299, _shape_param_310);  view_299 = _shape_param_310 = None
        mm_118: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_300, arg32_1)
        permute_117: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_300, [1, 0])
        mm_119: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_117, arg152_1);  permute_117 = None
        sum_155: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True, dtype = torch.float32);  view_300 = None
        view_301: "f32[4096]" = torch.ops.aten.view.default(sum_155, _shape_param_311);  sum_155 = _shape_param_311 = None
        convert_element_type_326: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_301, torch.bfloat16);  view_301 = None
        view_302: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_118, _shape_param_312);  mm_118 = _shape_param_312 = None
        convert_element_type_327: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_302, torch.float32);  view_302 = None
        add_220: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_287, convert_element_type_327);  mul_287 = convert_element_type_327 = None
        convert_element_type_328: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        add_221: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_197, convert_element_type_328);  add_197 = convert_element_type_328 = None
        convert_element_type_329: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_326, torch.float32);  convert_element_type_326 = None
        add_222: "f32[4096]" = torch.ops.aten.add.Tensor(add_198, convert_element_type_329);  add_198 = convert_element_type_329 = None
        permute_118: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_115, [0, 2, 1, 3]);  permute_115 = None
        view_303: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(permute_118, _shape_param_313);  permute_118 = _shape_param_313 = None
        clone_38: "bf16[8, 512, 4096]" = torch.ops.aten.clone.default(view_303, memory_format = torch.contiguous_format);  view_303 = None
        view_304: "bf16[4096, 4096]" = torch.ops.aten.view.default(clone_38, _shape_param_314);  clone_38 = _shape_param_314 = None
        mm_120: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_304, arg34_1)
        permute_119: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_304, [1, 0])
        mm_121: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_119, arg152_1);  permute_119 = None
        sum_156: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True, dtype = torch.float32);  view_304 = None
        view_305: "f32[4096]" = torch.ops.aten.view.default(sum_156, _shape_param_315);  sum_156 = _shape_param_315 = None
        convert_element_type_330: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_305, torch.bfloat16);  view_305 = None
        view_306: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_120, _shape_param_316);  mm_120 = _shape_param_316 = None
        convert_element_type_331: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_306, torch.float32);  view_306 = None
        add_223: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_220, convert_element_type_331);  add_220 = convert_element_type_331 = None
        convert_element_type_332: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        add_224: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_200, convert_element_type_332);  add_200 = convert_element_type_332 = None
        convert_element_type_333: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_330, torch.float32);  convert_element_type_330 = None
        add_225: "f32[4096]" = torch.ops.aten.add.Tensor(add_201, convert_element_type_333);  add_201 = convert_element_type_333 = None
        permute_120: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_291, [0, 2, 1, 3]);  mul_291 = None
        clone_39: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
        view_307: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_39, _shape_param_317);  clone_39 = _shape_param_317 = None
        view_308: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_307, _shape_param_318);  view_307 = _shape_param_318 = None
        mm_122: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_308, arg35_1)
        permute_121: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_308, [1, 0])
        mm_123: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_121, arg152_1);  permute_121 = arg152_1 = None
        sum_157: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True, dtype = torch.float32);  view_308 = None
        view_309: "f32[4096]" = torch.ops.aten.view.default(sum_157, _shape_param_319);  sum_157 = _shape_param_319 = None
        convert_element_type_334: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_309, torch.bfloat16);  view_309 = None
        view_310: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_122, _shape_param_320);  mm_122 = _shape_param_320 = None
        convert_element_type_335: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_310, torch.float32);  view_310 = None
        add_226: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_223, convert_element_type_335);  add_223 = convert_element_type_335 = None
        convert_element_type_336: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        add_227: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_203, convert_element_type_336);  add_203 = convert_element_type_336 = None
        convert_element_type_337: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_334, torch.float32);  convert_element_type_334 = None
        add_228: "f32[4096]" = torch.ops.aten.add.Tensor(add_204, convert_element_type_337);  add_204 = convert_element_type_337 = None
        mul_292: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_226, arg15_1)
        mul_293: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_292, 4096)
        sum_158: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True)
        mul_294: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_292, arg153_1);  mul_292 = None
        sum_159: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_294, [2], True);  mul_294 = None
        mul_295: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg153_1, sum_159);  sum_159 = None
        sub_57: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_293, sum_158);  mul_293 = sum_158 = None
        sub_58: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_57, mul_295);  sub_57 = mul_295 = None
        mul_296: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg154_1, sub_58);  arg154_1 = sub_58 = None
        mul_297: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_226, arg153_1);  arg153_1 = None
        sum_160: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 1]);  mul_297 = None
        sum_161: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_226, [0, 1]);  add_226 = None
        add_229: "f32[4096]" = torch.ops.aten.add.Tensor(add_205, sum_160);  add_205 = sum_160 = None
        add_230: "f32[4096]" = torch.ops.aten.add.Tensor(add_206, sum_161);  add_206 = sum_161 = None
        convert_element_type_338: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_296, torch.bfloat16)
        view_311: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_338, _shape_param_321);  convert_element_type_338 = _shape_param_321 = None
        mm_124: "bf16[4096, 16384]" = torch.ops.aten.mm.default(view_311, arg18_1)
        permute_122: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_311, [1, 0])
        mm_125: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_122, arg155_1);  permute_122 = arg155_1 = None
        sum_162: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True, dtype = torch.float32);  view_311 = None
        view_312: "f32[4096]" = torch.ops.aten.view.default(sum_162, _shape_param_322);  sum_162 = _shape_param_322 = None
        convert_element_type_339: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_312, torch.bfloat16);  view_312 = None
        view_313: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(mm_124, _shape_param_323);  mm_124 = _shape_param_323 = None
        convert_element_type_340: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_313, torch.float32);  view_313 = None
        convert_element_type_341: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        add_231: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_207, convert_element_type_341);  add_207 = convert_element_type_341 = None
        convert_element_type_342: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_339, torch.float32);  convert_element_type_339 = None
        add_232: "f32[4096]" = torch.ops.aten.add.Tensor(add_208, convert_element_type_342);  add_208 = convert_element_type_342 = None
        view_314: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(arg156_1, _shape_param_324);  arg156_1 = _shape_param_324 = None
        mul_298: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_314, 0.5)
        mul_299: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_340, mul_298);  mul_298 = None
        convert_element_type_343: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_314, torch.float32)
        pow_23: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_343, 3.0)
        mul_300: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_233: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_314, mul_300);  view_314 = mul_300 = None
        mul_301: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_233, 0.7978845608028654);  add_233 = None
        tanh_11: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_301);  mul_301 = None
        add_234: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_302: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_340, add_234);  convert_element_type_340 = add_234 = None
        convert_element_type_344: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_302, torch.bfloat16);  mul_302 = None
        mul_303: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_59: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_303);  mul_303 = None
        mul_304: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_299, sub_59);  mul_299 = sub_59 = None
        mul_305: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_304, 0.7978845608028654);  mul_304 = None
        convert_element_type_345: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_305, torch.bfloat16)
        mul_306: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_305, 0.044715);  mul_305 = None
        pow_24: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_343, 2.0);  convert_element_type_343 = None
        mul_307: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
        mul_308: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_306, mul_307);  mul_306 = mul_307 = None
        convert_element_type_346: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_308, torch.bfloat16);  mul_308 = None
        add_235: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_345, convert_element_type_346);  convert_element_type_345 = convert_element_type_346 = None
        mul_309: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_344, 0.5);  convert_element_type_344 = None
        add_236: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_235, mul_309);  add_235 = mul_309 = None
        view_315: "bf16[4096, 16384]" = torch.ops.aten.view.default(add_236, _shape_param_325);  add_236 = _shape_param_325 = None
        mm_126: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_315, arg21_1)
        permute_123: "bf16[16384, 4096]" = torch.ops.aten.permute.default(view_315, [1, 0])
        mm_127: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_123, arg157_1);  permute_123 = arg157_1 = None
        sum_163: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_315, [0], True, dtype = torch.float32);  view_315 = None
        view_316: "f32[16384]" = torch.ops.aten.view.default(sum_163, _shape_param_326);  sum_163 = _shape_param_326 = None
        convert_element_type_347: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_316, torch.bfloat16);  view_316 = None
        view_317: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_126, _shape_param_327);  mm_126 = _shape_param_327 = None
        convert_element_type_348: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_317, torch.float32);  view_317 = None
        add_237: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_296, convert_element_type_348);  mul_296 = convert_element_type_348 = None
        convert_element_type_349: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        add_238: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_214, convert_element_type_349);  add_214 = convert_element_type_349 = None
        convert_element_type_350: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_347, torch.float32);  convert_element_type_347 = None
        add_239: "f32[16384]" = torch.ops.aten.add.Tensor(add_215, convert_element_type_350);  add_215 = convert_element_type_350 = None
        mul_310: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_237, arg23_1)
        mul_311: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_310, 4096)
        sum_164: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_310, [2], True)
        mul_312: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_310, arg158_1);  mul_310 = None
        sum_165: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_312, [2], True);  mul_312 = None
        mul_313: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg158_1, sum_165);  sum_165 = None
        sub_60: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_311, sum_164);  mul_311 = sum_164 = None
        sub_61: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_60, mul_313);  sub_60 = mul_313 = None
        mul_314: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg159_1, sub_61);  arg159_1 = sub_61 = None
        mul_315: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_237, arg158_1);  arg158_1 = None
        sum_166: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_315, [0, 1]);  mul_315 = None
        sum_167: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_237, [0, 1]);  add_237 = None
        add_240: "f32[4096]" = torch.ops.aten.add.Tensor(add_216, sum_166);  add_216 = sum_166 = None
        add_241: "f32[4096]" = torch.ops.aten.add.Tensor(add_217, sum_167);  add_217 = sum_167 = None
        convert_element_type_351: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_314, torch.bfloat16)
        view_318: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_351, _shape_param_328);  convert_element_type_351 = _shape_param_328 = None
        mm_128: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_318, arg26_1)
        permute_124: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_318, [1, 0])
        mm_129: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_124, arg160_1);  permute_124 = arg160_1 = None
        sum_168: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True, dtype = torch.float32);  view_318 = None
        view_319: "f32[4096]" = torch.ops.aten.view.default(sum_168, _shape_param_329);  sum_168 = _shape_param_329 = None
        convert_element_type_352: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_319, torch.bfloat16);  view_319 = None
        view_320: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_128, _shape_param_330);  mm_128 = _shape_param_330 = None
        convert_element_type_353: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        add_242: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_218, convert_element_type_353);  add_218 = convert_element_type_353 = None
        convert_element_type_354: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_352, torch.float32);  convert_element_type_352 = None
        add_243: "f32[4096]" = torch.ops.aten.add.Tensor(add_219, convert_element_type_354);  add_219 = convert_element_type_354 = None
        view_321: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_320, _shape_param_331);  view_320 = _shape_param_331 = None
        permute_125: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_321, [0, 2, 1, 3]);  view_321 = None
        clone_40: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_322: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_332);  clone_40 = _shape_param_332 = None
        expand_11: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(arg161_1, _shape_param_333);  _shape_param_333 = None
        view_323: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_11, _shape_param_334);  expand_11 = _shape_param_334 = None
        permute_126: "bf16[512, 512, 512]" = torch.ops.aten.permute.default(view_323, [0, 2, 1]);  view_323 = None
        bmm_40: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(permute_126, view_322);  permute_126 = None
        bmm_41: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_322, arg162_1);  view_322 = arg162_1 = None
        view_324: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_40, _shape_param_335);  bmm_40 = _shape_param_335 = None
        view_325: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_41, _shape_param_336);  bmm_41 = _shape_param_336 = None
        convert_element_type_355: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_325, torch.float32);  view_325 = None
        convert_element_type_356: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(arg161_1, torch.float32);  arg161_1 = None
        mul_316: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_355, convert_element_type_356);  convert_element_type_355 = None
        sum_169: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_316, [-1], True)
        neg_10: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_356);  convert_element_type_356 = None
        fma_10: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_10, sum_169, mul_316);  neg_10 = sum_169 = mul_316 = None
        convert_element_type_357: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None
        view_326: "bf16[512, 512, 512]" = torch.ops.aten.view.default(convert_element_type_357, _shape_param_337);  convert_element_type_357 = _shape_param_337 = None
        bmm_42: "bf16[512, 64, 512]" = torch.ops.aten.bmm.default(arg163_1, view_326);  arg163_1 = None
        bmm_43: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_326, arg164_1);  view_326 = arg164_1 = None
        view_327: "bf16[8, 64, 64, 512]" = torch.ops.aten.view.default(bmm_42, _shape_param_338);  bmm_42 = _shape_param_338 = None
        view_328: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_43, _shape_param_339);  bmm_43 = _shape_param_339 = None
        mul_317: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_327, 0.3535533905932738);  view_327 = None
        permute_127: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_317, [0, 1, 3, 2]);  mul_317 = None
        mul_318: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_328, 0.3535533905932738);  view_328 = None
        permute_128: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
        clone_41: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_329: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_41, _shape_param_340);  clone_41 = _shape_param_340 = None
        view_330: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_329, _shape_param_341);  view_329 = _shape_param_341 = None
        mm_130: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_330, arg32_1)
        permute_129: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_330, [1, 0])
        mm_131: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_129, arg165_1);  permute_129 = None
        sum_170: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True, dtype = torch.float32);  view_330 = None
        view_331: "f32[4096]" = torch.ops.aten.view.default(sum_170, _shape_param_342);  sum_170 = _shape_param_342 = None
        convert_element_type_358: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_331, torch.bfloat16);  view_331 = None
        view_332: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_130, _shape_param_343);  mm_130 = _shape_param_343 = None
        convert_element_type_359: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_332, torch.float32);  view_332 = None
        add_244: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_314, convert_element_type_359);  mul_314 = convert_element_type_359 = None
        convert_element_type_360: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        add_245: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_221, convert_element_type_360);  add_221 = convert_element_type_360 = None
        convert_element_type_361: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_358, torch.float32);  convert_element_type_358 = None
        add_246: "f32[4096]" = torch.ops.aten.add.Tensor(add_222, convert_element_type_361);  add_222 = convert_element_type_361 = None
        permute_130: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_127, [0, 2, 1, 3]);  permute_127 = None
        view_333: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(permute_130, _shape_param_344);  permute_130 = _shape_param_344 = None
        clone_42: "bf16[8, 512, 4096]" = torch.ops.aten.clone.default(view_333, memory_format = torch.contiguous_format);  view_333 = None
        view_334: "bf16[4096, 4096]" = torch.ops.aten.view.default(clone_42, _shape_param_345);  clone_42 = _shape_param_345 = None
        mm_132: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_334, arg34_1)
        permute_131: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_334, [1, 0])
        mm_133: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_131, arg165_1);  permute_131 = None
        sum_171: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True, dtype = torch.float32);  view_334 = None
        view_335: "f32[4096]" = torch.ops.aten.view.default(sum_171, _shape_param_346);  sum_171 = _shape_param_346 = None
        convert_element_type_362: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_335, torch.bfloat16);  view_335 = None
        view_336: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_132, _shape_param_347);  mm_132 = _shape_param_347 = None
        convert_element_type_363: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_336, torch.float32);  view_336 = None
        add_247: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_244, convert_element_type_363);  add_244 = convert_element_type_363 = None
        convert_element_type_364: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        add_248: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_224, convert_element_type_364);  add_224 = convert_element_type_364 = None
        convert_element_type_365: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_362, torch.float32);  convert_element_type_362 = None
        add_249: "f32[4096]" = torch.ops.aten.add.Tensor(add_225, convert_element_type_365);  add_225 = convert_element_type_365 = None
        permute_132: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_318, [0, 2, 1, 3]);  mul_318 = None
        clone_43: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
        view_337: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_43, _shape_param_348);  clone_43 = _shape_param_348 = None
        view_338: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_337, _shape_param_349);  view_337 = _shape_param_349 = None
        mm_134: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_338, arg35_1)
        permute_133: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_338, [1, 0])
        mm_135: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_133, arg165_1);  permute_133 = arg165_1 = None
        sum_172: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True, dtype = torch.float32);  view_338 = None
        view_339: "f32[4096]" = torch.ops.aten.view.default(sum_172, _shape_param_350);  sum_172 = _shape_param_350 = None
        convert_element_type_366: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_339, torch.bfloat16);  view_339 = None
        view_340: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_134, _shape_param_351);  mm_134 = _shape_param_351 = None
        convert_element_type_367: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_340, torch.float32);  view_340 = None
        add_250: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_247, convert_element_type_367);  add_247 = convert_element_type_367 = None
        convert_element_type_368: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        add_251: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_227, convert_element_type_368);  add_227 = convert_element_type_368 = None
        convert_element_type_369: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_366, torch.float32);  convert_element_type_366 = None
        add_252: "f32[4096]" = torch.ops.aten.add.Tensor(add_228, convert_element_type_369);  add_228 = convert_element_type_369 = None
        mul_319: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_250, arg15_1);  arg15_1 = None
        mul_320: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_319, 4096)
        sum_173: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True)
        mul_321: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_319, arg166_1);  mul_319 = None
        sum_174: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True);  mul_321 = None
        mul_322: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg166_1, sum_174);  sum_174 = None
        sub_62: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_320, sum_173);  mul_320 = sum_173 = None
        sub_63: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_62, mul_322);  sub_62 = mul_322 = None
        mul_323: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(arg167_1, sub_63);  arg167_1 = sub_63 = None
        mul_324: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_250, arg166_1);  arg166_1 = None
        sum_175: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_324, [0, 1]);  mul_324 = None
        sum_176: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_250, [0, 1]);  add_250 = None
        add_253: "f32[4096]" = torch.ops.aten.add.Tensor(add_229, sum_175);  add_229 = sum_175 = None
        add_254: "f32[4096]" = torch.ops.aten.add.Tensor(add_230, sum_176);  add_230 = sum_176 = None
        convert_element_type_370: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_323, torch.bfloat16)
        view_341: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_370, _shape_param_352);  convert_element_type_370 = _shape_param_352 = None
        mm_136: "bf16[4096, 16384]" = torch.ops.aten.mm.default(view_341, arg18_1);  arg18_1 = None
        permute_134: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_341, [1, 0])
        mm_137: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_134, arg168_1);  permute_134 = arg168_1 = None
        sum_177: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True, dtype = torch.float32);  view_341 = None
        view_342: "f32[4096]" = torch.ops.aten.view.default(sum_177, _shape_param_353);  sum_177 = _shape_param_353 = None
        convert_element_type_371: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_342, torch.bfloat16);  view_342 = None
        view_343: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(mm_136, _shape_param_354);  mm_136 = _shape_param_354 = None
        convert_element_type_372: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_343, torch.float32);  view_343 = None
        convert_element_type_373: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        add_255: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_231, convert_element_type_373);  add_231 = convert_element_type_373 = None
        convert_element_type_374: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_371, torch.float32);  convert_element_type_371 = None
        add_256: "f32[4096]" = torch.ops.aten.add.Tensor(add_232, convert_element_type_374);  add_232 = convert_element_type_374 = None
        view_344: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(arg169_1, _shape_param_355);  arg169_1 = _shape_param_355 = None
        mul_325: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_344, 0.5)
        mul_326: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_372, mul_325);  mul_325 = None
        convert_element_type_375: "f32[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(view_344, torch.float32)
        pow_25: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_375, 3.0)
        mul_327: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_25, 0.044715);  pow_25 = None
        add_257: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_344, mul_327);  view_344 = mul_327 = None
        mul_328: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_257, 0.7978845608028654);  add_257 = None
        tanh_12: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_328);  mul_328 = None
        add_258: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
        mul_329: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_372, add_258);  convert_element_type_372 = add_258 = None
        convert_element_type_376: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_329, torch.bfloat16);  mul_329 = None
        mul_330: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
        sub_64: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_330);  mul_330 = None
        mul_331: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_326, sub_64);  mul_326 = sub_64 = None
        mul_332: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_331, 0.7978845608028654);  mul_331 = None
        convert_element_type_377: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_332, torch.bfloat16)
        mul_333: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_332, 0.044715);  mul_332 = None
        pow_26: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_375, 2.0);  convert_element_type_375 = None
        mul_334: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_26, 3.0);  pow_26 = None
        mul_335: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
        convert_element_type_378: "bf16[8, 512, 16384]" = torch.ops.prims.convert_element_type.default(mul_335, torch.bfloat16);  mul_335 = None
        add_259: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_377, convert_element_type_378);  convert_element_type_377 = convert_element_type_378 = None
        mul_336: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_376, 0.5);  convert_element_type_376 = None
        add_260: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_259, mul_336);  add_259 = mul_336 = None
        view_345: "bf16[4096, 16384]" = torch.ops.aten.view.default(add_260, _shape_param_356);  add_260 = _shape_param_356 = None
        mm_138: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_345, arg21_1);  arg21_1 = None
        permute_135: "bf16[16384, 4096]" = torch.ops.aten.permute.default(view_345, [1, 0])
        mm_139: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_135, arg170_1);  permute_135 = arg170_1 = None
        sum_178: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_345, [0], True, dtype = torch.float32);  view_345 = None
        view_346: "f32[16384]" = torch.ops.aten.view.default(sum_178, _shape_param_357);  sum_178 = _shape_param_357 = None
        convert_element_type_379: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_346, torch.bfloat16);  view_346 = None
        view_347: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_138, _shape_param_358);  mm_138 = _shape_param_358 = None
        convert_element_type_380: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_347, torch.float32);  view_347 = None
        add_261: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_323, convert_element_type_380);  mul_323 = convert_element_type_380 = None
        convert_element_type_381: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        add_262: "f32[16384, 4096]" = torch.ops.aten.add.Tensor(add_238, convert_element_type_381);  add_238 = convert_element_type_381 = None
        convert_element_type_382: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_379, torch.float32);  convert_element_type_379 = None
        add_263: "f32[16384]" = torch.ops.aten.add.Tensor(add_239, convert_element_type_382);  add_239 = convert_element_type_382 = None
        mul_337: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_261, arg23_1);  arg23_1 = None
        mul_338: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_337, 4096)
        sum_179: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_337, [2], True)
        convert_element_type_383: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(arg171_1, torch.float32);  arg171_1 = None
        sub_65: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_383, arg172_1);  convert_element_type_383 = arg172_1 = None
        mul_339: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_65, arg173_1);  sub_65 = None
        mul_340: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_337, mul_339);  mul_337 = None
        sum_180: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_340, [2], True);  mul_340 = None
        mul_341: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_339, sum_180);  sum_180 = None
        sub_66: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(mul_338, sum_179);  mul_338 = sum_179 = None
        sub_67: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(sub_66, mul_341);  sub_66 = mul_341 = None
        div_2: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(arg173_1, 4096);  arg173_1 = None
        mul_342: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(div_2, sub_67);  div_2 = sub_67 = None
        mul_343: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(add_261, mul_339);  mul_339 = None
        sum_181: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 1]);  mul_343 = None
        sum_182: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_261, [0, 1]);  add_261 = None
        add_264: "f32[4096]" = torch.ops.aten.add.Tensor(add_240, sum_181);  add_240 = sum_181 = None
        add_265: "f32[4096]" = torch.ops.aten.add.Tensor(add_241, sum_182);  add_241 = sum_182 = None
        convert_element_type_384: "bf16[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(mul_342, torch.bfloat16);  mul_342 = None
        view_348: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_384, _shape_param_359);  _shape_param_359 = None
        mm_140: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_348, arg26_1);  arg26_1 = None
        permute_136: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_348, [1, 0])
        mm_141: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_136, arg174_1);  permute_136 = arg174_1 = None
        sum_183: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_348, [0], True, dtype = torch.float32);  view_348 = None
        view_349: "f32[4096]" = torch.ops.aten.view.default(sum_183, _shape_param_360);  sum_183 = _shape_param_360 = None
        convert_element_type_385: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_349, torch.bfloat16);  view_349 = None
        view_350: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_140, _shape_param_361);  mm_140 = _shape_param_361 = None
        convert_element_type_386: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        add_266: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_242, convert_element_type_386);  add_242 = convert_element_type_386 = None
        convert_element_type_387: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_385, torch.float32);  convert_element_type_385 = None
        add_267: "f32[4096]" = torch.ops.aten.add.Tensor(add_243, convert_element_type_387);  add_243 = convert_element_type_387 = None
        view_351: "bf16[8, 512, 64, 64]" = torch.ops.aten.view.default(view_350, _shape_param_362);  view_350 = _shape_param_362 = None
        permute_137: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None
        clone_44: "bf16[8, 64, 512, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
        view_352: "bf16[512, 512, 64]" = torch.ops.aten.view.default(clone_44, _shape_param_363);  clone_44 = _shape_param_363 = None
        expand_12: "bf16[8, 64, 512, 512]" = torch.ops.aten.expand.default(arg175_1, _shape_param_364);  _shape_param_364 = None
        view_353: "bf16[512, 512, 512]" = torch.ops.aten.view.default(expand_12, _shape_param_365);  expand_12 = _shape_param_365 = None
        permute_138: "bf16[512, 512, 512]" = torch.ops.aten.permute.default(view_353, [0, 2, 1]);  view_353 = None
        bmm_44: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(permute_138, view_352);  permute_138 = None
        bmm_45: "bf16[512, 512, 512]" = torch.ops.aten.bmm.default(view_352, arg176_1);  view_352 = arg176_1 = None
        view_354: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_44, _shape_param_366);  bmm_44 = _shape_param_366 = None
        view_355: "bf16[8, 64, 512, 512]" = torch.ops.aten.view.default(bmm_45, _shape_param_367);  bmm_45 = _shape_param_367 = None
        convert_element_type_388: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(view_355, torch.float32);  view_355 = None
        convert_element_type_389: "f32[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(arg175_1, torch.float32);  arg175_1 = None
        mul_344: "f32[8, 64, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_388, convert_element_type_389);  convert_element_type_388 = None
        sum_184: "f32[8, 64, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [-1], True)
        neg_11: "f32[8, 64, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_389);  convert_element_type_389 = None
        fma_11: "f32[8, 64, 512, 512]" = torch.ops.prims.fma.default(neg_11, sum_184, mul_344);  neg_11 = sum_184 = mul_344 = None
        convert_element_type_390: "bf16[8, 64, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None
        view_356: "bf16[512, 512, 512]" = torch.ops.aten.view.default(convert_element_type_390, _shape_param_368);  convert_element_type_390 = _shape_param_368 = None
        bmm_46: "bf16[512, 64, 512]" = torch.ops.aten.bmm.default(arg177_1, view_356);  arg177_1 = None
        bmm_47: "bf16[512, 512, 64]" = torch.ops.aten.bmm.default(view_356, arg178_1);  view_356 = arg178_1 = None
        view_357: "bf16[8, 64, 64, 512]" = torch.ops.aten.view.default(bmm_46, _shape_param_369);  bmm_46 = _shape_param_369 = None
        view_358: "bf16[8, 64, 512, 64]" = torch.ops.aten.view.default(bmm_47, _shape_param_370);  bmm_47 = _shape_param_370 = None
        mul_345: "bf16[8, 64, 64, 512]" = torch.ops.aten.mul.Scalar(view_357, 0.3535533905932738);  view_357 = None
        permute_139: "bf16[8, 64, 512, 64]" = torch.ops.aten.permute.default(mul_345, [0, 1, 3, 2]);  mul_345 = None
        mul_346: "bf16[8, 64, 512, 64]" = torch.ops.aten.mul.Scalar(view_358, 0.3535533905932738);  view_358 = None
        permute_140: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
        clone_45: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_359: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_45, _shape_param_371);  clone_45 = _shape_param_371 = None
        view_360: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_359, _shape_param_372);  view_359 = _shape_param_372 = None
        mm_142: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_360, arg32_1);  arg32_1 = None
        permute_141: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_360, [1, 0])
        mm_143: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_141, arg179_1);  permute_141 = None
        sum_185: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_360, [0], True, dtype = torch.float32);  view_360 = None
        view_361: "f32[4096]" = torch.ops.aten.view.default(sum_185, _shape_param_373);  sum_185 = _shape_param_373 = None
        convert_element_type_391: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_361, torch.bfloat16);  view_361 = None
        view_362: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_142, _shape_param_374);  mm_142 = _shape_param_374 = None
        add_268: "bf16[8, 512, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_384, view_362);  convert_element_type_384 = view_362 = None
        convert_element_type_392: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        add_269: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_245, convert_element_type_392);  add_245 = convert_element_type_392 = None
        convert_element_type_393: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_391, torch.float32);  convert_element_type_391 = None
        add_270: "f32[4096]" = torch.ops.aten.add.Tensor(add_246, convert_element_type_393);  add_246 = convert_element_type_393 = None
        permute_142: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(permute_139, [0, 2, 1, 3]);  permute_139 = None
        view_363: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(permute_142, _shape_param_375);  permute_142 = _shape_param_375 = None
        clone_46: "bf16[8, 512, 4096]" = torch.ops.aten.clone.default(view_363, memory_format = torch.contiguous_format);  view_363 = None
        view_364: "bf16[4096, 4096]" = torch.ops.aten.view.default(clone_46, _shape_param_376);  clone_46 = _shape_param_376 = None
        mm_144: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_364, arg34_1);  arg34_1 = None
        permute_143: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_364, [1, 0])
        mm_145: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_143, arg179_1);  permute_143 = None
        sum_186: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_364, [0], True, dtype = torch.float32);  view_364 = None
        view_365: "f32[4096]" = torch.ops.aten.view.default(sum_186, _shape_param_377);  sum_186 = _shape_param_377 = None
        convert_element_type_394: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_365, torch.bfloat16);  view_365 = None
        view_366: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_144, _shape_param_378);  mm_144 = _shape_param_378 = None
        add_271: "bf16[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_268, view_366);  add_268 = view_366 = None
        convert_element_type_395: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        add_272: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_248, convert_element_type_395);  add_248 = convert_element_type_395 = None
        convert_element_type_396: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_394, torch.float32);  convert_element_type_394 = None
        add_273: "f32[4096]" = torch.ops.aten.add.Tensor(add_249, convert_element_type_396);  add_249 = convert_element_type_396 = None
        permute_144: "bf16[8, 512, 64, 64]" = torch.ops.aten.permute.default(mul_346, [0, 2, 1, 3]);  mul_346 = None
        clone_47: "bf16[8, 512, 64, 64]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        view_367: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(clone_47, _shape_param_379);  clone_47 = _shape_param_379 = None
        view_368: "bf16[4096, 4096]" = torch.ops.aten.view.default(view_367, _shape_param_380);  view_367 = _shape_param_380 = None
        mm_146: "bf16[4096, 4096]" = torch.ops.aten.mm.default(view_368, arg35_1);  arg35_1 = None
        permute_145: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_368, [1, 0])
        mm_147: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_145, arg179_1);  permute_145 = arg179_1 = None
        sum_187: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_368, [0], True, dtype = torch.float32);  view_368 = None
        view_369: "f32[4096]" = torch.ops.aten.view.default(sum_187, _shape_param_381);  sum_187 = _shape_param_381 = None
        convert_element_type_397: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_369, torch.bfloat16);  view_369 = None
        view_370: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_146, _shape_param_382);  mm_146 = _shape_param_382 = None
        add_274: "bf16[8, 512, 4096]" = torch.ops.aten.add.Tensor(add_271, view_370);  add_271 = view_370 = None
        convert_element_type_398: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        add_275: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_251, convert_element_type_398);  add_251 = convert_element_type_398 = None
        convert_element_type_399: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_397, torch.float32);  convert_element_type_397 = None
        add_276: "f32[4096]" = torch.ops.aten.add.Tensor(add_252, convert_element_type_399);  add_252 = convert_element_type_399 = None
        view_371: "bf16[4096, 4096]" = torch.ops.aten.view.default(add_274, _shape_param_383);  add_274 = _shape_param_383 = None
        mm_148: "bf16[4096, 128]" = torch.ops.aten.mm.default(view_371, arg180_1);  arg180_1 = None
        permute_146: "bf16[4096, 4096]" = torch.ops.aten.permute.default(view_371, [1, 0])
        mm_149: "bf16[4096, 128]" = torch.ops.aten.mm.default(permute_146, arg181_1);  permute_146 = arg181_1 = None
        sum_188: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_371, [0], True, dtype = torch.float32);  view_371 = None
        view_372: "f32[4096]" = torch.ops.aten.view.default(sum_188, _shape_param_384);  sum_188 = _shape_param_384 = None
        convert_element_type_400: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_372, torch.bfloat16);  view_372 = None
        view_373: "bf16[8, 512, 128]" = torch.ops.aten.view.default(mm_148, _shape_param_385);  mm_148 = _shape_param_385 = None
        convert_element_type_401: "f32[8, 512, 128]" = torch.ops.prims.convert_element_type.default(view_373, torch.float32);  view_373 = None
        convert_element_type_402: "f32[4096, 128]" = torch.ops.prims.convert_element_type.default(mm_149, torch.float32);  mm_149 = None
        convert_element_type_403: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_400, torch.float32);  convert_element_type_400 = None
        mul_347: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_401, arg182_1);  arg182_1 = None
        mul_348: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_347, 128)
        sum_189: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True)
        mul_349: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(mul_347, arg183_1);  mul_347 = None
        sum_190: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
        mul_350: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(arg183_1, sum_190);  sum_190 = None
        sub_68: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(mul_348, sum_189);  mul_348 = sum_189 = None
        sub_69: "f32[8, 512, 128]" = torch.ops.aten.sub.Tensor(sub_68, mul_350);  sub_68 = mul_350 = None
        mul_351: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(arg184_1, sub_69);  arg184_1 = sub_69 = None
        mul_352: "f32[8, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_401, arg183_1);  arg183_1 = None
        sum_191: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 1]);  mul_352 = None
        sum_192: "f32[128]" = torch.ops.aten.sum.dim_IntList(convert_element_type_401, [0, 1]);  convert_element_type_401 = None
        sum_193: "f32[1, 512, 128]" = torch.ops.aten.sum.dim_IntList(mul_351, [0], True, dtype = torch.float32)
        ge: "b8[1, 512]" = torch.ops.aten.ge.Scalar(arg185_1, 0)
        lt: "b8[1, 512]" = torch.ops.aten.lt.Scalar(arg185_1, 512)
        bitwise_and: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne_1: "b8[1, 512]" = torch.ops.aten.ne.Scalar(arg185_1, -1)
        bitwise_and_1: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne_1);  bitwise_and = ne_1 = None
        unsqueeze_1: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_2: "f32[512, 128]" = torch.ops.aten.full.default(_shape_param_386, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_386 = None
        _unsafe_masked_index_put_accumulate: "f32[512, 128]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_2, unsqueeze_1, [arg185_1], sum_193);  full_2 = unsqueeze_1 = arg185_1 = sum_193 = None
        expand_13: "i64[8, 512]" = torch.ops.aten.expand.default(arg186_1, _shape_param_387);  arg186_1 = _shape_param_387 = None
        ge_1: "b8[8, 512]" = torch.ops.aten.ge.Scalar(expand_13, 0)
        lt_1: "b8[8, 512]" = torch.ops.aten.lt.Scalar(expand_13, 2)
        bitwise_and_2: "b8[8, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_2: "b8[8, 512]" = torch.ops.aten.ne.Scalar(expand_13, -1)
        bitwise_and_3: "b8[8, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_2);  bitwise_and_2 = ne_2 = None
        unsqueeze_2: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_3: "f32[2, 128]" = torch.ops.aten.full.default(_shape_param_388, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_388 = None
        _unsafe_masked_index_put_accumulate_1: "f32[2, 128]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_3, unsqueeze_2, [expand_13], mul_351);  full_3 = unsqueeze_2 = expand_13 = None
        ge_2: "b8[8, 512]" = torch.ops.aten.ge.Scalar(arg187_1, 0)
        lt_2: "b8[8, 512]" = torch.ops.aten.lt.Scalar(arg187_1, 30000)
        bitwise_and_4: "b8[8, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_2, lt_2);  ge_2 = lt_2 = None
        ne_3: "b8[8, 512]" = torch.ops.aten.ne.Scalar(arg187_1, 0)
        bitwise_and_5: "b8[8, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, ne_3);  bitwise_and_4 = ne_3 = None
        unsqueeze_3: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_5, -1);  bitwise_and_5 = None
        full_4: "f32[30000, 128]" = torch.ops.aten.full.default(_shape_param_389, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_389 = None
        _unsafe_masked_index_put_accumulate_2: "f32[30000, 128]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_4, unsqueeze_3, [arg187_1], mul_351);  full_4 = unsqueeze_3 = arg187_1 = mul_351 = None
        add_277: "f32[30000, 128]" = torch.ops.aten.add.Tensor(convert_element_type_8, _unsafe_masked_index_put_accumulate_2);  convert_element_type_8 = _unsafe_masked_index_put_accumulate_2 = None
        return (convert_element_type_9, sum_5, sum_6, convert_element_type_16, convert_element_type_17, add_253, add_254, add_255, add_256, add_262, add_263, add_264, add_265, add_266, add_267, add_269, add_270, add_272, add_273, add_275, add_276, convert_element_type_402, convert_element_type_403, sum_191, sum_192, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1, add_277)



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
