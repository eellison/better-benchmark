"""
Standalone repro captured via capture_hook.
Label: T5ForConditionalGeneration
Pattern hash: cfe70883331c
Shape hash: ab455d2c
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
    def forward(self, arg0_1: "i64[8, 1024]", arg1_1: "bf16[32128, 512]", arg2_1: "i64[8, 1024]", arg3_1: "bf16[512]", arg4_1: "bf16[512, 512]", arg5_1: "bf16[512, 512]", arg6_1: "bf16[32, 8]", arg7_1: "bf16[512, 512]", arg8_1: "bf16[512, 512]", arg9_1: "bf16[512]", arg10_1: "bf16[512, 512]", arg11_1: "bf16[512]", arg12_1: "bf16[512, 512]", arg13_1: "bf16[512, 512]", arg14_1: "bf16[32, 8]", arg15_1: "bf16[512, 512]", arg16_1: "bf16[512, 512]", arg17_1: "bf16[512]", arg18_1: "bf16[2048, 512]", arg19_1: "bf16[512, 2048]", arg20_1: "bf16[512]", arg21_1: "bf16[512, 512]", arg22_1: "bf16[512, 512]", arg23_1: "bf16[512, 512]", arg24_1: "bf16[512, 512]", arg25_1: "bf16[512]", arg26_1: "bf16[2048, 512]", arg27_1: "bf16[512, 2048]", arg28_1: "bf16[512]", arg29_1: "bf16[512, 512]", arg30_1: "bf16[512, 512]", arg31_1: "bf16[512, 512]", arg32_1: "bf16[512, 512]", arg33_1: "bf16[512]", arg34_1: "bf16[2048, 512]", arg35_1: "bf16[512, 2048]", arg36_1: "bf16[512]", arg37_1: "bf16[512, 512]", arg38_1: "bf16[512, 512]", arg39_1: "bf16[512, 512]", arg40_1: "bf16[512, 512]", arg41_1: "bf16[512]", arg42_1: "bf16[2048, 512]", arg43_1: "bf16[512, 2048]", arg44_1: "bf16[512]", arg45_1: "bf16[512, 512]", arg46_1: "bf16[512, 512]", arg47_1: "bf16[512, 512]", arg48_1: "bf16[512, 512]", arg49_1: "bf16[512]", arg50_1: "bf16[2048, 512]", arg51_1: "bf16[512, 2048]", arg52_1: "bf16[512]", arg53_1: "bf16[512, 512]", arg54_1: "bf16[512, 512]", arg55_1: "bf16[512, 512]", arg56_1: "bf16[512, 512]", arg57_1: "bf16[512]", arg58_1: "bf16[2048, 512]", arg59_1: "bf16[512, 2048]", arg60_1: "bf16[512]", arg61_1: "bf16[512, 512]", arg62_1: "bf16[512, 512]", arg63_1: "bf16[512, 512]", arg64_1: "bf16[512]", arg65_1: "bf16[2048, 512]", arg66_1: "bf16[512, 2048]", arg67_1: "bf16[512]", arg68_1: "bf16[512, 512]", arg69_1: "bf16[512, 512]", arg70_1: "bf16[512, 512]", arg71_1: "bf16[512, 512]", arg72_1: "bf16[512]", arg73_1: "bf16[512, 512]", arg74_1: "bf16[512, 512]", arg75_1: "bf16[512, 512]", arg76_1: "bf16[512, 512]", arg77_1: "bf16[512]", arg78_1: "bf16[2048, 512]", arg79_1: "bf16[512, 2048]", arg80_1: "bf16[512]", arg81_1: "bf16[512, 512]", arg82_1: "bf16[512, 512]", arg83_1: "bf16[512, 512]", arg84_1: "bf16[512, 512]", arg85_1: "bf16[512]", arg86_1: "bf16[512, 512]", arg87_1: "bf16[512, 512]", arg88_1: "bf16[512, 512]", arg89_1: "bf16[512, 512]", arg90_1: "bf16[512]", arg91_1: "bf16[2048, 512]", arg92_1: "bf16[512, 2048]", arg93_1: "bf16[512]", arg94_1: "bf16[512, 512]", arg95_1: "bf16[512, 512]", arg96_1: "bf16[512, 512]", arg97_1: "bf16[512, 512]", arg98_1: "bf16[512]", arg99_1: "bf16[512, 512]", arg100_1: "bf16[512, 512]", arg101_1: "bf16[512, 512]", arg102_1: "bf16[512, 512]", arg103_1: "bf16[512]", arg104_1: "bf16[2048, 512]", arg105_1: "bf16[512, 2048]", arg106_1: "bf16[512]", arg107_1: "bf16[512, 512]", arg108_1: "bf16[512, 512]", arg109_1: "bf16[512, 512]", arg110_1: "bf16[512, 512]", arg111_1: "bf16[512]", arg112_1: "bf16[512, 512]", arg113_1: "bf16[512, 512]", arg114_1: "bf16[512, 512]", arg115_1: "bf16[512, 512]", arg116_1: "bf16[512]", arg117_1: "bf16[2048, 512]", arg118_1: "bf16[512, 2048]", arg119_1: "bf16[512]", arg120_1: "bf16[512, 512]", arg121_1: "bf16[512, 512]", arg122_1: "bf16[512, 512]", arg123_1: "bf16[512, 512]", arg124_1: "bf16[512]", arg125_1: "bf16[512, 512]", arg126_1: "bf16[512, 512]", arg127_1: "bf16[512, 512]", arg128_1: "bf16[512, 512]", arg129_1: "bf16[512]", arg130_1: "bf16[2048, 512]", arg131_1: "bf16[512, 2048]", arg132_1: "bf16[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451):
        # No stacktrace found for following nodes
        view: "i64[8192]" = torch.ops.aten.view.default(arg0_1, [-1]);  arg0_1 = None
        ne: "b8[8192]" = torch.ops.aten.ne.Scalar(view, -100)
        embedding: "bf16[8, 1024, 512]" = torch.ops.aten.embedding.default(arg1_1, arg2_1)
        convert_element_type: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32)
        pow_1: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2);  convert_element_type = None
        mean: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(embedding, rsqrt);  rsqrt = None
        convert_element_type_1: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg3_1, convert_element_type_1);  arg3_1 = convert_element_type_1 = None
        view_1: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_0);  _shape_param_0 = None
        permute: "bf16[512, 512]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_1, permute);  view_1 = permute = None
        view_2: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        view_3: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_2, _shape_param_2);  view_2 = _shape_param_2 = None
        permute_1: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        expand: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_1, _shape_param_3);  permute_1 = _shape_param_3 = None
        clone: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_4: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        view_5: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_5);  _shape_param_5 = None
        permute_2: "bf16[512, 512]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_1: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_5, permute_2);  view_5 = permute_2 = None
        view_6: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_1, _shape_param_6);  mm_1 = _shape_param_6 = None
        view_7: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_6, _shape_param_7);  view_6 = _shape_param_7 = None
        permute_3: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        permute_4: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand_1: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_4, _shape_param_8);  permute_4 = _shape_param_8 = None
        clone_1: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_8: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_1, _shape_param_9);  clone_1 = _shape_param_9 = None
        bmm: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_4, view_8);  view_4 = view_8 = None
        view_9: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm, _shape_param_10);  bmm = _shape_param_10 = None
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        iota_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        add_1: "i64[1024, 1]" = torch.ops.aten.add.Tensor(unsqueeze_1, 0);  unsqueeze_1 = None
        sub: "i64[1024, 1024]" = torch.ops.aten.sub.Tensor(unsqueeze, add_1);  unsqueeze = add_1 = None
        full: "i64[1024, 1024]" = torch.ops.aten.full.default(_shape_param_11, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_11 = None
        minimum: "i64[1024, 1024]" = torch.ops.aten.minimum.default(sub, full);  sub = full = None
        neg: "i64[1024, 1024]" = torch.ops.aten.neg.default(minimum);  minimum = None
        lt: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(neg, 16)
        convert_element_type_2: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
        div: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_2, 16);  convert_element_type_2 = None
        log: "f32[1024, 1024]" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log, 2.0794415416798357);  log = None
        mul_2: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_1, 16);  div_1 = None
        convert_element_type_3: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_2, torch.int64);  mul_2 = None
        add_2: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_3, 16);  convert_element_type_3 = None
        full_1: "i64[1024, 1024]" = torch.ops.aten.full.default(_shape_param_12, 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_12 = None
        minimum_1: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_2, full_1);  add_2 = full_1 = None
        where: "i64[1024, 1024]" = torch.ops.aten.where.self(lt, neg, minimum_1);  lt = neg = minimum_1 = None
        add_3: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(where, 0);  where = None
        embedding_1: "bf16[1024, 1024, 8]" = torch.ops.aten.embedding.default(arg6_1, add_3);  arg6_1 = add_3 = None
        permute_5: "bf16[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
        unsqueeze_2: "bf16[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_5, 0);  permute_5 = None
        iota_2: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_4: "i64[1024]" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None
        unsqueeze_3: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_4, 0);  add_4 = None
        unsqueeze_4: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        iota_3: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_5: "i64[1024]" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None
        unsqueeze_6: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_5, 0);  add_5 = None
        unsqueeze_7: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        unsqueeze_8: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        le: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_8);  unsqueeze_5 = unsqueeze_8 = None
        expand_2: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(le, _shape_param_13);  le = _shape_param_13 = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_3: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_2, full_2, full_3);  expand_2 = full_2 = full_3 = None
        add_6: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_2, where_1);  unsqueeze_2 = where_1 = None
        add_7: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_9, add_6);  view_9 = None
        convert_element_type_4: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_7, torch.float32);  add_7 = None
        amax: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_4, [-1], True)
        sub_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_4, amax);  convert_element_type_4 = amax = None
        exp: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_5: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        expand_3: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_5, _shape_param_14);  convert_element_type_5 = _shape_param_14 = None
        view_10: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_3, _shape_param_15);  expand_3 = _shape_param_15 = None
        view_11: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_16);  mul_1 = _shape_param_16 = None
        permute_6: "bf16[512, 512]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_2: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_11, permute_6);  view_11 = permute_6 = None
        view_12: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_2, _shape_param_17);  mm_2 = _shape_param_17 = None
        view_13: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_12, _shape_param_18);  view_12 = _shape_param_18 = None
        permute_7: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        expand_4: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_7, _shape_param_19);  permute_7 = _shape_param_19 = None
        clone_2: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_14: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_20);  clone_2 = _shape_param_20 = None
        bmm_1: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_10, view_14);  view_10 = view_14 = None
        view_15: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_21);  bmm_1 = _shape_param_21 = None
        permute_8: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        clone_3: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_16: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_3, _shape_param_22);  clone_3 = _shape_param_22 = None
        view_17: "bf16[8192, 512]" = torch.ops.aten.view.default(view_16, _shape_param_23);  view_16 = _shape_param_23 = None
        permute_9: "bf16[512, 512]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        mm_3: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_17, permute_9);  view_17 = permute_9 = None
        view_18: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_3, _shape_param_24);  mm_3 = _shape_param_24 = None
        add_8: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(embedding, view_18);  embedding = view_18 = None
        convert_element_type_6: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_8, torch.float32)
        pow_2: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_6, 2);  convert_element_type_6 = None
        mean_1: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_9: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_8, rsqrt_1);  rsqrt_1 = None
        convert_element_type_7: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        mul_4: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg9_1, convert_element_type_7);  arg9_1 = convert_element_type_7 = None
        view_19: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_4, _shape_param_25);  mul_4 = _shape_param_25 = None
        permute_10: "bf16[512, 512]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm_4: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_19, permute_10);  view_19 = permute_10 = None
        view_20: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_4, _shape_param_26);  mm_4 = _shape_param_26 = None
        view_21: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_20, _shape_param_27);  view_20 = _shape_param_27 = None
        permute_11: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        expand_5: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_11, _shape_param_28);  permute_11 = _shape_param_28 = None
        clone_4: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_22: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_29);  clone_4 = _shape_param_29 = None
        embedding_2: "bf16[8, 1024, 512]" = torch.ops.aten.embedding.default(arg1_1, arg2_1);  arg2_1 = None
        convert_element_type_8: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(embedding_2, torch.float32)
        pow_3: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_8, 2);  convert_element_type_8 = None
        mean_2: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        add_10: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_5: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(embedding_2, rsqrt_2);  rsqrt_2 = None
        convert_element_type_9: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        mul_6: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg11_1, convert_element_type_9);  arg11_1 = convert_element_type_9 = None
        view_23: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_6, _shape_param_30);  _shape_param_30 = None
        permute_12: "bf16[512, 512]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        mm_5: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_23, permute_12);  view_23 = permute_12 = None
        view_24: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_5, _shape_param_31);  mm_5 = _shape_param_31 = None
        view_25: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_24, _shape_param_32);  view_24 = _shape_param_32 = None
        permute_13: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        expand_6: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_13, _shape_param_33);  permute_13 = _shape_param_33 = None
        clone_5: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_26: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_5, _shape_param_34);  clone_5 = _shape_param_34 = None
        view_27: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_6, _shape_param_35);  _shape_param_35 = None
        permute_14: "bf16[512, 512]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        mm_6: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_27, permute_14);  view_27 = permute_14 = None
        view_28: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_6, _shape_param_36);  mm_6 = _shape_param_36 = None
        view_29: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_28, _shape_param_37);  view_28 = _shape_param_37 = None
        permute_15: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        permute_16: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        expand_7: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_16, _shape_param_38);  permute_16 = _shape_param_38 = None
        clone_6: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_30: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_6, _shape_param_39);  clone_6 = _shape_param_39 = None
        bmm_2: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_26, view_30);  view_26 = view_30 = None
        view_31: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_2, _shape_param_40);  bmm_2 = _shape_param_40 = None
        iota_4: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_9: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None
        iota_5: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_10: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(iota_5, 1);  iota_5 = None
        add_11: "i64[1024, 1]" = torch.ops.aten.add.Tensor(unsqueeze_10, 0);  unsqueeze_10 = None
        sub_2: "i64[1024, 1024]" = torch.ops.aten.sub.Tensor(unsqueeze_9, add_11);  unsqueeze_9 = add_11 = None
        gt: "b8[1024, 1024]" = torch.ops.aten.gt.Scalar(sub_2, 0)
        convert_element_type_10: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(gt, torch.int64);  gt = None
        mul_7: "i64[1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 16);  convert_element_type_10 = None
        add_12: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(mul_7, 0);  mul_7 = None
        abs_1: "i64[1024, 1024]" = torch.ops.aten.abs.default(sub_2);  sub_2 = None
        lt_1: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(abs_1, 8)
        convert_element_type_11: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
        div_3: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_11, 8);  convert_element_type_11 = None
        log_1: "f32[1024, 1024]" = torch.ops.aten.log.default(div_3);  div_3 = None
        div_4: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log_1, 2.772588722239781);  log_1 = None
        mul_8: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_4, 8);  div_4 = None
        convert_element_type_12: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_8, torch.int64);  mul_8 = None
        add_13: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_12, 8);  convert_element_type_12 = None
        full_4: "i64[1024, 1024]" = torch.ops.aten.full.default(_shape_param_41, 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_41 = None
        minimum_2: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_13, full_4);  add_13 = full_4 = None
        where_2: "i64[1024, 1024]" = torch.ops.aten.where.self(lt_1, abs_1, minimum_2);  lt_1 = abs_1 = minimum_2 = None
        add_14: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(add_12, where_2);  add_12 = where_2 = None
        embedding_3: "bf16[1024, 1024, 8]" = torch.ops.aten.embedding.default(arg14_1, add_14);  arg14_1 = add_14 = None
        permute_17: "bf16[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
        unsqueeze_11: "bf16[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_17, 0);  permute_17 = None
        full_5: "bf16[8, 1, 1024, 1024]" = torch.ops.aten.full.default(_shape_param_42, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_42 = None
        add_15: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_11, full_5);  unsqueeze_11 = full_5 = None
        add_16: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_31, add_15);  view_31 = None
        convert_element_type_13: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None
        amax_1: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_13, [-1], True)
        sub_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_13, amax_1);  convert_element_type_13 = amax_1 = None
        exp_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_2: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_14: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        expand_8: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_14, _shape_param_43);  convert_element_type_14 = _shape_param_43 = None
        view_32: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_8, _shape_param_44);  expand_8 = _shape_param_44 = None
        view_33: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_6, _shape_param_45);  mul_6 = _shape_param_45 = None
        permute_18: "bf16[512, 512]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_7: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_33, permute_18);  view_33 = permute_18 = None
        view_34: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_7, _shape_param_46);  mm_7 = _shape_param_46 = None
        view_35: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_34, _shape_param_47);  view_34 = _shape_param_47 = None
        permute_19: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
        expand_9: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_19, _shape_param_48);  permute_19 = _shape_param_48 = None
        clone_7: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_36: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_7, _shape_param_49);  clone_7 = _shape_param_49 = None
        bmm_3: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_32, view_36);  view_32 = view_36 = None
        view_37: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_50);  bmm_3 = _shape_param_50 = None
        permute_20: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        clone_8: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
        view_38: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_8, _shape_param_51);  clone_8 = _shape_param_51 = None
        view_39: "bf16[8192, 512]" = torch.ops.aten.view.default(view_38, _shape_param_52);  view_38 = _shape_param_52 = None
        permute_21: "bf16[512, 512]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        mm_8: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_39, permute_21);  view_39 = permute_21 = None
        view_40: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_8, _shape_param_53);  mm_8 = _shape_param_53 = None
        add_17: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(embedding_2, view_40);  embedding_2 = view_40 = None
        convert_element_type_15: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_17, torch.float32)
        pow_4: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_15, 2);  convert_element_type_15 = None
        mean_3: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
        add_18: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_9: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_17, rsqrt_3);  rsqrt_3 = None
        convert_element_type_16: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        mul_10: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg17_1, convert_element_type_16);  arg17_1 = convert_element_type_16 = None
        view_41: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_10, _shape_param_54);  mul_10 = _shape_param_54 = None
        permute_22: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        mm_9: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_41, permute_22);  view_41 = permute_22 = None
        view_42: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_9, _shape_param_55);  mm_9 = _shape_param_55 = None
        relu: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_42);  view_42 = None
        view_43: "bf16[8192, 2048]" = torch.ops.aten.view.default(relu, _shape_param_56);  relu = _shape_param_56 = None
        permute_23: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_10: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_43, permute_23);  view_43 = permute_23 = None
        view_44: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_10, _shape_param_57);  mm_10 = _shape_param_57 = None
        add_19: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_17, view_44);  add_17 = view_44 = None
        convert_element_type_17: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_19, torch.float32)
        pow_5: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_17, 2);  convert_element_type_17 = None
        mean_4: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
        add_20: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_11: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_19, rsqrt_4);  rsqrt_4 = None
        convert_element_type_18: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        mul_12: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg20_1, convert_element_type_18);  arg20_1 = convert_element_type_18 = None
        view_45: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_12, _shape_param_58);  _shape_param_58 = None
        permute_24: "bf16[512, 512]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        mm_11: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_45, permute_24);  view_45 = permute_24 = None
        view_46: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_11, _shape_param_59);  mm_11 = _shape_param_59 = None
        view_47: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_46, _shape_param_60);  view_46 = _shape_param_60 = None
        permute_25: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
        expand_10: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_25, _shape_param_61);  permute_25 = _shape_param_61 = None
        clone_9: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_48: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_9, _shape_param_62);  clone_9 = _shape_param_62 = None
        view_49: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_12, _shape_param_63);  _shape_param_63 = None
        permute_26: "bf16[512, 512]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_12: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_49, permute_26);  view_49 = permute_26 = None
        view_50: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_12, _shape_param_64);  mm_12 = _shape_param_64 = None
        view_51: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_50, _shape_param_65);  view_50 = _shape_param_65 = None
        permute_27: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        permute_28: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_27, [0, 1, 3, 2]);  permute_27 = None
        expand_11: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_28, _shape_param_66);  permute_28 = _shape_param_66 = None
        clone_10: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_52: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_10, _shape_param_67);  clone_10 = _shape_param_67 = None
        bmm_4: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_48, view_52);  view_48 = view_52 = None
        view_53: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_4, _shape_param_68);  bmm_4 = _shape_param_68 = None
        add_21: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_53, add_15);  view_53 = None
        convert_element_type_19: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_21, torch.float32);  add_21 = None
        amax_2: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_19, [-1], True)
        sub_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_19, amax_2);  convert_element_type_19 = amax_2 = None
        exp_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_3: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_20: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        expand_12: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_20, _shape_param_69);  convert_element_type_20 = _shape_param_69 = None
        view_54: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_12, _shape_param_70);  expand_12 = _shape_param_70 = None
        view_55: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_12, _shape_param_71);  mul_12 = _shape_param_71 = None
        permute_29: "bf16[512, 512]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_13: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_55, permute_29);  view_55 = permute_29 = None
        view_56: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_13, _shape_param_72);  mm_13 = _shape_param_72 = None
        view_57: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_56, _shape_param_73);  view_56 = _shape_param_73 = None
        permute_30: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
        expand_13: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_30, _shape_param_74);  permute_30 = _shape_param_74 = None
        clone_11: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_58: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_11, _shape_param_75);  clone_11 = _shape_param_75 = None
        bmm_5: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_54, view_58);  view_54 = view_58 = None
        view_59: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_5, _shape_param_76);  bmm_5 = _shape_param_76 = None
        permute_31: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
        clone_12: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
        view_60: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_12, _shape_param_77);  clone_12 = _shape_param_77 = None
        view_61: "bf16[8192, 512]" = torch.ops.aten.view.default(view_60, _shape_param_78);  view_60 = _shape_param_78 = None
        permute_32: "bf16[512, 512]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        mm_14: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_61, permute_32);  view_61 = permute_32 = None
        view_62: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_14, _shape_param_79);  mm_14 = _shape_param_79 = None
        add_22: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_19, view_62);  add_19 = view_62 = None
        convert_element_type_21: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_22, torch.float32)
        pow_6: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_21, 2);  convert_element_type_21 = None
        mean_5: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None
        add_23: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_13: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_22, rsqrt_5);  rsqrt_5 = None
        convert_element_type_22: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        mul_14: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg25_1, convert_element_type_22);  arg25_1 = convert_element_type_22 = None
        view_63: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_14, _shape_param_80);  mul_14 = _shape_param_80 = None
        permute_33: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        mm_15: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_63, permute_33);  view_63 = permute_33 = None
        view_64: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_15, _shape_param_81);  mm_15 = _shape_param_81 = None
        relu_1: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_64);  view_64 = None
        view_65: "bf16[8192, 2048]" = torch.ops.aten.view.default(relu_1, _shape_param_82);  relu_1 = _shape_param_82 = None
        permute_34: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        mm_16: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_65, permute_34);  view_65 = permute_34 = None
        view_66: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_16, _shape_param_83);  mm_16 = _shape_param_83 = None
        add_24: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_22, view_66);  add_22 = view_66 = None
        convert_element_type_23: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_24, torch.float32)
        pow_7: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_23, 2);  convert_element_type_23 = None
        mean_6: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
        add_25: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_15: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_24, rsqrt_6);  rsqrt_6 = None
        convert_element_type_24: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16);  mul_15 = None
        mul_16: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg28_1, convert_element_type_24);  arg28_1 = convert_element_type_24 = None
        view_67: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_16, _shape_param_84);  _shape_param_84 = None
        permute_35: "bf16[512, 512]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_17: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_67, permute_35);  view_67 = permute_35 = None
        view_68: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_17, _shape_param_85);  mm_17 = _shape_param_85 = None
        view_69: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_68, _shape_param_86);  view_68 = _shape_param_86 = None
        permute_36: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        expand_14: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_36, _shape_param_87);  permute_36 = _shape_param_87 = None
        clone_13: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_70: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_13, _shape_param_88);  clone_13 = _shape_param_88 = None
        view_71: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_16, _shape_param_89);  _shape_param_89 = None
        permute_37: "bf16[512, 512]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        mm_18: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_71, permute_37);  view_71 = permute_37 = None
        view_72: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_18, _shape_param_90);  mm_18 = _shape_param_90 = None
        view_73: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_72, _shape_param_91);  view_72 = _shape_param_91 = None
        permute_38: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
        permute_39: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_38, [0, 1, 3, 2]);  permute_38 = None
        expand_15: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_39, _shape_param_92);  permute_39 = _shape_param_92 = None
        clone_14: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_74: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_14, _shape_param_93);  clone_14 = _shape_param_93 = None
        bmm_6: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_70, view_74);  view_70 = view_74 = None
        view_75: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_6, _shape_param_94);  bmm_6 = _shape_param_94 = None
        add_26: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_75, add_15);  view_75 = None
        convert_element_type_25: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_26, torch.float32);  add_26 = None
        amax_3: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_25, [-1], True)
        sub_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_25, amax_3);  convert_element_type_25 = amax_3 = None
        exp_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_4: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_26: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        expand_16: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_26, _shape_param_95);  convert_element_type_26 = _shape_param_95 = None
        view_76: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_16, _shape_param_96);  expand_16 = _shape_param_96 = None
        view_77: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_16, _shape_param_97);  mul_16 = _shape_param_97 = None
        permute_40: "bf16[512, 512]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_19: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_77, permute_40);  view_77 = permute_40 = None
        view_78: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_19, _shape_param_98);  mm_19 = _shape_param_98 = None
        view_79: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_78, _shape_param_99);  view_78 = _shape_param_99 = None
        permute_41: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None
        expand_17: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_41, _shape_param_100);  permute_41 = _shape_param_100 = None
        clone_15: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_80: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_15, _shape_param_101);  clone_15 = _shape_param_101 = None
        bmm_7: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_76, view_80);  view_76 = view_80 = None
        view_81: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_102);  bmm_7 = _shape_param_102 = None
        permute_42: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None
        clone_16: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
        view_82: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_16, _shape_param_103);  clone_16 = _shape_param_103 = None
        view_83: "bf16[8192, 512]" = torch.ops.aten.view.default(view_82, _shape_param_104);  view_82 = _shape_param_104 = None
        permute_43: "bf16[512, 512]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        mm_20: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_83, permute_43);  view_83 = permute_43 = None
        view_84: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_20, _shape_param_105);  mm_20 = _shape_param_105 = None
        add_27: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_24, view_84);  add_24 = view_84 = None
        convert_element_type_27: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_27, torch.float32)
        pow_8: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_27, 2);  convert_element_type_27 = None
        mean_7: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
        add_28: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_17: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_27, rsqrt_7);  rsqrt_7 = None
        convert_element_type_28: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        mul_18: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg33_1, convert_element_type_28);  arg33_1 = convert_element_type_28 = None
        view_85: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_18, _shape_param_106);  mul_18 = _shape_param_106 = None
        permute_44: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm_21: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_85, permute_44);  view_85 = permute_44 = None
        view_86: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_21, _shape_param_107);  mm_21 = _shape_param_107 = None
        relu_2: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_86);  view_86 = None
        view_87: "bf16[8192, 2048]" = torch.ops.aten.view.default(relu_2, _shape_param_108);  relu_2 = _shape_param_108 = None
        permute_45: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        mm_22: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_87, permute_45);  view_87 = permute_45 = None
        view_88: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_22, _shape_param_109);  mm_22 = _shape_param_109 = None
        add_29: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_27, view_88);  add_27 = view_88 = None
        convert_element_type_29: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_29, torch.float32)
        pow_9: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_29, 2);  convert_element_type_29 = None
        mean_8: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None
        add_30: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_19: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_29, rsqrt_8);  rsqrt_8 = None
        convert_element_type_30: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16);  mul_19 = None
        mul_20: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg36_1, convert_element_type_30);  arg36_1 = convert_element_type_30 = None
        view_89: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_20, _shape_param_110);  _shape_param_110 = None
        permute_46: "bf16[512, 512]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_23: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_89, permute_46);  view_89 = permute_46 = None
        view_90: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_23, _shape_param_111);  mm_23 = _shape_param_111 = None
        view_91: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_90, _shape_param_112);  view_90 = _shape_param_112 = None
        permute_47: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_91, [0, 2, 1, 3]);  view_91 = None
        expand_18: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_47, _shape_param_113);  permute_47 = _shape_param_113 = None
        clone_17: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_92: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_17, _shape_param_114);  clone_17 = _shape_param_114 = None
        view_93: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_20, _shape_param_115);  _shape_param_115 = None
        permute_48: "bf16[512, 512]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        mm_24: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_93, permute_48);  view_93 = permute_48 = None
        view_94: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_24, _shape_param_116);  mm_24 = _shape_param_116 = None
        view_95: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_94, _shape_param_117);  view_94 = _shape_param_117 = None
        permute_49: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
        permute_50: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_49, [0, 1, 3, 2]);  permute_49 = None
        expand_19: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_50, _shape_param_118);  permute_50 = _shape_param_118 = None
        clone_18: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_96: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_18, _shape_param_119);  clone_18 = _shape_param_119 = None
        bmm_8: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_92, view_96);  view_92 = view_96 = None
        view_97: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_8, _shape_param_120);  bmm_8 = _shape_param_120 = None
        add_31: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_97, add_15);  view_97 = None
        convert_element_type_31: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_31, torch.float32);  add_31 = None
        amax_4: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_31, [-1], True)
        sub_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_31, amax_4);  convert_element_type_31 = amax_4 = None
        exp_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_5: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_8: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_32: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        expand_20: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_32, _shape_param_121);  convert_element_type_32 = _shape_param_121 = None
        view_98: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_20, _shape_param_122);  expand_20 = _shape_param_122 = None
        view_99: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_20, _shape_param_123);  mul_20 = _shape_param_123 = None
        permute_51: "bf16[512, 512]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        mm_25: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_99, permute_51);  view_99 = permute_51 = None
        view_100: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_25, _shape_param_124);  mm_25 = _shape_param_124 = None
        view_101: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_100, _shape_param_125);  view_100 = _shape_param_125 = None
        permute_52: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        expand_21: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_52, _shape_param_126);  permute_52 = _shape_param_126 = None
        clone_19: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_102: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_19, _shape_param_127);  clone_19 = _shape_param_127 = None
        bmm_9: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_98, view_102);  view_98 = view_102 = None
        view_103: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_9, _shape_param_128);  bmm_9 = _shape_param_128 = None
        permute_53: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_103, [0, 2, 1, 3]);  view_103 = None
        clone_20: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
        view_104: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_20, _shape_param_129);  clone_20 = _shape_param_129 = None
        view_105: "bf16[8192, 512]" = torch.ops.aten.view.default(view_104, _shape_param_130);  view_104 = _shape_param_130 = None
        permute_54: "bf16[512, 512]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        mm_26: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_105, permute_54);  view_105 = permute_54 = None
        view_106: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_26, _shape_param_131);  mm_26 = _shape_param_131 = None
        add_32: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_29, view_106);  add_29 = view_106 = None
        convert_element_type_33: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_32, torch.float32)
        pow_10: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_33, 2);  convert_element_type_33 = None
        mean_9: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
        add_33: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        mul_21: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_32, rsqrt_9);  rsqrt_9 = None
        convert_element_type_34: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        mul_22: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg41_1, convert_element_type_34);  arg41_1 = convert_element_type_34 = None
        view_107: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_22, _shape_param_132);  mul_22 = _shape_param_132 = None
        permute_55: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        mm_27: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_107, permute_55);  view_107 = permute_55 = None
        view_108: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_27, _shape_param_133);  mm_27 = _shape_param_133 = None
        relu_3: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_108);  view_108 = None
        view_109: "bf16[8192, 2048]" = torch.ops.aten.view.default(relu_3, _shape_param_134);  relu_3 = _shape_param_134 = None
        permute_56: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        mm_28: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_109, permute_56);  view_109 = permute_56 = None
        view_110: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_28, _shape_param_135);  mm_28 = _shape_param_135 = None
        add_34: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_32, view_110);  add_32 = view_110 = None
        convert_element_type_35: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_34, torch.float32)
        pow_11: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_35, 2);  convert_element_type_35 = None
        mean_10: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
        add_35: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_23: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_34, rsqrt_10);  rsqrt_10 = None
        convert_element_type_36: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_23, torch.bfloat16);  mul_23 = None
        mul_24: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg44_1, convert_element_type_36);  arg44_1 = convert_element_type_36 = None
        view_111: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_24, _shape_param_136);  _shape_param_136 = None
        permute_57: "bf16[512, 512]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        mm_29: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_111, permute_57);  view_111 = permute_57 = None
        view_112: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_29, _shape_param_137);  mm_29 = _shape_param_137 = None
        view_113: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_112, _shape_param_138);  view_112 = _shape_param_138 = None
        permute_58: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        expand_22: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_58, _shape_param_139);  permute_58 = _shape_param_139 = None
        clone_21: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_114: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_21, _shape_param_140);  clone_21 = _shape_param_140 = None
        view_115: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_24, _shape_param_141);  _shape_param_141 = None
        permute_59: "bf16[512, 512]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        mm_30: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_115, permute_59);  view_115 = permute_59 = None
        view_116: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_30, _shape_param_142);  mm_30 = _shape_param_142 = None
        view_117: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_116, _shape_param_143);  view_116 = _shape_param_143 = None
        permute_60: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        permute_61: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_60, [0, 1, 3, 2]);  permute_60 = None
        expand_23: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_61, _shape_param_144);  permute_61 = _shape_param_144 = None
        clone_22: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_118: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_22, _shape_param_145);  clone_22 = _shape_param_145 = None
        bmm_10: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_114, view_118);  view_114 = view_118 = None
        view_119: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_10, _shape_param_146);  bmm_10 = _shape_param_146 = None
        add_36: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_119, add_15);  view_119 = None
        convert_element_type_37: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_36, torch.float32);  add_36 = None
        amax_5: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_37, [-1], True)
        sub_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_37, amax_5);  convert_element_type_37 = amax_5 = None
        exp_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_6: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_9: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_38: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        expand_24: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_38, _shape_param_147);  convert_element_type_38 = _shape_param_147 = None
        view_120: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_24, _shape_param_148);  expand_24 = _shape_param_148 = None
        view_121: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_24, _shape_param_149);  mul_24 = _shape_param_149 = None
        permute_62: "bf16[512, 512]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        mm_31: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_121, permute_62);  view_121 = permute_62 = None
        view_122: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_31, _shape_param_150);  mm_31 = _shape_param_150 = None
        view_123: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_122, _shape_param_151);  view_122 = _shape_param_151 = None
        permute_63: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
        expand_25: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_63, _shape_param_152);  permute_63 = _shape_param_152 = None
        clone_23: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_124: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_23, _shape_param_153);  clone_23 = _shape_param_153 = None
        bmm_11: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_120, view_124);  view_120 = view_124 = None
        view_125: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_154);  bmm_11 = _shape_param_154 = None
        permute_64: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
        clone_24: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
        view_126: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_24, _shape_param_155);  clone_24 = _shape_param_155 = None
        view_127: "bf16[8192, 512]" = torch.ops.aten.view.default(view_126, _shape_param_156);  view_126 = _shape_param_156 = None
        permute_65: "bf16[512, 512]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
        mm_32: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_127, permute_65);  view_127 = permute_65 = None
        view_128: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_32, _shape_param_157);  mm_32 = _shape_param_157 = None
        add_37: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_34, view_128);  add_34 = view_128 = None
        convert_element_type_39: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_37, torch.float32)
        pow_12: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_39, 2);  convert_element_type_39 = None
        mean_11: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None
        add_38: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_25: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_37, rsqrt_11);  rsqrt_11 = None
        convert_element_type_40: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None
        mul_26: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg49_1, convert_element_type_40);  arg49_1 = convert_element_type_40 = None
        view_129: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_26, _shape_param_158);  mul_26 = _shape_param_158 = None
        permute_66: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_33: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_129, permute_66);  view_129 = permute_66 = None
        view_130: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_33, _shape_param_159);  mm_33 = _shape_param_159 = None
        relu_4: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_130);  view_130 = None
        view_131: "bf16[8192, 2048]" = torch.ops.aten.view.default(relu_4, _shape_param_160);  relu_4 = _shape_param_160 = None
        permute_67: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        mm_34: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_131, permute_67);  view_131 = permute_67 = None
        view_132: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_34, _shape_param_161);  mm_34 = _shape_param_161 = None
        add_39: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_37, view_132);  add_37 = view_132 = None
        convert_element_type_41: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_39, torch.float32)
        pow_13: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_41, 2);  convert_element_type_41 = None
        mean_12: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
        add_40: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_27: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_39, rsqrt_12);  rsqrt_12 = None
        convert_element_type_42: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16);  mul_27 = None
        mul_28: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg52_1, convert_element_type_42);  arg52_1 = convert_element_type_42 = None
        view_133: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_28, _shape_param_162);  _shape_param_162 = None
        permute_68: "bf16[512, 512]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        mm_35: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_133, permute_68);  view_133 = permute_68 = None
        view_134: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_35, _shape_param_163);  mm_35 = _shape_param_163 = None
        view_135: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_134, _shape_param_164);  view_134 = _shape_param_164 = None
        permute_69: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
        expand_26: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_69, _shape_param_165);  permute_69 = _shape_param_165 = None
        clone_25: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_136: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_25, _shape_param_166);  clone_25 = _shape_param_166 = None
        view_137: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_28, _shape_param_167);  _shape_param_167 = None
        permute_70: "bf16[512, 512]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        mm_36: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_137, permute_70);  view_137 = permute_70 = None
        view_138: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_36, _shape_param_168);  mm_36 = _shape_param_168 = None
        view_139: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_138, _shape_param_169);  view_138 = _shape_param_169 = None
        permute_71: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
        permute_72: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_71, [0, 1, 3, 2]);  permute_71 = None
        expand_27: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_72, _shape_param_170);  permute_72 = _shape_param_170 = None
        clone_26: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_140: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_26, _shape_param_171);  clone_26 = _shape_param_171 = None
        bmm_12: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_136, view_140);  view_136 = view_140 = None
        view_141: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_12, _shape_param_172);  bmm_12 = _shape_param_172 = None
        add_41: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_141, add_15);  view_141 = add_15 = None
        convert_element_type_43: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_41, torch.float32);  add_41 = None
        amax_6: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_43, [-1], True)
        sub_8: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_43, amax_6);  convert_element_type_43 = amax_6 = None
        exp_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_7: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_10: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_44: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        expand_28: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_44, _shape_param_173);  convert_element_type_44 = _shape_param_173 = None
        view_142: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_28, _shape_param_174);  expand_28 = _shape_param_174 = None
        view_143: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_28, _shape_param_175);  mul_28 = _shape_param_175 = None
        permute_73: "bf16[512, 512]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_37: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_143, permute_73);  view_143 = permute_73 = None
        view_144: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_37, _shape_param_176);  mm_37 = _shape_param_176 = None
        view_145: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_144, _shape_param_177);  view_144 = _shape_param_177 = None
        permute_74: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
        expand_29: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_74, _shape_param_178);  permute_74 = _shape_param_178 = None
        clone_27: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_146: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_27, _shape_param_179);  clone_27 = _shape_param_179 = None
        bmm_13: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_142, view_146);  view_142 = view_146 = None
        view_147: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_13, _shape_param_180);  bmm_13 = _shape_param_180 = None
        permute_75: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None
        clone_28: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
        view_148: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_28, _shape_param_181);  clone_28 = _shape_param_181 = None
        view_149: "bf16[8192, 512]" = torch.ops.aten.view.default(view_148, _shape_param_182);  view_148 = _shape_param_182 = None
        permute_76: "bf16[512, 512]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        mm_38: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_149, permute_76);  view_149 = permute_76 = None
        view_150: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_38, _shape_param_183);  mm_38 = _shape_param_183 = None
        add_42: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_39, view_150);  add_39 = view_150 = None
        convert_element_type_45: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_42, torch.float32)
        pow_14: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_45, 2);  convert_element_type_45 = None
        mean_13: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
        add_43: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_29: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_42, rsqrt_13);  rsqrt_13 = None
        convert_element_type_46: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        mul_30: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg57_1, convert_element_type_46);  arg57_1 = convert_element_type_46 = None
        view_151: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_30, _shape_param_184);  mul_30 = _shape_param_184 = None
        permute_77: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        mm_39: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_151, permute_77);  view_151 = permute_77 = None
        view_152: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_39, _shape_param_185);  mm_39 = _shape_param_185 = None
        relu_5: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_152);  view_152 = None
        view_153: "bf16[8192, 2048]" = torch.ops.aten.view.default(relu_5, _shape_param_186);  relu_5 = _shape_param_186 = None
        permute_78: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        mm_40: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_153, permute_78);  view_153 = permute_78 = None
        view_154: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_40, _shape_param_187);  mm_40 = _shape_param_187 = None
        add_44: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_42, view_154);  add_42 = view_154 = None
        convert_element_type_47: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_44, torch.float32)
        pow_15: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_47, 2);  convert_element_type_47 = None
        mean_14: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None
        add_45: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_31: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_44, rsqrt_14);  add_44 = rsqrt_14 = None
        convert_element_type_48: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_31, torch.bfloat16);  mul_31 = None
        mul_32: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg60_1, convert_element_type_48);  arg60_1 = convert_element_type_48 = None
        view_155: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_32, _shape_param_188);  _shape_param_188 = None
        permute_79: "bf16[512, 512]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        mm_41: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_155, permute_79);  view_155 = permute_79 = None
        view_156: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_41, _shape_param_189);  mm_41 = _shape_param_189 = None
        view_157: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_156, _shape_param_190);  view_156 = _shape_param_190 = None
        permute_80: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
        permute_81: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2]);  permute_80 = None
        expand_30: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_81, _shape_param_191);  permute_81 = _shape_param_191 = None
        clone_29: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_158: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_29, _shape_param_192);  clone_29 = _shape_param_192 = None
        bmm_14: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_22, view_158);  view_22 = view_158 = None
        view_159: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_14, _shape_param_193);  bmm_14 = _shape_param_193 = None
        convert_element_type_49: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_159, torch.float32);  view_159 = None
        amax_7: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_49, [-1], True)
        sub_9: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_49, amax_7);  convert_element_type_49 = amax_7 = None
        exp_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_8: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_50: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        expand_31: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_50, _shape_param_194);  convert_element_type_50 = _shape_param_194 = None
        view_160: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_31, _shape_param_195);  expand_31 = _shape_param_195 = None
        view_161: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_32, _shape_param_196);  _shape_param_196 = None
        permute_82: "bf16[512, 512]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        mm_42: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_161, permute_82);  view_161 = permute_82 = None
        view_162: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_42, _shape_param_197);  mm_42 = _shape_param_197 = None
        view_163: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_162, _shape_param_198);  view_162 = _shape_param_198 = None
        permute_83: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
        expand_32: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_83, _shape_param_199);  permute_83 = _shape_param_199 = None
        clone_30: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_164: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_30, _shape_param_200);  clone_30 = _shape_param_200 = None
        bmm_15: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_160, view_164);  view_160 = view_164 = None
        view_165: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_201);  bmm_15 = _shape_param_201 = None
        permute_84: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        clone_31: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_166: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_31, _shape_param_202);  clone_31 = _shape_param_202 = None
        view_167: "bf16[8192, 512]" = torch.ops.aten.view.default(view_166, _shape_param_203);  view_166 = _shape_param_203 = None
        permute_85: "bf16[512, 512]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_43: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_167, permute_85);  view_167 = permute_85 = None
        view_168: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_43, _shape_param_204);  mm_43 = _shape_param_204 = None
        add_46: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_8, view_168);  add_8 = view_168 = None
        convert_element_type_51: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_46, torch.float32)
        pow_16: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_51, 2);  convert_element_type_51 = None
        mean_15: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
        add_47: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_33: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_46, rsqrt_15);  rsqrt_15 = None
        convert_element_type_52: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_33, torch.bfloat16);  mul_33 = None
        mul_34: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg64_1, convert_element_type_52);  arg64_1 = convert_element_type_52 = None
        view_169: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_34, _shape_param_205);  mul_34 = _shape_param_205 = None
        permute_86: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        mm_44: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_169, permute_86);  view_169 = permute_86 = None
        view_170: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_44, _shape_param_206);  mm_44 = _shape_param_206 = None
        relu_6: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_170);  view_170 = None
        view_171: "bf16[8192, 2048]" = torch.ops.aten.view.default(relu_6, _shape_param_207);  relu_6 = _shape_param_207 = None
        permute_87: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        mm_45: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_171, permute_87);  view_171 = permute_87 = None
        view_172: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_45, _shape_param_208);  mm_45 = _shape_param_208 = None
        add_48: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_46, view_172);  add_46 = view_172 = None
        convert_element_type_53: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_48, torch.float32)
        pow_17: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_53, 2);  convert_element_type_53 = None
        mean_16: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
        add_49: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_35: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_48, rsqrt_16);  rsqrt_16 = None
        convert_element_type_54: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None
        mul_36: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg67_1, convert_element_type_54);  arg67_1 = convert_element_type_54 = None
        view_173: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_36, _shape_param_209);  _shape_param_209 = None
        permute_88: "bf16[512, 512]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        mm_46: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_173, permute_88);  view_173 = permute_88 = None
        view_174: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_46, _shape_param_210);  mm_46 = _shape_param_210 = None
        view_175: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_174, _shape_param_211);  view_174 = _shape_param_211 = None
        permute_89: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
        expand_33: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_89, _shape_param_212);  permute_89 = _shape_param_212 = None
        clone_32: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_176: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_32, _shape_param_213);  clone_32 = _shape_param_213 = None
        view_177: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_36, _shape_param_214);  _shape_param_214 = None
        permute_90: "bf16[512, 512]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_47: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_177, permute_90);  view_177 = permute_90 = None
        view_178: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_47, _shape_param_215);  mm_47 = _shape_param_215 = None
        view_179: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_178, _shape_param_216);  view_178 = _shape_param_216 = None
        permute_91: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_179, [0, 2, 1, 3]);  view_179 = None
        permute_92: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
        expand_34: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_92, _shape_param_217);  permute_92 = _shape_param_217 = None
        clone_33: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_180: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_33, _shape_param_218);  clone_33 = _shape_param_218 = None
        bmm_16: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_176, view_180);  view_176 = view_180 = None
        view_181: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_16, _shape_param_219);  bmm_16 = _shape_param_219 = None
        add_50: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_181, add_6);  view_181 = None
        convert_element_type_55: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_50, torch.float32);  add_50 = None
        amax_8: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_55, [-1], True)
        sub_10: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_55, amax_8);  convert_element_type_55 = amax_8 = None
        exp_8: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_9: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_12: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_56: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None
        expand_35: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_56, _shape_param_220);  convert_element_type_56 = _shape_param_220 = None
        view_182: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_35, _shape_param_221);  expand_35 = _shape_param_221 = None
        view_183: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_36, _shape_param_222);  mul_36 = _shape_param_222 = None
        permute_93: "bf16[512, 512]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_48: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_183, permute_93);  view_183 = permute_93 = None
        view_184: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_48, _shape_param_223);  mm_48 = _shape_param_223 = None
        view_185: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_184, _shape_param_224);  view_184 = _shape_param_224 = None
        permute_94: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        expand_36: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_94, _shape_param_225);  permute_94 = _shape_param_225 = None
        clone_34: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_186: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_34, _shape_param_226);  clone_34 = _shape_param_226 = None
        bmm_17: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_182, view_186);  view_182 = view_186 = None
        view_187: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_17, _shape_param_227);  bmm_17 = _shape_param_227 = None
        permute_95: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
        clone_35: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_188: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_35, _shape_param_228);  clone_35 = _shape_param_228 = None
        view_189: "bf16[8192, 512]" = torch.ops.aten.view.default(view_188, _shape_param_229);  view_188 = _shape_param_229 = None
        permute_96: "bf16[512, 512]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        mm_49: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_189, permute_96);  view_189 = permute_96 = None
        view_190: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_49, _shape_param_230);  mm_49 = _shape_param_230 = None
        add_51: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_48, view_190);  add_48 = view_190 = None
        convert_element_type_57: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_51, torch.float32)
        pow_18: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_57, 2);  convert_element_type_57 = None
        mean_17: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None
        add_52: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_37: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_51, rsqrt_17);  rsqrt_17 = None
        convert_element_type_58: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None
        mul_38: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg72_1, convert_element_type_58);  arg72_1 = convert_element_type_58 = None
        view_191: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_38, _shape_param_231);  mul_38 = _shape_param_231 = None
        permute_97: "bf16[512, 512]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        mm_50: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_191, permute_97);  view_191 = permute_97 = None
        view_192: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_50, _shape_param_232);  mm_50 = _shape_param_232 = None
        view_193: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_192, _shape_param_233);  view_192 = _shape_param_233 = None
        permute_98: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
        expand_37: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_98, _shape_param_234);  permute_98 = _shape_param_234 = None
        clone_36: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_194: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_235);  clone_36 = _shape_param_235 = None
        view_195: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_32, _shape_param_236);  _shape_param_236 = None
        permute_99: "bf16[512, 512]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        mm_51: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_195, permute_99);  view_195 = permute_99 = None
        view_196: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_51, _shape_param_237);  mm_51 = _shape_param_237 = None
        view_197: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_196, _shape_param_238);  view_196 = _shape_param_238 = None
        permute_100: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_197, [0, 2, 1, 3]);  view_197 = None
        permute_101: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2]);  permute_100 = None
        expand_38: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_101, _shape_param_239);  permute_101 = _shape_param_239 = None
        clone_37: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_198: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_37, _shape_param_240);  clone_37 = _shape_param_240 = None
        bmm_18: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_194, view_198);  view_194 = view_198 = None
        view_199: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_18, _shape_param_241);  bmm_18 = _shape_param_241 = None
        convert_element_type_59: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_199, torch.float32);  view_199 = None
        amax_9: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_59, [-1], True)
        sub_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_59, amax_9);  convert_element_type_59 = amax_9 = None
        exp_9: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_10: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_13: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_60: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None
        expand_39: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_60, _shape_param_242);  convert_element_type_60 = _shape_param_242 = None
        view_200: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_39, _shape_param_243);  expand_39 = _shape_param_243 = None
        view_201: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_32, _shape_param_244);  _shape_param_244 = None
        permute_102: "bf16[512, 512]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        mm_52: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_201, permute_102);  view_201 = permute_102 = None
        view_202: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_52, _shape_param_245);  mm_52 = _shape_param_245 = None
        view_203: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_202, _shape_param_246);  view_202 = _shape_param_246 = None
        permute_103: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
        expand_40: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_103, _shape_param_247);  permute_103 = _shape_param_247 = None
        clone_38: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_204: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_38, _shape_param_248);  clone_38 = _shape_param_248 = None
        bmm_19: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_200, view_204);  view_200 = view_204 = None
        view_205: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_249);  bmm_19 = _shape_param_249 = None
        permute_104: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
        clone_39: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_206: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_39, _shape_param_250);  clone_39 = _shape_param_250 = None
        view_207: "bf16[8192, 512]" = torch.ops.aten.view.default(view_206, _shape_param_251);  view_206 = _shape_param_251 = None
        permute_105: "bf16[512, 512]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        mm_53: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_207, permute_105);  view_207 = permute_105 = None
        view_208: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_53, _shape_param_252);  mm_53 = _shape_param_252 = None
        add_53: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_51, view_208);  add_51 = view_208 = None
        convert_element_type_61: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_53, torch.float32)
        pow_19: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_61, 2);  convert_element_type_61 = None
        mean_18: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
        add_54: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_39: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_53, rsqrt_18);  rsqrt_18 = None
        convert_element_type_62: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None
        mul_40: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg77_1, convert_element_type_62);  arg77_1 = convert_element_type_62 = None
        view_209: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_40, _shape_param_253);  mul_40 = _shape_param_253 = None
        permute_106: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        mm_54: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_209, permute_106);  view_209 = permute_106 = None
        view_210: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_54, _shape_param_254);  mm_54 = _shape_param_254 = None
        relu_7: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_210);  view_210 = None
        view_211: "bf16[8192, 2048]" = torch.ops.aten.view.default(relu_7, _shape_param_255);  relu_7 = _shape_param_255 = None
        permute_107: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        mm_55: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_211, permute_107);  view_211 = permute_107 = None
        view_212: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_55, _shape_param_256);  mm_55 = _shape_param_256 = None
        add_55: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_53, view_212);  add_53 = view_212 = None
        convert_element_type_63: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_55, torch.float32)
        pow_20: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_63, 2);  convert_element_type_63 = None
        mean_19: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
        add_56: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_41: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_55, rsqrt_19);  rsqrt_19 = None
        convert_element_type_64: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_41, torch.bfloat16);  mul_41 = None
        mul_42: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg80_1, convert_element_type_64);  arg80_1 = convert_element_type_64 = None
        view_213: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_42, _shape_param_257);  _shape_param_257 = None
        permute_108: "bf16[512, 512]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        mm_56: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_213, permute_108);  view_213 = permute_108 = None
        view_214: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_56, _shape_param_258);  mm_56 = _shape_param_258 = None
        view_215: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_214, _shape_param_259);  view_214 = _shape_param_259 = None
        permute_109: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
        expand_41: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_109, _shape_param_260);  permute_109 = _shape_param_260 = None
        clone_40: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_216: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_261);  clone_40 = _shape_param_261 = None
        view_217: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_42, _shape_param_262);  _shape_param_262 = None
        permute_110: "bf16[512, 512]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        mm_57: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_217, permute_110);  view_217 = permute_110 = None
        view_218: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_57, _shape_param_263);  mm_57 = _shape_param_263 = None
        view_219: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_218, _shape_param_264);  view_218 = _shape_param_264 = None
        permute_111: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_219, [0, 2, 1, 3]);  view_219 = None
        permute_112: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_111, [0, 1, 3, 2]);  permute_111 = None
        expand_42: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_112, _shape_param_265);  permute_112 = _shape_param_265 = None
        clone_41: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_220: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_41, _shape_param_266);  clone_41 = _shape_param_266 = None
        bmm_20: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_216, view_220);  view_216 = view_220 = None
        view_221: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_20, _shape_param_267);  bmm_20 = _shape_param_267 = None
        add_57: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_221, add_6);  view_221 = None
        convert_element_type_65: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_57, torch.float32);  add_57 = None
        amax_10: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_65, [-1], True)
        sub_12: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_65, amax_10);  convert_element_type_65 = amax_10 = None
        exp_10: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_11: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_14: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_66: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None
        expand_43: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_66, _shape_param_268);  convert_element_type_66 = _shape_param_268 = None
        view_222: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_43, _shape_param_269);  expand_43 = _shape_param_269 = None
        view_223: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_42, _shape_param_270);  mul_42 = _shape_param_270 = None
        permute_113: "bf16[512, 512]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_58: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_223, permute_113);  view_223 = permute_113 = None
        view_224: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_58, _shape_param_271);  mm_58 = _shape_param_271 = None
        view_225: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_224, _shape_param_272);  view_224 = _shape_param_272 = None
        permute_114: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
        expand_44: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_114, _shape_param_273);  permute_114 = _shape_param_273 = None
        clone_42: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_226: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_42, _shape_param_274);  clone_42 = _shape_param_274 = None
        bmm_21: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_222, view_226);  view_222 = view_226 = None
        view_227: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_21, _shape_param_275);  bmm_21 = _shape_param_275 = None
        permute_115: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
        clone_43: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_228: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_43, _shape_param_276);  clone_43 = _shape_param_276 = None
        view_229: "bf16[8192, 512]" = torch.ops.aten.view.default(view_228, _shape_param_277);  view_228 = _shape_param_277 = None
        permute_116: "bf16[512, 512]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        mm_59: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_229, permute_116);  view_229 = permute_116 = None
        view_230: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_59, _shape_param_278);  mm_59 = _shape_param_278 = None
        add_58: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_55, view_230);  add_55 = view_230 = None
        convert_element_type_67: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_58, torch.float32)
        pow_21: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_67, 2);  convert_element_type_67 = None
        mean_20: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None
        add_59: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        mul_43: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_58, rsqrt_20);  rsqrt_20 = None
        convert_element_type_68: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_43, torch.bfloat16);  mul_43 = None
        mul_44: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg85_1, convert_element_type_68);  arg85_1 = convert_element_type_68 = None
        view_231: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_44, _shape_param_279);  mul_44 = _shape_param_279 = None
        permute_117: "bf16[512, 512]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        mm_60: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_231, permute_117);  view_231 = permute_117 = None
        view_232: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_60, _shape_param_280);  mm_60 = _shape_param_280 = None
        view_233: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_232, _shape_param_281);  view_232 = _shape_param_281 = None
        permute_118: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None
        expand_45: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_118, _shape_param_282);  permute_118 = _shape_param_282 = None
        clone_44: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_234: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_44, _shape_param_283);  clone_44 = _shape_param_283 = None
        view_235: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_32, _shape_param_284);  _shape_param_284 = None
        permute_119: "bf16[512, 512]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        mm_61: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_235, permute_119);  view_235 = permute_119 = None
        view_236: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_61, _shape_param_285);  mm_61 = _shape_param_285 = None
        view_237: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_236, _shape_param_286);  view_236 = _shape_param_286 = None
        permute_120: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None
        permute_121: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_120, [0, 1, 3, 2]);  permute_120 = None
        expand_46: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_121, _shape_param_287);  permute_121 = _shape_param_287 = None
        clone_45: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_238: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_45, _shape_param_288);  clone_45 = _shape_param_288 = None
        bmm_22: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_234, view_238);  view_234 = view_238 = None
        view_239: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_22, _shape_param_289);  bmm_22 = _shape_param_289 = None
        convert_element_type_69: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        amax_11: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_69, [-1], True)
        sub_13: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_69, amax_11);  convert_element_type_69 = amax_11 = None
        exp_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_12: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_15: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_70: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None
        expand_47: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_70, _shape_param_290);  convert_element_type_70 = _shape_param_290 = None
        view_240: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_47, _shape_param_291);  expand_47 = _shape_param_291 = None
        view_241: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_32, _shape_param_292);  _shape_param_292 = None
        permute_122: "bf16[512, 512]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        mm_62: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_241, permute_122);  view_241 = permute_122 = None
        view_242: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_62, _shape_param_293);  mm_62 = _shape_param_293 = None
        view_243: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_242, _shape_param_294);  view_242 = _shape_param_294 = None
        permute_123: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None
        expand_48: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_123, _shape_param_295);  permute_123 = _shape_param_295 = None
        clone_46: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_244: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_46, _shape_param_296);  clone_46 = _shape_param_296 = None
        bmm_23: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_240, view_244);  view_240 = view_244 = None
        view_245: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_297);  bmm_23 = _shape_param_297 = None
        permute_124: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None
        clone_47: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
        view_246: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_47, _shape_param_298);  clone_47 = _shape_param_298 = None
        view_247: "bf16[8192, 512]" = torch.ops.aten.view.default(view_246, _shape_param_299);  view_246 = _shape_param_299 = None
        permute_125: "bf16[512, 512]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        mm_63: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_247, permute_125);  view_247 = permute_125 = None
        view_248: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_63, _shape_param_300);  mm_63 = _shape_param_300 = None
        add_60: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_58, view_248);  add_58 = view_248 = None
        convert_element_type_71: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_60, torch.float32)
        pow_22: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_71, 2);  convert_element_type_71 = None
        mean_21: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
        add_61: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_45: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_60, rsqrt_21);  rsqrt_21 = None
        convert_element_type_72: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        mul_46: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg90_1, convert_element_type_72);  arg90_1 = convert_element_type_72 = None
        view_249: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_46, _shape_param_301);  mul_46 = _shape_param_301 = None
        permute_126: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        mm_64: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_249, permute_126);  view_249 = permute_126 = None
        view_250: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_64, _shape_param_302);  mm_64 = _shape_param_302 = None
        relu_8: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_250);  view_250 = None
        view_251: "bf16[8192, 2048]" = torch.ops.aten.view.default(relu_8, _shape_param_303);  relu_8 = _shape_param_303 = None
        permute_127: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        mm_65: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_251, permute_127);  view_251 = permute_127 = None
        view_252: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_65, _shape_param_304);  mm_65 = _shape_param_304 = None
        add_62: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_60, view_252);  add_60 = view_252 = None
        convert_element_type_73: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_62, torch.float32)
        pow_23: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_73, 2);  convert_element_type_73 = None
        mean_22: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
        add_63: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_47: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_62, rsqrt_22);  rsqrt_22 = None
        convert_element_type_74: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_47, torch.bfloat16);  mul_47 = None
        mul_48: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg93_1, convert_element_type_74);  arg93_1 = convert_element_type_74 = None
        view_253: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_48, _shape_param_305);  _shape_param_305 = None
        permute_128: "bf16[512, 512]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        mm_66: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_253, permute_128);  view_253 = permute_128 = None
        view_254: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_66, _shape_param_306);  mm_66 = _shape_param_306 = None
        view_255: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_254, _shape_param_307);  view_254 = _shape_param_307 = None
        permute_129: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
        expand_49: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_129, _shape_param_308);  permute_129 = _shape_param_308 = None
        clone_48: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_256: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_48, _shape_param_309);  clone_48 = _shape_param_309 = None
        view_257: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_48, _shape_param_310);  _shape_param_310 = None
        permute_130: "bf16[512, 512]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        mm_67: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_257, permute_130);  view_257 = permute_130 = None
        view_258: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_67, _shape_param_311);  mm_67 = _shape_param_311 = None
        view_259: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_258, _shape_param_312);  view_258 = _shape_param_312 = None
        permute_131: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
        permute_132: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2]);  permute_131 = None
        expand_50: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_132, _shape_param_313);  permute_132 = _shape_param_313 = None
        clone_49: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_260: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_49, _shape_param_314);  clone_49 = _shape_param_314 = None
        bmm_24: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_256, view_260);  view_256 = view_260 = None
        view_261: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_24, _shape_param_315);  bmm_24 = _shape_param_315 = None
        add_64: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_261, add_6);  view_261 = None
        convert_element_type_75: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_64, torch.float32);  add_64 = None
        amax_12: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_75, [-1], True)
        sub_14: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_75, amax_12);  convert_element_type_75 = amax_12 = None
        exp_12: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_13: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_16: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_76: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None
        expand_51: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_76, _shape_param_316);  convert_element_type_76 = _shape_param_316 = None
        view_262: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_51, _shape_param_317);  expand_51 = _shape_param_317 = None
        view_263: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_48, _shape_param_318);  mul_48 = _shape_param_318 = None
        permute_133: "bf16[512, 512]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        mm_68: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_263, permute_133);  view_263 = permute_133 = None
        view_264: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_68, _shape_param_319);  mm_68 = _shape_param_319 = None
        view_265: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_264, _shape_param_320);  view_264 = _shape_param_320 = None
        permute_134: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
        expand_52: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_134, _shape_param_321);  permute_134 = _shape_param_321 = None
        clone_50: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_266: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_50, _shape_param_322);  clone_50 = _shape_param_322 = None
        bmm_25: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_262, view_266);  view_262 = view_266 = None
        view_267: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_25, _shape_param_323);  bmm_25 = _shape_param_323 = None
        permute_135: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
        clone_51: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_268: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_51, _shape_param_324);  clone_51 = _shape_param_324 = None
        view_269: "bf16[8192, 512]" = torch.ops.aten.view.default(view_268, _shape_param_325);  view_268 = _shape_param_325 = None
        permute_136: "bf16[512, 512]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        mm_69: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_269, permute_136);  view_269 = permute_136 = None
        view_270: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_69, _shape_param_326);  mm_69 = _shape_param_326 = None
        add_65: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_62, view_270);  add_62 = view_270 = None
        convert_element_type_77: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_65, torch.float32)
        pow_24: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_77, 2);  convert_element_type_77 = None
        mean_23: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None
        add_66: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_49: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_65, rsqrt_23);  rsqrt_23 = None
        convert_element_type_78: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        mul_50: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg98_1, convert_element_type_78);  arg98_1 = convert_element_type_78 = None
        view_271: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_50, _shape_param_327);  mul_50 = _shape_param_327 = None
        permute_137: "bf16[512, 512]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_70: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_271, permute_137);  view_271 = permute_137 = None
        view_272: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_70, _shape_param_328);  mm_70 = _shape_param_328 = None
        view_273: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_272, _shape_param_329);  view_272 = _shape_param_329 = None
        permute_138: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None
        expand_53: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_138, _shape_param_330);  permute_138 = _shape_param_330 = None
        clone_52: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_274: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_52, _shape_param_331);  clone_52 = _shape_param_331 = None
        view_275: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_32, _shape_param_332);  _shape_param_332 = None
        permute_139: "bf16[512, 512]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        mm_71: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_275, permute_139);  view_275 = permute_139 = None
        view_276: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_71, _shape_param_333);  mm_71 = _shape_param_333 = None
        view_277: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_276, _shape_param_334);  view_276 = _shape_param_334 = None
        permute_140: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_277, [0, 2, 1, 3]);  view_277 = None
        permute_141: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_140, [0, 1, 3, 2]);  permute_140 = None
        expand_54: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_141, _shape_param_335);  permute_141 = _shape_param_335 = None
        clone_53: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_278: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_53, _shape_param_336);  clone_53 = _shape_param_336 = None
        bmm_26: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_274, view_278);  view_274 = view_278 = None
        view_279: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_26, _shape_param_337);  bmm_26 = _shape_param_337 = None
        convert_element_type_79: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_279, torch.float32);  view_279 = None
        amax_13: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_79, [-1], True)
        sub_15: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_79, amax_13);  convert_element_type_79 = amax_13 = None
        exp_13: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_14: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_17: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_80: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None
        expand_55: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_80, _shape_param_338);  convert_element_type_80 = _shape_param_338 = None
        view_280: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_55, _shape_param_339);  expand_55 = _shape_param_339 = None
        view_281: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_32, _shape_param_340);  _shape_param_340 = None
        permute_142: "bf16[512, 512]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_72: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_281, permute_142);  view_281 = permute_142 = None
        view_282: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_72, _shape_param_341);  mm_72 = _shape_param_341 = None
        view_283: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_282, _shape_param_342);  view_282 = _shape_param_342 = None
        permute_143: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
        expand_56: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_143, _shape_param_343);  permute_143 = _shape_param_343 = None
        clone_54: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_284: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_54, _shape_param_344);  clone_54 = _shape_param_344 = None
        bmm_27: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_280, view_284);  view_280 = view_284 = None
        view_285: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_27, _shape_param_345);  bmm_27 = _shape_param_345 = None
        permute_144: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
        clone_55: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        view_286: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_55, _shape_param_346);  clone_55 = _shape_param_346 = None
        view_287: "bf16[8192, 512]" = torch.ops.aten.view.default(view_286, _shape_param_347);  view_286 = _shape_param_347 = None
        permute_145: "bf16[512, 512]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        mm_73: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_287, permute_145);  view_287 = permute_145 = None
        view_288: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_73, _shape_param_348);  mm_73 = _shape_param_348 = None
        add_67: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_65, view_288);  add_65 = view_288 = None
        convert_element_type_81: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_67, torch.float32)
        pow_25: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_81, 2);  convert_element_type_81 = None
        mean_24: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
        add_68: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_51: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_67, rsqrt_24);  rsqrt_24 = None
        convert_element_type_82: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_51, torch.bfloat16);  mul_51 = None
        mul_52: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg103_1, convert_element_type_82);  arg103_1 = convert_element_type_82 = None
        view_289: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_52, _shape_param_349);  mul_52 = _shape_param_349 = None
        permute_146: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        mm_74: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_289, permute_146);  view_289 = permute_146 = None
        view_290: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_74, _shape_param_350);  mm_74 = _shape_param_350 = None
        relu_9: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_290);  view_290 = None
        view_291: "bf16[8192, 2048]" = torch.ops.aten.view.default(relu_9, _shape_param_351);  relu_9 = _shape_param_351 = None
        permute_147: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        mm_75: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_291, permute_147);  view_291 = permute_147 = None
        view_292: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_75, _shape_param_352);  mm_75 = _shape_param_352 = None
        add_69: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_67, view_292);  add_67 = view_292 = None
        convert_element_type_83: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_69, torch.float32)
        pow_26: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_83, 2);  convert_element_type_83 = None
        mean_25: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None
        add_70: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_53: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_69, rsqrt_25);  rsqrt_25 = None
        convert_element_type_84: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16);  mul_53 = None
        mul_54: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg106_1, convert_element_type_84);  arg106_1 = convert_element_type_84 = None
        view_293: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_54, _shape_param_353);  _shape_param_353 = None
        permute_148: "bf16[512, 512]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        mm_76: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_293, permute_148);  view_293 = permute_148 = None
        view_294: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_76, _shape_param_354);  mm_76 = _shape_param_354 = None
        view_295: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_294, _shape_param_355);  view_294 = _shape_param_355 = None
        permute_149: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None
        expand_57: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_149, _shape_param_356);  permute_149 = _shape_param_356 = None
        clone_56: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_296: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_56, _shape_param_357);  clone_56 = _shape_param_357 = None
        view_297: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_54, _shape_param_358);  _shape_param_358 = None
        permute_150: "bf16[512, 512]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        mm_77: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_297, permute_150);  view_297 = permute_150 = None
        view_298: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_77, _shape_param_359);  mm_77 = _shape_param_359 = None
        view_299: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_298, _shape_param_360);  view_298 = _shape_param_360 = None
        permute_151: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_299, [0, 2, 1, 3]);  view_299 = None
        permute_152: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_151, [0, 1, 3, 2]);  permute_151 = None
        expand_58: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_152, _shape_param_361);  permute_152 = _shape_param_361 = None
        clone_57: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_300: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_57, _shape_param_362);  clone_57 = _shape_param_362 = None
        bmm_28: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_296, view_300);  view_296 = view_300 = None
        view_301: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_28, _shape_param_363);  bmm_28 = _shape_param_363 = None
        add_71: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_301, add_6);  view_301 = None
        convert_element_type_85: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_71, torch.float32);  add_71 = None
        amax_14: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_85, [-1], True)
        sub_16: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_85, amax_14);  convert_element_type_85 = amax_14 = None
        exp_14: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_15: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_18: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_86: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None
        expand_59: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_86, _shape_param_364);  convert_element_type_86 = _shape_param_364 = None
        view_302: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_59, _shape_param_365);  expand_59 = _shape_param_365 = None
        view_303: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_54, _shape_param_366);  mul_54 = _shape_param_366 = None
        permute_153: "bf16[512, 512]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        mm_78: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_303, permute_153);  view_303 = permute_153 = None
        view_304: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_78, _shape_param_367);  mm_78 = _shape_param_367 = None
        view_305: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_304, _shape_param_368);  view_304 = _shape_param_368 = None
        permute_154: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_305, [0, 2, 1, 3]);  view_305 = None
        expand_60: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_154, _shape_param_369);  permute_154 = _shape_param_369 = None
        clone_58: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_306: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_58, _shape_param_370);  clone_58 = _shape_param_370 = None
        bmm_29: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_302, view_306);  view_302 = view_306 = None
        view_307: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_29, _shape_param_371);  bmm_29 = _shape_param_371 = None
        permute_155: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
        clone_59: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_308: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_59, _shape_param_372);  clone_59 = _shape_param_372 = None
        view_309: "bf16[8192, 512]" = torch.ops.aten.view.default(view_308, _shape_param_373);  view_308 = _shape_param_373 = None
        permute_156: "bf16[512, 512]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        mm_79: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_309, permute_156);  view_309 = permute_156 = None
        view_310: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_79, _shape_param_374);  mm_79 = _shape_param_374 = None
        add_72: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_69, view_310);  add_69 = view_310 = None
        convert_element_type_87: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_72, torch.float32)
        pow_27: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_87, 2);  convert_element_type_87 = None
        mean_26: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
        add_73: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_55: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_72, rsqrt_26);  rsqrt_26 = None
        convert_element_type_88: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_55, torch.bfloat16);  mul_55 = None
        mul_56: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg111_1, convert_element_type_88);  arg111_1 = convert_element_type_88 = None
        view_311: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_56, _shape_param_375);  mul_56 = _shape_param_375 = None
        permute_157: "bf16[512, 512]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        mm_80: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_311, permute_157);  view_311 = permute_157 = None
        view_312: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_80, _shape_param_376);  mm_80 = _shape_param_376 = None
        view_313: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_312, _shape_param_377);  view_312 = _shape_param_377 = None
        permute_158: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None
        expand_61: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_158, _shape_param_378);  permute_158 = _shape_param_378 = None
        clone_60: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_314: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_60, _shape_param_379);  clone_60 = _shape_param_379 = None
        view_315: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_32, _shape_param_380);  _shape_param_380 = None
        permute_159: "bf16[512, 512]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        mm_81: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_315, permute_159);  view_315 = permute_159 = None
        view_316: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_81, _shape_param_381);  mm_81 = _shape_param_381 = None
        view_317: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_316, _shape_param_382);  view_316 = _shape_param_382 = None
        permute_160: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None
        permute_161: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_160, [0, 1, 3, 2]);  permute_160 = None
        expand_62: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_161, _shape_param_383);  permute_161 = _shape_param_383 = None
        clone_61: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_318: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_61, _shape_param_384);  clone_61 = _shape_param_384 = None
        bmm_30: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_314, view_318);  view_314 = view_318 = None
        view_319: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_30, _shape_param_385);  bmm_30 = _shape_param_385 = None
        convert_element_type_89: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_319, torch.float32);  view_319 = None
        amax_15: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_89, [-1], True)
        sub_17: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_89, amax_15);  convert_element_type_89 = amax_15 = None
        exp_15: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_16: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_19: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_90: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None
        expand_63: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_90, _shape_param_386);  convert_element_type_90 = _shape_param_386 = None
        view_320: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_63, _shape_param_387);  expand_63 = _shape_param_387 = None
        view_321: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_32, _shape_param_388);  _shape_param_388 = None
        permute_162: "bf16[512, 512]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        mm_82: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_321, permute_162);  view_321 = permute_162 = None
        view_322: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_82, _shape_param_389);  mm_82 = _shape_param_389 = None
        view_323: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_322, _shape_param_390);  view_322 = _shape_param_390 = None
        permute_163: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None
        expand_64: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_163, _shape_param_391);  permute_163 = _shape_param_391 = None
        clone_62: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_324: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_62, _shape_param_392);  clone_62 = _shape_param_392 = None
        bmm_31: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_320, view_324);  view_320 = view_324 = None
        view_325: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_31, _shape_param_393);  bmm_31 = _shape_param_393 = None
        permute_164: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_325, [0, 2, 1, 3]);  view_325 = None
        clone_63: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
        view_326: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_63, _shape_param_394);  clone_63 = _shape_param_394 = None
        view_327: "bf16[8192, 512]" = torch.ops.aten.view.default(view_326, _shape_param_395);  view_326 = _shape_param_395 = None
        permute_165: "bf16[512, 512]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        mm_83: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_327, permute_165);  view_327 = permute_165 = None
        view_328: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_83, _shape_param_396);  mm_83 = _shape_param_396 = None
        add_74: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_72, view_328);  add_72 = view_328 = None
        convert_element_type_91: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_74, torch.float32)
        pow_28: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_91, 2);  convert_element_type_91 = None
        mean_27: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None
        add_75: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_57: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_74, rsqrt_27);  rsqrt_27 = None
        convert_element_type_92: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_57, torch.bfloat16);  mul_57 = None
        mul_58: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg116_1, convert_element_type_92);  arg116_1 = convert_element_type_92 = None
        view_329: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_58, _shape_param_397);  mul_58 = _shape_param_397 = None
        permute_166: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_84: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_329, permute_166);  view_329 = permute_166 = None
        view_330: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_84, _shape_param_398);  mm_84 = _shape_param_398 = None
        relu_10: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_330);  view_330 = None
        view_331: "bf16[8192, 2048]" = torch.ops.aten.view.default(relu_10, _shape_param_399);  relu_10 = _shape_param_399 = None
        permute_167: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        mm_85: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_331, permute_167);  view_331 = permute_167 = None
        view_332: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_85, _shape_param_400);  mm_85 = _shape_param_400 = None
        add_76: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_74, view_332);  add_74 = view_332 = None
        convert_element_type_93: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_76, torch.float32)
        pow_29: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_93, 2);  convert_element_type_93 = None
        mean_28: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None
        add_77: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_59: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_76, rsqrt_28);  rsqrt_28 = None
        convert_element_type_94: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None
        mul_60: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg119_1, convert_element_type_94);  arg119_1 = convert_element_type_94 = None
        view_333: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_60, _shape_param_401);  _shape_param_401 = None
        permute_168: "bf16[512, 512]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        mm_86: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_333, permute_168);  view_333 = permute_168 = None
        view_334: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_86, _shape_param_402);  mm_86 = _shape_param_402 = None
        view_335: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_334, _shape_param_403);  view_334 = _shape_param_403 = None
        permute_169: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_335, [0, 2, 1, 3]);  view_335 = None
        expand_65: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_169, _shape_param_404);  permute_169 = _shape_param_404 = None
        clone_64: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_336: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_64, _shape_param_405);  clone_64 = _shape_param_405 = None
        view_337: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_60, _shape_param_406);  _shape_param_406 = None
        permute_170: "bf16[512, 512]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        mm_87: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_337, permute_170);  view_337 = permute_170 = None
        view_338: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_87, _shape_param_407);  mm_87 = _shape_param_407 = None
        view_339: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_338, _shape_param_408);  view_338 = _shape_param_408 = None
        permute_171: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
        permute_172: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_171, [0, 1, 3, 2]);  permute_171 = None
        expand_66: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_172, _shape_param_409);  permute_172 = _shape_param_409 = None
        clone_65: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_340: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_65, _shape_param_410);  clone_65 = _shape_param_410 = None
        bmm_32: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_336, view_340);  view_336 = view_340 = None
        view_341: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_32, _shape_param_411);  bmm_32 = _shape_param_411 = None
        add_78: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_341, add_6);  view_341 = add_6 = None
        convert_element_type_95: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_78, torch.float32);  add_78 = None
        amax_16: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_95, [-1], True)
        sub_18: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_95, amax_16);  convert_element_type_95 = amax_16 = None
        exp_16: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_17: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_20: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_96: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None
        expand_67: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_96, _shape_param_412);  convert_element_type_96 = _shape_param_412 = None
        view_342: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_67, _shape_param_413);  expand_67 = _shape_param_413 = None
        view_343: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_60, _shape_param_414);  mul_60 = _shape_param_414 = None
        permute_173: "bf16[512, 512]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        mm_88: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_343, permute_173);  view_343 = permute_173 = None
        view_344: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_88, _shape_param_415);  mm_88 = _shape_param_415 = None
        view_345: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_344, _shape_param_416);  view_344 = _shape_param_416 = None
        permute_174: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
        expand_68: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_174, _shape_param_417);  permute_174 = _shape_param_417 = None
        clone_66: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_346: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_66, _shape_param_418);  clone_66 = _shape_param_418 = None
        bmm_33: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_342, view_346);  view_342 = view_346 = None
        view_347: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_33, _shape_param_419);  bmm_33 = _shape_param_419 = None
        permute_175: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None
        clone_67: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
        view_348: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_67, _shape_param_420);  clone_67 = _shape_param_420 = None
        view_349: "bf16[8192, 512]" = torch.ops.aten.view.default(view_348, _shape_param_421);  view_348 = _shape_param_421 = None
        permute_176: "bf16[512, 512]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        mm_89: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_349, permute_176);  view_349 = permute_176 = None
        view_350: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_89, _shape_param_422);  mm_89 = _shape_param_422 = None
        add_79: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_76, view_350);  add_76 = view_350 = None
        convert_element_type_97: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_79, torch.float32)
        pow_30: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_97, 2);  convert_element_type_97 = None
        mean_29: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None
        add_80: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_61: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_79, rsqrt_29);  rsqrt_29 = None
        convert_element_type_98: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None
        mul_62: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg124_1, convert_element_type_98);  arg124_1 = convert_element_type_98 = None
        view_351: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_62, _shape_param_423);  mul_62 = _shape_param_423 = None
        permute_177: "bf16[512, 512]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        mm_90: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_351, permute_177);  view_351 = permute_177 = None
        view_352: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_90, _shape_param_424);  mm_90 = _shape_param_424 = None
        view_353: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_352, _shape_param_425);  view_352 = _shape_param_425 = None
        permute_178: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None
        expand_69: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_178, _shape_param_426);  permute_178 = _shape_param_426 = None
        clone_68: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_354: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_68, _shape_param_427);  clone_68 = _shape_param_427 = None
        view_355: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_32, _shape_param_428);  _shape_param_428 = None
        permute_179: "bf16[512, 512]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        mm_91: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_355, permute_179);  view_355 = permute_179 = None
        view_356: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_91, _shape_param_429);  mm_91 = _shape_param_429 = None
        view_357: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_356, _shape_param_430);  view_356 = _shape_param_430 = None
        permute_180: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None
        permute_181: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_180, [0, 1, 3, 2]);  permute_180 = None
        expand_70: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_181, _shape_param_431);  permute_181 = _shape_param_431 = None
        clone_69: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_358: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_69, _shape_param_432);  clone_69 = _shape_param_432 = None
        bmm_34: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_354, view_358);  view_354 = view_358 = None
        view_359: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_34, _shape_param_433);  bmm_34 = _shape_param_433 = None
        convert_element_type_99: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_359, torch.float32);  view_359 = None
        amax_17: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_99, [-1], True)
        sub_19: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_99, amax_17);  convert_element_type_99 = amax_17 = None
        exp_17: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_18: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_21: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_100: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None
        expand_71: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(convert_element_type_100, _shape_param_434);  convert_element_type_100 = _shape_param_434 = None
        view_360: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_71, _shape_param_435);  expand_71 = _shape_param_435 = None
        view_361: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_32, _shape_param_436);  _shape_param_436 = None
        permute_182: "bf16[512, 512]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        mm_92: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_361, permute_182);  view_361 = permute_182 = None
        view_362: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_92, _shape_param_437);  mm_92 = _shape_param_437 = None
        view_363: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_362, _shape_param_438);  view_362 = _shape_param_438 = None
        permute_183: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_363, [0, 2, 1, 3]);  view_363 = None
        expand_72: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_183, _shape_param_439);  permute_183 = _shape_param_439 = None
        clone_70: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_364: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_70, _shape_param_440);  clone_70 = _shape_param_440 = None
        bmm_35: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_360, view_364);  view_360 = view_364 = None
        view_365: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_35, _shape_param_441);  bmm_35 = _shape_param_441 = None
        permute_184: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_365, [0, 2, 1, 3]);  view_365 = None
        clone_71: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
        view_366: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_71, _shape_param_442);  clone_71 = _shape_param_442 = None
        view_367: "bf16[8192, 512]" = torch.ops.aten.view.default(view_366, _shape_param_443);  view_366 = _shape_param_443 = None
        permute_185: "bf16[512, 512]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        mm_93: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_367, permute_185);  view_367 = permute_185 = None
        view_368: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_93, _shape_param_444);  mm_93 = _shape_param_444 = None
        add_81: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_79, view_368);  add_79 = view_368 = None
        convert_element_type_101: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_81, torch.float32)
        pow_31: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_101, 2);  convert_element_type_101 = None
        mean_30: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
        add_82: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_63: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_81, rsqrt_30);  rsqrt_30 = None
        convert_element_type_102: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16);  mul_63 = None
        mul_64: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg129_1, convert_element_type_102);  arg129_1 = convert_element_type_102 = None
        view_369: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_64, _shape_param_445);  mul_64 = _shape_param_445 = None
        permute_186: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        mm_94: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_369, permute_186);  view_369 = permute_186 = None
        view_370: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_94, _shape_param_446);  mm_94 = _shape_param_446 = None
        relu_11: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_370);  view_370 = None
        view_371: "bf16[8192, 2048]" = torch.ops.aten.view.default(relu_11, _shape_param_447);  relu_11 = _shape_param_447 = None
        permute_187: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        mm_95: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_371, permute_187);  view_371 = permute_187 = None
        view_372: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_95, _shape_param_448);  mm_95 = _shape_param_448 = None
        add_83: "bf16[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_81, view_372);  add_81 = view_372 = None
        convert_element_type_103: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_83, torch.float32)
        pow_32: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_103, 2);  convert_element_type_103 = None
        mean_31: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None
        add_84: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_65: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_83, rsqrt_31);  add_83 = rsqrt_31 = None
        convert_element_type_104: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None
        mul_66: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg132_1, convert_element_type_104);  arg132_1 = convert_element_type_104 = None
        mul_67: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_66, 0.04419417382415922);  mul_66 = None
        view_373: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_67, _shape_param_449);  mul_67 = _shape_param_449 = None
        permute_188: "bf16[512, 32128]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm_96: "bf16[8192, 32128]" = torch.ops.aten.mm.default(view_373, permute_188);  view_373 = permute_188 = None
        view_374: "bf16[8, 1024, 32128]" = torch.ops.aten.view.default(mm_96, _shape_param_450);  mm_96 = _shape_param_450 = None
        view_375: "bf16[8192, 32128]" = torch.ops.aten.view.default(view_374, _shape_param_451);  _shape_param_451 = None
        convert_element_type_105: "f32[8192, 32128]" = torch.ops.prims.convert_element_type.default(view_375, torch.float32);  view_375 = None
        amax_18: "f32[8192, 1]" = torch.ops.aten.amax.default(convert_element_type_105, [1], True)
        sub_20: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(convert_element_type_105, amax_18);  convert_element_type_105 = amax_18 = None
        exp_18: "f32[8192, 32128]" = torch.ops.aten.exp.default(sub_20)
        sum_19: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [1], True);  exp_18 = None
        log_2: "f32[8192, 1]" = torch.ops.aten.log.default(sum_19);  sum_19 = None
        sub_21: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(sub_20, log_2);  sub_20 = log_2 = None
        convert_element_type_106: "bf16[8192, 32128]" = torch.ops.prims.convert_element_type.default(sub_21, torch.bfloat16);  sub_21 = None
        ne_1: "b8[8192]" = torch.ops.aten.ne.Scalar(view, -100)
        full_6: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "i64[8192]" = torch.ops.aten.where.self(ne_1, view, full_6);  ne_1 = full_6 = None
        unsqueeze_12: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where_3, 1);  where_3 = None
        gather: "bf16[8192, 1]" = torch.ops.aten.gather.default(convert_element_type_106, 1, unsqueeze_12);  convert_element_type_106 = unsqueeze_12 = None
        squeeze: "bf16[8192]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_1: "bf16[8192]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_7: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[8192]" = torch.ops.aten.where.self(ne, neg_1, full_7);  ne = neg_1 = full_7 = None
        sum_20: "bf16[]" = torch.ops.aten.sum.default(where_4);  where_4 = None
        ne_2: "b8[8192]" = torch.ops.aten.ne.Scalar(view, -100);  view = None
        sum_21: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_107: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_21, torch.bfloat16);  sum_21 = None
        div_22: "bf16[]" = torch.ops.aten.div.Tensor(sum_20, convert_element_type_107);  sum_20 = convert_element_type_107 = None
        return (mul_32, view_374, div_22)



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
