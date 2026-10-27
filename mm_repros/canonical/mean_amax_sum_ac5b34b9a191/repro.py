"""
Standalone repro captured via capture_hook.
Label: T5ForConditionalGeneration
Pattern hash: ac5b34b9a191
Shape hash: aeaa0947
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
    def forward(self, arg0_1: "f32[32128, 512]", arg1_1: "i64[8, 1024]", arg2_1: "f32[512]", arg3_1: "f32[512, 512]", arg4_1: "f32[512, 512]", arg5_1: "f32[512, 512]", arg6_1: "f32[32, 8]", arg7_1: "f32[512, 512]", arg8_1: "f32[512]", arg9_1: "f32[2048, 512]", arg10_1: "f32[512, 2048]", arg11_1: "f32[512]", arg12_1: "f32[512, 512]", arg13_1: "f32[512, 512]", arg14_1: "f32[512, 512]", arg15_1: "f32[512, 512]", arg16_1: "f32[512]", arg17_1: "f32[2048, 512]", arg18_1: "f32[512, 2048]", arg19_1: "f32[512]", arg20_1: "f32[512, 512]", arg21_1: "f32[512, 512]", arg22_1: "f32[512, 512]", arg23_1: "f32[512, 512]", arg24_1: "f32[512]", arg25_1: "f32[2048, 512]", arg26_1: "f32[512, 2048]", arg27_1: "f32[512]", arg28_1: "f32[512, 512]", arg29_1: "f32[512, 512]", arg30_1: "f32[512, 512]", arg31_1: "f32[512, 512]", arg32_1: "f32[512]", arg33_1: "f32[2048, 512]", arg34_1: "f32[512, 2048]", arg35_1: "f32[512]", arg36_1: "f32[512, 512]", arg37_1: "f32[512, 512]", arg38_1: "f32[512, 512]", arg39_1: "f32[512, 512]", arg40_1: "f32[512]", arg41_1: "f32[2048, 512]", arg42_1: "f32[512, 2048]", arg43_1: "f32[512]", arg44_1: "f32[512, 512]", arg45_1: "f32[512, 512]", arg46_1: "f32[512, 512]", arg47_1: "f32[512, 512]", arg48_1: "f32[512]", arg49_1: "f32[2048, 512]", arg50_1: "f32[512, 2048]", arg51_1: "f32[512]", arg52_1: "f32[512]", arg53_1: "f32[512, 512]", arg54_1: "f32[512, 512]", arg55_1: "f32[512, 512]", arg56_1: "f32[32, 8]", arg57_1: "f32[512, 512]", arg58_1: "f32[512]", arg59_1: "f32[512, 512]", arg60_1: "f32[512, 512]", arg61_1: "f32[512, 512]", arg62_1: "f32[512, 512]", arg63_1: "f32[512]", arg64_1: "f32[2048, 512]", arg65_1: "f32[512, 2048]", arg66_1: "f32[512]", arg67_1: "f32[512, 512]", arg68_1: "f32[512, 512]", arg69_1: "f32[512, 512]", arg70_1: "f32[512, 512]", arg71_1: "f32[512]", arg72_1: "f32[512, 512]", arg73_1: "f32[512, 512]", arg74_1: "f32[512, 512]", arg75_1: "f32[512, 512]", arg76_1: "f32[512]", arg77_1: "f32[2048, 512]", arg78_1: "f32[512, 2048]", arg79_1: "f32[512]", arg80_1: "f32[512, 512]", arg81_1: "f32[512, 512]", arg82_1: "f32[512, 512]", arg83_1: "f32[512, 512]", arg84_1: "f32[512]", arg85_1: "f32[512, 512]", arg86_1: "f32[512, 512]", arg87_1: "f32[512, 512]", arg88_1: "f32[512, 512]", arg89_1: "f32[512]", arg90_1: "f32[2048, 512]", arg91_1: "f32[512, 2048]", arg92_1: "f32[512]", arg93_1: "f32[512, 512]", arg94_1: "f32[512, 512]", arg95_1: "f32[512, 512]", arg96_1: "f32[512, 512]", arg97_1: "f32[512]", arg98_1: "f32[512, 512]", arg99_1: "f32[512, 512]", arg100_1: "f32[512, 512]", arg101_1: "f32[512, 512]", arg102_1: "f32[512]", arg103_1: "f32[2048, 512]", arg104_1: "f32[512, 2048]", arg105_1: "f32[512]", arg106_1: "f32[512, 512]", arg107_1: "f32[512, 512]", arg108_1: "f32[512, 512]", arg109_1: "f32[512, 512]", arg110_1: "f32[512]", arg111_1: "f32[512, 512]", arg112_1: "f32[512, 512]", arg113_1: "f32[512, 512]", arg114_1: "f32[512, 512]", arg115_1: "f32[512]", arg116_1: "f32[2048, 512]", arg117_1: "f32[512, 2048]", arg118_1: "f32[512]", arg119_1: "f32[512, 512]", arg120_1: "f32[512, 512]", arg121_1: "f32[512, 512]", arg122_1: "f32[512, 512]", arg123_1: "f32[512]", arg124_1: "f32[512, 512]", arg125_1: "f32[512, 512]", arg126_1: "f32[512, 512]", arg127_1: "f32[512, 512]", arg128_1: "f32[512]", arg129_1: "f32[2048, 512]", arg130_1: "f32[512, 2048]", arg131_1: "f32[512]", arg132_1: "i64[8, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480):
        # No stacktrace found for following nodes
        embedding: "f32[8, 1024, 512]" = torch.ops.aten.embedding.default(arg0_1, arg1_1);  arg1_1 = None
        full: "f32[8, 1, 1024, 1024]" = torch.ops.aten.full.default(_shape_param_0, 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        inductor_seeds: "i64[64]" = torch.ops.prims.inductor_seeds.default(64, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        gt: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt, embedding)
        mul_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        pow_1: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_1, 2)
        mean: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_1, rsqrt)
        mul_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg2_1, mul_2);  arg2_1 = mul_2 = None
        convert_element_type: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_1: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        permute: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        view: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_2);  convert_element_type_1 = _shape_param_2 = None
        mm: "bf16[8192, 512]" = torch.ops.aten.mm.default(view, permute)
        view_1: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm, _shape_param_3);  mm = _shape_param_3 = None
        view_2: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_1, _shape_param_4);  view_1 = _shape_param_4 = None
        permute_1: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        convert_element_type_2: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        permute_2: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_2, [1, 0]);  convert_element_type_2 = None
        mm_1: "bf16[8192, 512]" = torch.ops.aten.mm.default(view, permute_2)
        view_3: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_1, _shape_param_5);  mm_1 = _shape_param_5 = None
        view_4: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_3, _shape_param_6);  view_3 = _shape_param_6 = None
        permute_3: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        convert_element_type_3: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        permute_4: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0]);  convert_element_type_3 = None
        mm_2: "bf16[8192, 512]" = torch.ops.aten.mm.default(view, permute_4)
        view_5: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_2, _shape_param_7);  mm_2 = _shape_param_7 = None
        view_6: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_5, _shape_param_8);  view_5 = _shape_param_8 = None
        permute_5: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_6: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_1, _shape_param_9);  permute_1 = _shape_param_9 = None
        clone: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_7: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone, _shape_param_10);  clone = _shape_param_10 = None
        expand_1: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_6, _shape_param_11);  permute_6 = _shape_param_11 = None
        clone_1: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_8: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_1, _shape_param_12);  clone_1 = _shape_param_12 = None
        bmm: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_7, view_8)
        view_9: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm, _shape_param_13);  _shape_param_13 = None
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1024, 1]" = torch.ops.aten.unsqueeze.default(iota, 1)
        add_1: "i64[1024, 1]" = torch.ops.aten.add.Tensor(unsqueeze, 0);  unsqueeze = None
        unsqueeze_1: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(iota, 0)
        sub: "i64[1024, 1024]" = torch.ops.aten.sub.Tensor(unsqueeze_1, add_1);  unsqueeze_1 = add_1 = None
        gt_1: "b8[1024, 1024]" = torch.ops.aten.gt.Scalar(sub, 0)
        convert_element_type_4: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_1, torch.int64);  gt_1 = None
        mul_4: "i64[1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 16);  convert_element_type_4 = None
        add_2: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(mul_4, 0);  mul_4 = None
        abs_1: "i64[1024, 1024]" = torch.ops.aten.abs.default(sub)
        lt: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(abs_1, 8)
        convert_element_type_5: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
        div: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_5, 8);  convert_element_type_5 = None
        log: "f32[1024, 1024]" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
        mul_5: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None
        convert_element_type_6: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_5, torch.int64);  mul_5 = None
        add_3: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_6, 8);  convert_element_type_6 = None
        full_1: "i64[1024, 1024]" = torch.ops.aten.full.default(_shape_param_14, 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_14 = None
        minimum: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_3, full_1);  add_3 = full_1 = None
        where: "i64[1024, 1024]" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
        add_4: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(add_2, where);  add_2 = where = None
        embedding_1: "f32[1024, 1024, 8]" = torch.ops.aten.embedding.default(arg6_1, add_4);  arg6_1 = None
        permute_7: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1])
        unsqueeze_2: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None
        add_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_2, full);  unsqueeze_2 = full = None
        add_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_9, add_5);  view_9 = None
        convert_element_type_7: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        convert_element_type_8: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
        amax: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_8, [-1], True)
        sub_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_8, amax);  convert_element_type_8 = None
        exp: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_9: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1)
        inductor_random_1: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_15, inductor_lookup_seed_1, 'rand');  _shape_param_15 = inductor_lookup_seed_1 = None
        convert_element_type_10: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_1, torch.bfloat16);  inductor_random_1 = None
        gt_2: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_10, 0.1);  convert_element_type_10 = None
        mul_6: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_2, convert_element_type_9);  convert_element_type_9 = None
        mul_7: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_6, 1.1111111111111112);  mul_6 = None
        expand_2: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_7, _shape_param_16);  mul_7 = _shape_param_16 = None
        view_10: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_2, _shape_param_17);  expand_2 = _shape_param_17 = None
        expand_3: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_5, _shape_param_18);  permute_5 = _shape_param_18 = None
        clone_2: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_11: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_19);  clone_2 = _shape_param_19 = None
        bmm_1: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_10, view_11)
        view_12: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_20);  bmm_1 = _shape_param_20 = None
        permute_8: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        clone_3: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_13: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_3, _shape_param_21);  clone_3 = _shape_param_21 = None
        convert_element_type_11: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        permute_9: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_11, [1, 0]);  convert_element_type_11 = None
        view_14: "bf16[8192, 512]" = torch.ops.aten.view.default(view_13, _shape_param_22);  view_13 = _shape_param_22 = None
        mm_3: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_14, permute_9)
        view_15: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_3, _shape_param_23);  mm_3 = _shape_param_23 = None
        inductor_lookup_seed_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 2)
        inductor_random_2: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_24, inductor_lookup_seed_2, 'rand');  _shape_param_24 = inductor_lookup_seed_2 = None
        convert_element_type_12: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_2, torch.bfloat16);  inductor_random_2 = None
        gt_3: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_12, 0.1);  convert_element_type_12 = None
        mul_8: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_3, view_15);  view_15 = None
        mul_9: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_8, 1.1111111111111112);  mul_8 = None
        add_7: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(mul_1, mul_9);  mul_1 = mul_9 = None
        pow_2: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_7, 2)
        mean_1: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_8: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_10: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_7, rsqrt_1)
        mul_11: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg8_1, mul_10);  arg8_1 = mul_10 = None
        convert_element_type_13: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        convert_element_type_14: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        permute_10: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        view_16: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_14, _shape_param_25);  convert_element_type_14 = _shape_param_25 = None
        mm_4: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_16, permute_10)
        view_17: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_4, _shape_param_26);  mm_4 = _shape_param_26 = None
        relu: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_17);  view_17 = None
        inductor_lookup_seed_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 3)
        inductor_random_3: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default(_shape_param_27, inductor_lookup_seed_3, 'rand');  _shape_param_27 = inductor_lookup_seed_3 = None
        convert_element_type_15: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(inductor_random_3, torch.bfloat16);  inductor_random_3 = None
        gt_4: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(convert_element_type_15, 0.1);  convert_element_type_15 = None
        mul_12: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_4, relu)
        mul_13: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_12, 1.1111111111111112);  mul_12 = None
        convert_element_type_16: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        convert_element_type_17: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        permute_11: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_16, [1, 0]);  convert_element_type_16 = None
        view_18: "bf16[8192, 2048]" = torch.ops.aten.view.default(convert_element_type_17, _shape_param_28);  convert_element_type_17 = _shape_param_28 = None
        mm_5: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_18, permute_11)
        view_19: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_5, _shape_param_29);  mm_5 = _shape_param_29 = None
        inductor_lookup_seed_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 4)
        inductor_random_4: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_30, inductor_lookup_seed_4, 'rand');  _shape_param_30 = inductor_lookup_seed_4 = None
        convert_element_type_18: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_4, torch.bfloat16);  inductor_random_4 = None
        gt_5: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_18, 0.1);  convert_element_type_18 = None
        mul_14: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_5, view_19);  view_19 = None
        mul_15: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_14, 1.1111111111111112);  mul_14 = None
        add_9: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_7, mul_15);  mul_15 = None
        pow_3: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_9, 2)
        mean_2: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        add_10: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_16: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_9, rsqrt_2)
        mul_17: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg11_1, mul_16);  arg11_1 = mul_16 = None
        convert_element_type_19: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.bfloat16);  arg12_1 = None
        convert_element_type_20: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        permute_12: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_19, [1, 0]);  convert_element_type_19 = None
        view_20: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_20, _shape_param_31);  convert_element_type_20 = _shape_param_31 = None
        mm_6: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_20, permute_12)
        view_21: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_6, _shape_param_32);  mm_6 = _shape_param_32 = None
        view_22: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_21, _shape_param_33);  view_21 = _shape_param_33 = None
        permute_13: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        convert_element_type_21: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        permute_14: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_21, [1, 0]);  convert_element_type_21 = None
        mm_7: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_20, permute_14)
        view_23: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_7, _shape_param_34);  mm_7 = _shape_param_34 = None
        view_24: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_23, _shape_param_35);  view_23 = _shape_param_35 = None
        permute_15: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        convert_element_type_22: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.bfloat16);  arg14_1 = None
        permute_16: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_22, [1, 0]);  convert_element_type_22 = None
        mm_8: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_20, permute_16)
        view_25: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_8, _shape_param_36);  mm_8 = _shape_param_36 = None
        view_26: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_25, _shape_param_37);  view_25 = _shape_param_37 = None
        permute_17: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        permute_18: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        expand_4: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_13, _shape_param_38);  permute_13 = _shape_param_38 = None
        clone_4: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_27: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_39);  clone_4 = _shape_param_39 = None
        expand_5: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_18, _shape_param_40);  permute_18 = _shape_param_40 = None
        clone_5: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_28: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_5, _shape_param_41);  clone_5 = _shape_param_41 = None
        bmm_2: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_27, view_28)
        view_29: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_2, _shape_param_42);  bmm_2 = _shape_param_42 = None
        add_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_29, add_5);  view_29 = None
        convert_element_type_23: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None
        convert_element_type_24: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_23, torch.float32)
        amax_1: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_24, [-1], True)
        sub_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_24, amax_1);  convert_element_type_24 = None
        exp_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_2: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None
        convert_element_type_25: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        inductor_lookup_seed_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 5)
        inductor_random_5: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_43, inductor_lookup_seed_5, 'rand');  _shape_param_43 = inductor_lookup_seed_5 = None
        convert_element_type_26: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_5, torch.bfloat16);  inductor_random_5 = None
        gt_6: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_26, 0.1);  convert_element_type_26 = None
        mul_18: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_6, convert_element_type_25);  convert_element_type_25 = None
        mul_19: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_18, 1.1111111111111112);  mul_18 = None
        expand_6: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_19, _shape_param_44);  mul_19 = _shape_param_44 = None
        view_30: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_6, _shape_param_45);  expand_6 = _shape_param_45 = None
        expand_7: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_17, _shape_param_46);  permute_17 = _shape_param_46 = None
        clone_6: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_31: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_6, _shape_param_47);  clone_6 = _shape_param_47 = None
        bmm_3: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_30, view_31)
        view_32: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_48);  bmm_3 = _shape_param_48 = None
        permute_19: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        clone_7: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
        view_33: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_7, _shape_param_49);  clone_7 = _shape_param_49 = None
        convert_element_type_27: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg15_1, torch.bfloat16);  arg15_1 = None
        permute_20: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_27, [1, 0]);  convert_element_type_27 = None
        view_34: "bf16[8192, 512]" = torch.ops.aten.view.default(view_33, _shape_param_50);  view_33 = _shape_param_50 = None
        mm_9: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_34, permute_20)
        view_35: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_9, _shape_param_51);  mm_9 = _shape_param_51 = None
        inductor_lookup_seed_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 6)
        inductor_random_6: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_52, inductor_lookup_seed_6, 'rand');  _shape_param_52 = inductor_lookup_seed_6 = None
        convert_element_type_28: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_6, torch.bfloat16);  inductor_random_6 = None
        gt_7: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_28, 0.1);  convert_element_type_28 = None
        mul_20: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_7, view_35);  view_35 = None
        mul_21: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_20, 1.1111111111111112);  mul_20 = None
        add_12: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_9, mul_21);  mul_21 = None
        pow_4: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_12, 2)
        mean_3: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
        add_13: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_22: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_12, rsqrt_3)
        mul_23: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg16_1, mul_22);  arg16_1 = mul_22 = None
        convert_element_type_29: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        convert_element_type_30: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_23, torch.bfloat16);  mul_23 = None
        permute_21: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        view_36: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_30, _shape_param_53);  convert_element_type_30 = _shape_param_53 = None
        mm_10: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_36, permute_21)
        view_37: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_10, _shape_param_54);  mm_10 = _shape_param_54 = None
        relu_1: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_37);  view_37 = None
        inductor_lookup_seed_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 7)
        inductor_random_7: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default(_shape_param_55, inductor_lookup_seed_7, 'rand');  _shape_param_55 = inductor_lookup_seed_7 = None
        convert_element_type_31: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(inductor_random_7, torch.bfloat16);  inductor_random_7 = None
        gt_8: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(convert_element_type_31, 0.1);  convert_element_type_31 = None
        mul_24: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_8, relu_1)
        mul_25: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_24, 1.1111111111111112);  mul_24 = None
        convert_element_type_32: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.bfloat16);  arg18_1 = None
        convert_element_type_33: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None
        permute_22: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_32, [1, 0]);  convert_element_type_32 = None
        view_38: "bf16[8192, 2048]" = torch.ops.aten.view.default(convert_element_type_33, _shape_param_56);  convert_element_type_33 = _shape_param_56 = None
        mm_11: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_38, permute_22)
        view_39: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_11, _shape_param_57);  mm_11 = _shape_param_57 = None
        inductor_lookup_seed_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 8)
        inductor_random_8: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_58, inductor_lookup_seed_8, 'rand');  _shape_param_58 = inductor_lookup_seed_8 = None
        convert_element_type_34: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_8, torch.bfloat16);  inductor_random_8 = None
        gt_9: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_34, 0.1);  convert_element_type_34 = None
        mul_26: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_9, view_39);  view_39 = None
        mul_27: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_26, 1.1111111111111112);  mul_26 = None
        add_14: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_12, mul_27);  mul_27 = None
        pow_5: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_14, 2)
        mean_4: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
        add_15: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_28: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_14, rsqrt_4)
        mul_29: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg19_1, mul_28);  arg19_1 = mul_28 = None
        convert_element_type_35: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.bfloat16);  arg20_1 = None
        convert_element_type_36: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        permute_23: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_35, [1, 0]);  convert_element_type_35 = None
        view_40: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_36, _shape_param_59);  convert_element_type_36 = _shape_param_59 = None
        mm_12: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_40, permute_23)
        view_41: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_12, _shape_param_60);  mm_12 = _shape_param_60 = None
        view_42: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_41, _shape_param_61);  view_41 = _shape_param_61 = None
        permute_24: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        convert_element_type_37: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg21_1, torch.bfloat16);  arg21_1 = None
        permute_25: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_37, [1, 0]);  convert_element_type_37 = None
        mm_13: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_40, permute_25)
        view_43: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_13, _shape_param_62);  mm_13 = _shape_param_62 = None
        view_44: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_43, _shape_param_63);  view_43 = _shape_param_63 = None
        permute_26: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None
        convert_element_type_38: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg22_1, torch.bfloat16);  arg22_1 = None
        permute_27: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_38, [1, 0]);  convert_element_type_38 = None
        mm_14: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_40, permute_27)
        view_45: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_14, _shape_param_64);  mm_14 = _shape_param_64 = None
        view_46: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_45, _shape_param_65);  view_45 = _shape_param_65 = None
        permute_28: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        permute_29: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        expand_8: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_24, _shape_param_66);  permute_24 = _shape_param_66 = None
        clone_8: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_47: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_67);  clone_8 = _shape_param_67 = None
        expand_9: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_29, _shape_param_68);  permute_29 = _shape_param_68 = None
        clone_9: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_48: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_9, _shape_param_69);  clone_9 = _shape_param_69 = None
        bmm_4: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_47, view_48)
        view_49: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_4, _shape_param_70);  bmm_4 = _shape_param_70 = None
        add_16: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_49, add_5);  view_49 = None
        convert_element_type_39: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16);  add_16 = None
        convert_element_type_40: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_39, torch.float32)
        amax_2: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_40, [-1], True)
        sub_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_40, amax_2);  convert_element_type_40 = None
        exp_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_3: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = None
        convert_element_type_41: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        inductor_lookup_seed_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 9)
        inductor_random_9: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_71, inductor_lookup_seed_9, 'rand');  _shape_param_71 = inductor_lookup_seed_9 = None
        convert_element_type_42: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_9, torch.bfloat16);  inductor_random_9 = None
        gt_10: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_42, 0.1);  convert_element_type_42 = None
        mul_30: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_10, convert_element_type_41);  convert_element_type_41 = None
        mul_31: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_30, 1.1111111111111112);  mul_30 = None
        expand_10: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_31, _shape_param_72);  mul_31 = _shape_param_72 = None
        view_50: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_10, _shape_param_73);  expand_10 = _shape_param_73 = None
        expand_11: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_28, _shape_param_74);  permute_28 = _shape_param_74 = None
        clone_10: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_51: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_10, _shape_param_75);  clone_10 = _shape_param_75 = None
        bmm_5: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_50, view_51)
        view_52: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_5, _shape_param_76);  bmm_5 = _shape_param_76 = None
        permute_30: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        clone_11: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_53: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_11, _shape_param_77);  clone_11 = _shape_param_77 = None
        convert_element_type_43: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg23_1, torch.bfloat16);  arg23_1 = None
        permute_31: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_43, [1, 0]);  convert_element_type_43 = None
        view_54: "bf16[8192, 512]" = torch.ops.aten.view.default(view_53, _shape_param_78);  view_53 = _shape_param_78 = None
        mm_15: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_54, permute_31)
        view_55: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_15, _shape_param_79);  mm_15 = _shape_param_79 = None
        inductor_lookup_seed_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 10)
        inductor_random_10: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_80, inductor_lookup_seed_10, 'rand');  _shape_param_80 = inductor_lookup_seed_10 = None
        convert_element_type_44: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_10, torch.bfloat16);  inductor_random_10 = None
        gt_11: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_44, 0.1);  convert_element_type_44 = None
        mul_32: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_11, view_55);  view_55 = None
        mul_33: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_32, 1.1111111111111112);  mul_32 = None
        add_17: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_14, mul_33);  mul_33 = None
        pow_6: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_17, 2)
        mean_5: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None
        add_18: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_34: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_17, rsqrt_5)
        mul_35: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg24_1, mul_34);  arg24_1 = mul_34 = None
        convert_element_type_45: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg25_1, torch.bfloat16);  arg25_1 = None
        convert_element_type_46: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None
        permute_32: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_45, [1, 0]);  convert_element_type_45 = None
        view_56: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_46, _shape_param_81);  convert_element_type_46 = _shape_param_81 = None
        mm_16: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_56, permute_32)
        view_57: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_16, _shape_param_82);  mm_16 = _shape_param_82 = None
        relu_2: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_57);  view_57 = None
        inductor_lookup_seed_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 11)
        inductor_random_11: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default(_shape_param_83, inductor_lookup_seed_11, 'rand');  _shape_param_83 = inductor_lookup_seed_11 = None
        convert_element_type_47: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(inductor_random_11, torch.bfloat16);  inductor_random_11 = None
        gt_12: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(convert_element_type_47, 0.1);  convert_element_type_47 = None
        mul_36: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_12, relu_2)
        mul_37: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_36, 1.1111111111111112);  mul_36 = None
        convert_element_type_48: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg26_1, torch.bfloat16);  arg26_1 = None
        convert_element_type_49: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None
        permute_33: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_48, [1, 0]);  convert_element_type_48 = None
        view_58: "bf16[8192, 2048]" = torch.ops.aten.view.default(convert_element_type_49, _shape_param_84);  convert_element_type_49 = _shape_param_84 = None
        mm_17: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_58, permute_33)
        view_59: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_17, _shape_param_85);  mm_17 = _shape_param_85 = None
        inductor_lookup_seed_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 12)
        inductor_random_12: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_86, inductor_lookup_seed_12, 'rand');  _shape_param_86 = inductor_lookup_seed_12 = None
        convert_element_type_50: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_12, torch.bfloat16);  inductor_random_12 = None
        gt_13: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_50, 0.1);  convert_element_type_50 = None
        mul_38: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_13, view_59);  view_59 = None
        mul_39: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_38, 1.1111111111111112);  mul_38 = None
        add_19: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_17, mul_39);  mul_39 = None
        pow_7: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_19, 2)
        mean_6: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
        add_20: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_40: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_19, rsqrt_6)
        mul_41: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg27_1, mul_40);  arg27_1 = mul_40 = None
        convert_element_type_51: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg28_1, torch.bfloat16);  arg28_1 = None
        convert_element_type_52: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_41, torch.bfloat16);  mul_41 = None
        permute_34: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_51, [1, 0]);  convert_element_type_51 = None
        view_60: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_52, _shape_param_87);  convert_element_type_52 = _shape_param_87 = None
        mm_18: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_60, permute_34)
        view_61: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_18, _shape_param_88);  mm_18 = _shape_param_88 = None
        view_62: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_61, _shape_param_89);  view_61 = _shape_param_89 = None
        permute_35: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_62, [0, 2, 1, 3]);  view_62 = None
        convert_element_type_53: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg29_1, torch.bfloat16);  arg29_1 = None
        permute_36: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_53, [1, 0]);  convert_element_type_53 = None
        mm_19: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_60, permute_36)
        view_63: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_19, _shape_param_90);  mm_19 = _shape_param_90 = None
        view_64: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_63, _shape_param_91);  view_63 = _shape_param_91 = None
        permute_37: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        convert_element_type_54: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg30_1, torch.bfloat16);  arg30_1 = None
        permute_38: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_54, [1, 0]);  convert_element_type_54 = None
        mm_20: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_60, permute_38)
        view_65: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_20, _shape_param_92);  mm_20 = _shape_param_92 = None
        view_66: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_65, _shape_param_93);  view_65 = _shape_param_93 = None
        permute_39: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        permute_40: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2]);  permute_37 = None
        expand_12: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_35, _shape_param_94);  permute_35 = _shape_param_94 = None
        clone_12: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_67: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_95);  clone_12 = _shape_param_95 = None
        expand_13: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_40, _shape_param_96);  permute_40 = _shape_param_96 = None
        clone_13: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_68: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_13, _shape_param_97);  clone_13 = _shape_param_97 = None
        bmm_6: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_67, view_68)
        view_69: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_6, _shape_param_98);  bmm_6 = _shape_param_98 = None
        add_21: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_69, add_5);  view_69 = None
        convert_element_type_55: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16);  add_21 = None
        convert_element_type_56: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_55, torch.float32)
        amax_3: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_56, [-1], True)
        sub_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_56, amax_3);  convert_element_type_56 = None
        exp_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_4: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = None
        convert_element_type_57: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        inductor_lookup_seed_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 13)
        inductor_random_13: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_99, inductor_lookup_seed_13, 'rand');  _shape_param_99 = inductor_lookup_seed_13 = None
        convert_element_type_58: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_13, torch.bfloat16);  inductor_random_13 = None
        gt_14: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_58, 0.1);  convert_element_type_58 = None
        mul_42: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_14, convert_element_type_57);  convert_element_type_57 = None
        mul_43: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_42, 1.1111111111111112);  mul_42 = None
        expand_14: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_43, _shape_param_100);  mul_43 = _shape_param_100 = None
        view_70: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_14, _shape_param_101);  expand_14 = _shape_param_101 = None
        expand_15: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_39, _shape_param_102);  permute_39 = _shape_param_102 = None
        clone_14: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_71: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_14, _shape_param_103);  clone_14 = _shape_param_103 = None
        bmm_7: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_70, view_71)
        view_72: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_104);  bmm_7 = _shape_param_104 = None
        permute_41: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        clone_15: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
        view_73: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_15, _shape_param_105);  clone_15 = _shape_param_105 = None
        convert_element_type_59: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg31_1, torch.bfloat16);  arg31_1 = None
        permute_42: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_59, [1, 0]);  convert_element_type_59 = None
        view_74: "bf16[8192, 512]" = torch.ops.aten.view.default(view_73, _shape_param_106);  view_73 = _shape_param_106 = None
        mm_21: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_74, permute_42)
        view_75: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_21, _shape_param_107);  mm_21 = _shape_param_107 = None
        inductor_lookup_seed_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 14)
        inductor_random_14: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_108, inductor_lookup_seed_14, 'rand');  _shape_param_108 = inductor_lookup_seed_14 = None
        convert_element_type_60: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_14, torch.bfloat16);  inductor_random_14 = None
        gt_15: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_60, 0.1);  convert_element_type_60 = None
        mul_44: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_15, view_75);  view_75 = None
        mul_45: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_44, 1.1111111111111112);  mul_44 = None
        add_22: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_19, mul_45);  mul_45 = None
        pow_8: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_22, 2)
        mean_7: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
        add_23: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_46: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_22, rsqrt_7)
        mul_47: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg32_1, mul_46);  arg32_1 = mul_46 = None
        convert_element_type_61: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg33_1, torch.bfloat16);  arg33_1 = None
        convert_element_type_62: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_47, torch.bfloat16);  mul_47 = None
        permute_43: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_61, [1, 0]);  convert_element_type_61 = None
        view_76: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_62, _shape_param_109);  convert_element_type_62 = _shape_param_109 = None
        mm_22: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_76, permute_43)
        view_77: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_22, _shape_param_110);  mm_22 = _shape_param_110 = None
        relu_3: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_77);  view_77 = None
        inductor_lookup_seed_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 15)
        inductor_random_15: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default(_shape_param_111, inductor_lookup_seed_15, 'rand');  _shape_param_111 = inductor_lookup_seed_15 = None
        convert_element_type_63: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(inductor_random_15, torch.bfloat16);  inductor_random_15 = None
        gt_16: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(convert_element_type_63, 0.1);  convert_element_type_63 = None
        mul_48: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_16, relu_3)
        mul_49: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_48, 1.1111111111111112);  mul_48 = None
        convert_element_type_64: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg34_1, torch.bfloat16);  arg34_1 = None
        convert_element_type_65: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        permute_44: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_64, [1, 0]);  convert_element_type_64 = None
        view_78: "bf16[8192, 2048]" = torch.ops.aten.view.default(convert_element_type_65, _shape_param_112);  convert_element_type_65 = _shape_param_112 = None
        mm_23: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_78, permute_44)
        view_79: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_23, _shape_param_113);  mm_23 = _shape_param_113 = None
        inductor_lookup_seed_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 16)
        inductor_random_16: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_114, inductor_lookup_seed_16, 'rand');  _shape_param_114 = inductor_lookup_seed_16 = None
        convert_element_type_66: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_16, torch.bfloat16);  inductor_random_16 = None
        gt_17: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_66, 0.1);  convert_element_type_66 = None
        mul_50: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_17, view_79);  view_79 = None
        mul_51: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_50, 1.1111111111111112);  mul_50 = None
        add_24: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_22, mul_51);  mul_51 = None
        pow_9: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_24, 2)
        mean_8: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None
        add_25: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_52: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_24, rsqrt_8)
        mul_53: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg35_1, mul_52);  arg35_1 = mul_52 = None
        convert_element_type_67: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg36_1, torch.bfloat16);  arg36_1 = None
        convert_element_type_68: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16);  mul_53 = None
        permute_45: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_67, [1, 0]);  convert_element_type_67 = None
        view_80: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_68, _shape_param_115);  convert_element_type_68 = _shape_param_115 = None
        mm_24: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_80, permute_45)
        view_81: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_24, _shape_param_116);  mm_24 = _shape_param_116 = None
        view_82: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_81, _shape_param_117);  view_81 = _shape_param_117 = None
        permute_46: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        convert_element_type_69: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg37_1, torch.bfloat16);  arg37_1 = None
        permute_47: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_69, [1, 0]);  convert_element_type_69 = None
        mm_25: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_80, permute_47)
        view_83: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_25, _shape_param_118);  mm_25 = _shape_param_118 = None
        view_84: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_83, _shape_param_119);  view_83 = _shape_param_119 = None
        permute_48: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        convert_element_type_70: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg38_1, torch.bfloat16);  arg38_1 = None
        permute_49: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_70, [1, 0]);  convert_element_type_70 = None
        mm_26: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_80, permute_49)
        view_85: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_26, _shape_param_120);  mm_26 = _shape_param_120 = None
        view_86: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_85, _shape_param_121);  view_85 = _shape_param_121 = None
        permute_50: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        permute_51: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2]);  permute_48 = None
        expand_16: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_46, _shape_param_122);  permute_46 = _shape_param_122 = None
        clone_16: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_87: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_123);  clone_16 = _shape_param_123 = None
        expand_17: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_51, _shape_param_124);  permute_51 = _shape_param_124 = None
        clone_17: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_88: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_17, _shape_param_125);  clone_17 = _shape_param_125 = None
        bmm_8: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_87, view_88)
        view_89: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_8, _shape_param_126);  bmm_8 = _shape_param_126 = None
        add_26: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_89, add_5);  view_89 = None
        convert_element_type_71: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_26, torch.bfloat16);  add_26 = None
        convert_element_type_72: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_71, torch.float32)
        amax_4: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_72, [-1], True)
        sub_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_72, amax_4);  convert_element_type_72 = None
        exp_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_5: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = None
        convert_element_type_73: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        inductor_lookup_seed_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 17)
        inductor_random_17: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_127, inductor_lookup_seed_17, 'rand');  _shape_param_127 = inductor_lookup_seed_17 = None
        convert_element_type_74: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_17, torch.bfloat16);  inductor_random_17 = None
        gt_18: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_74, 0.1);  convert_element_type_74 = None
        mul_54: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_18, convert_element_type_73);  convert_element_type_73 = None
        mul_55: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_54, 1.1111111111111112);  mul_54 = None
        expand_18: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_55, _shape_param_128);  mul_55 = _shape_param_128 = None
        view_90: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_18, _shape_param_129);  expand_18 = _shape_param_129 = None
        expand_19: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_50, _shape_param_130);  permute_50 = _shape_param_130 = None
        clone_18: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_91: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_18, _shape_param_131);  clone_18 = _shape_param_131 = None
        bmm_9: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_90, view_91)
        view_92: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_9, _shape_param_132);  bmm_9 = _shape_param_132 = None
        permute_52: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
        clone_19: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
        view_93: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_19, _shape_param_133);  clone_19 = _shape_param_133 = None
        convert_element_type_75: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg39_1, torch.bfloat16);  arg39_1 = None
        permute_53: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_75, [1, 0]);  convert_element_type_75 = None
        view_94: "bf16[8192, 512]" = torch.ops.aten.view.default(view_93, _shape_param_134);  view_93 = _shape_param_134 = None
        mm_27: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_94, permute_53)
        view_95: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_27, _shape_param_135);  mm_27 = _shape_param_135 = None
        inductor_lookup_seed_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 18)
        inductor_random_18: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_136, inductor_lookup_seed_18, 'rand');  _shape_param_136 = inductor_lookup_seed_18 = None
        convert_element_type_76: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_18, torch.bfloat16);  inductor_random_18 = None
        gt_19: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_76, 0.1);  convert_element_type_76 = None
        mul_56: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_19, view_95);  view_95 = None
        mul_57: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_56, 1.1111111111111112);  mul_56 = None
        add_27: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_24, mul_57);  mul_57 = None
        pow_10: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_27, 2)
        mean_9: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
        add_28: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_58: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_27, rsqrt_9)
        mul_59: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg40_1, mul_58);  arg40_1 = mul_58 = None
        convert_element_type_77: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg41_1, torch.bfloat16);  arg41_1 = None
        convert_element_type_78: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None
        permute_54: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_77, [1, 0]);  convert_element_type_77 = None
        view_96: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_78, _shape_param_137);  convert_element_type_78 = _shape_param_137 = None
        mm_28: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_96, permute_54)
        view_97: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_28, _shape_param_138);  mm_28 = _shape_param_138 = None
        relu_4: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_97);  view_97 = None
        inductor_lookup_seed_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 19)
        inductor_random_19: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default(_shape_param_139, inductor_lookup_seed_19, 'rand');  _shape_param_139 = inductor_lookup_seed_19 = None
        convert_element_type_79: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(inductor_random_19, torch.bfloat16);  inductor_random_19 = None
        gt_20: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(convert_element_type_79, 0.1);  convert_element_type_79 = None
        mul_60: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_20, relu_4)
        mul_61: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_60, 1.1111111111111112);  mul_60 = None
        convert_element_type_80: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg42_1, torch.bfloat16);  arg42_1 = None
        convert_element_type_81: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None
        permute_55: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_80, [1, 0]);  convert_element_type_80 = None
        view_98: "bf16[8192, 2048]" = torch.ops.aten.view.default(convert_element_type_81, _shape_param_140);  convert_element_type_81 = _shape_param_140 = None
        mm_29: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_98, permute_55)
        view_99: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_29, _shape_param_141);  mm_29 = _shape_param_141 = None
        inductor_lookup_seed_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 20)
        inductor_random_20: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_142, inductor_lookup_seed_20, 'rand');  _shape_param_142 = inductor_lookup_seed_20 = None
        convert_element_type_82: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_20, torch.bfloat16);  inductor_random_20 = None
        gt_21: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_82, 0.1);  convert_element_type_82 = None
        mul_62: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_21, view_99);  view_99 = None
        mul_63: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_62, 1.1111111111111112);  mul_62 = None
        add_29: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_27, mul_63);  mul_63 = None
        pow_11: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_29, 2)
        mean_10: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
        add_30: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_64: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_29, rsqrt_10)
        mul_65: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg43_1, mul_64);  arg43_1 = mul_64 = None
        convert_element_type_83: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg44_1, torch.bfloat16);  arg44_1 = None
        convert_element_type_84: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None
        permute_56: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_83, [1, 0]);  convert_element_type_83 = None
        view_100: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_84, _shape_param_143);  convert_element_type_84 = _shape_param_143 = None
        mm_30: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_100, permute_56)
        view_101: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_30, _shape_param_144);  mm_30 = _shape_param_144 = None
        view_102: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_101, _shape_param_145);  view_101 = _shape_param_145 = None
        permute_57: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        convert_element_type_85: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg45_1, torch.bfloat16);  arg45_1 = None
        permute_58: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_85, [1, 0]);  convert_element_type_85 = None
        mm_31: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_100, permute_58)
        view_103: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_31, _shape_param_146);  mm_31 = _shape_param_146 = None
        view_104: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_103, _shape_param_147);  view_103 = _shape_param_147 = None
        permute_59: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        convert_element_type_86: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg46_1, torch.bfloat16);  arg46_1 = None
        permute_60: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_86, [1, 0]);  convert_element_type_86 = None
        mm_32: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_100, permute_60)
        view_105: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_32, _shape_param_148);  mm_32 = _shape_param_148 = None
        view_106: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_105, _shape_param_149);  view_105 = _shape_param_149 = None
        permute_61: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        permute_62: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2]);  permute_59 = None
        expand_20: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_57, _shape_param_150);  permute_57 = _shape_param_150 = None
        clone_20: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_107: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_151);  clone_20 = _shape_param_151 = None
        expand_21: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_62, _shape_param_152);  permute_62 = _shape_param_152 = None
        clone_21: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_108: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_21, _shape_param_153);  clone_21 = _shape_param_153 = None
        bmm_10: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_107, view_108)
        view_109: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_10, _shape_param_154);  bmm_10 = _shape_param_154 = None
        add_31: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_109, add_5);  view_109 = add_5 = None
        convert_element_type_87: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None
        convert_element_type_88: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_87, torch.float32)
        amax_5: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_88, [-1], True)
        sub_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_88, amax_5);  convert_element_type_88 = None
        exp_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_6: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = None
        convert_element_type_89: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        inductor_lookup_seed_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 21)
        inductor_random_21: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_155, inductor_lookup_seed_21, 'rand');  _shape_param_155 = inductor_lookup_seed_21 = None
        convert_element_type_90: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_21, torch.bfloat16);  inductor_random_21 = None
        gt_22: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_90, 0.1);  convert_element_type_90 = None
        mul_66: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_22, convert_element_type_89);  convert_element_type_89 = None
        mul_67: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_66, 1.1111111111111112);  mul_66 = None
        expand_22: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_67, _shape_param_156);  mul_67 = _shape_param_156 = None
        view_110: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_22, _shape_param_157);  expand_22 = _shape_param_157 = None
        expand_23: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_61, _shape_param_158);  permute_61 = _shape_param_158 = None
        clone_22: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_111: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_22, _shape_param_159);  clone_22 = _shape_param_159 = None
        bmm_11: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_110, view_111)
        view_112: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_160);  bmm_11 = _shape_param_160 = None
        permute_63: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        clone_23: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
        view_113: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_23, _shape_param_161);  clone_23 = _shape_param_161 = None
        convert_element_type_91: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg47_1, torch.bfloat16);  arg47_1 = None
        permute_64: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_91, [1, 0]);  convert_element_type_91 = None
        view_114: "bf16[8192, 512]" = torch.ops.aten.view.default(view_113, _shape_param_162);  view_113 = _shape_param_162 = None
        mm_33: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_114, permute_64)
        view_115: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_33, _shape_param_163);  mm_33 = _shape_param_163 = None
        inductor_lookup_seed_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 22)
        inductor_random_22: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_164, inductor_lookup_seed_22, 'rand');  _shape_param_164 = inductor_lookup_seed_22 = None
        convert_element_type_92: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_22, torch.bfloat16);  inductor_random_22 = None
        gt_23: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_92, 0.1);  convert_element_type_92 = None
        mul_68: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_23, view_115);  view_115 = None
        mul_69: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_68, 1.1111111111111112);  mul_68 = None
        add_32: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_29, mul_69);  mul_69 = None
        pow_12: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_32, 2)
        mean_11: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None
        add_33: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        mul_70: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_32, rsqrt_11)
        mul_71: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg48_1, mul_70);  arg48_1 = mul_70 = None
        convert_element_type_93: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg49_1, torch.bfloat16);  arg49_1 = None
        convert_element_type_94: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_71, torch.bfloat16);  mul_71 = None
        permute_65: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_93, [1, 0]);  convert_element_type_93 = None
        view_116: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_94, _shape_param_165);  convert_element_type_94 = _shape_param_165 = None
        mm_34: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_116, permute_65)
        view_117: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_34, _shape_param_166);  mm_34 = _shape_param_166 = None
        relu_5: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_117);  view_117 = None
        inductor_lookup_seed_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 23)
        inductor_random_23: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default(_shape_param_167, inductor_lookup_seed_23, 'rand');  _shape_param_167 = inductor_lookup_seed_23 = None
        convert_element_type_95: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(inductor_random_23, torch.bfloat16);  inductor_random_23 = None
        gt_24: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(convert_element_type_95, 0.1);  convert_element_type_95 = None
        mul_72: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_24, relu_5)
        mul_73: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_72, 1.1111111111111112);  mul_72 = None
        convert_element_type_96: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg50_1, torch.bfloat16);  arg50_1 = None
        convert_element_type_97: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(mul_73, torch.bfloat16);  mul_73 = None
        permute_66: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_96, [1, 0]);  convert_element_type_96 = None
        view_118: "bf16[8192, 2048]" = torch.ops.aten.view.default(convert_element_type_97, _shape_param_168);  convert_element_type_97 = _shape_param_168 = None
        mm_35: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_118, permute_66)
        view_119: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_35, _shape_param_169);  mm_35 = _shape_param_169 = None
        inductor_lookup_seed_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 24)
        inductor_random_24: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_170, inductor_lookup_seed_24, 'rand');  _shape_param_170 = inductor_lookup_seed_24 = None
        convert_element_type_98: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_24, torch.bfloat16);  inductor_random_24 = None
        gt_25: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_98, 0.1);  convert_element_type_98 = None
        mul_74: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_25, view_119);  view_119 = None
        mul_75: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_74, 1.1111111111111112);  mul_74 = None
        add_34: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_32, mul_75);  mul_75 = None
        pow_13: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_34, 2)
        mean_12: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
        add_35: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        mul_76: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_34, rsqrt_12)
        mul_77: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg51_1, mul_76);  arg51_1 = mul_76 = None
        inductor_lookup_seed_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 25)
        inductor_random_25: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_171, inductor_lookup_seed_25, 'rand');  _shape_param_171 = inductor_lookup_seed_25 = None
        gt_26: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_25, 0.1);  inductor_random_25 = None
        mul_78: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_26, mul_77);  mul_77 = None
        mul_79: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_78, 1.1111111111111112);  mul_78 = None
        add_36: "i64[1024]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze_3: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(add_36, 0);  add_36 = None
        unsqueeze_4: "i64[1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3)
        unsqueeze_6: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2)
        le: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_6, unsqueeze_5);  unsqueeze_6 = unsqueeze_5 = None
        expand_24: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(le, _shape_param_172);  le = _shape_param_172 = None
        full_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_3: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_24, full_2, full_3);  expand_24 = full_3 = None
        inductor_lookup_seed_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 26)
        inductor_random_26: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_173, inductor_lookup_seed_26, 'rand');  _shape_param_173 = inductor_lookup_seed_26 = None
        gt_27: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_26, 0.1);  inductor_random_26 = None
        mul_80: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_27, embedding)
        mul_81: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_80, 1.1111111111111112);  mul_80 = None
        pow_14: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_81, 2)
        mean_13: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
        add_37: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_82: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_81, rsqrt_13)
        mul_83: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg52_1, mul_82);  arg52_1 = mul_82 = None
        convert_element_type_99: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg53_1, torch.bfloat16);  arg53_1 = None
        convert_element_type_100: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_83, torch.bfloat16);  mul_83 = None
        permute_67: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_99, [1, 0]);  convert_element_type_99 = None
        view_120: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_100, _shape_param_174);  convert_element_type_100 = _shape_param_174 = None
        mm_36: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_120, permute_67)
        view_121: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_36, _shape_param_175);  mm_36 = _shape_param_175 = None
        view_122: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_121, _shape_param_176);  view_121 = _shape_param_176 = None
        permute_68: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        convert_element_type_101: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg54_1, torch.bfloat16);  arg54_1 = None
        permute_69: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_101, [1, 0]);  convert_element_type_101 = None
        mm_37: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_120, permute_69)
        view_123: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_37, _shape_param_177);  mm_37 = _shape_param_177 = None
        view_124: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_123, _shape_param_178);  view_123 = _shape_param_178 = None
        permute_70: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        convert_element_type_102: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg55_1, torch.bfloat16);  arg55_1 = None
        permute_71: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_102, [1, 0]);  convert_element_type_102 = None
        mm_38: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_120, permute_71)
        view_125: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_38, _shape_param_179);  mm_38 = _shape_param_179 = None
        view_126: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_125, _shape_param_180);  view_125 = _shape_param_180 = None
        permute_72: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        permute_73: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2]);  permute_70 = None
        expand_25: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_68, _shape_param_181);  permute_68 = _shape_param_181 = None
        clone_24: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_127: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_182);  clone_24 = _shape_param_182 = None
        expand_26: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_73, _shape_param_183);  permute_73 = _shape_param_183 = None
        clone_25: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_128: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_25, _shape_param_184);  clone_25 = _shape_param_184 = None
        bmm_12: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_127, view_128)
        view_129: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_12, _shape_param_185);  _shape_param_185 = None
        full_4: "i64[1024, 1024]" = torch.ops.aten.full.default(_shape_param_186, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_186 = None
        minimum_1: "i64[1024, 1024]" = torch.ops.aten.minimum.default(sub, full_4);  sub = full_4 = None
        neg: "i64[1024, 1024]" = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None
        lt_1: "b8[1024, 1024]" = torch.ops.aten.lt.Scalar(neg, 16)
        convert_element_type_103: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
        div_8: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(convert_element_type_103, 16);  convert_element_type_103 = None
        log_1: "f32[1024, 1024]" = torch.ops.aten.log.default(div_8);  div_8 = None
        div_9: "f32[1024, 1024]" = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
        mul_84: "f32[1024, 1024]" = torch.ops.aten.mul.Tensor(div_9, 16);  div_9 = None
        convert_element_type_104: "i64[1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_84, torch.int64);  mul_84 = None
        add_38: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_104, 16);  convert_element_type_104 = None
        full_5: "i64[1024, 1024]" = torch.ops.aten.full.default(_shape_param_187, 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_187 = None
        minimum_2: "i64[1024, 1024]" = torch.ops.aten.minimum.default(add_38, full_5);  add_38 = full_5 = None
        where_2: "i64[1024, 1024]" = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
        add_39: "i64[1024, 1024]" = torch.ops.aten.add.Tensor(where_2, 0);  where_2 = None
        embedding_2: "f32[1024, 1024, 8]" = torch.ops.aten.embedding.default(arg56_1, add_39);  arg56_1 = None
        permute_74: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding_2, [2, 0, 1])
        unsqueeze_7: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_74, 0);  permute_74 = None
        add_40: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_7, where_1);  unsqueeze_7 = where_1 = None
        add_41: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_129, add_40);  view_129 = None
        convert_element_type_105: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None
        convert_element_type_106: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_105, torch.float32);  convert_element_type_105 = None
        amax_6: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_106, [-1], True)
        sub_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_106, amax_6);  convert_element_type_106 = None
        exp_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_7: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_10: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = None
        convert_element_type_107: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        inductor_lookup_seed_27: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 27)
        inductor_random_27: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_188, inductor_lookup_seed_27, 'rand');  _shape_param_188 = inductor_lookup_seed_27 = None
        convert_element_type_108: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_27, torch.bfloat16);  inductor_random_27 = None
        gt_28: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_108, 0.1);  convert_element_type_108 = None
        mul_85: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_28, convert_element_type_107);  convert_element_type_107 = None
        mul_86: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_85, 1.1111111111111112);  mul_85 = None
        expand_27: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_86, _shape_param_189);  mul_86 = _shape_param_189 = None
        view_130: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_27, _shape_param_190);  expand_27 = _shape_param_190 = None
        expand_28: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_72, _shape_param_191);  permute_72 = _shape_param_191 = None
        clone_26: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_131: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_26, _shape_param_192);  clone_26 = _shape_param_192 = None
        bmm_13: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_130, view_131)
        view_132: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_13, _shape_param_193);  bmm_13 = _shape_param_193 = None
        permute_75: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
        clone_27: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
        view_133: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_27, _shape_param_194);  clone_27 = _shape_param_194 = None
        convert_element_type_109: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg57_1, torch.bfloat16);  arg57_1 = None
        permute_76: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_109, [1, 0]);  convert_element_type_109 = None
        view_134: "bf16[8192, 512]" = torch.ops.aten.view.default(view_133, _shape_param_195);  view_133 = _shape_param_195 = None
        mm_39: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_134, permute_76)
        view_135: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_39, _shape_param_196);  mm_39 = _shape_param_196 = None
        inductor_lookup_seed_28: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 28)
        inductor_random_28: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_197, inductor_lookup_seed_28, 'rand');  _shape_param_197 = inductor_lookup_seed_28 = None
        convert_element_type_110: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_28, torch.bfloat16);  inductor_random_28 = None
        gt_29: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_110, 0.1);  convert_element_type_110 = None
        mul_87: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_29, view_135);  view_135 = None
        mul_88: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_87, 1.1111111111111112);  mul_87 = None
        add_42: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(mul_81, mul_88);  mul_81 = mul_88 = None
        pow_15: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_42, 2)
        mean_14: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None
        add_43: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_89: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_42, rsqrt_14)
        mul_90: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg58_1, mul_89);  arg58_1 = mul_89 = None
        convert_element_type_111: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg59_1, torch.bfloat16);  arg59_1 = None
        convert_element_type_112: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16);  mul_90 = None
        permute_77: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_111, [1, 0]);  convert_element_type_111 = None
        view_136: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_112, _shape_param_198);  convert_element_type_112 = _shape_param_198 = None
        mm_40: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_136, permute_77)
        view_137: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_40, _shape_param_199);  mm_40 = _shape_param_199 = None
        view_138: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_137, _shape_param_200);  view_137 = _shape_param_200 = None
        permute_78: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        convert_element_type_113: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg60_1, torch.bfloat16);  arg60_1 = None
        convert_element_type_114: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16)
        permute_79: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_113, [1, 0]);  convert_element_type_113 = None
        view_139: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_114, _shape_param_201);  convert_element_type_114 = _shape_param_201 = None
        mm_41: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_139, permute_79)
        view_140: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_41, _shape_param_202);  mm_41 = _shape_param_202 = None
        view_141: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_140, _shape_param_203);  view_140 = _shape_param_203 = None
        permute_80: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
        convert_element_type_115: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg61_1, torch.bfloat16);  arg61_1 = None
        permute_81: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_115, [1, 0]);  convert_element_type_115 = None
        mm_42: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_139, permute_81)
        view_142: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_42, _shape_param_204);  mm_42 = _shape_param_204 = None
        view_143: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_142, _shape_param_205);  view_142 = _shape_param_205 = None
        permute_82: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_143, [0, 2, 1, 3]);  view_143 = None
        permute_83: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2]);  permute_80 = None
        expand_29: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_78, _shape_param_206);  permute_78 = _shape_param_206 = None
        clone_28: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_144: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_207);  clone_28 = _shape_param_207 = None
        expand_30: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_83, _shape_param_208);  permute_83 = _shape_param_208 = None
        clone_29: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_145: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_29, _shape_param_209);  clone_29 = _shape_param_209 = None
        bmm_14: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_144, view_145)
        view_146: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_14, _shape_param_210);  _shape_param_210 = None
        convert_element_type_116: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_146, torch.bfloat16);  view_146 = None
        convert_element_type_117: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_116, torch.float32);  convert_element_type_116 = None
        amax_7: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_117, [-1], True)
        sub_8: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_117, amax_7);  convert_element_type_117 = None
        exp_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_8: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = None
        convert_element_type_118: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        inductor_lookup_seed_29: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 29)
        inductor_random_29: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_211, inductor_lookup_seed_29, 'rand');  _shape_param_211 = inductor_lookup_seed_29 = None
        convert_element_type_119: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_29, torch.bfloat16);  inductor_random_29 = None
        gt_30: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_119, 0.1);  convert_element_type_119 = None
        mul_91: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_30, convert_element_type_118);  convert_element_type_118 = None
        mul_92: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_91, 1.1111111111111112);  mul_91 = None
        expand_31: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_92, _shape_param_212);  mul_92 = _shape_param_212 = None
        view_147: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_31, _shape_param_213);  expand_31 = _shape_param_213 = None
        expand_32: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_82, _shape_param_214);  permute_82 = _shape_param_214 = None
        clone_30: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_148: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_30, _shape_param_215);  clone_30 = _shape_param_215 = None
        bmm_15: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_147, view_148)
        view_149: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_216);  bmm_15 = _shape_param_216 = None
        permute_84: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
        clone_31: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_150: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_31, _shape_param_217);  clone_31 = _shape_param_217 = None
        convert_element_type_120: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg62_1, torch.bfloat16);  arg62_1 = None
        permute_85: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_120, [1, 0]);  convert_element_type_120 = None
        view_151: "bf16[8192, 512]" = torch.ops.aten.view.default(view_150, _shape_param_218);  view_150 = _shape_param_218 = None
        mm_43: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_151, permute_85)
        view_152: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_43, _shape_param_219);  mm_43 = _shape_param_219 = None
        inductor_lookup_seed_30: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 30)
        inductor_random_30: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_220, inductor_lookup_seed_30, 'rand');  _shape_param_220 = inductor_lookup_seed_30 = None
        convert_element_type_121: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_30, torch.bfloat16);  inductor_random_30 = None
        gt_31: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_121, 0.1);  convert_element_type_121 = None
        mul_93: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_31, view_152);  view_152 = None
        mul_94: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_93, 1.1111111111111112);  mul_93 = None
        add_44: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_42, mul_94);  mul_94 = None
        pow_16: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_44, 2)
        mean_15: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
        add_45: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_95: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_44, rsqrt_15)
        mul_96: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg63_1, mul_95);  arg63_1 = mul_95 = None
        convert_element_type_122: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg64_1, torch.bfloat16);  arg64_1 = None
        convert_element_type_123: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_96, torch.bfloat16);  mul_96 = None
        permute_86: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_122, [1, 0]);  convert_element_type_122 = None
        view_153: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_123, _shape_param_221);  convert_element_type_123 = _shape_param_221 = None
        mm_44: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_153, permute_86)
        view_154: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_44, _shape_param_222);  mm_44 = _shape_param_222 = None
        relu_6: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_154);  view_154 = None
        inductor_lookup_seed_31: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 31)
        inductor_random_31: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default(_shape_param_223, inductor_lookup_seed_31, 'rand');  _shape_param_223 = inductor_lookup_seed_31 = None
        convert_element_type_124: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(inductor_random_31, torch.bfloat16);  inductor_random_31 = None
        gt_32: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(convert_element_type_124, 0.1);  convert_element_type_124 = None
        mul_97: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_32, relu_6)
        mul_98: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_97, 1.1111111111111112);  mul_97 = None
        convert_element_type_125: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg65_1, torch.bfloat16);  arg65_1 = None
        convert_element_type_126: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(mul_98, torch.bfloat16);  mul_98 = None
        permute_87: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_125, [1, 0]);  convert_element_type_125 = None
        view_155: "bf16[8192, 2048]" = torch.ops.aten.view.default(convert_element_type_126, _shape_param_224);  convert_element_type_126 = _shape_param_224 = None
        mm_45: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_155, permute_87)
        view_156: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_45, _shape_param_225);  mm_45 = _shape_param_225 = None
        inductor_lookup_seed_32: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 32)
        inductor_random_32: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_226, inductor_lookup_seed_32, 'rand');  _shape_param_226 = inductor_lookup_seed_32 = None
        convert_element_type_127: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_32, torch.bfloat16);  inductor_random_32 = None
        gt_33: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_127, 0.1);  convert_element_type_127 = None
        mul_99: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_33, view_156);  view_156 = None
        mul_100: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_99, 1.1111111111111112);  mul_99 = None
        add_46: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_44, mul_100);  mul_100 = None
        pow_17: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_46, 2)
        mean_16: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
        add_47: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_101: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_46, rsqrt_16)
        mul_102: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg66_1, mul_101);  arg66_1 = mul_101 = None
        convert_element_type_128: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg67_1, torch.bfloat16);  arg67_1 = None
        convert_element_type_129: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_102, torch.bfloat16);  mul_102 = None
        permute_88: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_128, [1, 0]);  convert_element_type_128 = None
        view_157: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_129, _shape_param_227);  convert_element_type_129 = _shape_param_227 = None
        mm_46: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_157, permute_88)
        view_158: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_46, _shape_param_228);  mm_46 = _shape_param_228 = None
        view_159: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_158, _shape_param_229);  view_158 = _shape_param_229 = None
        permute_89: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
        convert_element_type_130: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg68_1, torch.bfloat16);  arg68_1 = None
        permute_90: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_130, [1, 0]);  convert_element_type_130 = None
        mm_47: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_157, permute_90)
        view_160: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_47, _shape_param_230);  mm_47 = _shape_param_230 = None
        view_161: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_160, _shape_param_231);  view_160 = _shape_param_231 = None
        permute_91: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
        convert_element_type_131: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg69_1, torch.bfloat16);  arg69_1 = None
        permute_92: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_131, [1, 0]);  convert_element_type_131 = None
        mm_48: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_157, permute_92)
        view_162: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_48, _shape_param_232);  mm_48 = _shape_param_232 = None
        view_163: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_162, _shape_param_233);  view_162 = _shape_param_233 = None
        permute_93: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
        permute_94: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
        expand_33: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_89, _shape_param_234);  permute_89 = _shape_param_234 = None
        clone_32: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_164: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_32, _shape_param_235);  clone_32 = _shape_param_235 = None
        expand_34: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_94, _shape_param_236);  permute_94 = _shape_param_236 = None
        clone_33: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_165: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_33, _shape_param_237);  clone_33 = _shape_param_237 = None
        bmm_16: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_164, view_165)
        view_166: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_16, _shape_param_238);  bmm_16 = _shape_param_238 = None
        add_48: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_166, add_40);  view_166 = None
        convert_element_type_132: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_48, torch.bfloat16);  add_48 = None
        convert_element_type_133: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_132, torch.float32)
        amax_8: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_133, [-1], True)
        sub_9: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_133, amax_8);  convert_element_type_133 = None
        exp_8: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_9: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_12: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = None
        convert_element_type_134: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None
        inductor_lookup_seed_33: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 33)
        inductor_random_33: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_239, inductor_lookup_seed_33, 'rand');  _shape_param_239 = inductor_lookup_seed_33 = None
        convert_element_type_135: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_33, torch.bfloat16);  inductor_random_33 = None
        gt_34: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_135, 0.1);  convert_element_type_135 = None
        mul_103: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_34, convert_element_type_134);  convert_element_type_134 = None
        mul_104: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_103, 1.1111111111111112);  mul_103 = None
        expand_35: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_104, _shape_param_240);  mul_104 = _shape_param_240 = None
        view_167: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_35, _shape_param_241);  expand_35 = _shape_param_241 = None
        expand_36: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_93, _shape_param_242);  permute_93 = _shape_param_242 = None
        clone_34: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_168: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_34, _shape_param_243);  clone_34 = _shape_param_243 = None
        bmm_17: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_167, view_168)
        view_169: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_17, _shape_param_244);  bmm_17 = _shape_param_244 = None
        permute_95: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
        clone_35: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_170: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_35, _shape_param_245);  clone_35 = _shape_param_245 = None
        convert_element_type_136: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg70_1, torch.bfloat16);  arg70_1 = None
        permute_96: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_136, [1, 0]);  convert_element_type_136 = None
        view_171: "bf16[8192, 512]" = torch.ops.aten.view.default(view_170, _shape_param_246);  view_170 = _shape_param_246 = None
        mm_49: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_171, permute_96)
        view_172: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_49, _shape_param_247);  mm_49 = _shape_param_247 = None
        inductor_lookup_seed_34: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 34)
        inductor_random_34: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_248, inductor_lookup_seed_34, 'rand');  _shape_param_248 = inductor_lookup_seed_34 = None
        convert_element_type_137: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_34, torch.bfloat16);  inductor_random_34 = None
        gt_35: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_137, 0.1);  convert_element_type_137 = None
        mul_105: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_35, view_172);  view_172 = None
        mul_106: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_105, 1.1111111111111112);  mul_105 = None
        add_49: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_46, mul_106);  mul_106 = None
        pow_18: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_49, 2)
        mean_17: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None
        add_50: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_107: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_49, rsqrt_17)
        mul_108: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg71_1, mul_107);  arg71_1 = mul_107 = None
        convert_element_type_138: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg72_1, torch.bfloat16);  arg72_1 = None
        convert_element_type_139: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_108, torch.bfloat16);  mul_108 = None
        permute_97: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_138, [1, 0]);  convert_element_type_138 = None
        view_173: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_139, _shape_param_249);  convert_element_type_139 = _shape_param_249 = None
        mm_50: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_173, permute_97)
        view_174: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_50, _shape_param_250);  mm_50 = _shape_param_250 = None
        view_175: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_174, _shape_param_251);  view_174 = _shape_param_251 = None
        permute_98: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
        convert_element_type_140: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg73_1, torch.bfloat16);  arg73_1 = None
        permute_99: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_140, [1, 0]);  convert_element_type_140 = None
        mm_51: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_139, permute_99)
        view_176: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_51, _shape_param_252);  mm_51 = _shape_param_252 = None
        view_177: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_176, _shape_param_253);  view_176 = _shape_param_253 = None
        permute_100: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None
        convert_element_type_141: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg74_1, torch.bfloat16);  arg74_1 = None
        permute_101: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_141, [1, 0]);  convert_element_type_141 = None
        mm_52: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_139, permute_101)
        view_178: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_52, _shape_param_254);  mm_52 = _shape_param_254 = None
        view_179: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_178, _shape_param_255);  view_178 = _shape_param_255 = None
        permute_102: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_179, [0, 2, 1, 3]);  view_179 = None
        permute_103: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2]);  permute_100 = None
        expand_37: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_98, _shape_param_256);  permute_98 = _shape_param_256 = None
        clone_36: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_180: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_257);  clone_36 = _shape_param_257 = None
        expand_38: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_103, _shape_param_258);  permute_103 = _shape_param_258 = None
        clone_37: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_181: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_37, _shape_param_259);  clone_37 = _shape_param_259 = None
        bmm_18: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_180, view_181)
        view_182: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_18, _shape_param_260);  _shape_param_260 = None
        convert_element_type_142: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_182, torch.bfloat16);  view_182 = None
        convert_element_type_143: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_142, torch.float32);  convert_element_type_142 = None
        amax_9: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_143, [-1], True)
        sub_10: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_143, amax_9);  convert_element_type_143 = None
        exp_9: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_10: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_13: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = None
        convert_element_type_144: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None
        inductor_lookup_seed_35: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 35)
        inductor_random_35: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_261, inductor_lookup_seed_35, 'rand');  _shape_param_261 = inductor_lookup_seed_35 = None
        convert_element_type_145: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_35, torch.bfloat16);  inductor_random_35 = None
        gt_36: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_145, 0.1);  convert_element_type_145 = None
        mul_109: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_36, convert_element_type_144);  convert_element_type_144 = None
        mul_110: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_109, 1.1111111111111112);  mul_109 = None
        expand_39: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_110, _shape_param_262);  mul_110 = _shape_param_262 = None
        view_183: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_39, _shape_param_263);  expand_39 = _shape_param_263 = None
        expand_40: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_102, _shape_param_264);  permute_102 = _shape_param_264 = None
        clone_38: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_184: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_38, _shape_param_265);  clone_38 = _shape_param_265 = None
        bmm_19: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_183, view_184)
        view_185: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_266);  bmm_19 = _shape_param_266 = None
        permute_104: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        clone_39: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_186: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_39, _shape_param_267);  clone_39 = _shape_param_267 = None
        convert_element_type_146: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg75_1, torch.bfloat16);  arg75_1 = None
        permute_105: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_146, [1, 0]);  convert_element_type_146 = None
        view_187: "bf16[8192, 512]" = torch.ops.aten.view.default(view_186, _shape_param_268);  view_186 = _shape_param_268 = None
        mm_53: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_187, permute_105)
        view_188: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_53, _shape_param_269);  mm_53 = _shape_param_269 = None
        inductor_lookup_seed_36: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 36)
        inductor_random_36: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_270, inductor_lookup_seed_36, 'rand');  _shape_param_270 = inductor_lookup_seed_36 = None
        convert_element_type_147: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_36, torch.bfloat16);  inductor_random_36 = None
        gt_37: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_147, 0.1);  convert_element_type_147 = None
        mul_111: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_37, view_188);  view_188 = None
        mul_112: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_111, 1.1111111111111112);  mul_111 = None
        add_51: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_49, mul_112);  mul_112 = None
        pow_19: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_51, 2)
        mean_18: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
        add_52: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_113: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_51, rsqrt_18)
        mul_114: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg76_1, mul_113);  arg76_1 = mul_113 = None
        convert_element_type_148: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg77_1, torch.bfloat16);  arg77_1 = None
        convert_element_type_149: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_114, torch.bfloat16);  mul_114 = None
        permute_106: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_148, [1, 0]);  convert_element_type_148 = None
        view_189: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_149, _shape_param_271);  convert_element_type_149 = _shape_param_271 = None
        mm_54: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_189, permute_106)
        view_190: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_54, _shape_param_272);  mm_54 = _shape_param_272 = None
        relu_7: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_190);  view_190 = None
        inductor_lookup_seed_37: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 37)
        inductor_random_37: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default(_shape_param_273, inductor_lookup_seed_37, 'rand');  _shape_param_273 = inductor_lookup_seed_37 = None
        convert_element_type_150: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(inductor_random_37, torch.bfloat16);  inductor_random_37 = None
        gt_38: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(convert_element_type_150, 0.1);  convert_element_type_150 = None
        mul_115: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_38, relu_7)
        mul_116: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_115, 1.1111111111111112);  mul_115 = None
        convert_element_type_151: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg78_1, torch.bfloat16);  arg78_1 = None
        convert_element_type_152: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(mul_116, torch.bfloat16);  mul_116 = None
        permute_107: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_151, [1, 0]);  convert_element_type_151 = None
        view_191: "bf16[8192, 2048]" = torch.ops.aten.view.default(convert_element_type_152, _shape_param_274);  convert_element_type_152 = _shape_param_274 = None
        mm_55: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_191, permute_107)
        view_192: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_55, _shape_param_275);  mm_55 = _shape_param_275 = None
        inductor_lookup_seed_38: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 38)
        inductor_random_38: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_276, inductor_lookup_seed_38, 'rand');  _shape_param_276 = inductor_lookup_seed_38 = None
        convert_element_type_153: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_38, torch.bfloat16);  inductor_random_38 = None
        gt_39: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_153, 0.1);  convert_element_type_153 = None
        mul_117: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_39, view_192);  view_192 = None
        mul_118: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_117, 1.1111111111111112);  mul_117 = None
        add_53: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_51, mul_118);  mul_118 = None
        pow_20: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_53, 2)
        mean_19: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
        add_54: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_119: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_53, rsqrt_19)
        mul_120: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg79_1, mul_119);  arg79_1 = mul_119 = None
        convert_element_type_154: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg80_1, torch.bfloat16);  arg80_1 = None
        convert_element_type_155: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None
        permute_108: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_154, [1, 0]);  convert_element_type_154 = None
        view_193: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_155, _shape_param_277);  convert_element_type_155 = _shape_param_277 = None
        mm_56: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_193, permute_108)
        view_194: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_56, _shape_param_278);  mm_56 = _shape_param_278 = None
        view_195: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_194, _shape_param_279);  view_194 = _shape_param_279 = None
        permute_109: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None
        convert_element_type_156: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg81_1, torch.bfloat16);  arg81_1 = None
        permute_110: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_156, [1, 0]);  convert_element_type_156 = None
        mm_57: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_193, permute_110)
        view_196: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_57, _shape_param_280);  mm_57 = _shape_param_280 = None
        view_197: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_196, _shape_param_281);  view_196 = _shape_param_281 = None
        permute_111: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_197, [0, 2, 1, 3]);  view_197 = None
        convert_element_type_157: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg82_1, torch.bfloat16);  arg82_1 = None
        permute_112: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_157, [1, 0]);  convert_element_type_157 = None
        mm_58: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_193, permute_112)
        view_198: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_58, _shape_param_282);  mm_58 = _shape_param_282 = None
        view_199: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_198, _shape_param_283);  view_198 = _shape_param_283 = None
        permute_113: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_199, [0, 2, 1, 3]);  view_199 = None
        permute_114: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_111, [0, 1, 3, 2]);  permute_111 = None
        expand_41: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_109, _shape_param_284);  permute_109 = _shape_param_284 = None
        clone_40: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_200: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_285);  clone_40 = _shape_param_285 = None
        expand_42: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_114, _shape_param_286);  permute_114 = _shape_param_286 = None
        clone_41: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_201: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_41, _shape_param_287);  clone_41 = _shape_param_287 = None
        bmm_20: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_200, view_201)
        view_202: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_20, _shape_param_288);  bmm_20 = _shape_param_288 = None
        add_55: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_202, add_40);  view_202 = None
        convert_element_type_158: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None
        convert_element_type_159: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_158, torch.float32)
        amax_10: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_159, [-1], True)
        sub_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_159, amax_10);  convert_element_type_159 = None
        exp_10: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_11: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_14: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = None
        convert_element_type_160: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None
        inductor_lookup_seed_39: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 39)
        inductor_random_39: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_289, inductor_lookup_seed_39, 'rand');  _shape_param_289 = inductor_lookup_seed_39 = None
        convert_element_type_161: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_39, torch.bfloat16);  inductor_random_39 = None
        gt_40: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_161, 0.1);  convert_element_type_161 = None
        mul_121: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_40, convert_element_type_160);  convert_element_type_160 = None
        mul_122: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_121, 1.1111111111111112);  mul_121 = None
        expand_43: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_122, _shape_param_290);  mul_122 = _shape_param_290 = None
        view_203: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_43, _shape_param_291);  expand_43 = _shape_param_291 = None
        expand_44: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_113, _shape_param_292);  permute_113 = _shape_param_292 = None
        clone_42: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_204: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_42, _shape_param_293);  clone_42 = _shape_param_293 = None
        bmm_21: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_203, view_204)
        view_205: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_21, _shape_param_294);  bmm_21 = _shape_param_294 = None
        permute_115: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
        clone_43: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_206: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_43, _shape_param_295);  clone_43 = _shape_param_295 = None
        convert_element_type_162: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg83_1, torch.bfloat16);  arg83_1 = None
        permute_116: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_162, [1, 0]);  convert_element_type_162 = None
        view_207: "bf16[8192, 512]" = torch.ops.aten.view.default(view_206, _shape_param_296);  view_206 = _shape_param_296 = None
        mm_59: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_207, permute_116)
        view_208: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_59, _shape_param_297);  mm_59 = _shape_param_297 = None
        inductor_lookup_seed_40: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 40)
        inductor_random_40: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_298, inductor_lookup_seed_40, 'rand');  _shape_param_298 = inductor_lookup_seed_40 = None
        convert_element_type_163: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_40, torch.bfloat16);  inductor_random_40 = None
        gt_41: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_163, 0.1);  convert_element_type_163 = None
        mul_123: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_41, view_208);  view_208 = None
        mul_124: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_123, 1.1111111111111112);  mul_123 = None
        add_56: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_53, mul_124);  mul_124 = None
        pow_21: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_56, 2)
        mean_20: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None
        add_57: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        mul_125: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_56, rsqrt_20)
        mul_126: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg84_1, mul_125);  arg84_1 = mul_125 = None
        convert_element_type_164: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg85_1, torch.bfloat16);  arg85_1 = None
        convert_element_type_165: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16);  mul_126 = None
        permute_117: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_164, [1, 0]);  convert_element_type_164 = None
        view_209: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_165, _shape_param_299);  convert_element_type_165 = _shape_param_299 = None
        mm_60: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_209, permute_117)
        view_210: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_60, _shape_param_300);  mm_60 = _shape_param_300 = None
        view_211: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_210, _shape_param_301);  view_210 = _shape_param_301 = None
        permute_118: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None
        convert_element_type_166: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg86_1, torch.bfloat16);  arg86_1 = None
        permute_119: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_166, [1, 0]);  convert_element_type_166 = None
        mm_61: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_139, permute_119)
        view_212: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_61, _shape_param_302);  mm_61 = _shape_param_302 = None
        view_213: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_212, _shape_param_303);  view_212 = _shape_param_303 = None
        permute_120: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
        convert_element_type_167: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg87_1, torch.bfloat16);  arg87_1 = None
        permute_121: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_167, [1, 0]);  convert_element_type_167 = None
        mm_62: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_139, permute_121)
        view_214: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_62, _shape_param_304);  mm_62 = _shape_param_304 = None
        view_215: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_214, _shape_param_305);  view_214 = _shape_param_305 = None
        permute_122: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
        permute_123: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_120, [0, 1, 3, 2]);  permute_120 = None
        expand_45: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_118, _shape_param_306);  permute_118 = _shape_param_306 = None
        clone_44: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_216: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_44, _shape_param_307);  clone_44 = _shape_param_307 = None
        expand_46: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_123, _shape_param_308);  permute_123 = _shape_param_308 = None
        clone_45: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_217: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_45, _shape_param_309);  clone_45 = _shape_param_309 = None
        bmm_22: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_216, view_217)
        view_218: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_22, _shape_param_310);  _shape_param_310 = None
        convert_element_type_168: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_218, torch.bfloat16);  view_218 = None
        convert_element_type_169: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_168, torch.float32);  convert_element_type_168 = None
        amax_11: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_169, [-1], True)
        sub_12: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_169, amax_11);  convert_element_type_169 = None
        exp_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_12: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_15: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = None
        convert_element_type_170: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None
        inductor_lookup_seed_41: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 41)
        inductor_random_41: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_311, inductor_lookup_seed_41, 'rand');  _shape_param_311 = inductor_lookup_seed_41 = None
        convert_element_type_171: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_41, torch.bfloat16);  inductor_random_41 = None
        gt_42: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_171, 0.1);  convert_element_type_171 = None
        mul_127: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_42, convert_element_type_170);  convert_element_type_170 = None
        mul_128: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_127, 1.1111111111111112);  mul_127 = None
        expand_47: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_128, _shape_param_312);  mul_128 = _shape_param_312 = None
        view_219: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_47, _shape_param_313);  expand_47 = _shape_param_313 = None
        expand_48: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_122, _shape_param_314);  permute_122 = _shape_param_314 = None
        clone_46: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_220: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_46, _shape_param_315);  clone_46 = _shape_param_315 = None
        bmm_23: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_219, view_220)
        view_221: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_316);  bmm_23 = _shape_param_316 = None
        permute_124: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_221, [0, 2, 1, 3]);  view_221 = None
        clone_47: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
        view_222: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_47, _shape_param_317);  clone_47 = _shape_param_317 = None
        convert_element_type_172: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg88_1, torch.bfloat16);  arg88_1 = None
        permute_125: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_172, [1, 0]);  convert_element_type_172 = None
        view_223: "bf16[8192, 512]" = torch.ops.aten.view.default(view_222, _shape_param_318);  view_222 = _shape_param_318 = None
        mm_63: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_223, permute_125)
        view_224: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_63, _shape_param_319);  mm_63 = _shape_param_319 = None
        inductor_lookup_seed_42: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 42)
        inductor_random_42: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_320, inductor_lookup_seed_42, 'rand');  _shape_param_320 = inductor_lookup_seed_42 = None
        convert_element_type_173: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_42, torch.bfloat16);  inductor_random_42 = None
        gt_43: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_173, 0.1);  convert_element_type_173 = None
        mul_129: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_43, view_224);  view_224 = None
        mul_130: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_129, 1.1111111111111112);  mul_129 = None
        add_58: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_56, mul_130);  mul_130 = None
        pow_22: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_58, 2)
        mean_21: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
        add_59: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        mul_131: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_58, rsqrt_21)
        mul_132: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg89_1, mul_131);  arg89_1 = mul_131 = None
        convert_element_type_174: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg90_1, torch.bfloat16);  arg90_1 = None
        convert_element_type_175: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_132, torch.bfloat16);  mul_132 = None
        permute_126: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_174, [1, 0]);  convert_element_type_174 = None
        view_225: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_175, _shape_param_321);  convert_element_type_175 = _shape_param_321 = None
        mm_64: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_225, permute_126)
        view_226: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_64, _shape_param_322);  mm_64 = _shape_param_322 = None
        relu_8: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_226);  view_226 = None
        inductor_lookup_seed_43: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 43)
        inductor_random_43: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default(_shape_param_323, inductor_lookup_seed_43, 'rand');  _shape_param_323 = inductor_lookup_seed_43 = None
        convert_element_type_176: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(inductor_random_43, torch.bfloat16);  inductor_random_43 = None
        gt_44: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(convert_element_type_176, 0.1);  convert_element_type_176 = None
        mul_133: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_44, relu_8)
        mul_134: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_133, 1.1111111111111112);  mul_133 = None
        convert_element_type_177: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg91_1, torch.bfloat16);  arg91_1 = None
        convert_element_type_178: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(mul_134, torch.bfloat16);  mul_134 = None
        permute_127: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_177, [1, 0]);  convert_element_type_177 = None
        view_227: "bf16[8192, 2048]" = torch.ops.aten.view.default(convert_element_type_178, _shape_param_324);  convert_element_type_178 = _shape_param_324 = None
        mm_65: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_227, permute_127)
        view_228: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_65, _shape_param_325);  mm_65 = _shape_param_325 = None
        inductor_lookup_seed_44: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 44)
        inductor_random_44: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_326, inductor_lookup_seed_44, 'rand');  _shape_param_326 = inductor_lookup_seed_44 = None
        convert_element_type_179: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_44, torch.bfloat16);  inductor_random_44 = None
        gt_45: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_179, 0.1);  convert_element_type_179 = None
        mul_135: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_45, view_228);  view_228 = None
        mul_136: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_135, 1.1111111111111112);  mul_135 = None
        add_60: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_58, mul_136);  mul_136 = None
        pow_23: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_60, 2)
        mean_22: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
        add_61: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_137: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_60, rsqrt_22)
        mul_138: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg92_1, mul_137);  arg92_1 = mul_137 = None
        convert_element_type_180: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg93_1, torch.bfloat16);  arg93_1 = None
        convert_element_type_181: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_138, torch.bfloat16);  mul_138 = None
        permute_128: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_180, [1, 0]);  convert_element_type_180 = None
        view_229: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_181, _shape_param_327);  convert_element_type_181 = _shape_param_327 = None
        mm_66: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_229, permute_128)
        view_230: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_66, _shape_param_328);  mm_66 = _shape_param_328 = None
        view_231: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_230, _shape_param_329);  view_230 = _shape_param_329 = None
        permute_129: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_231, [0, 2, 1, 3]);  view_231 = None
        convert_element_type_182: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg94_1, torch.bfloat16);  arg94_1 = None
        permute_130: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_182, [1, 0]);  convert_element_type_182 = None
        mm_67: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_229, permute_130)
        view_232: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_67, _shape_param_330);  mm_67 = _shape_param_330 = None
        view_233: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_232, _shape_param_331);  view_232 = _shape_param_331 = None
        permute_131: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None
        convert_element_type_183: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg95_1, torch.bfloat16);  arg95_1 = None
        permute_132: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_183, [1, 0]);  convert_element_type_183 = None
        mm_68: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_229, permute_132)
        view_234: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_68, _shape_param_332);  mm_68 = _shape_param_332 = None
        view_235: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_234, _shape_param_333);  view_234 = _shape_param_333 = None
        permute_133: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_235, [0, 2, 1, 3]);  view_235 = None
        permute_134: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2]);  permute_131 = None
        expand_49: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_129, _shape_param_334);  permute_129 = _shape_param_334 = None
        clone_48: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_236: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_48, _shape_param_335);  clone_48 = _shape_param_335 = None
        expand_50: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_134, _shape_param_336);  permute_134 = _shape_param_336 = None
        clone_49: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_237: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_49, _shape_param_337);  clone_49 = _shape_param_337 = None
        bmm_24: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_236, view_237)
        view_238: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_24, _shape_param_338);  bmm_24 = _shape_param_338 = None
        add_62: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_238, add_40);  view_238 = None
        convert_element_type_184: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_62, torch.bfloat16);  add_62 = None
        convert_element_type_185: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_184, torch.float32)
        amax_12: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_185, [-1], True)
        sub_13: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_185, amax_12);  convert_element_type_185 = None
        exp_12: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_13: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_16: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = None
        convert_element_type_186: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None
        inductor_lookup_seed_45: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 45)
        inductor_random_45: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_339, inductor_lookup_seed_45, 'rand');  _shape_param_339 = inductor_lookup_seed_45 = None
        convert_element_type_187: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_45, torch.bfloat16);  inductor_random_45 = None
        gt_46: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_187, 0.1);  convert_element_type_187 = None
        mul_139: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_46, convert_element_type_186);  convert_element_type_186 = None
        mul_140: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_139, 1.1111111111111112);  mul_139 = None
        expand_51: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_140, _shape_param_340);  mul_140 = _shape_param_340 = None
        view_239: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_51, _shape_param_341);  expand_51 = _shape_param_341 = None
        expand_52: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_133, _shape_param_342);  permute_133 = _shape_param_342 = None
        clone_50: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_240: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_50, _shape_param_343);  clone_50 = _shape_param_343 = None
        bmm_25: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_239, view_240)
        view_241: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_25, _shape_param_344);  bmm_25 = _shape_param_344 = None
        permute_135: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_241, [0, 2, 1, 3]);  view_241 = None
        clone_51: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_242: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_51, _shape_param_345);  clone_51 = _shape_param_345 = None
        convert_element_type_188: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg96_1, torch.bfloat16);  arg96_1 = None
        permute_136: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_188, [1, 0]);  convert_element_type_188 = None
        view_243: "bf16[8192, 512]" = torch.ops.aten.view.default(view_242, _shape_param_346);  view_242 = _shape_param_346 = None
        mm_69: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_243, permute_136)
        view_244: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_69, _shape_param_347);  mm_69 = _shape_param_347 = None
        inductor_lookup_seed_46: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 46)
        inductor_random_46: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_348, inductor_lookup_seed_46, 'rand');  _shape_param_348 = inductor_lookup_seed_46 = None
        convert_element_type_189: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_46, torch.bfloat16);  inductor_random_46 = None
        gt_47: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_189, 0.1);  convert_element_type_189 = None
        mul_141: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_47, view_244);  view_244 = None
        mul_142: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_141, 1.1111111111111112);  mul_141 = None
        add_63: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_60, mul_142);  mul_142 = None
        pow_24: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_63, 2)
        mean_23: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None
        add_64: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_143: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_63, rsqrt_23)
        mul_144: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg97_1, mul_143);  arg97_1 = mul_143 = None
        convert_element_type_190: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg98_1, torch.bfloat16);  arg98_1 = None
        convert_element_type_191: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_144, torch.bfloat16);  mul_144 = None
        permute_137: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_190, [1, 0]);  convert_element_type_190 = None
        view_245: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_191, _shape_param_349);  convert_element_type_191 = _shape_param_349 = None
        mm_70: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_245, permute_137)
        view_246: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_70, _shape_param_350);  mm_70 = _shape_param_350 = None
        view_247: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_246, _shape_param_351);  view_246 = _shape_param_351 = None
        permute_138: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
        convert_element_type_192: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg99_1, torch.bfloat16);  arg99_1 = None
        permute_139: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_192, [1, 0]);  convert_element_type_192 = None
        mm_71: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_139, permute_139)
        view_248: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_71, _shape_param_352);  mm_71 = _shape_param_352 = None
        view_249: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_248, _shape_param_353);  view_248 = _shape_param_353 = None
        permute_140: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
        convert_element_type_193: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg100_1, torch.bfloat16);  arg100_1 = None
        permute_141: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_193, [1, 0]);  convert_element_type_193 = None
        mm_72: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_139, permute_141)
        view_250: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_72, _shape_param_354);  mm_72 = _shape_param_354 = None
        view_251: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_250, _shape_param_355);  view_250 = _shape_param_355 = None
        permute_142: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
        permute_143: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_140, [0, 1, 3, 2]);  permute_140 = None
        expand_53: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_138, _shape_param_356);  permute_138 = _shape_param_356 = None
        clone_52: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_252: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_52, _shape_param_357);  clone_52 = _shape_param_357 = None
        expand_54: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_143, _shape_param_358);  permute_143 = _shape_param_358 = None
        clone_53: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_253: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_53, _shape_param_359);  clone_53 = _shape_param_359 = None
        bmm_26: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_252, view_253)
        view_254: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_26, _shape_param_360);  _shape_param_360 = None
        convert_element_type_194: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_254, torch.bfloat16);  view_254 = None
        convert_element_type_195: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_194, torch.float32);  convert_element_type_194 = None
        amax_13: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_195, [-1], True)
        sub_14: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_195, amax_13);  convert_element_type_195 = None
        exp_13: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_14: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_17: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = None
        convert_element_type_196: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None
        inductor_lookup_seed_47: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 47)
        inductor_random_47: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_361, inductor_lookup_seed_47, 'rand');  _shape_param_361 = inductor_lookup_seed_47 = None
        convert_element_type_197: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_47, torch.bfloat16);  inductor_random_47 = None
        gt_48: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_197, 0.1);  convert_element_type_197 = None
        mul_145: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_48, convert_element_type_196);  convert_element_type_196 = None
        mul_146: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_145, 1.1111111111111112);  mul_145 = None
        expand_55: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_146, _shape_param_362);  mul_146 = _shape_param_362 = None
        view_255: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_55, _shape_param_363);  expand_55 = _shape_param_363 = None
        expand_56: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_142, _shape_param_364);  permute_142 = _shape_param_364 = None
        clone_54: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_256: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_54, _shape_param_365);  clone_54 = _shape_param_365 = None
        bmm_27: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_255, view_256)
        view_257: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_27, _shape_param_366);  bmm_27 = _shape_param_366 = None
        permute_144: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None
        clone_55: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        view_258: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_55, _shape_param_367);  clone_55 = _shape_param_367 = None
        convert_element_type_198: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg101_1, torch.bfloat16);  arg101_1 = None
        permute_145: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_198, [1, 0]);  convert_element_type_198 = None
        view_259: "bf16[8192, 512]" = torch.ops.aten.view.default(view_258, _shape_param_368);  view_258 = _shape_param_368 = None
        mm_73: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_259, permute_145)
        view_260: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_73, _shape_param_369);  mm_73 = _shape_param_369 = None
        inductor_lookup_seed_48: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 48)
        inductor_random_48: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_370, inductor_lookup_seed_48, 'rand');  _shape_param_370 = inductor_lookup_seed_48 = None
        convert_element_type_199: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_48, torch.bfloat16);  inductor_random_48 = None
        gt_49: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_199, 0.1);  convert_element_type_199 = None
        mul_147: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_49, view_260);  view_260 = None
        mul_148: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_147, 1.1111111111111112);  mul_147 = None
        add_65: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_63, mul_148);  mul_148 = None
        pow_25: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_65, 2)
        mean_24: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
        add_66: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_149: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_65, rsqrt_24)
        mul_150: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg102_1, mul_149);  arg102_1 = mul_149 = None
        convert_element_type_200: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg103_1, torch.bfloat16);  arg103_1 = None
        convert_element_type_201: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_150, torch.bfloat16);  mul_150 = None
        permute_146: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_200, [1, 0]);  convert_element_type_200 = None
        view_261: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_201, _shape_param_371);  convert_element_type_201 = _shape_param_371 = None
        mm_74: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_261, permute_146)
        view_262: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_74, _shape_param_372);  mm_74 = _shape_param_372 = None
        relu_9: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_262);  view_262 = None
        inductor_lookup_seed_49: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 49)
        inductor_random_49: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default(_shape_param_373, inductor_lookup_seed_49, 'rand');  _shape_param_373 = inductor_lookup_seed_49 = None
        convert_element_type_202: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(inductor_random_49, torch.bfloat16);  inductor_random_49 = None
        gt_50: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(convert_element_type_202, 0.1);  convert_element_type_202 = None
        mul_151: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_50, relu_9)
        mul_152: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_151, 1.1111111111111112);  mul_151 = None
        convert_element_type_203: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg104_1, torch.bfloat16);  arg104_1 = None
        convert_element_type_204: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None
        permute_147: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_203, [1, 0]);  convert_element_type_203 = None
        view_263: "bf16[8192, 2048]" = torch.ops.aten.view.default(convert_element_type_204, _shape_param_374);  convert_element_type_204 = _shape_param_374 = None
        mm_75: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_263, permute_147)
        view_264: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_75, _shape_param_375);  mm_75 = _shape_param_375 = None
        inductor_lookup_seed_50: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 50)
        inductor_random_50: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_376, inductor_lookup_seed_50, 'rand');  _shape_param_376 = inductor_lookup_seed_50 = None
        convert_element_type_205: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_50, torch.bfloat16);  inductor_random_50 = None
        gt_51: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_205, 0.1);  convert_element_type_205 = None
        mul_153: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_51, view_264);  view_264 = None
        mul_154: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_153, 1.1111111111111112);  mul_153 = None
        add_67: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_65, mul_154);  mul_154 = None
        pow_26: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_67, 2)
        mean_25: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None
        add_68: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_155: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_67, rsqrt_25)
        mul_156: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg105_1, mul_155);  arg105_1 = mul_155 = None
        convert_element_type_206: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg106_1, torch.bfloat16);  arg106_1 = None
        convert_element_type_207: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_156, torch.bfloat16);  mul_156 = None
        permute_148: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_206, [1, 0]);  convert_element_type_206 = None
        view_265: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_207, _shape_param_377);  convert_element_type_207 = _shape_param_377 = None
        mm_76: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_265, permute_148)
        view_266: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_76, _shape_param_378);  mm_76 = _shape_param_378 = None
        view_267: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_266, _shape_param_379);  view_266 = _shape_param_379 = None
        permute_149: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
        convert_element_type_208: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg107_1, torch.bfloat16);  arg107_1 = None
        permute_150: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_208, [1, 0]);  convert_element_type_208 = None
        mm_77: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_265, permute_150)
        view_268: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_77, _shape_param_380);  mm_77 = _shape_param_380 = None
        view_269: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_268, _shape_param_381);  view_268 = _shape_param_381 = None
        permute_151: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
        convert_element_type_209: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg108_1, torch.bfloat16);  arg108_1 = None
        permute_152: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_209, [1, 0]);  convert_element_type_209 = None
        mm_78: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_265, permute_152)
        view_270: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_78, _shape_param_382);  mm_78 = _shape_param_382 = None
        view_271: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_270, _shape_param_383);  view_270 = _shape_param_383 = None
        permute_153: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_271, [0, 2, 1, 3]);  view_271 = None
        permute_154: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_151, [0, 1, 3, 2]);  permute_151 = None
        expand_57: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_149, _shape_param_384);  permute_149 = _shape_param_384 = None
        clone_56: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_272: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_56, _shape_param_385);  clone_56 = _shape_param_385 = None
        expand_58: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_154, _shape_param_386);  permute_154 = _shape_param_386 = None
        clone_57: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_273: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_57, _shape_param_387);  clone_57 = _shape_param_387 = None
        bmm_28: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_272, view_273)
        view_274: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_28, _shape_param_388);  bmm_28 = _shape_param_388 = None
        add_69: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_274, add_40);  view_274 = None
        convert_element_type_210: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16);  add_69 = None
        convert_element_type_211: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_210, torch.float32)
        amax_14: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_211, [-1], True)
        sub_15: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_211, amax_14);  convert_element_type_211 = None
        exp_14: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_15: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_18: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = None
        convert_element_type_212: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None
        inductor_lookup_seed_51: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 51)
        inductor_random_51: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_389, inductor_lookup_seed_51, 'rand');  _shape_param_389 = inductor_lookup_seed_51 = None
        convert_element_type_213: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_51, torch.bfloat16);  inductor_random_51 = None
        gt_52: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_213, 0.1);  convert_element_type_213 = None
        mul_157: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_52, convert_element_type_212);  convert_element_type_212 = None
        mul_158: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_157, 1.1111111111111112);  mul_157 = None
        expand_59: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_158, _shape_param_390);  mul_158 = _shape_param_390 = None
        view_275: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_59, _shape_param_391);  expand_59 = _shape_param_391 = None
        expand_60: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_153, _shape_param_392);  permute_153 = _shape_param_392 = None
        clone_58: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_276: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_58, _shape_param_393);  clone_58 = _shape_param_393 = None
        bmm_29: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_275, view_276)
        view_277: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_29, _shape_param_394);  bmm_29 = _shape_param_394 = None
        permute_155: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_277, [0, 2, 1, 3]);  view_277 = None
        clone_59: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_278: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_59, _shape_param_395);  clone_59 = _shape_param_395 = None
        convert_element_type_214: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg109_1, torch.bfloat16);  arg109_1 = None
        permute_156: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_214, [1, 0]);  convert_element_type_214 = None
        view_279: "bf16[8192, 512]" = torch.ops.aten.view.default(view_278, _shape_param_396);  view_278 = _shape_param_396 = None
        mm_79: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_279, permute_156)
        view_280: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_79, _shape_param_397);  mm_79 = _shape_param_397 = None
        inductor_lookup_seed_52: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 52)
        inductor_random_52: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_398, inductor_lookup_seed_52, 'rand');  _shape_param_398 = inductor_lookup_seed_52 = None
        convert_element_type_215: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_52, torch.bfloat16);  inductor_random_52 = None
        gt_53: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_215, 0.1);  convert_element_type_215 = None
        mul_159: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_53, view_280);  view_280 = None
        mul_160: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_159, 1.1111111111111112);  mul_159 = None
        add_70: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_67, mul_160);  mul_160 = None
        pow_27: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_70, 2)
        mean_26: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
        add_71: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_161: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_70, rsqrt_26)
        mul_162: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg110_1, mul_161);  arg110_1 = mul_161 = None
        convert_element_type_216: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg111_1, torch.bfloat16);  arg111_1 = None
        convert_element_type_217: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_162, torch.bfloat16);  mul_162 = None
        permute_157: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_216, [1, 0]);  convert_element_type_216 = None
        view_281: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_217, _shape_param_399);  convert_element_type_217 = _shape_param_399 = None
        mm_80: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_281, permute_157)
        view_282: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_80, _shape_param_400);  mm_80 = _shape_param_400 = None
        view_283: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_282, _shape_param_401);  view_282 = _shape_param_401 = None
        permute_158: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
        convert_element_type_218: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg112_1, torch.bfloat16);  arg112_1 = None
        permute_159: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_218, [1, 0]);  convert_element_type_218 = None
        mm_81: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_139, permute_159)
        view_284: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_81, _shape_param_402);  mm_81 = _shape_param_402 = None
        view_285: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_284, _shape_param_403);  view_284 = _shape_param_403 = None
        permute_160: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
        convert_element_type_219: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg113_1, torch.bfloat16);  arg113_1 = None
        permute_161: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_219, [1, 0]);  convert_element_type_219 = None
        mm_82: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_139, permute_161)
        view_286: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_82, _shape_param_404);  mm_82 = _shape_param_404 = None
        view_287: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_286, _shape_param_405);  view_286 = _shape_param_405 = None
        permute_162: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_287, [0, 2, 1, 3]);  view_287 = None
        permute_163: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_160, [0, 1, 3, 2]);  permute_160 = None
        expand_61: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_158, _shape_param_406);  permute_158 = _shape_param_406 = None
        clone_60: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_288: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_60, _shape_param_407);  clone_60 = _shape_param_407 = None
        expand_62: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_163, _shape_param_408);  permute_163 = _shape_param_408 = None
        clone_61: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_289: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_61, _shape_param_409);  clone_61 = _shape_param_409 = None
        bmm_30: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_288, view_289)
        view_290: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_30, _shape_param_410);  _shape_param_410 = None
        convert_element_type_220: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_290, torch.bfloat16);  view_290 = None
        convert_element_type_221: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_220, torch.float32);  convert_element_type_220 = None
        amax_15: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_221, [-1], True)
        sub_16: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_221, amax_15);  convert_element_type_221 = None
        exp_15: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_16: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_19: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = None
        convert_element_type_222: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None
        inductor_lookup_seed_53: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 53)
        inductor_random_53: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_411, inductor_lookup_seed_53, 'rand');  _shape_param_411 = inductor_lookup_seed_53 = None
        convert_element_type_223: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_53, torch.bfloat16);  inductor_random_53 = None
        gt_54: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_223, 0.1);  convert_element_type_223 = None
        mul_163: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_54, convert_element_type_222);  convert_element_type_222 = None
        mul_164: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_163, 1.1111111111111112);  mul_163 = None
        expand_63: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_164, _shape_param_412);  mul_164 = _shape_param_412 = None
        view_291: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_63, _shape_param_413);  expand_63 = _shape_param_413 = None
        expand_64: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_162, _shape_param_414);  permute_162 = _shape_param_414 = None
        clone_62: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_292: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_62, _shape_param_415);  clone_62 = _shape_param_415 = None
        bmm_31: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_291, view_292)
        view_293: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_31, _shape_param_416);  bmm_31 = _shape_param_416 = None
        permute_164: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None
        clone_63: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
        view_294: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_63, _shape_param_417);  clone_63 = _shape_param_417 = None
        convert_element_type_224: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg114_1, torch.bfloat16);  arg114_1 = None
        permute_165: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_224, [1, 0]);  convert_element_type_224 = None
        view_295: "bf16[8192, 512]" = torch.ops.aten.view.default(view_294, _shape_param_418);  view_294 = _shape_param_418 = None
        mm_83: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_295, permute_165)
        view_296: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_83, _shape_param_419);  mm_83 = _shape_param_419 = None
        inductor_lookup_seed_54: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 54)
        inductor_random_54: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_420, inductor_lookup_seed_54, 'rand');  _shape_param_420 = inductor_lookup_seed_54 = None
        convert_element_type_225: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_54, torch.bfloat16);  inductor_random_54 = None
        gt_55: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_225, 0.1);  convert_element_type_225 = None
        mul_165: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_55, view_296);  view_296 = None
        mul_166: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_165, 1.1111111111111112);  mul_165 = None
        add_72: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_70, mul_166);  mul_166 = None
        pow_28: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_72, 2)
        mean_27: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None
        add_73: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_167: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_72, rsqrt_27)
        mul_168: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg115_1, mul_167);  arg115_1 = mul_167 = None
        convert_element_type_226: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg116_1, torch.bfloat16);  arg116_1 = None
        convert_element_type_227: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_168, torch.bfloat16);  mul_168 = None
        permute_166: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_226, [1, 0]);  convert_element_type_226 = None
        view_297: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_227, _shape_param_421);  convert_element_type_227 = _shape_param_421 = None
        mm_84: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_297, permute_166)
        view_298: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_84, _shape_param_422);  mm_84 = _shape_param_422 = None
        relu_10: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_298);  view_298 = None
        inductor_lookup_seed_55: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 55)
        inductor_random_55: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default(_shape_param_423, inductor_lookup_seed_55, 'rand');  _shape_param_423 = inductor_lookup_seed_55 = None
        convert_element_type_228: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(inductor_random_55, torch.bfloat16);  inductor_random_55 = None
        gt_56: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(convert_element_type_228, 0.1);  convert_element_type_228 = None
        mul_169: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_56, relu_10)
        mul_170: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_169, 1.1111111111111112);  mul_169 = None
        convert_element_type_229: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg117_1, torch.bfloat16);  arg117_1 = None
        convert_element_type_230: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(mul_170, torch.bfloat16);  mul_170 = None
        permute_167: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_229, [1, 0]);  convert_element_type_229 = None
        view_299: "bf16[8192, 2048]" = torch.ops.aten.view.default(convert_element_type_230, _shape_param_424);  convert_element_type_230 = _shape_param_424 = None
        mm_85: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_299, permute_167)
        view_300: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_85, _shape_param_425);  mm_85 = _shape_param_425 = None
        inductor_lookup_seed_56: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 56)
        inductor_random_56: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_426, inductor_lookup_seed_56, 'rand');  _shape_param_426 = inductor_lookup_seed_56 = None
        convert_element_type_231: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_56, torch.bfloat16);  inductor_random_56 = None
        gt_57: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_231, 0.1);  convert_element_type_231 = None
        mul_171: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_57, view_300);  view_300 = None
        mul_172: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_171, 1.1111111111111112);  mul_171 = None
        add_74: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_72, mul_172);  mul_172 = None
        pow_29: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_74, 2)
        mean_28: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None
        add_75: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_173: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_74, rsqrt_28)
        mul_174: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg118_1, mul_173);  arg118_1 = mul_173 = None
        convert_element_type_232: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg119_1, torch.bfloat16);  arg119_1 = None
        convert_element_type_233: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_174, torch.bfloat16);  mul_174 = None
        permute_168: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_232, [1, 0]);  convert_element_type_232 = None
        view_301: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_233, _shape_param_427);  convert_element_type_233 = _shape_param_427 = None
        mm_86: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_301, permute_168)
        view_302: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_86, _shape_param_428);  mm_86 = _shape_param_428 = None
        view_303: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_302, _shape_param_429);  view_302 = _shape_param_429 = None
        permute_169: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
        convert_element_type_234: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg120_1, torch.bfloat16);  arg120_1 = None
        permute_170: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_234, [1, 0]);  convert_element_type_234 = None
        mm_87: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_301, permute_170)
        view_304: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_87, _shape_param_430);  mm_87 = _shape_param_430 = None
        view_305: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_304, _shape_param_431);  view_304 = _shape_param_431 = None
        permute_171: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_305, [0, 2, 1, 3]);  view_305 = None
        convert_element_type_235: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg121_1, torch.bfloat16);  arg121_1 = None
        permute_172: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_235, [1, 0]);  convert_element_type_235 = None
        mm_88: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_301, permute_172)
        view_306: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_88, _shape_param_432);  mm_88 = _shape_param_432 = None
        view_307: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_306, _shape_param_433);  view_306 = _shape_param_433 = None
        permute_173: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
        permute_174: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_171, [0, 1, 3, 2]);  permute_171 = None
        expand_65: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_169, _shape_param_434);  permute_169 = _shape_param_434 = None
        clone_64: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_308: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_64, _shape_param_435);  clone_64 = _shape_param_435 = None
        expand_66: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_174, _shape_param_436);  permute_174 = _shape_param_436 = None
        clone_65: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_309: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_65, _shape_param_437);  clone_65 = _shape_param_437 = None
        bmm_32: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_308, view_309)
        view_310: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_32, _shape_param_438);  bmm_32 = _shape_param_438 = None
        add_76: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_310, add_40);  view_310 = add_40 = None
        convert_element_type_236: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_76, torch.bfloat16);  add_76 = None
        convert_element_type_237: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_236, torch.float32)
        amax_16: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_237, [-1], True)
        sub_17: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_237, amax_16);  convert_element_type_237 = None
        exp_16: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_17: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_20: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = None
        convert_element_type_238: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None
        inductor_lookup_seed_57: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 57)
        inductor_random_57: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_439, inductor_lookup_seed_57, 'rand');  _shape_param_439 = inductor_lookup_seed_57 = None
        convert_element_type_239: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_57, torch.bfloat16);  inductor_random_57 = None
        gt_58: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_239, 0.1);  convert_element_type_239 = None
        mul_175: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_58, convert_element_type_238);  convert_element_type_238 = None
        mul_176: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_175, 1.1111111111111112);  mul_175 = None
        expand_67: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_176, _shape_param_440);  mul_176 = _shape_param_440 = None
        view_311: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_67, _shape_param_441);  expand_67 = _shape_param_441 = None
        expand_68: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_173, _shape_param_442);  permute_173 = _shape_param_442 = None
        clone_66: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_312: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_66, _shape_param_443);  clone_66 = _shape_param_443 = None
        bmm_33: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_311, view_312)
        view_313: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_33, _shape_param_444);  bmm_33 = _shape_param_444 = None
        permute_175: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None
        clone_67: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
        view_314: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_67, _shape_param_445);  clone_67 = _shape_param_445 = None
        convert_element_type_240: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg122_1, torch.bfloat16);  arg122_1 = None
        permute_176: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_240, [1, 0]);  convert_element_type_240 = None
        view_315: "bf16[8192, 512]" = torch.ops.aten.view.default(view_314, _shape_param_446);  view_314 = _shape_param_446 = None
        mm_89: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_315, permute_176)
        view_316: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_89, _shape_param_447);  mm_89 = _shape_param_447 = None
        inductor_lookup_seed_58: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 58)
        inductor_random_58: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_448, inductor_lookup_seed_58, 'rand');  _shape_param_448 = inductor_lookup_seed_58 = None
        convert_element_type_241: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_58, torch.bfloat16);  inductor_random_58 = None
        gt_59: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_241, 0.1);  convert_element_type_241 = None
        mul_177: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_59, view_316);  view_316 = None
        mul_178: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_177, 1.1111111111111112);  mul_177 = None
        add_77: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_74, mul_178);  mul_178 = None
        pow_30: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_77, 2)
        mean_29: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None
        add_78: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_179: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_77, rsqrt_29)
        mul_180: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg123_1, mul_179);  arg123_1 = mul_179 = None
        convert_element_type_242: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg124_1, torch.bfloat16);  arg124_1 = None
        convert_element_type_243: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_180, torch.bfloat16);  mul_180 = None
        permute_177: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_242, [1, 0]);  convert_element_type_242 = None
        view_317: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_243, _shape_param_449);  convert_element_type_243 = _shape_param_449 = None
        mm_90: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_317, permute_177)
        view_318: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_90, _shape_param_450);  mm_90 = _shape_param_450 = None
        view_319: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_318, _shape_param_451);  view_318 = _shape_param_451 = None
        permute_178: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None
        convert_element_type_244: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg125_1, torch.bfloat16);  arg125_1 = None
        permute_179: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_244, [1, 0]);  convert_element_type_244 = None
        mm_91: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_139, permute_179)
        view_320: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_91, _shape_param_452);  mm_91 = _shape_param_452 = None
        view_321: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_320, _shape_param_453);  view_320 = _shape_param_453 = None
        permute_180: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_321, [0, 2, 1, 3]);  view_321 = None
        convert_element_type_245: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg126_1, torch.bfloat16);  arg126_1 = None
        permute_181: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_245, [1, 0]);  convert_element_type_245 = None
        mm_92: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_139, permute_181)
        view_322: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_92, _shape_param_454);  mm_92 = _shape_param_454 = None
        view_323: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_322, _shape_param_455);  view_322 = _shape_param_455 = None
        permute_182: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None
        permute_183: "bf16[8, 8, 64, 1024]" = torch.ops.aten.permute.default(permute_180, [0, 1, 3, 2]);  permute_180 = None
        expand_69: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_178, _shape_param_456);  permute_178 = _shape_param_456 = None
        clone_68: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_324: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_68, _shape_param_457);  clone_68 = _shape_param_457 = None
        expand_70: "bf16[8, 8, 64, 1024]" = torch.ops.aten.expand.default(permute_183, _shape_param_458);  permute_183 = _shape_param_458 = None
        clone_69: "bf16[8, 8, 64, 1024]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_325: "bf16[64, 64, 1024]" = torch.ops.aten.view.default(clone_69, _shape_param_459);  clone_69 = _shape_param_459 = None
        bmm_34: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_324, view_325)
        view_326: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_34, _shape_param_460);  _shape_param_460 = None
        convert_element_type_246: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_326, torch.bfloat16);  view_326 = None
        convert_element_type_247: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_246, torch.float32);  convert_element_type_246 = None
        amax_17: "f32[8, 8, 1024, 1]" = torch.ops.aten.amax.default(convert_element_type_247, [-1], True)
        sub_18: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_247, amax_17);  convert_element_type_247 = None
        exp_17: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_18: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_21: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = None
        convert_element_type_248: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None
        inductor_lookup_seed_59: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 59)
        inductor_random_59: "f32[8, 8, 1024, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_461, inductor_lookup_seed_59, 'rand');  _shape_param_461 = inductor_lookup_seed_59 = None
        convert_element_type_249: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_59, torch.bfloat16);  inductor_random_59 = None
        gt_60: "b8[8, 8, 1024, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_249, 0.1);  convert_element_type_249 = None
        mul_181: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(gt_60, convert_element_type_248);  convert_element_type_248 = None
        mul_182: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_181, 1.1111111111111112);  mul_181 = None
        expand_71: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.expand.default(mul_182, _shape_param_462);  mul_182 = _shape_param_462 = None
        view_327: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(expand_71, _shape_param_463);  expand_71 = _shape_param_463 = None
        expand_72: "bf16[8, 8, 1024, 64]" = torch.ops.aten.expand.default(permute_182, _shape_param_464);  permute_182 = _shape_param_464 = None
        clone_70: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_328: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_70, _shape_param_465);  clone_70 = _shape_param_465 = None
        bmm_35: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_327, view_328)
        view_329: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_35, _shape_param_466);  bmm_35 = _shape_param_466 = None
        permute_184: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_329, [0, 2, 1, 3]);  view_329 = None
        clone_71: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
        view_330: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_71, _shape_param_467);  clone_71 = _shape_param_467 = None
        convert_element_type_250: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg127_1, torch.bfloat16);  arg127_1 = None
        permute_185: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_250, [1, 0]);  convert_element_type_250 = None
        view_331: "bf16[8192, 512]" = torch.ops.aten.view.default(view_330, _shape_param_468);  view_330 = _shape_param_468 = None
        mm_93: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_331, permute_185)
        view_332: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_93, _shape_param_469);  mm_93 = _shape_param_469 = None
        inductor_lookup_seed_60: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 60)
        inductor_random_60: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_470, inductor_lookup_seed_60, 'rand');  _shape_param_470 = inductor_lookup_seed_60 = None
        convert_element_type_251: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_60, torch.bfloat16);  inductor_random_60 = None
        gt_61: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_251, 0.1);  convert_element_type_251 = None
        mul_183: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_61, view_332);  view_332 = None
        mul_184: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_183, 1.1111111111111112);  mul_183 = None
        add_79: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_77, mul_184);  mul_184 = None
        pow_31: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_79, 2)
        mean_30: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
        add_80: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_185: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_79, rsqrt_30)
        mul_186: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg128_1, mul_185);  arg128_1 = mul_185 = None
        convert_element_type_252: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg129_1, torch.bfloat16);  arg129_1 = None
        convert_element_type_253: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_186, torch.bfloat16);  mul_186 = None
        permute_186: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_252, [1, 0]);  convert_element_type_252 = None
        view_333: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_253, _shape_param_471);  convert_element_type_253 = _shape_param_471 = None
        mm_94: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_333, permute_186)
        view_334: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_94, _shape_param_472);  mm_94 = _shape_param_472 = None
        relu_11: "bf16[8, 1024, 2048]" = torch.ops.aten.relu.default(view_334);  view_334 = None
        inductor_lookup_seed_61: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 61)
        inductor_random_61: "f32[8, 1024, 2048]" = torch.ops.prims.inductor_random.default(_shape_param_473, inductor_lookup_seed_61, 'rand');  _shape_param_473 = inductor_lookup_seed_61 = None
        convert_element_type_254: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(inductor_random_61, torch.bfloat16);  inductor_random_61 = None
        gt_62: "b8[8, 1024, 2048]" = torch.ops.aten.gt.Scalar(convert_element_type_254, 0.1);  convert_element_type_254 = None
        mul_187: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(gt_62, relu_11)
        mul_188: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(mul_187, 1.1111111111111112);  mul_187 = None
        convert_element_type_255: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg130_1, torch.bfloat16);  arg130_1 = None
        convert_element_type_256: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(mul_188, torch.bfloat16);  mul_188 = None
        permute_187: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_255, [1, 0]);  convert_element_type_255 = None
        view_335: "bf16[8192, 2048]" = torch.ops.aten.view.default(convert_element_type_256, _shape_param_474);  convert_element_type_256 = _shape_param_474 = None
        mm_95: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_335, permute_187)
        view_336: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_95, _shape_param_475);  mm_95 = _shape_param_475 = None
        inductor_lookup_seed_62: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 62)
        inductor_random_62: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_476, inductor_lookup_seed_62, 'rand');  _shape_param_476 = inductor_lookup_seed_62 = None
        convert_element_type_257: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_62, torch.bfloat16);  inductor_random_62 = None
        gt_63: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_257, 0.1);  convert_element_type_257 = None
        mul_189: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_63, view_336);  view_336 = None
        mul_190: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_189, 1.1111111111111112);  mul_189 = None
        add_81: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_79, mul_190);  mul_190 = None
        pow_32: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_81, 2)
        mean_31: "f32[8, 1024, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None
        add_82: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_191: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_81, rsqrt_31)
        mul_192: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg131_1, mul_191);  arg131_1 = mul_191 = None
        inductor_lookup_seed_63: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 63);  inductor_seeds = None
        inductor_random_63: "f32[8, 1024, 512]" = torch.ops.prims.inductor_random.default(_shape_param_477, inductor_lookup_seed_63, 'rand');  _shape_param_477 = inductor_lookup_seed_63 = None
        gt_64: "b8[8, 1024, 512]" = torch.ops.aten.gt.Scalar(inductor_random_63, 0.1);  inductor_random_63 = None
        mul_193: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_64, mul_192);  mul_192 = None
        mul_194: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_193, 1.1111111111111112);  mul_193 = None
        mul_195: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_194, 0.04419417382415922);  mul_194 = None
        convert_element_type_258: "bf16[32128, 512]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_259: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(mul_195, torch.bfloat16);  mul_195 = None
        permute_188: "bf16[512, 32128]" = torch.ops.aten.permute.default(convert_element_type_258, [1, 0]);  convert_element_type_258 = None
        view_337: "bf16[8192, 512]" = torch.ops.aten.view.default(convert_element_type_259, _shape_param_478);  convert_element_type_259 = _shape_param_478 = None
        mm_96: "bf16[8192, 32128]" = torch.ops.aten.mm.default(view_337, permute_188)
        view_338: "bf16[8, 1024, 32128]" = torch.ops.aten.view.default(mm_96, _shape_param_479);  mm_96 = _shape_param_479 = None
        view_339: "bf16[8192, 32128]" = torch.ops.aten.view.default(view_338, _shape_param_480);  _shape_param_480 = None
        view_340: "i64[8192]" = torch.ops.aten.view.default(arg132_1, [-1]);  arg132_1 = None
        convert_element_type_260: "f32[8192, 32128]" = torch.ops.prims.convert_element_type.default(view_339, torch.float32);  view_339 = None
        amax_18: "f32[8192, 1]" = torch.ops.aten.amax.default(convert_element_type_260, [1], True)
        sub_19: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(convert_element_type_260, amax_18);  convert_element_type_260 = None
        exp_18: "f32[8192, 32128]" = torch.ops.aten.exp.default(sub_19)
        sum_19: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [1], True);  exp_18 = None
        log_2: "f32[8192, 1]" = torch.ops.aten.log.default(sum_19);  sum_19 = None
        sub_20: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(sub_19, log_2);  sub_19 = None
        convert_element_type_261: "bf16[8192, 32128]" = torch.ops.prims.convert_element_type.default(sub_20, torch.bfloat16);  sub_20 = None
        convert_element_type_262: "f32[8192, 32128]" = torch.ops.prims.convert_element_type.default(convert_element_type_261, torch.float32);  convert_element_type_261 = None
        ne: "b8[8192]" = torch.ops.aten.ne.Scalar(view_340, -100)
        full_6: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "i64[8192]" = torch.ops.aten.where.self(ne, view_340, full_6);  view_340 = full_6 = None
        unsqueeze_8: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where_3, 1);  where_3 = None
        gather: "f32[8192, 1]" = torch.ops.aten.gather.default(convert_element_type_262, 1, unsqueeze_8);  convert_element_type_262 = unsqueeze_8 = None
        squeeze: "f32[8192]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_1: "f32[8192]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_4: "f32[8192]" = torch.ops.aten.where.self(ne, neg_1, full_2);  neg_1 = full_2 = None
        sum_20: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_263: "f32[]" = torch.ops.prims.convert_element_type.default(sum_20, torch.float32);  sum_20 = None
        sum_21: "f32[]" = torch.ops.aten.sum.default(where_4);  where_4 = None
        div_22: "f32[]" = torch.ops.aten.div.Tensor(sum_21, convert_element_type_263);  sum_21 = None
        permute_189: "bf16[32128, 512]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
        permute_190: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
        le_1: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        permute_191: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        permute_192: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        permute_193: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_327, [0, 2, 1]);  view_327 = None
        permute_194: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_328, [0, 2, 1]);  view_328 = None
        permute_195: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_324, [0, 2, 1]);  view_324 = None
        permute_196: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_325, [0, 2, 1]);  view_325 = None
        permute_197: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
        permute_198: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
        permute_199: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
        permute_200: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        permute_201: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None
        permute_202: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_312, [0, 2, 1]);  view_312 = None
        permute_203: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_308, [0, 2, 1]);  view_308 = None
        permute_204: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_309, [0, 2, 1]);  view_309 = None
        permute_205: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
        permute_206: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
        permute_207: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
        permute_208: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        le_2: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        permute_209: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
        permute_210: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        permute_211: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_291, [0, 2, 1]);  view_291 = None
        permute_212: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_292, [0, 2, 1]);  view_292 = None
        permute_213: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_288, [0, 2, 1]);  view_288 = None
        permute_214: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_289, [0, 2, 1]);  view_289 = None
        permute_215: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
        permute_216: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
        permute_217: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
        permute_218: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        permute_219: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None
        permute_220: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_276, [0, 2, 1]);  view_276 = None
        permute_221: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_272, [0, 2, 1]);  view_272 = None
        permute_222: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_273, [0, 2, 1]);  view_273 = None
        permute_223: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        permute_224: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
        permute_225: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
        permute_226: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
        le_3: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        permute_227: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
        permute_228: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        permute_229: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
        permute_230: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
        permute_231: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None
        permute_232: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
        permute_233: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        permute_234: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
        permute_235: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
        permute_236: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        permute_237: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_239, [0, 2, 1]);  view_239 = None
        permute_238: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_240, [0, 2, 1]);  view_240 = None
        permute_239: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_236, [0, 2, 1]);  view_236 = None
        permute_240: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_237, [0, 2, 1]);  view_237 = None
        permute_241: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        permute_242: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        permute_243: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
        permute_244: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
        le_4: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        permute_245: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
        permute_246: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        permute_247: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_219, [0, 2, 1]);  view_219 = None
        permute_248: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_220, [0, 2, 1]);  view_220 = None
        permute_249: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_216, [0, 2, 1]);  view_216 = None
        permute_250: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_217, [0, 2, 1]);  view_217 = None
        permute_251: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        permute_252: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        permute_253: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
        permute_254: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
        permute_255: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None
        permute_256: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_204, [0, 2, 1]);  view_204 = None
        permute_257: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_200, [0, 2, 1]);  view_200 = None
        permute_258: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_201, [0, 2, 1]);  view_201 = None
        permute_259: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        permute_260: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        permute_261: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        permute_262: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        le_5: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        permute_263: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
        permute_264: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
        permute_265: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None
        permute_266: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_184, [0, 2, 1]);  view_184 = None
        permute_267: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_180, [0, 2, 1]);  view_180 = None
        permute_268: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1]);  view_181 = None
        permute_269: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        permute_270: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        permute_271: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        permute_272: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        permute_273: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
        permute_274: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
        permute_275: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None
        permute_276: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None
        permute_277: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        permute_278: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        permute_279: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        permute_280: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        le_6: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        permute_281: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        permute_282: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        permute_283: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
        permute_284: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_148, [0, 2, 1]);  view_148 = None
        permute_285: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        permute_286: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        permute_287: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        permute_288: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        permute_289: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        permute_290: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        permute_291: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
        permute_292: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None
        permute_293: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_127, [0, 2, 1]);  view_127 = None
        permute_294: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None
        permute_295: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        permute_296: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        permute_297: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        permute_298: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        le_7: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        permute_299: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        permute_300: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        permute_301: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
        permute_302: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
        permute_303: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
        permute_304: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1]);  view_108 = None
        permute_305: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        permute_306: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        permute_307: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        permute_308: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        le_8: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        permute_309: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        permute_310: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        permute_311: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_90, [0, 2, 1]);  view_90 = None
        permute_312: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_313: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
        permute_314: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1]);  view_88 = None
        permute_315: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        permute_316: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        permute_317: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        permute_318: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        le_9: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        permute_319: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        permute_320: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        permute_321: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None
        permute_322: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None
        permute_323: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_67, [0, 2, 1]);  view_67 = None
        permute_324: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
        permute_325: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        permute_326: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        permute_327: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        permute_328: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        le_10: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        permute_329: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        permute_330: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        permute_331: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
        permute_332: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
        permute_333: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_47, [0, 2, 1]);  view_47 = None
        permute_334: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1]);  view_48 = None
        permute_335: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        permute_336: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        permute_337: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        permute_338: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        le_11: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        permute_339: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        permute_340: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        permute_341: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
        permute_342: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_343: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
        permute_344: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
        permute_345: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        permute_346: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        permute_347: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_348: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        le_12: "b8[8, 1024, 2048]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        permute_349: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_350: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        permute_351: "bf16[64, 1024, 1024]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_352: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        permute_353: "bf16[64, 64, 1024]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        permute_354: "bf16[64, 1024, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
        permute_355: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        permute_356: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_357: "bf16[512, 512]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (embedding, gt, rsqrt, view, bmm, add_4, embedding_1, amax, sum_1, gt_2, view_14, gt_3, add_7, rsqrt_1, view_16, gt_4, view_18, gt_5, add_9, rsqrt_2, view_20, convert_element_type_23, amax_1, sum_2, gt_6, view_34, gt_7, add_12, rsqrt_3, view_36, gt_8, view_38, gt_9, add_14, rsqrt_4, view_40, convert_element_type_39, amax_2, sum_3, gt_10, view_54, gt_11, add_17, rsqrt_5, view_56, gt_12, view_58, gt_13, add_19, rsqrt_6, view_60, convert_element_type_55, amax_3, sum_4, gt_14, view_74, gt_15, add_22, rsqrt_7, view_76, gt_16, view_78, gt_17, add_24, rsqrt_8, view_80, convert_element_type_71, amax_4, sum_5, gt_18, view_94, gt_19, add_27, rsqrt_9, view_96, gt_20, view_98, gt_21, add_29, rsqrt_10, view_100, convert_element_type_87, amax_5, sum_6, gt_22, view_114, gt_23, add_32, rsqrt_11, view_116, gt_24, view_118, gt_25, add_34, rsqrt_12, gt_26, mul_79, unsqueeze_4, gt_27, rsqrt_13, view_120, bmm_12, add_39, embedding_2, amax_6, sum_7, gt_28, view_134, gt_29, add_42, rsqrt_14, view_136, view_139, bmm_14, amax_7, sum_8, gt_30, view_151, gt_31, add_44, rsqrt_15, view_153, gt_32, view_155, gt_33, add_46, rsqrt_16, view_157, convert_element_type_132, amax_8, sum_9, gt_34, view_171, gt_35, add_49, rsqrt_17, view_173, bmm_18, amax_9, sum_10, gt_36, view_187, gt_37, add_51, rsqrt_18, view_189, gt_38, view_191, gt_39, add_53, rsqrt_19, view_193, convert_element_type_158, amax_10, sum_11, gt_40, view_207, gt_41, add_56, rsqrt_20, view_209, bmm_22, amax_11, sum_12, gt_42, view_223, gt_43, add_58, rsqrt_21, view_225, gt_44, view_227, gt_45, add_60, rsqrt_22, view_229, convert_element_type_184, amax_12, sum_13, gt_46, view_243, gt_47, add_63, rsqrt_23, view_245, bmm_26, amax_13, sum_14, gt_48, view_259, gt_49, add_65, rsqrt_24, view_261, gt_50, view_263, gt_51, add_67, rsqrt_25, view_265, convert_element_type_210, amax_14, sum_15, gt_52, view_279, gt_53, add_70, rsqrt_26, view_281, bmm_30, amax_15, sum_16, gt_54, view_295, gt_55, add_72, rsqrt_27, view_297, gt_56, view_299, gt_57, add_74, rsqrt_28, view_301, convert_element_type_236, amax_16, sum_17, gt_58, view_315, gt_59, add_77, rsqrt_29, view_317, bmm_34, amax_17, sum_18, gt_60, view_331, gt_61, add_79, rsqrt_30, view_333, gt_62, view_335, gt_63, add_81, rsqrt_31, gt_64, view_337, view_338, amax_18, log_2, convert_element_type_263, div_22, permute_189, permute_190, le_1, permute_191, permute_192, permute_193, permute_194, permute_195, permute_196, permute_197, permute_198, permute_199, permute_200, permute_201, permute_202, permute_203, permute_204, permute_205, permute_206, permute_207, permute_208, le_2, permute_209, permute_210, permute_211, permute_212, permute_213, permute_214, permute_215, permute_216, permute_217, permute_218, permute_219, permute_220, permute_221, permute_222, permute_223, permute_224, permute_225, permute_226, le_3, permute_227, permute_228, permute_229, permute_230, permute_231, permute_232, permute_233, permute_234, permute_235, permute_236, permute_237, permute_238, permute_239, permute_240, permute_241, permute_242, permute_243, permute_244, le_4, permute_245, permute_246, permute_247, permute_248, permute_249, permute_250, permute_251, permute_252, permute_253, permute_254, permute_255, permute_256, permute_257, permute_258, permute_259, permute_260, permute_261, permute_262, le_5, permute_263, permute_264, permute_265, permute_266, permute_267, permute_268, permute_269, permute_270, permute_271, permute_272, permute_273, permute_274, permute_275, permute_276, permute_277, permute_278, permute_279, permute_280, le_6, permute_281, permute_282, permute_283, permute_284, permute_285, permute_286, permute_287, permute_288, permute_289, permute_290, permute_291, permute_292, permute_293, permute_294, permute_295, permute_296, permute_297, permute_298, le_7, permute_299, permute_300, permute_301, permute_302, permute_303, permute_304, permute_305, permute_306, permute_307, permute_308, le_8, permute_309, permute_310, permute_311, permute_312, permute_313, permute_314, permute_315, permute_316, permute_317, permute_318, le_9, permute_319, permute_320, permute_321, permute_322, permute_323, permute_324, permute_325, permute_326, permute_327, permute_328, le_10, permute_329, permute_330, permute_331, permute_332, permute_333, permute_334, permute_335, permute_336, permute_337, permute_338, le_11, permute_339, permute_340, permute_341, permute_342, permute_343, permute_344, permute_345, permute_346, permute_347, permute_348, le_12, permute_349, permute_350, permute_351, permute_352, permute_353, permute_354, permute_355, permute_356, permute_357)



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
