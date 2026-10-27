"""
Standalone repro captured via capture_hook.
Label: AllenaiLongformerBase
Pattern hash: fc94eecab4c0
Shape hash: aed8d19a
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
    def forward(self, arg0_1: "f32[8, 1024, 768]", arg1_1: "f32[768]", arg2_1: "f32[768, 768]", arg3_1: "f32[768]", arg4_1: "f32[768, 768]", arg5_1: "f32[768]", arg6_1: "f32[768, 768]", arg7_1: "f32[8, 1024]", arg8_1: "b8[8, 1024]", arg9_1: "f32[768]", arg10_1: "f32[768, 768]", arg11_1: "f32[768]", arg12_1: "f32[768]", arg13_1: "f32[3072]", arg14_1: "f32[3072, 768]", arg15_1: "f32[768]", arg16_1: "f32[768, 3072]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[768]", arg20_1: "f32[768, 768]", arg21_1: "f32[768]", arg22_1: "f32[768, 768]", arg23_1: "f32[768]", arg24_1: "f32[768, 768]", arg25_1: "f32[768]", arg26_1: "f32[768, 768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[3072]", arg30_1: "f32[3072, 768]", arg31_1: "f32[768]", arg32_1: "f32[768, 3072]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[768, 768]", arg37_1: "f32[768]", arg38_1: "f32[768, 768]", arg39_1: "f32[768]", arg40_1: "f32[768, 768]", arg41_1: "f32[768]", arg42_1: "f32[768, 768]", arg43_1: "f32[768]", arg44_1: "f32[768]", arg45_1: "f32[3072]", arg46_1: "f32[3072, 768]", arg47_1: "f32[768]", arg48_1: "f32[768, 3072]", arg49_1: "f32[768]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768, 768]", arg53_1: "f32[768]", arg54_1: "f32[768, 768]", arg55_1: "f32[768]", arg56_1: "f32[768, 768]", arg57_1: "f32[768]", arg58_1: "f32[768, 768]", arg59_1: "f32[768]", arg60_1: "f32[768]", arg61_1: "f32[3072]", arg62_1: "f32[3072, 768]", arg63_1: "f32[768]", arg64_1: "f32[768, 3072]", arg65_1: "f32[768]", arg66_1: "f32[768]", arg67_1: "f32[768]", arg68_1: "f32[768, 768]", arg69_1: "f32[768]", arg70_1: "f32[768, 768]", arg71_1: "f32[768]", arg72_1: "f32[768, 768]", arg73_1: "f32[768]", arg74_1: "f32[768, 768]", arg75_1: "f32[768]", arg76_1: "f32[768]", arg77_1: "f32[3072]", arg78_1: "f32[3072, 768]", arg79_1: "f32[768]", arg80_1: "f32[768, 3072]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[768]", arg84_1: "f32[768, 768]", arg85_1: "f32[768]", arg86_1: "f32[768, 768]", arg87_1: "f32[768]", arg88_1: "f32[768, 768]", arg89_1: "f32[768]", arg90_1: "f32[768, 768]", arg91_1: "f32[768]", arg92_1: "f32[768]", arg93_1: "f32[3072]", arg94_1: "f32[3072, 768]", arg95_1: "f32[768]", arg96_1: "f32[768, 3072]", arg97_1: "f32[768]", arg98_1: "f32[768]", arg99_1: "f32[768]", arg100_1: "f32[768, 768]", arg101_1: "f32[768]", arg102_1: "f32[768, 768]", arg103_1: "f32[768]", arg104_1: "f32[768, 768]", arg105_1: "f32[768]", arg106_1: "f32[768, 768]", arg107_1: "f32[768]", arg108_1: "f32[768]", arg109_1: "f32[3072]", arg110_1: "f32[3072, 768]", arg111_1: "f32[768]", arg112_1: "f32[768, 3072]", arg113_1: "f32[768]", arg114_1: "f32[768]", arg115_1: "f32[768]", arg116_1: "f32[768, 768]", arg117_1: "f32[768]", arg118_1: "f32[768, 768]", arg119_1: "f32[768]", arg120_1: "f32[768, 768]", arg121_1: "f32[768]", arg122_1: "f32[768, 768]", arg123_1: "f32[768]", arg124_1: "f32[768]", arg125_1: "f32[3072]", arg126_1: "f32[3072, 768]", arg127_1: "f32[768]", arg128_1: "f32[768, 3072]", arg129_1: "f32[768]", arg130_1: "f32[768]", arg131_1: "f32[768]", arg132_1: "f32[768, 768]", arg133_1: "f32[768]", arg134_1: "f32[768, 768]", arg135_1: "f32[768]", arg136_1: "f32[768, 768]", arg137_1: "f32[768]", arg138_1: "f32[768, 768]", arg139_1: "f32[768]", arg140_1: "f32[768]", arg141_1: "f32[3072]", arg142_1: "f32[3072, 768]", arg143_1: "f32[768]", arg144_1: "f32[768, 3072]", arg145_1: "f32[768]", arg146_1: "f32[768]", arg147_1: "f32[768]", arg148_1: "f32[768, 768]", arg149_1: "f32[768]", arg150_1: "f32[768, 768]", arg151_1: "f32[768]", arg152_1: "f32[768, 768]", arg153_1: "f32[768]", arg154_1: "f32[768, 768]", arg155_1: "f32[768]", arg156_1: "f32[768]", arg157_1: "f32[3072]", arg158_1: "f32[3072, 768]", arg159_1: "f32[768]", arg160_1: "f32[768, 3072]", arg161_1: "f32[768]", arg162_1: "f32[768]", arg163_1: "f32[768]", arg164_1: "f32[768, 768]", arg165_1: "f32[768]", arg166_1: "f32[768, 768]", arg167_1: "f32[768]", arg168_1: "f32[768, 768]", arg169_1: "f32[768]", arg170_1: "f32[768, 768]", arg171_1: "f32[768]", arg172_1: "f32[768]", arg173_1: "f32[3072]", arg174_1: "f32[3072, 768]", arg175_1: "f32[768]", arg176_1: "f32[768, 3072]", arg177_1: "f32[768]", arg178_1: "f32[768]", arg179_1: "f32[768]", arg180_1: "f32[768, 768]", arg181_1: "f32[768]", arg182_1: "f32[768, 768]", arg183_1: "f32[768]", arg184_1: "f32[768, 768]", arg185_1: "f32[768]", arg186_1: "f32[768, 768]", arg187_1: "f32[768]", arg188_1: "f32[768]", arg189_1: "f32[3072]", arg190_1: "f32[3072, 768]", arg191_1: "f32[768]", arg192_1: "f32[768, 3072]", arg193_1: "f32[768]", arg194_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574):
        # No stacktrace found for following nodes
        permute: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(arg0_1, [1, 0, 2])
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_1: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        convert_element_type_2: "bf16[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(permute, torch.bfloat16);  permute = None
        permute_1: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        clone: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(convert_element_type_2, memory_format = torch.contiguous_format);  convert_element_type_2 = None
        view: "bf16[8192, 768]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(view, permute_1)
        view_1: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        add: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_1, convert_element_type);  view_1 = convert_element_type = None
        convert_element_type_3: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_4: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        permute_2: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        mm_1: "bf16[8192, 768]" = torch.ops.aten.mm.default(view, permute_2)
        view_2: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_1, _shape_param_2);  mm_1 = _shape_param_2 = None
        add_1: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_2, convert_element_type_3);  view_2 = convert_element_type_3 = None
        convert_element_type_5: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_6: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        permute_3: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_6, [1, 0]);  convert_element_type_6 = None
        mm_2: "bf16[8192, 768]" = torch.ops.aten.mm.default(view, permute_3)
        view_3: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_2, _shape_param_3);  mm_2 = _shape_param_3 = None
        add_2: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_3, convert_element_type_5);  view_3 = convert_element_type_5 = None
        div: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add, 8.0);  add = None
        view_4: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_1, _shape_param_4);  add_1 = _shape_param_4 = None
        permute_4: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_4, [1, 0, 2, 3]);  view_4 = None
        permute_5: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_4, [0, 2, 1, 3]);  permute_4 = None
        view_5: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_5, _shape_param_5);  permute_5 = _shape_param_5 = None
        view_6: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_5, _shape_param_6);  view_5 = _shape_param_6 = None
        as_strided: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_6, _shape_param_7, _shape_param_8);  view_6 = _shape_param_7 = _shape_param_8 = None
        unsqueeze: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided, 4);  as_strided = None
        permute_6: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze, [0, 1, 4, 2, 3]);  unsqueeze = None
        view_7: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div, _shape_param_9);  div = _shape_param_9 = None
        permute_7: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_7, [1, 0, 2, 3]);  view_7 = None
        permute_8: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_7, [0, 2, 1, 3]);  permute_7 = None
        view_8: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_8, _shape_param_10);  permute_8 = _shape_param_10 = None
        view_9: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_8, _shape_param_11);  view_8 = _shape_param_11 = None
        as_strided_1: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_9, _shape_param_12, _shape_param_13);  view_9 = _shape_param_12 = _shape_param_13 = None
        unsqueeze_1: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_1, 4);  as_strided_1 = None
        clone_1: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_1, memory_format = torch.contiguous_format);  unsqueeze_1 = None
        view_10: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_14);  clone_1 = _shape_param_14 = None
        permute_9: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_6, [0, 1, 4, 3, 2]);  permute_6 = None
        clone_2: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
        view_11: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_2, _shape_param_15);  clone_2 = _shape_param_15 = None
        bmm: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_10, view_11)
        view_12: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm, _shape_param_16);  bmm = _shape_param_16 = None
        permute_10: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_12, [0, 1, 2, 4, 3]);  view_12 = None
        view_13: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_10, _shape_param_17);  permute_10 = _shape_param_17 = None
        constant_pad_nd: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_13, [0, 0, 0, 1], 0.0);  view_13 = None
        view_14: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd, _shape_param_18);  constant_pad_nd = _shape_param_18 = None
        full: "bf16[96, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_19, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_19 = None
        slice_1: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_14, 2, 0, 256)
        slice_2: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 257);  slice_1 = None
        slice_3: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full, 1, 0, -1)
        slice_4: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_3, 3, 256, 9223372036854775807)
        copy: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_2);  slice_2 = None
        slice_scatter: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy, 3, 256, 9223372036854775807);  copy = None
        slice_scatter_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter, 1, 0, -1);  slice_scatter = None
        select: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_14, 1, -1)
        slice_5: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select, 1, 256, 9223372036854775807);  select = None
        slice_6: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_5, 2, 0, 257);  slice_5 = None
        select_1: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
        slice_7: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_1, 2, 256, 9223372036854775807)
        copy_1: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_7, slice_6);  slice_7 = slice_6 = None
        slice_scatter_2: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_1, copy_1, 2, 256, 9223372036854775807);  select_1 = copy_1 = None
        select_scatter: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_1, slice_scatter_2, 1, -1);  slice_scatter_1 = slice_scatter_2 = None
        slice_8: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_14, 2, -257, -1)
        slice_9: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_8, 3, 257, 9223372036854775807);  slice_8 = None
        slice_10: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_11: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_10, 3, 0, 256)
        copy_2: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_11, slice_9);  slice_11 = slice_9 = None
        slice_scatter_3: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_10, copy_2, 3, 0, 256);  slice_10 = copy_2 = None
        slice_scatter_4: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter, slice_scatter_3, 1, 1, 9223372036854775807);  select_scatter = slice_scatter_3 = None
        select_2: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_14, 1, 0);  view_14 = None
        slice_12: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_2, 1, 0, 255);  select_2 = None
        slice_13: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_12, 2, -255, 9223372036854775807);  slice_12 = None
        select_3: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_4, 1, 0)
        slice_14: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_3, 1, 1, 256)
        slice_15: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_14, 2, 1, 256)
        copy_3: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_15, slice_13);  slice_15 = slice_13 = None
        slice_scatter_5: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_14, copy_3, 2, 1, 256);  slice_14 = copy_3 = None
        slice_scatter_6: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_3, slice_scatter_5, 1, 1, 256);  select_3 = slice_scatter_5 = None
        select_scatter_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_4, slice_scatter_6, 1, 0);  slice_scatter_4 = slice_scatter_6 = None
        full_1: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_20, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_20 = None
        iota: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_2: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_2, unsqueeze_3);  unsqueeze_2 = unsqueeze_3 = None
        le: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub, 0);  sub = None
        scalar_tensor: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where: "bf16[256, 257]" = torch.ops.aten.where.self(le, full_1, scalar_tensor);  le = full_1 = scalar_tensor = None
        rev: "bf16[256, 257]" = torch.ops.prims.rev.default(where, [0]);  where = None
        unsqueeze_4: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev, 0);  rev = None
        unsqueeze_5: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        rev_1: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_5, [1, 3])
        expand: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_5, _shape_param_21);  _shape_param_21 = None
        view_15: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_1, _shape_param_22);  select_scatter_1 = _shape_param_22 = None
        permute_11: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        slice_16: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_11, 1, 0, 256)
        slice_17: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_16, 3, 0, 257)
        full_2: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_23, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_23 = None
        permute_12: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_2, [0, 2, 1, 3]);  full_2 = None
        convert_element_type_7: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand, torch.bool);  expand = None
        where_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_7, permute_12, slice_17)
        copy_4: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_17, where_1);  slice_17 = where_1 = None
        slice_scatter_7: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_16, copy_4, 3, 0, 257);  slice_16 = copy_4 = None
        slice_scatter_8: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_11, slice_scatter_7, 1, 0, 256);  permute_11 = slice_scatter_7 = None
        permute_13: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_8, [0, 2, 1, 3]);  slice_scatter_8 = None
        view_16: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_13, _shape_param_24);  permute_13 = _shape_param_24 = None
        expand_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_1, _shape_param_25);  _shape_param_25 = None
        view_17: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_16, _shape_param_26);  view_16 = _shape_param_26 = None
        permute_14: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        slice_18: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_14, 1, -256, 9223372036854775807)
        slice_19: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_18, 3, -257, 9223372036854775807)
        convert_element_type_8: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_1, torch.bool);  expand_1 = None
        where_2: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_8, permute_12, slice_19)
        copy_5: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_19, where_2);  slice_19 = where_2 = None
        slice_scatter_9: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_18, copy_5, 3, -257, 9223372036854775807);  slice_18 = copy_5 = None
        slice_scatter_10: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_14, slice_scatter_9, 1, -256, 9223372036854775807);  permute_14 = slice_scatter_9 = None
        permute_15: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_10, [0, 2, 1, 3])
        ne: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg7_1, 0);  arg7_1 = None
        unsqueeze_6: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne, 2);  ne = None
        unsqueeze_7: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 3);  unsqueeze_6 = None
        convert_element_type_9: "bf16[8, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_7, torch.bfloat16)
        full_3: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[8, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_7, full_3, convert_element_type_9);  unsqueeze_7 = full_3 = convert_element_type_9 = None
        full_4: "bf16[8, 1024, 1, 1]" = torch.ops.aten.full.default(_shape_param_27, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_27 = None
        permute_16: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(full_4, [0, 2, 1, 3]);  full_4 = None
        view_18: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_16, _shape_param_28);  permute_16 = _shape_param_28 = None
        permute_17: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(where_3, [0, 2, 1, 3]);  where_3 = None
        view_19: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_17, _shape_param_29);  permute_17 = _shape_param_29 = None
        view_20: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_18, _shape_param_30);  view_18 = _shape_param_30 = None
        as_strided_2: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_20, _shape_param_31, _shape_param_32);  view_20 = _shape_param_31 = _shape_param_32 = None
        view_21: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_19, _shape_param_33);  view_19 = _shape_param_33 = None
        as_strided_3: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_21, _shape_param_34, _shape_param_35);  view_21 = _shape_param_34 = _shape_param_35 = None
        unsqueeze_8: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_2, 4);  as_strided_2 = None
        permute_18: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_8, [0, 1, 2, 4, 3]);  unsqueeze_8 = None
        unsqueeze_9: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_3, 4);  as_strided_3 = None
        permute_19: "bf16[8, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_9, [0, 1, 4, 2, 3]);  unsqueeze_9 = None
        mul: "bf16[8, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_18, permute_19);  permute_18 = permute_19 = None
        view_22: "bf16[8, 3, 512, 512]" = torch.ops.aten.view.default(mul, _shape_param_36);  mul = _shape_param_36 = None
        constant_pad_nd_1: "bf16[8, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_22, [0, 0, 0, 1], 0.0);  view_22 = None
        view_23: "bf16[8, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_1, _shape_param_37);  constant_pad_nd_1 = _shape_param_37 = None
        full_5: "bf16[8, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_38, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_38 = None
        slice_20: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_23, 2, 0, 256)
        slice_21: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_20, 3, 0, 257);  slice_20 = None
        slice_22: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_5, 1, 0, -1)
        slice_23: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_22, 3, 256, 9223372036854775807)
        copy_6: "bf16[8, 3, 256, 257]" = torch.ops.aten.copy.default(slice_23, slice_21);  slice_23 = slice_21 = None
        slice_scatter_11: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_22, copy_6, 3, 256, 9223372036854775807);  slice_22 = copy_6 = None
        slice_scatter_12: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_11, 1, 0, -1);  full_5 = slice_scatter_11 = None
        select_4: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_23, 1, -1)
        slice_24: "bf16[8, 256, 513]" = torch.ops.aten.slice.Tensor(select_4, 1, 256, 9223372036854775807);  select_4 = None
        slice_25: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(slice_24, 2, 0, 257);  slice_24 = None
        select_5: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_12, 1, -1)
        slice_26: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(select_5, 2, 256, 9223372036854775807)
        copy_7: "bf16[8, 256, 257]" = torch.ops.aten.copy.default(slice_26, slice_25);  slice_26 = slice_25 = None
        slice_scatter_13: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_5, copy_7, 2, 256, 9223372036854775807);  select_5 = copy_7 = None
        select_scatter_2: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_12, slice_scatter_13, 1, -1);  slice_scatter_12 = slice_scatter_13 = None
        slice_27: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_23, 2, -257, -1)
        slice_28: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_27, 3, 257, 9223372036854775807);  slice_27 = None
        slice_29: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_2, 1, 1, 9223372036854775807)
        slice_30: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_29, 3, 0, 256)
        copy_8: "bf16[8, 3, 256, 256]" = torch.ops.aten.copy.default(slice_30, slice_28);  slice_30 = slice_28 = None
        slice_scatter_14: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_29, copy_8, 3, 0, 256);  slice_29 = copy_8 = None
        slice_scatter_15: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_2, slice_scatter_14, 1, 1, 9223372036854775807);  select_scatter_2 = slice_scatter_14 = None
        select_6: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_23, 1, 0);  view_23 = None
        slice_31: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_6, 1, 0, 255);  select_6 = None
        slice_32: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_31, 2, -255, 9223372036854775807);  slice_31 = None
        select_7: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_15, 1, 0)
        slice_33: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_7, 1, 1, 256)
        slice_34: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_33, 2, 1, 256)
        copy_9: "bf16[8, 255, 255]" = torch.ops.aten.copy.default(slice_34, slice_32);  slice_34 = slice_32 = None
        slice_scatter_16: "bf16[8, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_33, copy_9, 2, 1, 256);  slice_33 = copy_9 = None
        slice_scatter_17: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_7, slice_scatter_16, 1, 1, 256);  select_7 = slice_scatter_16 = None
        select_scatter_3: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_15, slice_scatter_17, 1, 0);  slice_scatter_15 = slice_scatter_17 = None
        expand_2: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_5, _shape_param_39);  _shape_param_39 = None
        view_24: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_3, _shape_param_40);  select_scatter_3 = _shape_param_40 = None
        permute_20: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        slice_35: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_20, 1, 0, 256)
        slice_36: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_35, 3, 0, 257)
        full_6: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_41, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_41 = None
        convert_element_type_10: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_2, torch.bool);  expand_2 = None
        where_4: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_10, full_6, slice_36);  convert_element_type_10 = None
        copy_10: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_36, where_4);  slice_36 = where_4 = None
        slice_scatter_18: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_35, copy_10, 3, 0, 257);  slice_35 = copy_10 = None
        slice_scatter_19: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_20, slice_scatter_18, 1, 0, 256);  permute_20 = slice_scatter_18 = None
        permute_21: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_19, [0, 2, 1, 3]);  slice_scatter_19 = None
        view_25: "bf16[8, 4, 256, 513]" = torch.ops.aten.view.default(permute_21, _shape_param_42);  permute_21 = _shape_param_42 = None
        expand_3: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(rev_1, _shape_param_43);  _shape_param_43 = None
        view_26: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_25, _shape_param_44);  view_25 = _shape_param_44 = None
        permute_22: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        slice_37: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_22, 1, -256, 9223372036854775807)
        slice_38: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_37, 3, -257, 9223372036854775807)
        convert_element_type_11: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_3, torch.bool);  expand_3 = None
        where_5: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_11, full_6, slice_38);  convert_element_type_11 = full_6 = None
        copy_11: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_38, where_5);  slice_38 = where_5 = None
        slice_scatter_20: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_37, copy_11, 3, -257, 9223372036854775807);  slice_37 = copy_11 = None
        slice_scatter_21: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_22, slice_scatter_20, 1, -256, 9223372036854775807);  permute_22 = slice_scatter_20 = None
        permute_23: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_21, [0, 2, 1, 3])
        permute_24: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_15, [0, 2, 1, 3]);  permute_15 = None
        permute_25: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_23, [0, 2, 1, 3]);  permute_23 = None
        add_3: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_24, permute_25);  permute_24 = None
        permute_26: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_3, [0, 2, 1, 3]);  add_3 = None
        permute_27: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_26, [0, 2, 1, 3]);  permute_26 = None
        convert_element_type_12: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_27, torch.float32);  permute_27 = None
        clone_3: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_12, memory_format = torch.contiguous_format);  convert_element_type_12 = None
        amax: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_3, [-1], True)
        sub_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_3, amax);  clone_3 = None
        exp: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        unsqueeze_10: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 2);  arg8_1 = None
        unsqueeze_11: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None
        full_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_11, full_7, div_1);  div_1 = None
        convert_element_type_13: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_6, torch.bfloat16);  where_6 = None
        inductor_seeds: "i64[36]" = torch.ops.prims.inductor_seeds.default(36, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default(_shape_param_45, inductor_lookup_seed, 'rand');  _shape_param_45 = inductor_lookup_seed = None
        convert_element_type_14: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(convert_element_type_14, 0.1);  convert_element_type_14 = None
        mul_1: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt, convert_element_type_13);  convert_element_type_13 = None
        mul_2: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_1, 1.1111111111111112);  mul_1 = None
        view_27: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_2, _shape_param_46);  add_2 = _shape_param_46 = None
        permute_28: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_27, [1, 0, 2, 3]);  view_27 = None
        permute_29: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_2, [0, 2, 1, 3]);  mul_2 = None
        clone_4: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_28: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_4, _shape_param_47);  clone_4 = _shape_param_47 = None
        permute_30: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_28, [0, 2, 1, 3]);  permute_28 = None
        view_29: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_30, _shape_param_48);  permute_30 = _shape_param_48 = None
        constant_pad_nd_2: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_29, _shape_param_49, -1.0);  view_29 = _shape_param_49 = None
        as_strided_4: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_2, _shape_param_50, _shape_param_51);  constant_pad_nd_2 = _shape_param_50 = _shape_param_51 = None
        constant_pad_nd_3: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_28, _shape_param_52, 0.0);  view_28 = _shape_param_52 = None
        view_30: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_3, _shape_param_53);  constant_pad_nd_3 = _shape_param_53 = None
        slice_39: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_30, 2, 0, -256);  view_30 = None
        view_31: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_39, _shape_param_54);  slice_39 = _shape_param_54 = None
        slice_40: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_31, 3, 0, -1);  view_31 = None
        unsqueeze_12: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_40, 4);  slice_40 = None
        unsqueeze_13: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_4, 4);  as_strided_4 = None
        view_32: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_12, _shape_param_55);  unsqueeze_12 = _shape_param_55 = None
        clone_5: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_13, memory_format = torch.contiguous_format);  unsqueeze_13 = None
        view_33: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_5, _shape_param_56);  clone_5 = _shape_param_56 = None
        bmm_1: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_32, view_33)
        view_34: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_57);  bmm_1 = _shape_param_57 = None
        permute_31: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_34, [0, 1, 2, 4, 3]);  view_34 = None
        view_35: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_31, _shape_param_58);  permute_31 = _shape_param_58 = None
        view_36: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_35, _shape_param_59);  view_35 = _shape_param_59 = None
        permute_32: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        permute_33: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_32, [1, 0, 2, 3]);  permute_32 = None
        clone_6: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        view_37: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_6, _shape_param_60);  clone_6 = _shape_param_60 = None
        permute_34: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_37, [1, 0, 2]);  view_37 = None
        convert_element_type_15: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        convert_element_type_16: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        permute_35: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_16, [1, 0]);  convert_element_type_16 = None
        clone_7: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_38: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_7, _shape_param_61);  clone_7 = _shape_param_61 = None
        mm_3: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_38, permute_35)
        view_39: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_3, _shape_param_62);  mm_3 = _shape_param_62 = None
        add_4: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_39, convert_element_type_15);  view_39 = convert_element_type_15 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1)
        inductor_random_1: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_63, inductor_lookup_seed_1, 'rand');  _shape_param_63 = inductor_lookup_seed_1 = None
        convert_element_type_17: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_1, torch.bfloat16);  inductor_random_1 = None
        gt_1: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_17, 0.1);  convert_element_type_17 = None
        mul_3: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_1, add_4);  add_4 = None
        mul_4: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_3, 1.1111111111111112);  mul_3 = None
        add_5: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_4, arg0_1);  mul_4 = arg0_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean[1];  var_mean = None
        add_6: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_2: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_5, getitem_1);  add_5 = getitem_1 = None
        mul_5: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_5, arg11_1);  arg11_1 = None
        add_7: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_6, arg12_1);  mul_6 = arg12_1 = None
        convert_element_type_18: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        convert_element_type_19: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.bfloat16);  arg14_1 = None
        convert_element_type_20: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16)
        view_40: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_20, _shape_param_64);  convert_element_type_20 = _shape_param_64 = None
        permute_36: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_19, [1, 0]);  convert_element_type_19 = None
        addmm: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(convert_element_type_18, view_40, permute_36);  convert_element_type_18 = None
        view_41: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm, _shape_param_65);  _shape_param_65 = None
        convert_element_type_21: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_7: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.5)
        mul_8: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.7071067811865476);  convert_element_type_21 = None
        erf: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_8: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_7, add_8);  mul_7 = add_8 = None
        convert_element_type_22: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        convert_element_type_23: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg15_1, torch.bfloat16);  arg15_1 = None
        convert_element_type_24: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.bfloat16);  arg16_1 = None
        view_42: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_22, _shape_param_66);  convert_element_type_22 = _shape_param_66 = None
        permute_37: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        addmm_1: "bf16[8192, 768]" = torch.ops.aten.addmm.default(convert_element_type_23, view_42, permute_37);  convert_element_type_23 = None
        view_43: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_1, _shape_param_67);  addmm_1 = _shape_param_67 = None
        inductor_lookup_seed_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 2)
        inductor_random_2: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_68, inductor_lookup_seed_2, 'rand');  _shape_param_68 = inductor_lookup_seed_2 = None
        convert_element_type_25: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_2, torch.bfloat16);  inductor_random_2 = None
        gt_2: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_25, 0.1);  convert_element_type_25 = None
        mul_10: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_2, view_43);  view_43 = None
        mul_11: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_10, 1.1111111111111112);  mul_10 = None
        add_9: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_11, add_7);  mul_11 = add_7 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_2: "f32[8, 1024, 1]" = var_mean_1[0]
        getitem_3: "f32[8, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        add_10: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_3: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_3);  add_9 = getitem_3 = None
        mul_12: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
        mul_13: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_12, arg17_1);  arg17_1 = None
        add_11: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_13, arg18_1);  mul_13 = arg18_1 = None
        permute_38: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_11, [1, 0, 2])
        convert_element_type_26: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg19_1, torch.bfloat16);  arg19_1 = None
        convert_element_type_27: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.bfloat16);  arg20_1 = None
        convert_element_type_28: "bf16[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(permute_38, torch.bfloat16);  permute_38 = None
        permute_39: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_27, [1, 0]);  convert_element_type_27 = None
        clone_8: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(convert_element_type_28, memory_format = torch.contiguous_format);  convert_element_type_28 = None
        view_44: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_8, _shape_param_69);  clone_8 = _shape_param_69 = None
        mm_4: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_44, permute_39)
        view_45: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_4, _shape_param_70);  mm_4 = _shape_param_70 = None
        add_12: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_45, convert_element_type_26);  view_45 = convert_element_type_26 = None
        convert_element_type_29: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg21_1, torch.bfloat16);  arg21_1 = None
        convert_element_type_30: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg22_1, torch.bfloat16);  arg22_1 = None
        permute_40: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_30, [1, 0]);  convert_element_type_30 = None
        mm_5: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_44, permute_40)
        view_46: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_5, _shape_param_71);  mm_5 = _shape_param_71 = None
        add_13: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_46, convert_element_type_29);  view_46 = convert_element_type_29 = None
        convert_element_type_31: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg23_1, torch.bfloat16);  arg23_1 = None
        convert_element_type_32: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg24_1, torch.bfloat16);  arg24_1 = None
        permute_41: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_32, [1, 0]);  convert_element_type_32 = None
        mm_6: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_44, permute_41)
        view_47: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_6, _shape_param_72);  mm_6 = _shape_param_72 = None
        add_14: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_47, convert_element_type_31);  view_47 = convert_element_type_31 = None
        div_2: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_12, 8.0);  add_12 = None
        view_48: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_13, _shape_param_73);  add_13 = _shape_param_73 = None
        permute_42: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_48, [1, 0, 2, 3]);  view_48 = None
        permute_43: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_42, [0, 2, 1, 3]);  permute_42 = None
        view_49: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_43, _shape_param_74);  permute_43 = _shape_param_74 = None
        view_50: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_49, _shape_param_75);  view_49 = _shape_param_75 = None
        as_strided_5: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_50, _shape_param_76, _shape_param_77);  view_50 = _shape_param_76 = _shape_param_77 = None
        unsqueeze_14: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_5, 4);  as_strided_5 = None
        permute_44: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_14, [0, 1, 4, 2, 3]);  unsqueeze_14 = None
        view_51: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_2, _shape_param_78);  div_2 = _shape_param_78 = None
        permute_45: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_51, [1, 0, 2, 3]);  view_51 = None
        permute_46: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_45, [0, 2, 1, 3]);  permute_45 = None
        view_52: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_46, _shape_param_79);  permute_46 = _shape_param_79 = None
        view_53: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_52, _shape_param_80);  view_52 = _shape_param_80 = None
        as_strided_6: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_53, _shape_param_81, _shape_param_82);  view_53 = _shape_param_81 = _shape_param_82 = None
        unsqueeze_15: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_6, 4);  as_strided_6 = None
        clone_9: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_15, memory_format = torch.contiguous_format);  unsqueeze_15 = None
        view_54: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_9, _shape_param_83);  clone_9 = _shape_param_83 = None
        permute_47: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_44, [0, 1, 4, 3, 2]);  permute_44 = None
        clone_10: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_55: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_10, _shape_param_84);  clone_10 = _shape_param_84 = None
        bmm_2: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_54, view_55)
        view_56: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_2, _shape_param_85);  bmm_2 = _shape_param_85 = None
        permute_48: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_56, [0, 1, 2, 4, 3]);  view_56 = None
        view_57: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_48, _shape_param_86);  permute_48 = _shape_param_86 = None
        constant_pad_nd_4: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_57, [0, 0, 0, 1], 0.0);  view_57 = None
        view_58: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_4, _shape_param_87);  constant_pad_nd_4 = _shape_param_87 = None
        slice_41: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_58, 2, 0, 256)
        slice_42: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_41, 3, 0, 257);  slice_41 = None
        copy_12: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_42);  slice_42 = None
        slice_scatter_22: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_12, 3, 256, 9223372036854775807);  copy_12 = None
        slice_scatter_23: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_22, 1, 0, -1);  slice_scatter_22 = None
        select_8: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_58, 1, -1)
        slice_43: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_8, 1, 256, 9223372036854775807);  select_8 = None
        slice_44: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_43, 2, 0, 257);  slice_43 = None
        select_9: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_23, 1, -1)
        slice_45: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_9, 2, 256, 9223372036854775807)
        copy_13: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_45, slice_44);  slice_45 = slice_44 = None
        slice_scatter_24: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_9, copy_13, 2, 256, 9223372036854775807);  select_9 = copy_13 = None
        select_scatter_4: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_23, slice_scatter_24, 1, -1);  slice_scatter_23 = slice_scatter_24 = None
        slice_46: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_58, 2, -257, -1)
        slice_47: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_46, 3, 257, 9223372036854775807);  slice_46 = None
        slice_48: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_4, 1, 1, 9223372036854775807)
        slice_49: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_48, 3, 0, 256)
        copy_14: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_49, slice_47);  slice_49 = slice_47 = None
        slice_scatter_25: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_48, copy_14, 3, 0, 256);  slice_48 = copy_14 = None
        slice_scatter_26: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_4, slice_scatter_25, 1, 1, 9223372036854775807);  select_scatter_4 = slice_scatter_25 = None
        select_10: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_58, 1, 0);  view_58 = None
        slice_50: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_10, 1, 0, 255);  select_10 = None
        slice_51: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_50, 2, -255, 9223372036854775807);  slice_50 = None
        select_11: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_26, 1, 0)
        slice_52: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_11, 1, 1, 256)
        slice_53: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_52, 2, 1, 256)
        copy_15: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_53, slice_51);  slice_53 = slice_51 = None
        slice_scatter_27: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_52, copy_15, 2, 1, 256);  slice_52 = copy_15 = None
        slice_scatter_28: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_11, slice_scatter_27, 1, 1, 256);  select_11 = slice_scatter_27 = None
        select_scatter_5: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_26, slice_scatter_28, 1, 0);  slice_scatter_26 = slice_scatter_28 = None
        view_59: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_5, _shape_param_88);  select_scatter_5 = _shape_param_88 = None
        permute_49: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
        slice_54: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_49, 1, 0, 256)
        slice_55: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_54, 3, 0, 257)
        where_7: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_7, permute_12, slice_55)
        copy_16: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_55, where_7);  slice_55 = where_7 = None
        slice_scatter_29: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_54, copy_16, 3, 0, 257);  slice_54 = copy_16 = None
        slice_scatter_30: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_49, slice_scatter_29, 1, 0, 256);  permute_49 = slice_scatter_29 = None
        permute_50: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_30, [0, 2, 1, 3]);  slice_scatter_30 = None
        view_60: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_50, _shape_param_89);  permute_50 = _shape_param_89 = None
        view_61: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_60, _shape_param_90);  view_60 = _shape_param_90 = None
        permute_51: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
        slice_56: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_51, 1, -256, 9223372036854775807)
        slice_57: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_56, 3, -257, 9223372036854775807)
        where_8: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_8, permute_12, slice_57)
        copy_17: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_57, where_8);  slice_57 = where_8 = None
        slice_scatter_31: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_56, copy_17, 3, -257, 9223372036854775807);  slice_56 = copy_17 = None
        slice_scatter_32: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_51, slice_scatter_31, 1, -256, 9223372036854775807);  permute_51 = slice_scatter_31 = None
        permute_52: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_32, [0, 2, 1, 3]);  slice_scatter_32 = None
        permute_53: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_52, [0, 2, 1, 3]);  permute_52 = None
        add_15: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_53, permute_25);  permute_53 = None
        permute_54: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_15, [0, 2, 1, 3]);  add_15 = None
        permute_55: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_54, [0, 2, 1, 3]);  permute_54 = None
        convert_element_type_33: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_55, torch.float32)
        clone_11: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_33, memory_format = torch.contiguous_format);  convert_element_type_33 = None
        amax_1: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_11, [-1], True)
        sub_4: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_11, amax_1);  clone_11 = None
        exp_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None
        where_9: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_11, full_7, div_3);  div_3 = None
        convert_element_type_34: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_9, torch.bfloat16);  where_9 = None
        inductor_lookup_seed_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 3)
        inductor_random_3: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default(_shape_param_91, inductor_lookup_seed_3, 'rand');  _shape_param_91 = inductor_lookup_seed_3 = None
        convert_element_type_35: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(inductor_random_3, torch.bfloat16);  inductor_random_3 = None
        gt_3: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(convert_element_type_35, 0.1);  convert_element_type_35 = None
        mul_14: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_3, convert_element_type_34);  convert_element_type_34 = None
        mul_15: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_14, 1.1111111111111112);  mul_14 = None
        view_62: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_14, _shape_param_92);  add_14 = _shape_param_92 = None
        permute_56: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_62, [1, 0, 2, 3]);  view_62 = None
        permute_57: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_15, [0, 2, 1, 3]);  mul_15 = None
        clone_12: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
        view_63: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_12, _shape_param_93);  clone_12 = _shape_param_93 = None
        permute_58: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_56, [0, 2, 1, 3]);  permute_56 = None
        view_64: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_58, _shape_param_94);  permute_58 = _shape_param_94 = None
        constant_pad_nd_5: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_64, _shape_param_95, -1.0);  view_64 = _shape_param_95 = None
        as_strided_7: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_5, _shape_param_96, _shape_param_97);  constant_pad_nd_5 = _shape_param_96 = _shape_param_97 = None
        constant_pad_nd_6: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_63, _shape_param_98, 0.0);  view_63 = _shape_param_98 = None
        view_65: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_6, _shape_param_99);  constant_pad_nd_6 = _shape_param_99 = None
        slice_58: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_65, 2, 0, -256);  view_65 = None
        view_66: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_58, _shape_param_100);  slice_58 = _shape_param_100 = None
        slice_59: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_66, 3, 0, -1);  view_66 = None
        unsqueeze_16: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_59, 4);  slice_59 = None
        unsqueeze_17: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_7, 4);  as_strided_7 = None
        view_67: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_16, _shape_param_101);  unsqueeze_16 = _shape_param_101 = None
        clone_13: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_17, memory_format = torch.contiguous_format);  unsqueeze_17 = None
        view_68: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_13, _shape_param_102);  clone_13 = _shape_param_102 = None
        bmm_3: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_67, view_68)
        view_69: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_103);  bmm_3 = _shape_param_103 = None
        permute_59: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_69, [0, 1, 2, 4, 3]);  view_69 = None
        view_70: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_59, _shape_param_104);  permute_59 = _shape_param_104 = None
        view_71: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_70, _shape_param_105);  view_70 = _shape_param_105 = None
        permute_60: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        permute_61: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_60, [1, 0, 2, 3]);  permute_60 = None
        clone_14: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
        view_72: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_14, _shape_param_106);  clone_14 = _shape_param_106 = None
        permute_62: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_72, [1, 0, 2]);  view_72 = None
        convert_element_type_36: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg25_1, torch.bfloat16);  arg25_1 = None
        convert_element_type_37: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg26_1, torch.bfloat16);  arg26_1 = None
        permute_63: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_37, [1, 0]);  convert_element_type_37 = None
        clone_15: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_73: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_15, _shape_param_107);  clone_15 = _shape_param_107 = None
        mm_7: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_73, permute_63)
        view_74: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_7, _shape_param_108);  mm_7 = _shape_param_108 = None
        add_16: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_74, convert_element_type_36);  view_74 = convert_element_type_36 = None
        inductor_lookup_seed_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 4)
        inductor_random_4: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_109, inductor_lookup_seed_4, 'rand');  _shape_param_109 = inductor_lookup_seed_4 = None
        convert_element_type_38: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_4, torch.bfloat16);  inductor_random_4 = None
        gt_4: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_38, 0.1);  convert_element_type_38 = None
        mul_16: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_4, add_16);  add_16 = None
        mul_17: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_16, 1.1111111111111112);  mul_16 = None
        add_17: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_17, add_11);  mul_17 = add_11 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_4: "f32[8, 1024, 1]" = var_mean_2[0]
        getitem_5: "f32[8, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
        add_18: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_5: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_5);  add_17 = getitem_5 = None
        mul_18: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_2);  sub_5 = None
        mul_19: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg27_1);  arg27_1 = None
        add_19: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_19, arg28_1);  mul_19 = arg28_1 = None
        convert_element_type_39: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg29_1, torch.bfloat16);  arg29_1 = None
        convert_element_type_40: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg30_1, torch.bfloat16);  arg30_1 = None
        convert_element_type_41: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16)
        view_75: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_41, _shape_param_110);  convert_element_type_41 = _shape_param_110 = None
        permute_64: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_40, [1, 0]);  convert_element_type_40 = None
        addmm_2: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(convert_element_type_39, view_75, permute_64);  convert_element_type_39 = None
        view_76: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_2, _shape_param_111);  _shape_param_111 = None
        convert_element_type_42: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_76, torch.float32);  view_76 = None
        mul_20: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 0.5)
        mul_21: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 0.7071067811865476);  convert_element_type_42 = None
        erf_1: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_21);  mul_21 = None
        add_20: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_22: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_20, add_20);  mul_20 = add_20 = None
        convert_element_type_43: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None
        convert_element_type_44: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg31_1, torch.bfloat16);  arg31_1 = None
        convert_element_type_45: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg32_1, torch.bfloat16);  arg32_1 = None
        view_77: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_43, _shape_param_112);  convert_element_type_43 = _shape_param_112 = None
        permute_65: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_45, [1, 0]);  convert_element_type_45 = None
        addmm_3: "bf16[8192, 768]" = torch.ops.aten.addmm.default(convert_element_type_44, view_77, permute_65);  convert_element_type_44 = None
        view_78: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_3, _shape_param_113);  addmm_3 = _shape_param_113 = None
        inductor_lookup_seed_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 5)
        inductor_random_5: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_114, inductor_lookup_seed_5, 'rand');  _shape_param_114 = inductor_lookup_seed_5 = None
        convert_element_type_46: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_5, torch.bfloat16);  inductor_random_5 = None
        gt_5: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_46, 0.1);  convert_element_type_46 = None
        mul_23: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_5, view_78);  view_78 = None
        mul_24: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_23, 1.1111111111111112);  mul_23 = None
        add_21: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_24, add_19);  mul_24 = add_19 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_6: "f32[8, 1024, 1]" = var_mean_3[0]
        getitem_7: "f32[8, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
        add_22: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_6: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_7);  add_21 = getitem_7 = None
        mul_25: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
        mul_26: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_25, arg33_1);  arg33_1 = None
        add_23: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_26, arg34_1);  mul_26 = arg34_1 = None
        permute_66: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_23, [1, 0, 2])
        convert_element_type_47: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg35_1, torch.bfloat16);  arg35_1 = None
        convert_element_type_48: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg36_1, torch.bfloat16);  arg36_1 = None
        convert_element_type_49: "bf16[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(permute_66, torch.bfloat16);  permute_66 = None
        permute_67: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_48, [1, 0]);  convert_element_type_48 = None
        clone_16: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(convert_element_type_49, memory_format = torch.contiguous_format);  convert_element_type_49 = None
        view_79: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_16, _shape_param_115);  clone_16 = _shape_param_115 = None
        mm_8: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_79, permute_67)
        view_80: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_8, _shape_param_116);  mm_8 = _shape_param_116 = None
        add_24: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_80, convert_element_type_47);  view_80 = convert_element_type_47 = None
        convert_element_type_50: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg37_1, torch.bfloat16);  arg37_1 = None
        convert_element_type_51: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg38_1, torch.bfloat16);  arg38_1 = None
        permute_68: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_51, [1, 0]);  convert_element_type_51 = None
        mm_9: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_79, permute_68)
        view_81: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_9, _shape_param_117);  mm_9 = _shape_param_117 = None
        add_25: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_81, convert_element_type_50);  view_81 = convert_element_type_50 = None
        convert_element_type_52: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg39_1, torch.bfloat16);  arg39_1 = None
        convert_element_type_53: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg40_1, torch.bfloat16);  arg40_1 = None
        permute_69: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_53, [1, 0]);  convert_element_type_53 = None
        mm_10: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_79, permute_69)
        view_82: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_10, _shape_param_118);  mm_10 = _shape_param_118 = None
        add_26: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_82, convert_element_type_52);  view_82 = convert_element_type_52 = None
        div_4: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_24, 8.0);  add_24 = None
        view_83: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_25, _shape_param_119);  add_25 = _shape_param_119 = None
        permute_70: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_83, [1, 0, 2, 3]);  view_83 = None
        permute_71: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_70, [0, 2, 1, 3]);  permute_70 = None
        view_84: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_71, _shape_param_120);  permute_71 = _shape_param_120 = None
        view_85: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_84, _shape_param_121);  view_84 = _shape_param_121 = None
        as_strided_8: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_85, _shape_param_122, _shape_param_123);  view_85 = _shape_param_122 = _shape_param_123 = None
        unsqueeze_18: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_8, 4);  as_strided_8 = None
        permute_72: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_18, [0, 1, 4, 2, 3]);  unsqueeze_18 = None
        view_86: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_4, _shape_param_124);  div_4 = _shape_param_124 = None
        permute_73: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_86, [1, 0, 2, 3]);  view_86 = None
        permute_74: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_73, [0, 2, 1, 3]);  permute_73 = None
        view_87: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_74, _shape_param_125);  permute_74 = _shape_param_125 = None
        view_88: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_87, _shape_param_126);  view_87 = _shape_param_126 = None
        as_strided_9: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_88, _shape_param_127, _shape_param_128);  view_88 = _shape_param_127 = _shape_param_128 = None
        unsqueeze_19: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_9, 4);  as_strided_9 = None
        clone_17: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_19, memory_format = torch.contiguous_format);  unsqueeze_19 = None
        view_89: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_17, _shape_param_129);  clone_17 = _shape_param_129 = None
        permute_75: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_72, [0, 1, 4, 3, 2]);  permute_72 = None
        clone_18: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
        view_90: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_18, _shape_param_130);  clone_18 = _shape_param_130 = None
        bmm_4: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_89, view_90)
        view_91: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_4, _shape_param_131);  bmm_4 = _shape_param_131 = None
        permute_76: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_91, [0, 1, 2, 4, 3]);  view_91 = None
        view_92: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_76, _shape_param_132);  permute_76 = _shape_param_132 = None
        constant_pad_nd_7: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_92, [0, 0, 0, 1], 0.0);  view_92 = None
        view_93: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_7, _shape_param_133);  constant_pad_nd_7 = _shape_param_133 = None
        slice_60: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_93, 2, 0, 256)
        slice_61: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_60, 3, 0, 257);  slice_60 = None
        copy_18: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_61);  slice_61 = None
        slice_scatter_33: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_18, 3, 256, 9223372036854775807);  copy_18 = None
        slice_scatter_34: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_33, 1, 0, -1);  slice_scatter_33 = None
        select_12: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_93, 1, -1)
        slice_62: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_12, 1, 256, 9223372036854775807);  select_12 = None
        slice_63: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_62, 2, 0, 257);  slice_62 = None
        select_13: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_34, 1, -1)
        slice_64: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_13, 2, 256, 9223372036854775807)
        copy_19: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_64, slice_63);  slice_64 = slice_63 = None
        slice_scatter_35: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_13, copy_19, 2, 256, 9223372036854775807);  select_13 = copy_19 = None
        select_scatter_6: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_34, slice_scatter_35, 1, -1);  slice_scatter_34 = slice_scatter_35 = None
        slice_65: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_93, 2, -257, -1)
        slice_66: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_65, 3, 257, 9223372036854775807);  slice_65 = None
        slice_67: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_6, 1, 1, 9223372036854775807)
        slice_68: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_67, 3, 0, 256)
        copy_20: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_68, slice_66);  slice_68 = slice_66 = None
        slice_scatter_36: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_67, copy_20, 3, 0, 256);  slice_67 = copy_20 = None
        slice_scatter_37: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_6, slice_scatter_36, 1, 1, 9223372036854775807);  select_scatter_6 = slice_scatter_36 = None
        select_14: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_93, 1, 0);  view_93 = None
        slice_69: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_14, 1, 0, 255);  select_14 = None
        slice_70: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_69, 2, -255, 9223372036854775807);  slice_69 = None
        select_15: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_37, 1, 0)
        slice_71: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_15, 1, 1, 256)
        slice_72: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_71, 2, 1, 256)
        copy_21: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_72, slice_70);  slice_72 = slice_70 = None
        slice_scatter_38: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_71, copy_21, 2, 1, 256);  slice_71 = copy_21 = None
        slice_scatter_39: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_15, slice_scatter_38, 1, 1, 256);  select_15 = slice_scatter_38 = None
        select_scatter_7: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_37, slice_scatter_39, 1, 0);  slice_scatter_37 = slice_scatter_39 = None
        view_94: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_7, _shape_param_134);  select_scatter_7 = _shape_param_134 = None
        permute_77: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        slice_73: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_77, 1, 0, 256)
        slice_74: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_73, 3, 0, 257)
        where_10: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_7, permute_12, slice_74)
        copy_22: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_74, where_10);  slice_74 = where_10 = None
        slice_scatter_40: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_73, copy_22, 3, 0, 257);  slice_73 = copy_22 = None
        slice_scatter_41: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_77, slice_scatter_40, 1, 0, 256);  permute_77 = slice_scatter_40 = None
        permute_78: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_41, [0, 2, 1, 3]);  slice_scatter_41 = None
        view_95: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_78, _shape_param_135);  permute_78 = _shape_param_135 = None
        view_96: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_95, _shape_param_136);  view_95 = _shape_param_136 = None
        permute_79: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        slice_75: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_79, 1, -256, 9223372036854775807)
        slice_76: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_75, 3, -257, 9223372036854775807)
        where_11: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_8, permute_12, slice_76)
        copy_23: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_76, where_11);  slice_76 = where_11 = None
        slice_scatter_42: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_75, copy_23, 3, -257, 9223372036854775807);  slice_75 = copy_23 = None
        slice_scatter_43: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_79, slice_scatter_42, 1, -256, 9223372036854775807);  permute_79 = slice_scatter_42 = None
        permute_80: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_43, [0, 2, 1, 3]);  slice_scatter_43 = None
        permute_81: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_80, [0, 2, 1, 3]);  permute_80 = None
        add_27: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_81, permute_25);  permute_81 = None
        permute_82: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_27, [0, 2, 1, 3]);  add_27 = None
        permute_83: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_82, [0, 2, 1, 3]);  permute_82 = None
        convert_element_type_54: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_83, torch.float32)
        clone_19: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_54, memory_format = torch.contiguous_format);  convert_element_type_54 = None
        amax_2: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_19, [-1], True)
        sub_7: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_19, amax_2);  clone_19 = None
        exp_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_5: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = None
        where_12: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_11, full_7, div_5);  div_5 = None
        convert_element_type_55: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_12, torch.bfloat16);  where_12 = None
        inductor_lookup_seed_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 6)
        inductor_random_6: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default(_shape_param_137, inductor_lookup_seed_6, 'rand');  _shape_param_137 = inductor_lookup_seed_6 = None
        convert_element_type_56: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(inductor_random_6, torch.bfloat16);  inductor_random_6 = None
        gt_6: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(convert_element_type_56, 0.1);  convert_element_type_56 = None
        mul_27: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_6, convert_element_type_55);  convert_element_type_55 = None
        mul_28: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_27, 1.1111111111111112);  mul_27 = None
        view_97: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_26, _shape_param_138);  add_26 = _shape_param_138 = None
        permute_84: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_97, [1, 0, 2, 3]);  view_97 = None
        permute_85: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_28, [0, 2, 1, 3]);  mul_28 = None
        clone_20: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
        view_98: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_20, _shape_param_139);  clone_20 = _shape_param_139 = None
        permute_86: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_84, [0, 2, 1, 3]);  permute_84 = None
        view_99: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_86, _shape_param_140);  permute_86 = _shape_param_140 = None
        constant_pad_nd_8: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_99, _shape_param_141, -1.0);  view_99 = _shape_param_141 = None
        as_strided_10: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_8, _shape_param_142, _shape_param_143);  constant_pad_nd_8 = _shape_param_142 = _shape_param_143 = None
        constant_pad_nd_9: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_98, _shape_param_144, 0.0);  view_98 = _shape_param_144 = None
        view_100: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_9, _shape_param_145);  constant_pad_nd_9 = _shape_param_145 = None
        slice_77: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_100, 2, 0, -256);  view_100 = None
        view_101: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_77, _shape_param_146);  slice_77 = _shape_param_146 = None
        slice_78: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_101, 3, 0, -1);  view_101 = None
        unsqueeze_20: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_78, 4);  slice_78 = None
        unsqueeze_21: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_10, 4);  as_strided_10 = None
        view_102: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_20, _shape_param_147);  unsqueeze_20 = _shape_param_147 = None
        clone_21: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_21, memory_format = torch.contiguous_format);  unsqueeze_21 = None
        view_103: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_21, _shape_param_148);  clone_21 = _shape_param_148 = None
        bmm_5: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_102, view_103)
        view_104: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_5, _shape_param_149);  bmm_5 = _shape_param_149 = None
        permute_87: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_104, [0, 1, 2, 4, 3]);  view_104 = None
        view_105: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_87, _shape_param_150);  permute_87 = _shape_param_150 = None
        view_106: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_105, _shape_param_151);  view_105 = _shape_param_151 = None
        permute_88: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        permute_89: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_88, [1, 0, 2, 3]);  permute_88 = None
        clone_22: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
        view_107: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_22, _shape_param_152);  clone_22 = _shape_param_152 = None
        permute_90: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_107, [1, 0, 2]);  view_107 = None
        convert_element_type_57: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg41_1, torch.bfloat16);  arg41_1 = None
        convert_element_type_58: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg42_1, torch.bfloat16);  arg42_1 = None
        permute_91: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_58, [1, 0]);  convert_element_type_58 = None
        clone_23: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        view_108: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_23, _shape_param_153);  clone_23 = _shape_param_153 = None
        mm_11: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_108, permute_91)
        view_109: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_11, _shape_param_154);  mm_11 = _shape_param_154 = None
        add_28: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_109, convert_element_type_57);  view_109 = convert_element_type_57 = None
        inductor_lookup_seed_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 7)
        inductor_random_7: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_155, inductor_lookup_seed_7, 'rand');  _shape_param_155 = inductor_lookup_seed_7 = None
        convert_element_type_59: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_7, torch.bfloat16);  inductor_random_7 = None
        gt_7: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_59, 0.1);  convert_element_type_59 = None
        mul_29: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_7, add_28);  add_28 = None
        mul_30: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_29, 1.1111111111111112);  mul_29 = None
        add_29: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_30, add_23);  mul_30 = add_23 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_8: "f32[8, 1024, 1]" = var_mean_4[0]
        getitem_9: "f32[8, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
        add_30: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_8: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_29, getitem_9);  add_29 = getitem_9 = None
        mul_31: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_4);  sub_8 = None
        mul_32: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_31, arg43_1);  arg43_1 = None
        add_31: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_32, arg44_1);  mul_32 = arg44_1 = None
        convert_element_type_60: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg45_1, torch.bfloat16);  arg45_1 = None
        convert_element_type_61: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg46_1, torch.bfloat16);  arg46_1 = None
        convert_element_type_62: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16)
        view_110: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_62, _shape_param_156);  convert_element_type_62 = _shape_param_156 = None
        permute_92: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_61, [1, 0]);  convert_element_type_61 = None
        addmm_4: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(convert_element_type_60, view_110, permute_92);  convert_element_type_60 = None
        view_111: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_4, _shape_param_157);  _shape_param_157 = None
        convert_element_type_63: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_111, torch.float32);  view_111 = None
        mul_33: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 0.5)
        mul_34: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 0.7071067811865476);  convert_element_type_63 = None
        erf_2: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_32: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_35: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_33, add_32);  mul_33 = add_32 = None
        convert_element_type_64: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None
        convert_element_type_65: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg47_1, torch.bfloat16);  arg47_1 = None
        convert_element_type_66: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg48_1, torch.bfloat16);  arg48_1 = None
        view_112: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_64, _shape_param_158);  convert_element_type_64 = _shape_param_158 = None
        permute_93: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_66, [1, 0]);  convert_element_type_66 = None
        addmm_5: "bf16[8192, 768]" = torch.ops.aten.addmm.default(convert_element_type_65, view_112, permute_93);  convert_element_type_65 = None
        view_113: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_5, _shape_param_159);  addmm_5 = _shape_param_159 = None
        inductor_lookup_seed_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 8)
        inductor_random_8: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_160, inductor_lookup_seed_8, 'rand');  _shape_param_160 = inductor_lookup_seed_8 = None
        convert_element_type_67: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_8, torch.bfloat16);  inductor_random_8 = None
        gt_8: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_67, 0.1);  convert_element_type_67 = None
        mul_36: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_8, view_113);  view_113 = None
        mul_37: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_36, 1.1111111111111112);  mul_36 = None
        add_33: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_37, add_31);  mul_37 = add_31 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_10: "f32[8, 1024, 1]" = var_mean_5[0]
        getitem_11: "f32[8, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
        add_34: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_9: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_33, getitem_11);  add_33 = getitem_11 = None
        mul_38: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = None
        mul_39: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_38, arg49_1);  arg49_1 = None
        add_35: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_39, arg50_1);  mul_39 = arg50_1 = None
        permute_94: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_35, [1, 0, 2])
        convert_element_type_68: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg51_1, torch.bfloat16);  arg51_1 = None
        convert_element_type_69: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg52_1, torch.bfloat16);  arg52_1 = None
        convert_element_type_70: "bf16[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(permute_94, torch.bfloat16);  permute_94 = None
        permute_95: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_69, [1, 0]);  convert_element_type_69 = None
        clone_24: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(convert_element_type_70, memory_format = torch.contiguous_format);  convert_element_type_70 = None
        view_114: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_24, _shape_param_161);  clone_24 = _shape_param_161 = None
        mm_12: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_114, permute_95)
        view_115: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_12, _shape_param_162);  mm_12 = _shape_param_162 = None
        add_36: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_115, convert_element_type_68);  view_115 = convert_element_type_68 = None
        convert_element_type_71: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg53_1, torch.bfloat16);  arg53_1 = None
        convert_element_type_72: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg54_1, torch.bfloat16);  arg54_1 = None
        permute_96: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_72, [1, 0]);  convert_element_type_72 = None
        mm_13: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_114, permute_96)
        view_116: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_13, _shape_param_163);  mm_13 = _shape_param_163 = None
        add_37: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_116, convert_element_type_71);  view_116 = convert_element_type_71 = None
        convert_element_type_73: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg55_1, torch.bfloat16);  arg55_1 = None
        convert_element_type_74: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg56_1, torch.bfloat16);  arg56_1 = None
        permute_97: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_74, [1, 0]);  convert_element_type_74 = None
        mm_14: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_114, permute_97)
        view_117: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_14, _shape_param_164);  mm_14 = _shape_param_164 = None
        add_38: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_117, convert_element_type_73);  view_117 = convert_element_type_73 = None
        div_6: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_36, 8.0);  add_36 = None
        view_118: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_37, _shape_param_165);  add_37 = _shape_param_165 = None
        permute_98: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_118, [1, 0, 2, 3]);  view_118 = None
        permute_99: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_98, [0, 2, 1, 3]);  permute_98 = None
        view_119: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_99, _shape_param_166);  permute_99 = _shape_param_166 = None
        view_120: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_119, _shape_param_167);  view_119 = _shape_param_167 = None
        as_strided_11: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_120, _shape_param_168, _shape_param_169);  view_120 = _shape_param_168 = _shape_param_169 = None
        unsqueeze_22: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_11, 4);  as_strided_11 = None
        permute_100: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_22, [0, 1, 4, 2, 3]);  unsqueeze_22 = None
        view_121: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_6, _shape_param_170);  div_6 = _shape_param_170 = None
        permute_101: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_121, [1, 0, 2, 3]);  view_121 = None
        permute_102: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_101, [0, 2, 1, 3]);  permute_101 = None
        view_122: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_102, _shape_param_171);  permute_102 = _shape_param_171 = None
        view_123: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_122, _shape_param_172);  view_122 = _shape_param_172 = None
        as_strided_12: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_123, _shape_param_173, _shape_param_174);  view_123 = _shape_param_173 = _shape_param_174 = None
        unsqueeze_23: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_12, 4);  as_strided_12 = None
        clone_25: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_23, memory_format = torch.contiguous_format);  unsqueeze_23 = None
        view_124: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_25, _shape_param_175);  clone_25 = _shape_param_175 = None
        permute_103: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_100, [0, 1, 4, 3, 2]);  permute_100 = None
        clone_26: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
        view_125: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_26, _shape_param_176);  clone_26 = _shape_param_176 = None
        bmm_6: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_124, view_125)
        view_126: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_6, _shape_param_177);  bmm_6 = _shape_param_177 = None
        permute_104: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_126, [0, 1, 2, 4, 3]);  view_126 = None
        view_127: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_104, _shape_param_178);  permute_104 = _shape_param_178 = None
        constant_pad_nd_10: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_127, [0, 0, 0, 1], 0.0);  view_127 = None
        view_128: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_10, _shape_param_179);  constant_pad_nd_10 = _shape_param_179 = None
        slice_79: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_128, 2, 0, 256)
        slice_80: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_79, 3, 0, 257);  slice_79 = None
        copy_24: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_80);  slice_80 = None
        slice_scatter_44: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_24, 3, 256, 9223372036854775807);  copy_24 = None
        slice_scatter_45: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_44, 1, 0, -1);  slice_scatter_44 = None
        select_16: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_128, 1, -1)
        slice_81: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_16, 1, 256, 9223372036854775807);  select_16 = None
        slice_82: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_81, 2, 0, 257);  slice_81 = None
        select_17: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_45, 1, -1)
        slice_83: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_17, 2, 256, 9223372036854775807)
        copy_25: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_83, slice_82);  slice_83 = slice_82 = None
        slice_scatter_46: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_17, copy_25, 2, 256, 9223372036854775807);  select_17 = copy_25 = None
        select_scatter_8: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_45, slice_scatter_46, 1, -1);  slice_scatter_45 = slice_scatter_46 = None
        slice_84: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_128, 2, -257, -1)
        slice_85: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_84, 3, 257, 9223372036854775807);  slice_84 = None
        slice_86: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_8, 1, 1, 9223372036854775807)
        slice_87: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_86, 3, 0, 256)
        copy_26: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_87, slice_85);  slice_87 = slice_85 = None
        slice_scatter_47: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_86, copy_26, 3, 0, 256);  slice_86 = copy_26 = None
        slice_scatter_48: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_8, slice_scatter_47, 1, 1, 9223372036854775807);  select_scatter_8 = slice_scatter_47 = None
        select_18: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_128, 1, 0);  view_128 = None
        slice_88: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_18, 1, 0, 255);  select_18 = None
        slice_89: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_88, 2, -255, 9223372036854775807);  slice_88 = None
        select_19: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_48, 1, 0)
        slice_90: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_19, 1, 1, 256)
        slice_91: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_90, 2, 1, 256)
        copy_27: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_91, slice_89);  slice_91 = slice_89 = None
        slice_scatter_49: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_90, copy_27, 2, 1, 256);  slice_90 = copy_27 = None
        slice_scatter_50: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_19, slice_scatter_49, 1, 1, 256);  select_19 = slice_scatter_49 = None
        select_scatter_9: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_48, slice_scatter_50, 1, 0);  slice_scatter_48 = slice_scatter_50 = None
        view_129: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_9, _shape_param_180);  select_scatter_9 = _shape_param_180 = None
        permute_105: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        slice_92: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_105, 1, 0, 256)
        slice_93: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_92, 3, 0, 257)
        where_13: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_7, permute_12, slice_93)
        copy_28: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_93, where_13);  slice_93 = where_13 = None
        slice_scatter_51: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_92, copy_28, 3, 0, 257);  slice_92 = copy_28 = None
        slice_scatter_52: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_105, slice_scatter_51, 1, 0, 256);  permute_105 = slice_scatter_51 = None
        permute_106: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_52, [0, 2, 1, 3]);  slice_scatter_52 = None
        view_130: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_106, _shape_param_181);  permute_106 = _shape_param_181 = None
        view_131: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_130, _shape_param_182);  view_130 = _shape_param_182 = None
        permute_107: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
        slice_94: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_107, 1, -256, 9223372036854775807)
        slice_95: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_94, 3, -257, 9223372036854775807)
        where_14: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_8, permute_12, slice_95)
        copy_29: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_95, where_14);  slice_95 = where_14 = None
        slice_scatter_53: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_94, copy_29, 3, -257, 9223372036854775807);  slice_94 = copy_29 = None
        slice_scatter_54: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_107, slice_scatter_53, 1, -256, 9223372036854775807);  permute_107 = slice_scatter_53 = None
        permute_108: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_54, [0, 2, 1, 3]);  slice_scatter_54 = None
        permute_109: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_108, [0, 2, 1, 3]);  permute_108 = None
        add_39: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_109, permute_25);  permute_109 = None
        permute_110: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_39, [0, 2, 1, 3]);  add_39 = None
        permute_111: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_110, [0, 2, 1, 3]);  permute_110 = None
        convert_element_type_75: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_111, torch.float32)
        clone_27: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_75, memory_format = torch.contiguous_format);  convert_element_type_75 = None
        amax_3: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_27, [-1], True)
        sub_10: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_27, amax_3);  clone_27 = None
        exp_3: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = None
        where_15: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_11, full_7, div_7);  div_7 = None
        convert_element_type_76: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_15, torch.bfloat16);  where_15 = None
        inductor_lookup_seed_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 9)
        inductor_random_9: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default(_shape_param_183, inductor_lookup_seed_9, 'rand');  _shape_param_183 = inductor_lookup_seed_9 = None
        convert_element_type_77: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(inductor_random_9, torch.bfloat16);  inductor_random_9 = None
        gt_9: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(convert_element_type_77, 0.1);  convert_element_type_77 = None
        mul_40: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_9, convert_element_type_76);  convert_element_type_76 = None
        mul_41: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_40, 1.1111111111111112);  mul_40 = None
        view_132: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_38, _shape_param_184);  add_38 = _shape_param_184 = None
        permute_112: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_132, [1, 0, 2, 3]);  view_132 = None
        permute_113: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_41, [0, 2, 1, 3]);  mul_41 = None
        clone_28: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
        view_133: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_28, _shape_param_185);  clone_28 = _shape_param_185 = None
        permute_114: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_112, [0, 2, 1, 3]);  permute_112 = None
        view_134: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_114, _shape_param_186);  permute_114 = _shape_param_186 = None
        constant_pad_nd_11: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_134, _shape_param_187, -1.0);  view_134 = _shape_param_187 = None
        as_strided_13: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_11, _shape_param_188, _shape_param_189);  constant_pad_nd_11 = _shape_param_188 = _shape_param_189 = None
        constant_pad_nd_12: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_133, _shape_param_190, 0.0);  view_133 = _shape_param_190 = None
        view_135: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_12, _shape_param_191);  constant_pad_nd_12 = _shape_param_191 = None
        slice_96: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_135, 2, 0, -256);  view_135 = None
        view_136: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_96, _shape_param_192);  slice_96 = _shape_param_192 = None
        slice_97: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_136, 3, 0, -1);  view_136 = None
        unsqueeze_24: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_97, 4);  slice_97 = None
        unsqueeze_25: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_13, 4);  as_strided_13 = None
        view_137: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_24, _shape_param_193);  unsqueeze_24 = _shape_param_193 = None
        clone_29: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_25, memory_format = torch.contiguous_format);  unsqueeze_25 = None
        view_138: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_29, _shape_param_194);  clone_29 = _shape_param_194 = None
        bmm_7: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_137, view_138)
        view_139: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_195);  bmm_7 = _shape_param_195 = None
        permute_115: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_139, [0, 1, 2, 4, 3]);  view_139 = None
        view_140: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_115, _shape_param_196);  permute_115 = _shape_param_196 = None
        view_141: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_140, _shape_param_197);  view_140 = _shape_param_197 = None
        permute_116: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
        permute_117: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_116, [1, 0, 2, 3]);  permute_116 = None
        clone_30: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_142: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_30, _shape_param_198);  clone_30 = _shape_param_198 = None
        permute_118: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_142, [1, 0, 2]);  view_142 = None
        convert_element_type_78: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg57_1, torch.bfloat16);  arg57_1 = None
        convert_element_type_79: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg58_1, torch.bfloat16);  arg58_1 = None
        permute_119: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_79, [1, 0]);  convert_element_type_79 = None
        clone_31: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
        view_143: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_31, _shape_param_199);  clone_31 = _shape_param_199 = None
        mm_15: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_143, permute_119)
        view_144: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_15, _shape_param_200);  mm_15 = _shape_param_200 = None
        add_40: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_144, convert_element_type_78);  view_144 = convert_element_type_78 = None
        inductor_lookup_seed_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 10)
        inductor_random_10: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_201, inductor_lookup_seed_10, 'rand');  _shape_param_201 = inductor_lookup_seed_10 = None
        convert_element_type_80: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_10, torch.bfloat16);  inductor_random_10 = None
        gt_10: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_80, 0.1);  convert_element_type_80 = None
        mul_42: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_10, add_40);  add_40 = None
        mul_43: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_42, 1.1111111111111112);  mul_42 = None
        add_41: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_43, add_35);  mul_43 = add_35 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_12: "f32[8, 1024, 1]" = var_mean_6[0]
        getitem_13: "f32[8, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
        add_42: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_11: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_13);  add_41 = getitem_13 = None
        mul_44: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_6);  sub_11 = None
        mul_45: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_44, arg59_1);  arg59_1 = None
        add_43: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_45, arg60_1);  mul_45 = arg60_1 = None
        convert_element_type_81: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg61_1, torch.bfloat16);  arg61_1 = None
        convert_element_type_82: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg62_1, torch.bfloat16);  arg62_1 = None
        convert_element_type_83: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16)
        view_145: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_83, _shape_param_202);  convert_element_type_83 = _shape_param_202 = None
        permute_120: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_82, [1, 0]);  convert_element_type_82 = None
        addmm_6: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(convert_element_type_81, view_145, permute_120);  convert_element_type_81 = None
        view_146: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_6, _shape_param_203);  _shape_param_203 = None
        convert_element_type_84: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_146, torch.float32);  view_146 = None
        mul_46: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_84, 0.5)
        mul_47: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_84, 0.7071067811865476);  convert_element_type_84 = None
        erf_3: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_44: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_48: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_44);  mul_46 = add_44 = None
        convert_element_type_85: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16);  mul_48 = None
        convert_element_type_86: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg63_1, torch.bfloat16);  arg63_1 = None
        convert_element_type_87: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg64_1, torch.bfloat16);  arg64_1 = None
        view_147: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_85, _shape_param_204);  convert_element_type_85 = _shape_param_204 = None
        permute_121: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_87, [1, 0]);  convert_element_type_87 = None
        addmm_7: "bf16[8192, 768]" = torch.ops.aten.addmm.default(convert_element_type_86, view_147, permute_121);  convert_element_type_86 = None
        view_148: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_7, _shape_param_205);  addmm_7 = _shape_param_205 = None
        inductor_lookup_seed_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 11)
        inductor_random_11: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_206, inductor_lookup_seed_11, 'rand');  _shape_param_206 = inductor_lookup_seed_11 = None
        convert_element_type_88: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_11, torch.bfloat16);  inductor_random_11 = None
        gt_11: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_88, 0.1);  convert_element_type_88 = None
        mul_49: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_11, view_148);  view_148 = None
        mul_50: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_49, 1.1111111111111112);  mul_49 = None
        add_45: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_50, add_43);  mul_50 = add_43 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_14: "f32[8, 1024, 1]" = var_mean_7[0]
        getitem_15: "f32[8, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
        add_46: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_12: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_15);  add_45 = getitem_15 = None
        mul_51: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = None
        mul_52: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_51, arg65_1);  arg65_1 = None
        add_47: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_52, arg66_1);  mul_52 = arg66_1 = None
        permute_122: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_47, [1, 0, 2])
        convert_element_type_89: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg67_1, torch.bfloat16);  arg67_1 = None
        convert_element_type_90: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg68_1, torch.bfloat16);  arg68_1 = None
        convert_element_type_91: "bf16[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(permute_122, torch.bfloat16);  permute_122 = None
        permute_123: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_90, [1, 0]);  convert_element_type_90 = None
        clone_32: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(convert_element_type_91, memory_format = torch.contiguous_format);  convert_element_type_91 = None
        view_149: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_32, _shape_param_207);  clone_32 = _shape_param_207 = None
        mm_16: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_149, permute_123)
        view_150: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_16, _shape_param_208);  mm_16 = _shape_param_208 = None
        add_48: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_150, convert_element_type_89);  view_150 = convert_element_type_89 = None
        convert_element_type_92: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg69_1, torch.bfloat16);  arg69_1 = None
        convert_element_type_93: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg70_1, torch.bfloat16);  arg70_1 = None
        permute_124: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_93, [1, 0]);  convert_element_type_93 = None
        mm_17: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_149, permute_124)
        view_151: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_17, _shape_param_209);  mm_17 = _shape_param_209 = None
        add_49: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_151, convert_element_type_92);  view_151 = convert_element_type_92 = None
        convert_element_type_94: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg71_1, torch.bfloat16);  arg71_1 = None
        convert_element_type_95: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg72_1, torch.bfloat16);  arg72_1 = None
        permute_125: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_95, [1, 0]);  convert_element_type_95 = None
        mm_18: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_149, permute_125)
        view_152: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_18, _shape_param_210);  mm_18 = _shape_param_210 = None
        add_50: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_152, convert_element_type_94);  view_152 = convert_element_type_94 = None
        div_8: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_48, 8.0);  add_48 = None
        view_153: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_49, _shape_param_211);  add_49 = _shape_param_211 = None
        permute_126: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_153, [1, 0, 2, 3]);  view_153 = None
        permute_127: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_126, [0, 2, 1, 3]);  permute_126 = None
        view_154: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_127, _shape_param_212);  permute_127 = _shape_param_212 = None
        view_155: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_154, _shape_param_213);  view_154 = _shape_param_213 = None
        as_strided_14: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_155, _shape_param_214, _shape_param_215);  view_155 = _shape_param_214 = _shape_param_215 = None
        unsqueeze_26: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_14, 4);  as_strided_14 = None
        permute_128: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_26, [0, 1, 4, 2, 3]);  unsqueeze_26 = None
        view_156: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_8, _shape_param_216);  div_8 = _shape_param_216 = None
        permute_129: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_156, [1, 0, 2, 3]);  view_156 = None
        permute_130: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_129, [0, 2, 1, 3]);  permute_129 = None
        view_157: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_130, _shape_param_217);  permute_130 = _shape_param_217 = None
        view_158: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_157, _shape_param_218);  view_157 = _shape_param_218 = None
        as_strided_15: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_158, _shape_param_219, _shape_param_220);  view_158 = _shape_param_219 = _shape_param_220 = None
        unsqueeze_27: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_15, 4);  as_strided_15 = None
        clone_33: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_27, memory_format = torch.contiguous_format);  unsqueeze_27 = None
        view_159: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_33, _shape_param_221);  clone_33 = _shape_param_221 = None
        permute_131: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_128, [0, 1, 4, 3, 2]);  permute_128 = None
        clone_34: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_131, memory_format = torch.contiguous_format);  permute_131 = None
        view_160: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_34, _shape_param_222);  clone_34 = _shape_param_222 = None
        bmm_8: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_159, view_160)
        view_161: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_8, _shape_param_223);  bmm_8 = _shape_param_223 = None
        permute_132: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_161, [0, 1, 2, 4, 3]);  view_161 = None
        view_162: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_132, _shape_param_224);  permute_132 = _shape_param_224 = None
        constant_pad_nd_13: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_162, [0, 0, 0, 1], 0.0);  view_162 = None
        view_163: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_13, _shape_param_225);  constant_pad_nd_13 = _shape_param_225 = None
        slice_98: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_163, 2, 0, 256)
        slice_99: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_98, 3, 0, 257);  slice_98 = None
        copy_30: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_99);  slice_99 = None
        slice_scatter_55: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_30, 3, 256, 9223372036854775807);  copy_30 = None
        slice_scatter_56: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_55, 1, 0, -1);  slice_scatter_55 = None
        select_20: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_163, 1, -1)
        slice_100: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_20, 1, 256, 9223372036854775807);  select_20 = None
        slice_101: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_100, 2, 0, 257);  slice_100 = None
        select_21: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_56, 1, -1)
        slice_102: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_21, 2, 256, 9223372036854775807)
        copy_31: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_102, slice_101);  slice_102 = slice_101 = None
        slice_scatter_57: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_21, copy_31, 2, 256, 9223372036854775807);  select_21 = copy_31 = None
        select_scatter_10: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_56, slice_scatter_57, 1, -1);  slice_scatter_56 = slice_scatter_57 = None
        slice_103: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_163, 2, -257, -1)
        slice_104: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_103, 3, 257, 9223372036854775807);  slice_103 = None
        slice_105: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_10, 1, 1, 9223372036854775807)
        slice_106: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_105, 3, 0, 256)
        copy_32: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_106, slice_104);  slice_106 = slice_104 = None
        slice_scatter_58: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_105, copy_32, 3, 0, 256);  slice_105 = copy_32 = None
        slice_scatter_59: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_10, slice_scatter_58, 1, 1, 9223372036854775807);  select_scatter_10 = slice_scatter_58 = None
        select_22: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_163, 1, 0);  view_163 = None
        slice_107: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_22, 1, 0, 255);  select_22 = None
        slice_108: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_107, 2, -255, 9223372036854775807);  slice_107 = None
        select_23: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_59, 1, 0)
        slice_109: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_23, 1, 1, 256)
        slice_110: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_109, 2, 1, 256)
        copy_33: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_110, slice_108);  slice_110 = slice_108 = None
        slice_scatter_60: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_109, copy_33, 2, 1, 256);  slice_109 = copy_33 = None
        slice_scatter_61: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_23, slice_scatter_60, 1, 1, 256);  select_23 = slice_scatter_60 = None
        select_scatter_11: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_59, slice_scatter_61, 1, 0);  slice_scatter_59 = slice_scatter_61 = None
        view_164: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_11, _shape_param_226);  select_scatter_11 = _shape_param_226 = None
        permute_133: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        slice_111: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_133, 1, 0, 256)
        slice_112: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_111, 3, 0, 257)
        where_16: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_7, permute_12, slice_112)
        copy_34: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_112, where_16);  slice_112 = where_16 = None
        slice_scatter_62: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_111, copy_34, 3, 0, 257);  slice_111 = copy_34 = None
        slice_scatter_63: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_133, slice_scatter_62, 1, 0, 256);  permute_133 = slice_scatter_62 = None
        permute_134: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_63, [0, 2, 1, 3]);  slice_scatter_63 = None
        view_165: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_134, _shape_param_227);  permute_134 = _shape_param_227 = None
        view_166: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_165, _shape_param_228);  view_165 = _shape_param_228 = None
        permute_135: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        slice_113: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_135, 1, -256, 9223372036854775807)
        slice_114: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_113, 3, -257, 9223372036854775807)
        where_17: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_8, permute_12, slice_114)
        copy_35: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_114, where_17);  slice_114 = where_17 = None
        slice_scatter_64: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_113, copy_35, 3, -257, 9223372036854775807);  slice_113 = copy_35 = None
        slice_scatter_65: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_135, slice_scatter_64, 1, -256, 9223372036854775807);  permute_135 = slice_scatter_64 = None
        permute_136: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_65, [0, 2, 1, 3]);  slice_scatter_65 = None
        permute_137: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_136, [0, 2, 1, 3]);  permute_136 = None
        add_51: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_137, permute_25);  permute_137 = None
        permute_138: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_51, [0, 2, 1, 3]);  add_51 = None
        permute_139: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_138, [0, 2, 1, 3]);  permute_138 = None
        convert_element_type_96: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_139, torch.float32)
        clone_35: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_96, memory_format = torch.contiguous_format);  convert_element_type_96 = None
        amax_4: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_35, [-1], True)
        sub_13: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_35, amax_4);  clone_35 = None
        exp_4: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_9: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = None
        where_18: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_11, full_7, div_9);  div_9 = None
        convert_element_type_97: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_18, torch.bfloat16);  where_18 = None
        inductor_lookup_seed_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 12)
        inductor_random_12: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default(_shape_param_229, inductor_lookup_seed_12, 'rand');  _shape_param_229 = inductor_lookup_seed_12 = None
        convert_element_type_98: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(inductor_random_12, torch.bfloat16);  inductor_random_12 = None
        gt_12: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(convert_element_type_98, 0.1);  convert_element_type_98 = None
        mul_53: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_12, convert_element_type_97);  convert_element_type_97 = None
        mul_54: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_53, 1.1111111111111112);  mul_53 = None
        view_167: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_50, _shape_param_230);  add_50 = _shape_param_230 = None
        permute_140: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_167, [1, 0, 2, 3]);  view_167 = None
        permute_141: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_54, [0, 2, 1, 3]);  mul_54 = None
        clone_36: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None
        view_168: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_36, _shape_param_231);  clone_36 = _shape_param_231 = None
        permute_142: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_140, [0, 2, 1, 3]);  permute_140 = None
        view_169: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_142, _shape_param_232);  permute_142 = _shape_param_232 = None
        constant_pad_nd_14: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_169, _shape_param_233, -1.0);  view_169 = _shape_param_233 = None
        as_strided_16: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_14, _shape_param_234, _shape_param_235);  constant_pad_nd_14 = _shape_param_234 = _shape_param_235 = None
        constant_pad_nd_15: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_168, _shape_param_236, 0.0);  view_168 = _shape_param_236 = None
        view_170: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_15, _shape_param_237);  constant_pad_nd_15 = _shape_param_237 = None
        slice_115: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_170, 2, 0, -256);  view_170 = None
        view_171: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_115, _shape_param_238);  slice_115 = _shape_param_238 = None
        slice_116: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_171, 3, 0, -1);  view_171 = None
        unsqueeze_28: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_116, 4);  slice_116 = None
        unsqueeze_29: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_16, 4);  as_strided_16 = None
        view_172: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_28, _shape_param_239);  unsqueeze_28 = _shape_param_239 = None
        clone_37: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_29, memory_format = torch.contiguous_format);  unsqueeze_29 = None
        view_173: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_37, _shape_param_240);  clone_37 = _shape_param_240 = None
        bmm_9: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_172, view_173)
        view_174: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_9, _shape_param_241);  bmm_9 = _shape_param_241 = None
        permute_143: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_174, [0, 1, 2, 4, 3]);  view_174 = None
        view_175: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_143, _shape_param_242);  permute_143 = _shape_param_242 = None
        view_176: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_175, _shape_param_243);  view_175 = _shape_param_243 = None
        permute_144: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None
        permute_145: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_144, [1, 0, 2, 3]);  permute_144 = None
        clone_38: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
        view_177: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_38, _shape_param_244);  clone_38 = _shape_param_244 = None
        permute_146: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_177, [1, 0, 2]);  view_177 = None
        convert_element_type_99: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg73_1, torch.bfloat16);  arg73_1 = None
        convert_element_type_100: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg74_1, torch.bfloat16);  arg74_1 = None
        permute_147: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_100, [1, 0]);  convert_element_type_100 = None
        clone_39: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
        view_178: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_39, _shape_param_245);  clone_39 = _shape_param_245 = None
        mm_19: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_178, permute_147)
        view_179: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_19, _shape_param_246);  mm_19 = _shape_param_246 = None
        add_52: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_179, convert_element_type_99);  view_179 = convert_element_type_99 = None
        inductor_lookup_seed_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 13)
        inductor_random_13: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_247, inductor_lookup_seed_13, 'rand');  _shape_param_247 = inductor_lookup_seed_13 = None
        convert_element_type_101: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_13, torch.bfloat16);  inductor_random_13 = None
        gt_13: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_101, 0.1);  convert_element_type_101 = None
        mul_55: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_13, add_52);  add_52 = None
        mul_56: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_55, 1.1111111111111112);  mul_55 = None
        add_53: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_56, add_47);  mul_56 = add_47 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_16: "f32[8, 1024, 1]" = var_mean_8[0]
        getitem_17: "f32[8, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
        add_54: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_14: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_53, getitem_17);  add_53 = getitem_17 = None
        mul_57: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_8);  sub_14 = None
        mul_58: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_57, arg75_1);  arg75_1 = None
        add_55: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_58, arg76_1);  mul_58 = arg76_1 = None
        convert_element_type_102: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg77_1, torch.bfloat16);  arg77_1 = None
        convert_element_type_103: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg78_1, torch.bfloat16);  arg78_1 = None
        convert_element_type_104: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16)
        view_180: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_104, _shape_param_248);  convert_element_type_104 = _shape_param_248 = None
        permute_148: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_103, [1, 0]);  convert_element_type_103 = None
        addmm_8: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(convert_element_type_102, view_180, permute_148);  convert_element_type_102 = None
        view_181: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_8, _shape_param_249);  _shape_param_249 = None
        convert_element_type_105: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_181, torch.float32);  view_181 = None
        mul_59: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_105, 0.5)
        mul_60: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_105, 0.7071067811865476);  convert_element_type_105 = None
        erf_4: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
        add_56: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_61: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_59, add_56);  mul_59 = add_56 = None
        convert_element_type_106: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None
        convert_element_type_107: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg79_1, torch.bfloat16);  arg79_1 = None
        convert_element_type_108: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg80_1, torch.bfloat16);  arg80_1 = None
        view_182: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_106, _shape_param_250);  convert_element_type_106 = _shape_param_250 = None
        permute_149: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_108, [1, 0]);  convert_element_type_108 = None
        addmm_9: "bf16[8192, 768]" = torch.ops.aten.addmm.default(convert_element_type_107, view_182, permute_149);  convert_element_type_107 = None
        view_183: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_9, _shape_param_251);  addmm_9 = _shape_param_251 = None
        inductor_lookup_seed_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 14)
        inductor_random_14: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_252, inductor_lookup_seed_14, 'rand');  _shape_param_252 = inductor_lookup_seed_14 = None
        convert_element_type_109: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_14, torch.bfloat16);  inductor_random_14 = None
        gt_14: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_109, 0.1);  convert_element_type_109 = None
        mul_62: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_14, view_183);  view_183 = None
        mul_63: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_62, 1.1111111111111112);  mul_62 = None
        add_57: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_63, add_55);  mul_63 = add_55 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_18: "f32[8, 1024, 1]" = var_mean_9[0]
        getitem_19: "f32[8, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
        add_58: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_15: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_19);  add_57 = getitem_19 = None
        mul_64: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
        mul_65: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_64, arg81_1);  arg81_1 = None
        add_59: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_65, arg82_1);  mul_65 = arg82_1 = None
        permute_150: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_59, [1, 0, 2])
        convert_element_type_110: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg83_1, torch.bfloat16);  arg83_1 = None
        convert_element_type_111: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg84_1, torch.bfloat16);  arg84_1 = None
        convert_element_type_112: "bf16[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(permute_150, torch.bfloat16);  permute_150 = None
        permute_151: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_111, [1, 0]);  convert_element_type_111 = None
        clone_40: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(convert_element_type_112, memory_format = torch.contiguous_format);  convert_element_type_112 = None
        view_184: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_40, _shape_param_253);  clone_40 = _shape_param_253 = None
        mm_20: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_184, permute_151)
        view_185: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_20, _shape_param_254);  mm_20 = _shape_param_254 = None
        add_60: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_185, convert_element_type_110);  view_185 = convert_element_type_110 = None
        convert_element_type_113: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg85_1, torch.bfloat16);  arg85_1 = None
        convert_element_type_114: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg86_1, torch.bfloat16);  arg86_1 = None
        permute_152: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_114, [1, 0]);  convert_element_type_114 = None
        mm_21: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_184, permute_152)
        view_186: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_21, _shape_param_255);  mm_21 = _shape_param_255 = None
        add_61: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_186, convert_element_type_113);  view_186 = convert_element_type_113 = None
        convert_element_type_115: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg87_1, torch.bfloat16);  arg87_1 = None
        convert_element_type_116: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg88_1, torch.bfloat16);  arg88_1 = None
        permute_153: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_116, [1, 0]);  convert_element_type_116 = None
        mm_22: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_184, permute_153)
        view_187: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_22, _shape_param_256);  mm_22 = _shape_param_256 = None
        add_62: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_187, convert_element_type_115);  view_187 = convert_element_type_115 = None
        div_10: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_60, 8.0);  add_60 = None
        view_188: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_61, _shape_param_257);  add_61 = _shape_param_257 = None
        permute_154: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_188, [1, 0, 2, 3]);  view_188 = None
        permute_155: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_154, [0, 2, 1, 3]);  permute_154 = None
        view_189: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_155, _shape_param_258);  permute_155 = _shape_param_258 = None
        view_190: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_189, _shape_param_259);  view_189 = _shape_param_259 = None
        as_strided_17: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_190, _shape_param_260, _shape_param_261);  view_190 = _shape_param_260 = _shape_param_261 = None
        unsqueeze_30: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_17, 4);  as_strided_17 = None
        permute_156: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_30, [0, 1, 4, 2, 3]);  unsqueeze_30 = None
        view_191: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_10, _shape_param_262);  div_10 = _shape_param_262 = None
        permute_157: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_191, [1, 0, 2, 3]);  view_191 = None
        permute_158: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_157, [0, 2, 1, 3]);  permute_157 = None
        view_192: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_158, _shape_param_263);  permute_158 = _shape_param_263 = None
        view_193: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_192, _shape_param_264);  view_192 = _shape_param_264 = None
        as_strided_18: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_193, _shape_param_265, _shape_param_266);  view_193 = _shape_param_265 = _shape_param_266 = None
        unsqueeze_31: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_18, 4);  as_strided_18 = None
        clone_41: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_31, memory_format = torch.contiguous_format);  unsqueeze_31 = None
        view_194: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_41, _shape_param_267);  clone_41 = _shape_param_267 = None
        permute_159: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_156, [0, 1, 4, 3, 2]);  permute_156 = None
        clone_42: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        view_195: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_42, _shape_param_268);  clone_42 = _shape_param_268 = None
        bmm_10: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_194, view_195)
        view_196: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_10, _shape_param_269);  bmm_10 = _shape_param_269 = None
        permute_160: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_196, [0, 1, 2, 4, 3]);  view_196 = None
        view_197: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_160, _shape_param_270);  permute_160 = _shape_param_270 = None
        constant_pad_nd_16: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_197, [0, 0, 0, 1], 0.0);  view_197 = None
        view_198: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_16, _shape_param_271);  constant_pad_nd_16 = _shape_param_271 = None
        slice_117: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_198, 2, 0, 256)
        slice_118: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_117, 3, 0, 257);  slice_117 = None
        copy_36: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_118);  slice_118 = None
        slice_scatter_66: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_36, 3, 256, 9223372036854775807);  copy_36 = None
        slice_scatter_67: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_66, 1, 0, -1);  slice_scatter_66 = None
        select_24: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_198, 1, -1)
        slice_119: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_24, 1, 256, 9223372036854775807);  select_24 = None
        slice_120: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_119, 2, 0, 257);  slice_119 = None
        select_25: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_67, 1, -1)
        slice_121: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_25, 2, 256, 9223372036854775807)
        copy_37: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_121, slice_120);  slice_121 = slice_120 = None
        slice_scatter_68: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_25, copy_37, 2, 256, 9223372036854775807);  select_25 = copy_37 = None
        select_scatter_12: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_67, slice_scatter_68, 1, -1);  slice_scatter_67 = slice_scatter_68 = None
        slice_122: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_198, 2, -257, -1)
        slice_123: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_122, 3, 257, 9223372036854775807);  slice_122 = None
        slice_124: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_12, 1, 1, 9223372036854775807)
        slice_125: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_124, 3, 0, 256)
        copy_38: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_125, slice_123);  slice_125 = slice_123 = None
        slice_scatter_69: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_124, copy_38, 3, 0, 256);  slice_124 = copy_38 = None
        slice_scatter_70: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_12, slice_scatter_69, 1, 1, 9223372036854775807);  select_scatter_12 = slice_scatter_69 = None
        select_26: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_198, 1, 0);  view_198 = None
        slice_126: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_26, 1, 0, 255);  select_26 = None
        slice_127: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_126, 2, -255, 9223372036854775807);  slice_126 = None
        select_27: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_70, 1, 0)
        slice_128: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_27, 1, 1, 256)
        slice_129: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_128, 2, 1, 256)
        copy_39: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_129, slice_127);  slice_129 = slice_127 = None
        slice_scatter_71: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_128, copy_39, 2, 1, 256);  slice_128 = copy_39 = None
        slice_scatter_72: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_27, slice_scatter_71, 1, 1, 256);  select_27 = slice_scatter_71 = None
        select_scatter_13: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_70, slice_scatter_72, 1, 0);  slice_scatter_70 = slice_scatter_72 = None
        view_199: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_13, _shape_param_272);  select_scatter_13 = _shape_param_272 = None
        permute_161: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_199, [0, 2, 1, 3]);  view_199 = None
        slice_130: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_161, 1, 0, 256)
        slice_131: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_130, 3, 0, 257)
        where_19: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_7, permute_12, slice_131)
        copy_40: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_131, where_19);  slice_131 = where_19 = None
        slice_scatter_73: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_130, copy_40, 3, 0, 257);  slice_130 = copy_40 = None
        slice_scatter_74: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_161, slice_scatter_73, 1, 0, 256);  permute_161 = slice_scatter_73 = None
        permute_162: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_74, [0, 2, 1, 3]);  slice_scatter_74 = None
        view_200: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_162, _shape_param_273);  permute_162 = _shape_param_273 = None
        view_201: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_200, _shape_param_274);  view_200 = _shape_param_274 = None
        permute_163: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None
        slice_132: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_163, 1, -256, 9223372036854775807)
        slice_133: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_132, 3, -257, 9223372036854775807)
        where_20: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_8, permute_12, slice_133)
        copy_41: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_133, where_20);  slice_133 = where_20 = None
        slice_scatter_75: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_132, copy_41, 3, -257, 9223372036854775807);  slice_132 = copy_41 = None
        slice_scatter_76: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_163, slice_scatter_75, 1, -256, 9223372036854775807);  permute_163 = slice_scatter_75 = None
        permute_164: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_76, [0, 2, 1, 3]);  slice_scatter_76 = None
        permute_165: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_164, [0, 2, 1, 3]);  permute_164 = None
        add_63: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_165, permute_25);  permute_165 = None
        permute_166: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_63, [0, 2, 1, 3]);  add_63 = None
        permute_167: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_166, [0, 2, 1, 3]);  permute_166 = None
        convert_element_type_117: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_167, torch.float32)
        clone_43: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_117, memory_format = torch.contiguous_format);  convert_element_type_117 = None
        amax_5: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_43, [-1], True)
        sub_16: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_43, amax_5);  clone_43 = None
        exp_5: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_11: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = None
        where_21: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_11, full_7, div_11);  div_11 = None
        convert_element_type_118: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_21, torch.bfloat16);  where_21 = None
        inductor_lookup_seed_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 15)
        inductor_random_15: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default(_shape_param_275, inductor_lookup_seed_15, 'rand');  _shape_param_275 = inductor_lookup_seed_15 = None
        convert_element_type_119: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(inductor_random_15, torch.bfloat16);  inductor_random_15 = None
        gt_15: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(convert_element_type_119, 0.1);  convert_element_type_119 = None
        mul_66: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_15, convert_element_type_118);  convert_element_type_118 = None
        mul_67: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_66, 1.1111111111111112);  mul_66 = None
        view_202: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_62, _shape_param_276);  add_62 = _shape_param_276 = None
        permute_168: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_202, [1, 0, 2, 3]);  view_202 = None
        permute_169: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_67, [0, 2, 1, 3]);  mul_67 = None
        clone_44: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        view_203: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_44, _shape_param_277);  clone_44 = _shape_param_277 = None
        permute_170: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_168, [0, 2, 1, 3]);  permute_168 = None
        view_204: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_170, _shape_param_278);  permute_170 = _shape_param_278 = None
        constant_pad_nd_17: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_204, _shape_param_279, -1.0);  view_204 = _shape_param_279 = None
        as_strided_19: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_17, _shape_param_280, _shape_param_281);  constant_pad_nd_17 = _shape_param_280 = _shape_param_281 = None
        constant_pad_nd_18: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_203, _shape_param_282, 0.0);  view_203 = _shape_param_282 = None
        view_205: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_18, _shape_param_283);  constant_pad_nd_18 = _shape_param_283 = None
        slice_134: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_205, 2, 0, -256);  view_205 = None
        view_206: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_134, _shape_param_284);  slice_134 = _shape_param_284 = None
        slice_135: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_206, 3, 0, -1);  view_206 = None
        unsqueeze_32: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_135, 4);  slice_135 = None
        unsqueeze_33: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_19, 4);  as_strided_19 = None
        view_207: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_32, _shape_param_285);  unsqueeze_32 = _shape_param_285 = None
        clone_45: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_33, memory_format = torch.contiguous_format);  unsqueeze_33 = None
        view_208: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_45, _shape_param_286);  clone_45 = _shape_param_286 = None
        bmm_11: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_207, view_208)
        view_209: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_287);  bmm_11 = _shape_param_287 = None
        permute_171: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_209, [0, 1, 2, 4, 3]);  view_209 = None
        view_210: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_171, _shape_param_288);  permute_171 = _shape_param_288 = None
        view_211: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_210, _shape_param_289);  view_210 = _shape_param_289 = None
        permute_172: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None
        permute_173: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_172, [1, 0, 2, 3]);  permute_172 = None
        clone_46: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_173, memory_format = torch.contiguous_format);  permute_173 = None
        view_212: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_46, _shape_param_290);  clone_46 = _shape_param_290 = None
        permute_174: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_212, [1, 0, 2]);  view_212 = None
        convert_element_type_120: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg89_1, torch.bfloat16);  arg89_1 = None
        convert_element_type_121: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg90_1, torch.bfloat16);  arg90_1 = None
        permute_175: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_121, [1, 0]);  convert_element_type_121 = None
        clone_47: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_174, memory_format = torch.contiguous_format);  permute_174 = None
        view_213: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_47, _shape_param_291);  clone_47 = _shape_param_291 = None
        mm_23: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_213, permute_175)
        view_214: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_23, _shape_param_292);  mm_23 = _shape_param_292 = None
        add_64: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_214, convert_element_type_120);  view_214 = convert_element_type_120 = None
        inductor_lookup_seed_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 16)
        inductor_random_16: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_293, inductor_lookup_seed_16, 'rand');  _shape_param_293 = inductor_lookup_seed_16 = None
        convert_element_type_122: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_16, torch.bfloat16);  inductor_random_16 = None
        gt_16: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_122, 0.1);  convert_element_type_122 = None
        mul_68: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_16, add_64);  add_64 = None
        mul_69: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_68, 1.1111111111111112);  mul_68 = None
        add_65: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_69, add_59);  mul_69 = add_59 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_20: "f32[8, 1024, 1]" = var_mean_10[0]
        getitem_21: "f32[8, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
        add_66: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_17: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_65, getitem_21);  add_65 = getitem_21 = None
        mul_70: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_10);  sub_17 = None
        mul_71: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_70, arg91_1);  arg91_1 = None
        add_67: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_71, arg92_1);  mul_71 = arg92_1 = None
        convert_element_type_123: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg93_1, torch.bfloat16);  arg93_1 = None
        convert_element_type_124: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg94_1, torch.bfloat16);  arg94_1 = None
        convert_element_type_125: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_67, torch.bfloat16)
        view_215: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_125, _shape_param_294);  convert_element_type_125 = _shape_param_294 = None
        permute_176: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_124, [1, 0]);  convert_element_type_124 = None
        addmm_10: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(convert_element_type_123, view_215, permute_176);  convert_element_type_123 = None
        view_216: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_10, _shape_param_295);  _shape_param_295 = None
        convert_element_type_126: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_216, torch.float32);  view_216 = None
        mul_72: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_126, 0.5)
        mul_73: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_126, 0.7071067811865476);  convert_element_type_126 = None
        erf_5: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_73);  mul_73 = None
        add_68: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_74: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_72, add_68);  mul_72 = add_68 = None
        convert_element_type_127: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_74, torch.bfloat16);  mul_74 = None
        convert_element_type_128: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg95_1, torch.bfloat16);  arg95_1 = None
        convert_element_type_129: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg96_1, torch.bfloat16);  arg96_1 = None
        view_217: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_127, _shape_param_296);  convert_element_type_127 = _shape_param_296 = None
        permute_177: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_129, [1, 0]);  convert_element_type_129 = None
        addmm_11: "bf16[8192, 768]" = torch.ops.aten.addmm.default(convert_element_type_128, view_217, permute_177);  convert_element_type_128 = None
        view_218: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_11, _shape_param_297);  addmm_11 = _shape_param_297 = None
        inductor_lookup_seed_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 17)
        inductor_random_17: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_298, inductor_lookup_seed_17, 'rand');  _shape_param_298 = inductor_lookup_seed_17 = None
        convert_element_type_130: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_17, torch.bfloat16);  inductor_random_17 = None
        gt_17: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_130, 0.1);  convert_element_type_130 = None
        mul_75: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_17, view_218);  view_218 = None
        mul_76: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_75, 1.1111111111111112);  mul_75 = None
        add_69: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_76, add_67);  mul_76 = add_67 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
        getitem_22: "f32[8, 1024, 1]" = var_mean_11[0]
        getitem_23: "f32[8, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
        add_70: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_18: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_69, getitem_23);  add_69 = getitem_23 = None
        mul_77: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = None
        mul_78: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_77, arg97_1);  arg97_1 = None
        add_71: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_78, arg98_1);  mul_78 = arg98_1 = None
        permute_178: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_71, [1, 0, 2])
        convert_element_type_131: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg99_1, torch.bfloat16);  arg99_1 = None
        convert_element_type_132: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg100_1, torch.bfloat16);  arg100_1 = None
        convert_element_type_133: "bf16[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(permute_178, torch.bfloat16);  permute_178 = None
        permute_179: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_132, [1, 0]);  convert_element_type_132 = None
        clone_48: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(convert_element_type_133, memory_format = torch.contiguous_format);  convert_element_type_133 = None
        view_219: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_48, _shape_param_299);  clone_48 = _shape_param_299 = None
        mm_24: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_219, permute_179)
        view_220: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_24, _shape_param_300);  mm_24 = _shape_param_300 = None
        add_72: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_220, convert_element_type_131);  view_220 = convert_element_type_131 = None
        convert_element_type_134: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg101_1, torch.bfloat16);  arg101_1 = None
        convert_element_type_135: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg102_1, torch.bfloat16);  arg102_1 = None
        permute_180: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_135, [1, 0]);  convert_element_type_135 = None
        mm_25: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_219, permute_180)
        view_221: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_25, _shape_param_301);  mm_25 = _shape_param_301 = None
        add_73: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_221, convert_element_type_134);  view_221 = convert_element_type_134 = None
        convert_element_type_136: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg103_1, torch.bfloat16);  arg103_1 = None
        convert_element_type_137: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg104_1, torch.bfloat16);  arg104_1 = None
        permute_181: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_137, [1, 0]);  convert_element_type_137 = None
        mm_26: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_219, permute_181)
        view_222: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_26, _shape_param_302);  mm_26 = _shape_param_302 = None
        add_74: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_222, convert_element_type_136);  view_222 = convert_element_type_136 = None
        div_12: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_72, 8.0);  add_72 = None
        view_223: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_73, _shape_param_303);  add_73 = _shape_param_303 = None
        permute_182: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_223, [1, 0, 2, 3]);  view_223 = None
        permute_183: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_182, [0, 2, 1, 3]);  permute_182 = None
        view_224: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_183, _shape_param_304);  permute_183 = _shape_param_304 = None
        view_225: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_224, _shape_param_305);  view_224 = _shape_param_305 = None
        as_strided_20: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_225, _shape_param_306, _shape_param_307);  view_225 = _shape_param_306 = _shape_param_307 = None
        unsqueeze_34: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_20, 4);  as_strided_20 = None
        permute_184: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_34, [0, 1, 4, 2, 3]);  unsqueeze_34 = None
        view_226: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_12, _shape_param_308);  div_12 = _shape_param_308 = None
        permute_185: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_226, [1, 0, 2, 3]);  view_226 = None
        permute_186: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_185, [0, 2, 1, 3]);  permute_185 = None
        view_227: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_186, _shape_param_309);  permute_186 = _shape_param_309 = None
        view_228: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_227, _shape_param_310);  view_227 = _shape_param_310 = None
        as_strided_21: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_228, _shape_param_311, _shape_param_312);  view_228 = _shape_param_311 = _shape_param_312 = None
        unsqueeze_35: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_21, 4);  as_strided_21 = None
        clone_49: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_35, memory_format = torch.contiguous_format);  unsqueeze_35 = None
        view_229: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_49, _shape_param_313);  clone_49 = _shape_param_313 = None
        permute_187: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_184, [0, 1, 4, 3, 2]);  permute_184 = None
        clone_50: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
        view_230: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_50, _shape_param_314);  clone_50 = _shape_param_314 = None
        bmm_12: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_229, view_230)
        view_231: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_12, _shape_param_315);  bmm_12 = _shape_param_315 = None
        permute_188: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_231, [0, 1, 2, 4, 3]);  view_231 = None
        view_232: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_188, _shape_param_316);  permute_188 = _shape_param_316 = None
        constant_pad_nd_19: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_232, [0, 0, 0, 1], 0.0);  view_232 = None
        view_233: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_19, _shape_param_317);  constant_pad_nd_19 = _shape_param_317 = None
        slice_136: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_233, 2, 0, 256)
        slice_137: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_136, 3, 0, 257);  slice_136 = None
        copy_42: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_137);  slice_137 = None
        slice_scatter_77: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_42, 3, 256, 9223372036854775807);  copy_42 = None
        slice_scatter_78: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_77, 1, 0, -1);  slice_scatter_77 = None
        select_28: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_233, 1, -1)
        slice_138: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_28, 1, 256, 9223372036854775807);  select_28 = None
        slice_139: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_138, 2, 0, 257);  slice_138 = None
        select_29: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_78, 1, -1)
        slice_140: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_29, 2, 256, 9223372036854775807)
        copy_43: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_140, slice_139);  slice_140 = slice_139 = None
        slice_scatter_79: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_29, copy_43, 2, 256, 9223372036854775807);  select_29 = copy_43 = None
        select_scatter_14: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_78, slice_scatter_79, 1, -1);  slice_scatter_78 = slice_scatter_79 = None
        slice_141: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_233, 2, -257, -1)
        slice_142: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_141, 3, 257, 9223372036854775807);  slice_141 = None
        slice_143: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_14, 1, 1, 9223372036854775807)
        slice_144: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_143, 3, 0, 256)
        copy_44: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_144, slice_142);  slice_144 = slice_142 = None
        slice_scatter_80: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_143, copy_44, 3, 0, 256);  slice_143 = copy_44 = None
        slice_scatter_81: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_14, slice_scatter_80, 1, 1, 9223372036854775807);  select_scatter_14 = slice_scatter_80 = None
        select_30: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_233, 1, 0);  view_233 = None
        slice_145: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_30, 1, 0, 255);  select_30 = None
        slice_146: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_145, 2, -255, 9223372036854775807);  slice_145 = None
        select_31: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_81, 1, 0)
        slice_147: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_31, 1, 1, 256)
        slice_148: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_147, 2, 1, 256)
        copy_45: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_148, slice_146);  slice_148 = slice_146 = None
        slice_scatter_82: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_147, copy_45, 2, 1, 256);  slice_147 = copy_45 = None
        slice_scatter_83: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_31, slice_scatter_82, 1, 1, 256);  select_31 = slice_scatter_82 = None
        select_scatter_15: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_81, slice_scatter_83, 1, 0);  slice_scatter_81 = slice_scatter_83 = None
        view_234: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_15, _shape_param_318);  select_scatter_15 = _shape_param_318 = None
        permute_189: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        slice_149: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_189, 1, 0, 256)
        slice_150: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_149, 3, 0, 257)
        where_22: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_7, permute_12, slice_150)
        copy_46: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_150, where_22);  slice_150 = where_22 = None
        slice_scatter_84: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_149, copy_46, 3, 0, 257);  slice_149 = copy_46 = None
        slice_scatter_85: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_189, slice_scatter_84, 1, 0, 256);  permute_189 = slice_scatter_84 = None
        permute_190: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_85, [0, 2, 1, 3]);  slice_scatter_85 = None
        view_235: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_190, _shape_param_319);  permute_190 = _shape_param_319 = None
        view_236: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_235, _shape_param_320);  view_235 = _shape_param_320 = None
        permute_191: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        slice_151: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_191, 1, -256, 9223372036854775807)
        slice_152: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_151, 3, -257, 9223372036854775807)
        where_23: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_8, permute_12, slice_152)
        copy_47: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_152, where_23);  slice_152 = where_23 = None
        slice_scatter_86: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_151, copy_47, 3, -257, 9223372036854775807);  slice_151 = copy_47 = None
        slice_scatter_87: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_191, slice_scatter_86, 1, -256, 9223372036854775807);  permute_191 = slice_scatter_86 = None
        permute_192: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_87, [0, 2, 1, 3]);  slice_scatter_87 = None
        permute_193: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_192, [0, 2, 1, 3]);  permute_192 = None
        add_75: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_193, permute_25);  permute_193 = None
        permute_194: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_75, [0, 2, 1, 3]);  add_75 = None
        permute_195: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_194, [0, 2, 1, 3]);  permute_194 = None
        convert_element_type_138: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_195, torch.float32)
        clone_51: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_138, memory_format = torch.contiguous_format);  convert_element_type_138 = None
        amax_6: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_51, [-1], True)
        sub_19: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_51, amax_6);  clone_51 = None
        exp_6: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_13: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = None
        where_24: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_11, full_7, div_13);  div_13 = None
        convert_element_type_139: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_24, torch.bfloat16);  where_24 = None
        inductor_lookup_seed_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 18)
        inductor_random_18: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default(_shape_param_321, inductor_lookup_seed_18, 'rand');  _shape_param_321 = inductor_lookup_seed_18 = None
        convert_element_type_140: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(inductor_random_18, torch.bfloat16);  inductor_random_18 = None
        gt_18: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(convert_element_type_140, 0.1);  convert_element_type_140 = None
        mul_79: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_18, convert_element_type_139);  convert_element_type_139 = None
        mul_80: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_79, 1.1111111111111112);  mul_79 = None
        view_237: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_74, _shape_param_322);  add_74 = _shape_param_322 = None
        permute_196: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_237, [1, 0, 2, 3]);  view_237 = None
        permute_197: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_80, [0, 2, 1, 3]);  mul_80 = None
        clone_52: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
        view_238: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_52, _shape_param_323);  clone_52 = _shape_param_323 = None
        permute_198: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_196, [0, 2, 1, 3]);  permute_196 = None
        view_239: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_198, _shape_param_324);  permute_198 = _shape_param_324 = None
        constant_pad_nd_20: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_239, _shape_param_325, -1.0);  view_239 = _shape_param_325 = None
        as_strided_22: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_20, _shape_param_326, _shape_param_327);  constant_pad_nd_20 = _shape_param_326 = _shape_param_327 = None
        constant_pad_nd_21: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_238, _shape_param_328, 0.0);  view_238 = _shape_param_328 = None
        view_240: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_21, _shape_param_329);  constant_pad_nd_21 = _shape_param_329 = None
        slice_153: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_240, 2, 0, -256);  view_240 = None
        view_241: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_153, _shape_param_330);  slice_153 = _shape_param_330 = None
        slice_154: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_241, 3, 0, -1);  view_241 = None
        unsqueeze_36: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_154, 4);  slice_154 = None
        unsqueeze_37: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_22, 4);  as_strided_22 = None
        view_242: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_36, _shape_param_331);  unsqueeze_36 = _shape_param_331 = None
        clone_53: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_37, memory_format = torch.contiguous_format);  unsqueeze_37 = None
        view_243: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_53, _shape_param_332);  clone_53 = _shape_param_332 = None
        bmm_13: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_242, view_243)
        view_244: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_13, _shape_param_333);  bmm_13 = _shape_param_333 = None
        permute_199: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_244, [0, 1, 2, 4, 3]);  view_244 = None
        view_245: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_199, _shape_param_334);  permute_199 = _shape_param_334 = None
        view_246: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_245, _shape_param_335);  view_245 = _shape_param_335 = None
        permute_200: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        permute_201: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_200, [1, 0, 2, 3]);  permute_200 = None
        clone_54: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
        view_247: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_54, _shape_param_336);  clone_54 = _shape_param_336 = None
        permute_202: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_247, [1, 0, 2]);  view_247 = None
        convert_element_type_141: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg105_1, torch.bfloat16);  arg105_1 = None
        convert_element_type_142: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg106_1, torch.bfloat16);  arg106_1 = None
        permute_203: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_142, [1, 0]);  convert_element_type_142 = None
        clone_55: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
        view_248: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_55, _shape_param_337);  clone_55 = _shape_param_337 = None
        mm_27: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_248, permute_203)
        view_249: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_27, _shape_param_338);  mm_27 = _shape_param_338 = None
        add_76: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_249, convert_element_type_141);  view_249 = convert_element_type_141 = None
        inductor_lookup_seed_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 19)
        inductor_random_19: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_339, inductor_lookup_seed_19, 'rand');  _shape_param_339 = inductor_lookup_seed_19 = None
        convert_element_type_143: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_19, torch.bfloat16);  inductor_random_19 = None
        gt_19: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_143, 0.1);  convert_element_type_143 = None
        mul_81: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_19, add_76);  add_76 = None
        mul_82: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_81, 1.1111111111111112);  mul_81 = None
        add_77: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_82, add_71);  mul_82 = add_71 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_24: "f32[8, 1024, 1]" = var_mean_12[0]
        getitem_25: "f32[8, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
        add_78: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_20: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_25);  add_77 = getitem_25 = None
        mul_83: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_12);  sub_20 = None
        mul_84: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_83, arg107_1);  arg107_1 = None
        add_79: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_84, arg108_1);  mul_84 = arg108_1 = None
        convert_element_type_144: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg109_1, torch.bfloat16);  arg109_1 = None
        convert_element_type_145: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg110_1, torch.bfloat16);  arg110_1 = None
        convert_element_type_146: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16)
        view_250: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_146, _shape_param_340);  convert_element_type_146 = _shape_param_340 = None
        permute_204: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_145, [1, 0]);  convert_element_type_145 = None
        addmm_12: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(convert_element_type_144, view_250, permute_204);  convert_element_type_144 = None
        view_251: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_12, _shape_param_341);  _shape_param_341 = None
        convert_element_type_147: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_251, torch.float32);  view_251 = None
        mul_85: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_147, 0.5)
        mul_86: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_147, 0.7071067811865476);  convert_element_type_147 = None
        erf_6: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_86);  mul_86 = None
        add_80: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_87: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_85, add_80);  mul_85 = add_80 = None
        convert_element_type_148: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_87, torch.bfloat16);  mul_87 = None
        convert_element_type_149: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg111_1, torch.bfloat16);  arg111_1 = None
        convert_element_type_150: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg112_1, torch.bfloat16);  arg112_1 = None
        view_252: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_148, _shape_param_342);  convert_element_type_148 = _shape_param_342 = None
        permute_205: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_150, [1, 0]);  convert_element_type_150 = None
        addmm_13: "bf16[8192, 768]" = torch.ops.aten.addmm.default(convert_element_type_149, view_252, permute_205);  convert_element_type_149 = None
        view_253: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_13, _shape_param_343);  addmm_13 = _shape_param_343 = None
        inductor_lookup_seed_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 20)
        inductor_random_20: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_344, inductor_lookup_seed_20, 'rand');  _shape_param_344 = inductor_lookup_seed_20 = None
        convert_element_type_151: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_20, torch.bfloat16);  inductor_random_20 = None
        gt_20: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_151, 0.1);  convert_element_type_151 = None
        mul_88: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_20, view_253);  view_253 = None
        mul_89: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_88, 1.1111111111111112);  mul_88 = None
        add_81: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_89, add_79);  mul_89 = add_79 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_26: "f32[8, 1024, 1]" = var_mean_13[0]
        getitem_27: "f32[8, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
        add_82: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_21: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_27);  add_81 = getitem_27 = None
        mul_90: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = None
        mul_91: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_90, arg113_1);  arg113_1 = None
        add_83: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_91, arg114_1);  mul_91 = arg114_1 = None
        permute_206: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_83, [1, 0, 2])
        convert_element_type_152: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg115_1, torch.bfloat16);  arg115_1 = None
        convert_element_type_153: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg116_1, torch.bfloat16);  arg116_1 = None
        convert_element_type_154: "bf16[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(permute_206, torch.bfloat16);  permute_206 = None
        permute_207: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_153, [1, 0]);  convert_element_type_153 = None
        clone_56: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(convert_element_type_154, memory_format = torch.contiguous_format);  convert_element_type_154 = None
        view_254: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_56, _shape_param_345);  clone_56 = _shape_param_345 = None
        mm_28: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_254, permute_207)
        view_255: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_28, _shape_param_346);  mm_28 = _shape_param_346 = None
        add_84: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_255, convert_element_type_152);  view_255 = convert_element_type_152 = None
        convert_element_type_155: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg117_1, torch.bfloat16);  arg117_1 = None
        convert_element_type_156: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg118_1, torch.bfloat16);  arg118_1 = None
        permute_208: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_156, [1, 0]);  convert_element_type_156 = None
        mm_29: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_254, permute_208)
        view_256: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_29, _shape_param_347);  mm_29 = _shape_param_347 = None
        add_85: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_256, convert_element_type_155);  view_256 = convert_element_type_155 = None
        convert_element_type_157: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg119_1, torch.bfloat16);  arg119_1 = None
        convert_element_type_158: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg120_1, torch.bfloat16);  arg120_1 = None
        permute_209: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_158, [1, 0]);  convert_element_type_158 = None
        mm_30: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_254, permute_209)
        view_257: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_30, _shape_param_348);  mm_30 = _shape_param_348 = None
        add_86: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_257, convert_element_type_157);  view_257 = convert_element_type_157 = None
        div_14: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_84, 8.0);  add_84 = None
        view_258: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_85, _shape_param_349);  add_85 = _shape_param_349 = None
        permute_210: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_258, [1, 0, 2, 3]);  view_258 = None
        permute_211: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_210, [0, 2, 1, 3]);  permute_210 = None
        view_259: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_211, _shape_param_350);  permute_211 = _shape_param_350 = None
        view_260: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_259, _shape_param_351);  view_259 = _shape_param_351 = None
        as_strided_23: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_260, _shape_param_352, _shape_param_353);  view_260 = _shape_param_352 = _shape_param_353 = None
        unsqueeze_38: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_23, 4);  as_strided_23 = None
        permute_212: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_38, [0, 1, 4, 2, 3]);  unsqueeze_38 = None
        view_261: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_14, _shape_param_354);  div_14 = _shape_param_354 = None
        permute_213: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_261, [1, 0, 2, 3]);  view_261 = None
        permute_214: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_213, [0, 2, 1, 3]);  permute_213 = None
        view_262: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_214, _shape_param_355);  permute_214 = _shape_param_355 = None
        view_263: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_262, _shape_param_356);  view_262 = _shape_param_356 = None
        as_strided_24: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_263, _shape_param_357, _shape_param_358);  view_263 = _shape_param_357 = _shape_param_358 = None
        unsqueeze_39: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_24, 4);  as_strided_24 = None
        clone_57: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_39, memory_format = torch.contiguous_format);  unsqueeze_39 = None
        view_264: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_57, _shape_param_359);  clone_57 = _shape_param_359 = None
        permute_215: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_212, [0, 1, 4, 3, 2]);  permute_212 = None
        clone_58: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_265: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_58, _shape_param_360);  clone_58 = _shape_param_360 = None
        bmm_14: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_264, view_265)
        view_266: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_14, _shape_param_361);  bmm_14 = _shape_param_361 = None
        permute_216: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_266, [0, 1, 2, 4, 3]);  view_266 = None
        view_267: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_216, _shape_param_362);  permute_216 = _shape_param_362 = None
        constant_pad_nd_22: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_267, [0, 0, 0, 1], 0.0);  view_267 = None
        view_268: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_22, _shape_param_363);  constant_pad_nd_22 = _shape_param_363 = None
        slice_155: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_268, 2, 0, 256)
        slice_156: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_155, 3, 0, 257);  slice_155 = None
        copy_48: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_156);  slice_156 = None
        slice_scatter_88: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_48, 3, 256, 9223372036854775807);  copy_48 = None
        slice_scatter_89: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_88, 1, 0, -1);  slice_scatter_88 = None
        select_32: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_268, 1, -1)
        slice_157: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_32, 1, 256, 9223372036854775807);  select_32 = None
        slice_158: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_157, 2, 0, 257);  slice_157 = None
        select_33: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_89, 1, -1)
        slice_159: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_33, 2, 256, 9223372036854775807)
        copy_49: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_159, slice_158);  slice_159 = slice_158 = None
        slice_scatter_90: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_33, copy_49, 2, 256, 9223372036854775807);  select_33 = copy_49 = None
        select_scatter_16: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_89, slice_scatter_90, 1, -1);  slice_scatter_89 = slice_scatter_90 = None
        slice_160: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_268, 2, -257, -1)
        slice_161: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_160, 3, 257, 9223372036854775807);  slice_160 = None
        slice_162: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_16, 1, 1, 9223372036854775807)
        slice_163: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_162, 3, 0, 256)
        copy_50: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_163, slice_161);  slice_163 = slice_161 = None
        slice_scatter_91: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_162, copy_50, 3, 0, 256);  slice_162 = copy_50 = None
        slice_scatter_92: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_16, slice_scatter_91, 1, 1, 9223372036854775807);  select_scatter_16 = slice_scatter_91 = None
        select_34: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_268, 1, 0);  view_268 = None
        slice_164: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_34, 1, 0, 255);  select_34 = None
        slice_165: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_164, 2, -255, 9223372036854775807);  slice_164 = None
        select_35: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_92, 1, 0)
        slice_166: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_35, 1, 1, 256)
        slice_167: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_166, 2, 1, 256)
        copy_51: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_167, slice_165);  slice_167 = slice_165 = None
        slice_scatter_93: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_166, copy_51, 2, 1, 256);  slice_166 = copy_51 = None
        slice_scatter_94: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_35, slice_scatter_93, 1, 1, 256);  select_35 = slice_scatter_93 = None
        select_scatter_17: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_92, slice_scatter_94, 1, 0);  slice_scatter_92 = slice_scatter_94 = None
        view_269: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_17, _shape_param_364);  select_scatter_17 = _shape_param_364 = None
        permute_217: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
        slice_168: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_217, 1, 0, 256)
        slice_169: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_168, 3, 0, 257)
        where_25: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_7, permute_12, slice_169)
        copy_52: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_169, where_25);  slice_169 = where_25 = None
        slice_scatter_95: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_168, copy_52, 3, 0, 257);  slice_168 = copy_52 = None
        slice_scatter_96: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_217, slice_scatter_95, 1, 0, 256);  permute_217 = slice_scatter_95 = None
        permute_218: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_96, [0, 2, 1, 3]);  slice_scatter_96 = None
        view_270: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_218, _shape_param_365);  permute_218 = _shape_param_365 = None
        view_271: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_270, _shape_param_366);  view_270 = _shape_param_366 = None
        permute_219: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_271, [0, 2, 1, 3]);  view_271 = None
        slice_170: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_219, 1, -256, 9223372036854775807)
        slice_171: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_170, 3, -257, 9223372036854775807)
        where_26: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_8, permute_12, slice_171)
        copy_53: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_171, where_26);  slice_171 = where_26 = None
        slice_scatter_97: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_170, copy_53, 3, -257, 9223372036854775807);  slice_170 = copy_53 = None
        slice_scatter_98: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_219, slice_scatter_97, 1, -256, 9223372036854775807);  permute_219 = slice_scatter_97 = None
        permute_220: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_98, [0, 2, 1, 3]);  slice_scatter_98 = None
        permute_221: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_220, [0, 2, 1, 3]);  permute_220 = None
        add_87: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_221, permute_25);  permute_221 = None
        permute_222: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_87, [0, 2, 1, 3]);  add_87 = None
        permute_223: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_222, [0, 2, 1, 3]);  permute_222 = None
        convert_element_type_159: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_223, torch.float32)
        clone_59: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_159, memory_format = torch.contiguous_format);  convert_element_type_159 = None
        amax_7: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_59, [-1], True)
        sub_22: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_59, amax_7);  clone_59 = None
        exp_7: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_15: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = None
        where_27: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_11, full_7, div_15);  div_15 = None
        convert_element_type_160: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_27, torch.bfloat16);  where_27 = None
        inductor_lookup_seed_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 21)
        inductor_random_21: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default(_shape_param_367, inductor_lookup_seed_21, 'rand');  _shape_param_367 = inductor_lookup_seed_21 = None
        convert_element_type_161: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(inductor_random_21, torch.bfloat16);  inductor_random_21 = None
        gt_21: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(convert_element_type_161, 0.1);  convert_element_type_161 = None
        mul_92: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_21, convert_element_type_160);  convert_element_type_160 = None
        mul_93: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_92, 1.1111111111111112);  mul_92 = None
        view_272: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_86, _shape_param_368);  add_86 = _shape_param_368 = None
        permute_224: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_272, [1, 0, 2, 3]);  view_272 = None
        permute_225: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_93, [0, 2, 1, 3]);  mul_93 = None
        clone_60: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_273: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_60, _shape_param_369);  clone_60 = _shape_param_369 = None
        permute_226: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_224, [0, 2, 1, 3]);  permute_224 = None
        view_274: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_226, _shape_param_370);  permute_226 = _shape_param_370 = None
        constant_pad_nd_23: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_274, _shape_param_371, -1.0);  view_274 = _shape_param_371 = None
        as_strided_25: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_23, _shape_param_372, _shape_param_373);  constant_pad_nd_23 = _shape_param_372 = _shape_param_373 = None
        constant_pad_nd_24: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_273, _shape_param_374, 0.0);  view_273 = _shape_param_374 = None
        view_275: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_24, _shape_param_375);  constant_pad_nd_24 = _shape_param_375 = None
        slice_172: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_275, 2, 0, -256);  view_275 = None
        view_276: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_172, _shape_param_376);  slice_172 = _shape_param_376 = None
        slice_173: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_276, 3, 0, -1);  view_276 = None
        unsqueeze_40: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_173, 4);  slice_173 = None
        unsqueeze_41: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_25, 4);  as_strided_25 = None
        view_277: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_40, _shape_param_377);  unsqueeze_40 = _shape_param_377 = None
        clone_61: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_41, memory_format = torch.contiguous_format);  unsqueeze_41 = None
        view_278: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_61, _shape_param_378);  clone_61 = _shape_param_378 = None
        bmm_15: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_277, view_278)
        view_279: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_379);  bmm_15 = _shape_param_379 = None
        permute_227: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_279, [0, 1, 2, 4, 3]);  view_279 = None
        view_280: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_227, _shape_param_380);  permute_227 = _shape_param_380 = None
        view_281: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_280, _shape_param_381);  view_280 = _shape_param_381 = None
        permute_228: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None
        permute_229: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_228, [1, 0, 2, 3]);  permute_228 = None
        clone_62: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_229, memory_format = torch.contiguous_format);  permute_229 = None
        view_282: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_62, _shape_param_382);  clone_62 = _shape_param_382 = None
        permute_230: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_282, [1, 0, 2]);  view_282 = None
        convert_element_type_162: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg121_1, torch.bfloat16);  arg121_1 = None
        convert_element_type_163: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg122_1, torch.bfloat16);  arg122_1 = None
        permute_231: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_163, [1, 0]);  convert_element_type_163 = None
        clone_63: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
        view_283: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_63, _shape_param_383);  clone_63 = _shape_param_383 = None
        mm_31: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_283, permute_231)
        view_284: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_31, _shape_param_384);  mm_31 = _shape_param_384 = None
        add_88: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_284, convert_element_type_162);  view_284 = convert_element_type_162 = None
        inductor_lookup_seed_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 22)
        inductor_random_22: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_385, inductor_lookup_seed_22, 'rand');  _shape_param_385 = inductor_lookup_seed_22 = None
        convert_element_type_164: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_22, torch.bfloat16);  inductor_random_22 = None
        gt_22: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_164, 0.1);  convert_element_type_164 = None
        mul_94: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_22, add_88);  add_88 = None
        mul_95: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_94, 1.1111111111111112);  mul_94 = None
        add_89: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_95, add_83);  mul_95 = add_83 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_28: "f32[8, 1024, 1]" = var_mean_14[0]
        getitem_29: "f32[8, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
        add_90: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_23: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_89, getitem_29);  add_89 = getitem_29 = None
        mul_96: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_14);  sub_23 = None
        mul_97: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_96, arg123_1);  arg123_1 = None
        add_91: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_97, arg124_1);  mul_97 = arg124_1 = None
        convert_element_type_165: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg125_1, torch.bfloat16);  arg125_1 = None
        convert_element_type_166: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg126_1, torch.bfloat16);  arg126_1 = None
        convert_element_type_167: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16)
        view_285: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_167, _shape_param_386);  convert_element_type_167 = _shape_param_386 = None
        permute_232: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_166, [1, 0]);  convert_element_type_166 = None
        addmm_14: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(convert_element_type_165, view_285, permute_232);  convert_element_type_165 = None
        view_286: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_14, _shape_param_387);  _shape_param_387 = None
        convert_element_type_168: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_286, torch.float32);  view_286 = None
        mul_98: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_168, 0.5)
        mul_99: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_168, 0.7071067811865476);  convert_element_type_168 = None
        erf_7: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
        add_92: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_100: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_98, add_92);  mul_98 = add_92 = None
        convert_element_type_169: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_100, torch.bfloat16);  mul_100 = None
        convert_element_type_170: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg127_1, torch.bfloat16);  arg127_1 = None
        convert_element_type_171: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg128_1, torch.bfloat16);  arg128_1 = None
        view_287: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_169, _shape_param_388);  convert_element_type_169 = _shape_param_388 = None
        permute_233: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_171, [1, 0]);  convert_element_type_171 = None
        addmm_15: "bf16[8192, 768]" = torch.ops.aten.addmm.default(convert_element_type_170, view_287, permute_233);  convert_element_type_170 = None
        view_288: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_15, _shape_param_389);  addmm_15 = _shape_param_389 = None
        inductor_lookup_seed_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 23)
        inductor_random_23: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_390, inductor_lookup_seed_23, 'rand');  _shape_param_390 = inductor_lookup_seed_23 = None
        convert_element_type_172: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_23, torch.bfloat16);  inductor_random_23 = None
        gt_23: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_172, 0.1);  convert_element_type_172 = None
        mul_101: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_23, view_288);  view_288 = None
        mul_102: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_101, 1.1111111111111112);  mul_101 = None
        add_93: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_102, add_91);  mul_102 = add_91 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
        getitem_30: "f32[8, 1024, 1]" = var_mean_15[0]
        getitem_31: "f32[8, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
        add_94: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_24: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_93, getitem_31);  add_93 = getitem_31 = None
        mul_103: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = None
        mul_104: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_103, arg129_1);  arg129_1 = None
        add_95: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_104, arg130_1);  mul_104 = arg130_1 = None
        permute_234: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_95, [1, 0, 2])
        convert_element_type_173: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg131_1, torch.bfloat16);  arg131_1 = None
        convert_element_type_174: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg132_1, torch.bfloat16);  arg132_1 = None
        convert_element_type_175: "bf16[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(permute_234, torch.bfloat16);  permute_234 = None
        permute_235: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_174, [1, 0]);  convert_element_type_174 = None
        clone_64: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(convert_element_type_175, memory_format = torch.contiguous_format);  convert_element_type_175 = None
        view_289: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_64, _shape_param_391);  clone_64 = _shape_param_391 = None
        mm_32: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_289, permute_235)
        view_290: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_32, _shape_param_392);  mm_32 = _shape_param_392 = None
        add_96: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_290, convert_element_type_173);  view_290 = convert_element_type_173 = None
        convert_element_type_176: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg133_1, torch.bfloat16);  arg133_1 = None
        convert_element_type_177: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg134_1, torch.bfloat16);  arg134_1 = None
        permute_236: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_177, [1, 0]);  convert_element_type_177 = None
        mm_33: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_289, permute_236)
        view_291: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_33, _shape_param_393);  mm_33 = _shape_param_393 = None
        add_97: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_291, convert_element_type_176);  view_291 = convert_element_type_176 = None
        convert_element_type_178: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg135_1, torch.bfloat16);  arg135_1 = None
        convert_element_type_179: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg136_1, torch.bfloat16);  arg136_1 = None
        permute_237: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_179, [1, 0]);  convert_element_type_179 = None
        mm_34: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_289, permute_237)
        view_292: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_34, _shape_param_394);  mm_34 = _shape_param_394 = None
        add_98: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_292, convert_element_type_178);  view_292 = convert_element_type_178 = None
        div_16: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_96, 8.0);  add_96 = None
        view_293: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_97, _shape_param_395);  add_97 = _shape_param_395 = None
        permute_238: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_293, [1, 0, 2, 3]);  view_293 = None
        permute_239: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_238, [0, 2, 1, 3]);  permute_238 = None
        view_294: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_239, _shape_param_396);  permute_239 = _shape_param_396 = None
        view_295: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_294, _shape_param_397);  view_294 = _shape_param_397 = None
        as_strided_26: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_295, _shape_param_398, _shape_param_399);  view_295 = _shape_param_398 = _shape_param_399 = None
        unsqueeze_42: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_26, 4);  as_strided_26 = None
        permute_240: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_42, [0, 1, 4, 2, 3]);  unsqueeze_42 = None
        view_296: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_16, _shape_param_400);  div_16 = _shape_param_400 = None
        permute_241: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_296, [1, 0, 2, 3]);  view_296 = None
        permute_242: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_241, [0, 2, 1, 3]);  permute_241 = None
        view_297: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_242, _shape_param_401);  permute_242 = _shape_param_401 = None
        view_298: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_297, _shape_param_402);  view_297 = _shape_param_402 = None
        as_strided_27: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_298, _shape_param_403, _shape_param_404);  view_298 = _shape_param_403 = _shape_param_404 = None
        unsqueeze_43: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_27, 4);  as_strided_27 = None
        clone_65: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_43, memory_format = torch.contiguous_format);  unsqueeze_43 = None
        view_299: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_65, _shape_param_405);  clone_65 = _shape_param_405 = None
        permute_243: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_240, [0, 1, 4, 3, 2]);  permute_240 = None
        clone_66: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
        view_300: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_66, _shape_param_406);  clone_66 = _shape_param_406 = None
        bmm_16: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_299, view_300)
        view_301: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_16, _shape_param_407);  bmm_16 = _shape_param_407 = None
        permute_244: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_301, [0, 1, 2, 4, 3]);  view_301 = None
        view_302: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_244, _shape_param_408);  permute_244 = _shape_param_408 = None
        constant_pad_nd_25: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_302, [0, 0, 0, 1], 0.0);  view_302 = None
        view_303: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_25, _shape_param_409);  constant_pad_nd_25 = _shape_param_409 = None
        slice_174: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_303, 2, 0, 256)
        slice_175: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_174, 3, 0, 257);  slice_174 = None
        copy_54: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_175);  slice_175 = None
        slice_scatter_99: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_54, 3, 256, 9223372036854775807);  copy_54 = None
        slice_scatter_100: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_99, 1, 0, -1);  slice_scatter_99 = None
        select_36: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_303, 1, -1)
        slice_176: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_36, 1, 256, 9223372036854775807);  select_36 = None
        slice_177: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_176, 2, 0, 257);  slice_176 = None
        select_37: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_100, 1, -1)
        slice_178: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_37, 2, 256, 9223372036854775807)
        copy_55: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_178, slice_177);  slice_178 = slice_177 = None
        slice_scatter_101: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_37, copy_55, 2, 256, 9223372036854775807);  select_37 = copy_55 = None
        select_scatter_18: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_100, slice_scatter_101, 1, -1);  slice_scatter_100 = slice_scatter_101 = None
        slice_179: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_303, 2, -257, -1)
        slice_180: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_179, 3, 257, 9223372036854775807);  slice_179 = None
        slice_181: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_18, 1, 1, 9223372036854775807)
        slice_182: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_181, 3, 0, 256)
        copy_56: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_182, slice_180);  slice_182 = slice_180 = None
        slice_scatter_102: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_181, copy_56, 3, 0, 256);  slice_181 = copy_56 = None
        slice_scatter_103: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_18, slice_scatter_102, 1, 1, 9223372036854775807);  select_scatter_18 = slice_scatter_102 = None
        select_38: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_303, 1, 0);  view_303 = None
        slice_183: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_38, 1, 0, 255);  select_38 = None
        slice_184: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_183, 2, -255, 9223372036854775807);  slice_183 = None
        select_39: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_103, 1, 0)
        slice_185: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_39, 1, 1, 256)
        slice_186: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_185, 2, 1, 256)
        copy_57: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_186, slice_184);  slice_186 = slice_184 = None
        slice_scatter_104: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_185, copy_57, 2, 1, 256);  slice_185 = copy_57 = None
        slice_scatter_105: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_39, slice_scatter_104, 1, 1, 256);  select_39 = slice_scatter_104 = None
        select_scatter_19: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_103, slice_scatter_105, 1, 0);  slice_scatter_103 = slice_scatter_105 = None
        view_304: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_19, _shape_param_410);  select_scatter_19 = _shape_param_410 = None
        permute_245: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None
        slice_187: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_245, 1, 0, 256)
        slice_188: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_187, 3, 0, 257)
        where_28: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_7, permute_12, slice_188)
        copy_58: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_188, where_28);  slice_188 = where_28 = None
        slice_scatter_106: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_187, copy_58, 3, 0, 257);  slice_187 = copy_58 = None
        slice_scatter_107: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_245, slice_scatter_106, 1, 0, 256);  permute_245 = slice_scatter_106 = None
        permute_246: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_107, [0, 2, 1, 3]);  slice_scatter_107 = None
        view_305: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_246, _shape_param_411);  permute_246 = _shape_param_411 = None
        view_306: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_305, _shape_param_412);  view_305 = _shape_param_412 = None
        permute_247: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None
        slice_189: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_247, 1, -256, 9223372036854775807)
        slice_190: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_189, 3, -257, 9223372036854775807)
        where_29: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_8, permute_12, slice_190)
        copy_59: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_190, where_29);  slice_190 = where_29 = None
        slice_scatter_108: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_189, copy_59, 3, -257, 9223372036854775807);  slice_189 = copy_59 = None
        slice_scatter_109: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_247, slice_scatter_108, 1, -256, 9223372036854775807);  permute_247 = slice_scatter_108 = None
        permute_248: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_109, [0, 2, 1, 3]);  slice_scatter_109 = None
        permute_249: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_248, [0, 2, 1, 3]);  permute_248 = None
        add_99: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_249, permute_25);  permute_249 = None
        permute_250: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_99, [0, 2, 1, 3]);  add_99 = None
        permute_251: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_250, [0, 2, 1, 3]);  permute_250 = None
        convert_element_type_180: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_251, torch.float32)
        clone_67: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_180, memory_format = torch.contiguous_format);  convert_element_type_180 = None
        amax_8: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_67, [-1], True)
        sub_25: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_67, amax_8);  clone_67 = None
        exp_8: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_17: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = None
        where_30: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_11, full_7, div_17);  div_17 = None
        convert_element_type_181: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_30, torch.bfloat16);  where_30 = None
        inductor_lookup_seed_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 24)
        inductor_random_24: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default(_shape_param_413, inductor_lookup_seed_24, 'rand');  _shape_param_413 = inductor_lookup_seed_24 = None
        convert_element_type_182: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(inductor_random_24, torch.bfloat16);  inductor_random_24 = None
        gt_24: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(convert_element_type_182, 0.1);  convert_element_type_182 = None
        mul_105: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_24, convert_element_type_181);  convert_element_type_181 = None
        mul_106: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_105, 1.1111111111111112);  mul_105 = None
        view_307: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_98, _shape_param_414);  add_98 = _shape_param_414 = None
        permute_252: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_307, [1, 0, 2, 3]);  view_307 = None
        permute_253: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_106, [0, 2, 1, 3]);  mul_106 = None
        clone_68: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
        view_308: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_68, _shape_param_415);  clone_68 = _shape_param_415 = None
        permute_254: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_252, [0, 2, 1, 3]);  permute_252 = None
        view_309: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_254, _shape_param_416);  permute_254 = _shape_param_416 = None
        constant_pad_nd_26: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_309, _shape_param_417, -1.0);  view_309 = _shape_param_417 = None
        as_strided_28: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_26, _shape_param_418, _shape_param_419);  constant_pad_nd_26 = _shape_param_418 = _shape_param_419 = None
        constant_pad_nd_27: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_308, _shape_param_420, 0.0);  view_308 = _shape_param_420 = None
        view_310: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_27, _shape_param_421);  constant_pad_nd_27 = _shape_param_421 = None
        slice_191: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_310, 2, 0, -256);  view_310 = None
        view_311: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_191, _shape_param_422);  slice_191 = _shape_param_422 = None
        slice_192: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_311, 3, 0, -1);  view_311 = None
        unsqueeze_44: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_192, 4);  slice_192 = None
        unsqueeze_45: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_28, 4);  as_strided_28 = None
        view_312: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_44, _shape_param_423);  unsqueeze_44 = _shape_param_423 = None
        clone_69: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_45, memory_format = torch.contiguous_format);  unsqueeze_45 = None
        view_313: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_69, _shape_param_424);  clone_69 = _shape_param_424 = None
        bmm_17: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_312, view_313)
        view_314: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_17, _shape_param_425);  bmm_17 = _shape_param_425 = None
        permute_255: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_314, [0, 1, 2, 4, 3]);  view_314 = None
        view_315: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_255, _shape_param_426);  permute_255 = _shape_param_426 = None
        view_316: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_315, _shape_param_427);  view_315 = _shape_param_427 = None
        permute_256: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
        permute_257: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_256, [1, 0, 2, 3]);  permute_256 = None
        clone_70: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
        view_317: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_70, _shape_param_428);  clone_70 = _shape_param_428 = None
        permute_258: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_317, [1, 0, 2]);  view_317 = None
        convert_element_type_183: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg137_1, torch.bfloat16);  arg137_1 = None
        convert_element_type_184: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg138_1, torch.bfloat16);  arg138_1 = None
        permute_259: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_184, [1, 0]);  convert_element_type_184 = None
        clone_71: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
        view_318: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_71, _shape_param_429);  clone_71 = _shape_param_429 = None
        mm_35: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_318, permute_259)
        view_319: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_35, _shape_param_430);  mm_35 = _shape_param_430 = None
        add_100: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_319, convert_element_type_183);  view_319 = convert_element_type_183 = None
        inductor_lookup_seed_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 25)
        inductor_random_25: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_431, inductor_lookup_seed_25, 'rand');  _shape_param_431 = inductor_lookup_seed_25 = None
        convert_element_type_185: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_25, torch.bfloat16);  inductor_random_25 = None
        gt_25: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_185, 0.1);  convert_element_type_185 = None
        mul_107: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_25, add_100);  add_100 = None
        mul_108: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_107, 1.1111111111111112);  mul_107 = None
        add_101: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_108, add_95);  mul_108 = add_95 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
        getitem_32: "f32[8, 1024, 1]" = var_mean_16[0]
        getitem_33: "f32[8, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
        add_102: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_26: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_101, getitem_33);  add_101 = getitem_33 = None
        mul_109: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_16);  sub_26 = None
        mul_110: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_109, arg139_1);  arg139_1 = None
        add_103: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_110, arg140_1);  mul_110 = arg140_1 = None
        convert_element_type_186: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg141_1, torch.bfloat16);  arg141_1 = None
        convert_element_type_187: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg142_1, torch.bfloat16);  arg142_1 = None
        convert_element_type_188: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16)
        view_320: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_188, _shape_param_432);  convert_element_type_188 = _shape_param_432 = None
        permute_260: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_187, [1, 0]);  convert_element_type_187 = None
        addmm_16: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(convert_element_type_186, view_320, permute_260);  convert_element_type_186 = None
        view_321: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_16, _shape_param_433);  _shape_param_433 = None
        convert_element_type_189: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_321, torch.float32);  view_321 = None
        mul_111: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_189, 0.5)
        mul_112: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_189, 0.7071067811865476);  convert_element_type_189 = None
        erf_8: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_104: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_113: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_111, add_104);  mul_111 = add_104 = None
        convert_element_type_190: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_113, torch.bfloat16);  mul_113 = None
        convert_element_type_191: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg143_1, torch.bfloat16);  arg143_1 = None
        convert_element_type_192: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg144_1, torch.bfloat16);  arg144_1 = None
        view_322: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_190, _shape_param_434);  convert_element_type_190 = _shape_param_434 = None
        permute_261: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_192, [1, 0]);  convert_element_type_192 = None
        addmm_17: "bf16[8192, 768]" = torch.ops.aten.addmm.default(convert_element_type_191, view_322, permute_261);  convert_element_type_191 = None
        view_323: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_17, _shape_param_435);  addmm_17 = _shape_param_435 = None
        inductor_lookup_seed_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 26)
        inductor_random_26: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_436, inductor_lookup_seed_26, 'rand');  _shape_param_436 = inductor_lookup_seed_26 = None
        convert_element_type_193: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_26, torch.bfloat16);  inductor_random_26 = None
        gt_26: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_193, 0.1);  convert_element_type_193 = None
        mul_114: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_26, view_323);  view_323 = None
        mul_115: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_114, 1.1111111111111112);  mul_114 = None
        add_105: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_115, add_103);  mul_115 = add_103 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
        getitem_34: "f32[8, 1024, 1]" = var_mean_17[0]
        getitem_35: "f32[8, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
        add_106: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_27: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_105, getitem_35);  add_105 = getitem_35 = None
        mul_116: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = None
        mul_117: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_116, arg145_1);  arg145_1 = None
        add_107: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_117, arg146_1);  mul_117 = arg146_1 = None
        permute_262: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_107, [1, 0, 2])
        convert_element_type_194: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg147_1, torch.bfloat16);  arg147_1 = None
        convert_element_type_195: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg148_1, torch.bfloat16);  arg148_1 = None
        convert_element_type_196: "bf16[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(permute_262, torch.bfloat16);  permute_262 = None
        permute_263: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_195, [1, 0]);  convert_element_type_195 = None
        clone_72: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(convert_element_type_196, memory_format = torch.contiguous_format);  convert_element_type_196 = None
        view_324: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_72, _shape_param_437);  clone_72 = _shape_param_437 = None
        mm_36: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_324, permute_263)
        view_325: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_36, _shape_param_438);  mm_36 = _shape_param_438 = None
        add_108: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_325, convert_element_type_194);  view_325 = convert_element_type_194 = None
        convert_element_type_197: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg149_1, torch.bfloat16);  arg149_1 = None
        convert_element_type_198: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg150_1, torch.bfloat16);  arg150_1 = None
        permute_264: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_198, [1, 0]);  convert_element_type_198 = None
        mm_37: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_324, permute_264)
        view_326: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_37, _shape_param_439);  mm_37 = _shape_param_439 = None
        add_109: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_326, convert_element_type_197);  view_326 = convert_element_type_197 = None
        convert_element_type_199: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg151_1, torch.bfloat16);  arg151_1 = None
        convert_element_type_200: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg152_1, torch.bfloat16);  arg152_1 = None
        permute_265: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_200, [1, 0]);  convert_element_type_200 = None
        mm_38: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_324, permute_265)
        view_327: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_38, _shape_param_440);  mm_38 = _shape_param_440 = None
        add_110: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_327, convert_element_type_199);  view_327 = convert_element_type_199 = None
        div_18: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_108, 8.0);  add_108 = None
        view_328: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_109, _shape_param_441);  add_109 = _shape_param_441 = None
        permute_266: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_328, [1, 0, 2, 3]);  view_328 = None
        permute_267: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_266, [0, 2, 1, 3]);  permute_266 = None
        view_329: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_267, _shape_param_442);  permute_267 = _shape_param_442 = None
        view_330: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_329, _shape_param_443);  view_329 = _shape_param_443 = None
        as_strided_29: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_330, _shape_param_444, _shape_param_445);  view_330 = _shape_param_444 = _shape_param_445 = None
        unsqueeze_46: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_29, 4);  as_strided_29 = None
        permute_268: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_46, [0, 1, 4, 2, 3]);  unsqueeze_46 = None
        view_331: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_18, _shape_param_446);  div_18 = _shape_param_446 = None
        permute_269: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_331, [1, 0, 2, 3]);  view_331 = None
        permute_270: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_269, [0, 2, 1, 3]);  permute_269 = None
        view_332: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_270, _shape_param_447);  permute_270 = _shape_param_447 = None
        view_333: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_332, _shape_param_448);  view_332 = _shape_param_448 = None
        as_strided_30: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_333, _shape_param_449, _shape_param_450);  view_333 = _shape_param_449 = _shape_param_450 = None
        unsqueeze_47: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_30, 4);  as_strided_30 = None
        clone_73: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_47, memory_format = torch.contiguous_format);  unsqueeze_47 = None
        view_334: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_73, _shape_param_451);  clone_73 = _shape_param_451 = None
        permute_271: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_268, [0, 1, 4, 3, 2]);  permute_268 = None
        clone_74: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_271, memory_format = torch.contiguous_format);  permute_271 = None
        view_335: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_74, _shape_param_452);  clone_74 = _shape_param_452 = None
        bmm_18: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_334, view_335)
        view_336: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_18, _shape_param_453);  bmm_18 = _shape_param_453 = None
        permute_272: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_336, [0, 1, 2, 4, 3]);  view_336 = None
        view_337: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_272, _shape_param_454);  permute_272 = _shape_param_454 = None
        constant_pad_nd_28: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_337, [0, 0, 0, 1], 0.0);  view_337 = None
        view_338: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_28, _shape_param_455);  constant_pad_nd_28 = _shape_param_455 = None
        slice_193: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_338, 2, 0, 256)
        slice_194: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_193, 3, 0, 257);  slice_193 = None
        copy_60: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_194);  slice_194 = None
        slice_scatter_110: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_60, 3, 256, 9223372036854775807);  copy_60 = None
        slice_scatter_111: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_110, 1, 0, -1);  slice_scatter_110 = None
        select_40: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_338, 1, -1)
        slice_195: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_40, 1, 256, 9223372036854775807);  select_40 = None
        slice_196: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_195, 2, 0, 257);  slice_195 = None
        select_41: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_111, 1, -1)
        slice_197: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_41, 2, 256, 9223372036854775807)
        copy_61: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_197, slice_196);  slice_197 = slice_196 = None
        slice_scatter_112: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_41, copy_61, 2, 256, 9223372036854775807);  select_41 = copy_61 = None
        select_scatter_20: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_111, slice_scatter_112, 1, -1);  slice_scatter_111 = slice_scatter_112 = None
        slice_198: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_338, 2, -257, -1)
        slice_199: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_198, 3, 257, 9223372036854775807);  slice_198 = None
        slice_200: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_20, 1, 1, 9223372036854775807)
        slice_201: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_200, 3, 0, 256)
        copy_62: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_201, slice_199);  slice_201 = slice_199 = None
        slice_scatter_113: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_200, copy_62, 3, 0, 256);  slice_200 = copy_62 = None
        slice_scatter_114: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_20, slice_scatter_113, 1, 1, 9223372036854775807);  select_scatter_20 = slice_scatter_113 = None
        select_42: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_338, 1, 0);  view_338 = None
        slice_202: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_42, 1, 0, 255);  select_42 = None
        slice_203: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_202, 2, -255, 9223372036854775807);  slice_202 = None
        select_43: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_114, 1, 0)
        slice_204: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_43, 1, 1, 256)
        slice_205: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_204, 2, 1, 256)
        copy_63: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_205, slice_203);  slice_205 = slice_203 = None
        slice_scatter_115: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_204, copy_63, 2, 1, 256);  slice_204 = copy_63 = None
        slice_scatter_116: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_43, slice_scatter_115, 1, 1, 256);  select_43 = slice_scatter_115 = None
        select_scatter_21: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_114, slice_scatter_116, 1, 0);  slice_scatter_114 = slice_scatter_116 = None
        view_339: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_21, _shape_param_456);  select_scatter_21 = _shape_param_456 = None
        permute_273: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
        slice_206: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_273, 1, 0, 256)
        slice_207: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_206, 3, 0, 257)
        where_31: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_7, permute_12, slice_207)
        copy_64: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_207, where_31);  slice_207 = where_31 = None
        slice_scatter_117: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_206, copy_64, 3, 0, 257);  slice_206 = copy_64 = None
        slice_scatter_118: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_273, slice_scatter_117, 1, 0, 256);  permute_273 = slice_scatter_117 = None
        permute_274: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_118, [0, 2, 1, 3]);  slice_scatter_118 = None
        view_340: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_274, _shape_param_457);  permute_274 = _shape_param_457 = None
        view_341: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_340, _shape_param_458);  view_340 = _shape_param_458 = None
        permute_275: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None
        slice_208: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_275, 1, -256, 9223372036854775807)
        slice_209: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_208, 3, -257, 9223372036854775807)
        where_32: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_8, permute_12, slice_209)
        copy_65: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_209, where_32);  slice_209 = where_32 = None
        slice_scatter_119: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_208, copy_65, 3, -257, 9223372036854775807);  slice_208 = copy_65 = None
        slice_scatter_120: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_275, slice_scatter_119, 1, -256, 9223372036854775807);  permute_275 = slice_scatter_119 = None
        permute_276: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_120, [0, 2, 1, 3]);  slice_scatter_120 = None
        permute_277: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_276, [0, 2, 1, 3]);  permute_276 = None
        add_111: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_277, permute_25);  permute_277 = None
        permute_278: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_111, [0, 2, 1, 3]);  add_111 = None
        permute_279: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_278, [0, 2, 1, 3]);  permute_278 = None
        convert_element_type_201: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_279, torch.float32)
        clone_75: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_201, memory_format = torch.contiguous_format);  convert_element_type_201 = None
        amax_9: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_75, [-1], True)
        sub_28: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_75, amax_9);  clone_75 = None
        exp_9: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_19: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = None
        where_33: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_11, full_7, div_19);  div_19 = None
        convert_element_type_202: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_33, torch.bfloat16);  where_33 = None
        inductor_lookup_seed_27: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 27)
        inductor_random_27: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default(_shape_param_459, inductor_lookup_seed_27, 'rand');  _shape_param_459 = inductor_lookup_seed_27 = None
        convert_element_type_203: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(inductor_random_27, torch.bfloat16);  inductor_random_27 = None
        gt_27: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(convert_element_type_203, 0.1);  convert_element_type_203 = None
        mul_118: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_27, convert_element_type_202);  convert_element_type_202 = None
        mul_119: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_118, 1.1111111111111112);  mul_118 = None
        view_342: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_110, _shape_param_460);  add_110 = _shape_param_460 = None
        permute_280: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_342, [1, 0, 2, 3]);  view_342 = None
        permute_281: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_119, [0, 2, 1, 3]);  mul_119 = None
        clone_76: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_281, memory_format = torch.contiguous_format);  permute_281 = None
        view_343: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_76, _shape_param_461);  clone_76 = _shape_param_461 = None
        permute_282: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_280, [0, 2, 1, 3]);  permute_280 = None
        view_344: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_282, _shape_param_462);  permute_282 = _shape_param_462 = None
        constant_pad_nd_29: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_344, _shape_param_463, -1.0);  view_344 = _shape_param_463 = None
        as_strided_31: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_29, _shape_param_464, _shape_param_465);  constant_pad_nd_29 = _shape_param_464 = _shape_param_465 = None
        constant_pad_nd_30: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_343, _shape_param_466, 0.0);  view_343 = _shape_param_466 = None
        view_345: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_30, _shape_param_467);  constant_pad_nd_30 = _shape_param_467 = None
        slice_210: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_345, 2, 0, -256);  view_345 = None
        view_346: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_210, _shape_param_468);  slice_210 = _shape_param_468 = None
        slice_211: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_346, 3, 0, -1);  view_346 = None
        unsqueeze_48: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_211, 4);  slice_211 = None
        unsqueeze_49: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_31, 4);  as_strided_31 = None
        view_347: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_48, _shape_param_469);  unsqueeze_48 = _shape_param_469 = None
        clone_77: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_49, memory_format = torch.contiguous_format);  unsqueeze_49 = None
        view_348: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_77, _shape_param_470);  clone_77 = _shape_param_470 = None
        bmm_19: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_347, view_348)
        view_349: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_471);  bmm_19 = _shape_param_471 = None
        permute_283: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_349, [0, 1, 2, 4, 3]);  view_349 = None
        view_350: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_283, _shape_param_472);  permute_283 = _shape_param_472 = None
        view_351: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_350, _shape_param_473);  view_350 = _shape_param_473 = None
        permute_284: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None
        permute_285: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_284, [1, 0, 2, 3]);  permute_284 = None
        clone_78: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_285, memory_format = torch.contiguous_format);  permute_285 = None
        view_352: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_78, _shape_param_474);  clone_78 = _shape_param_474 = None
        permute_286: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_352, [1, 0, 2]);  view_352 = None
        convert_element_type_204: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg153_1, torch.bfloat16);  arg153_1 = None
        convert_element_type_205: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg154_1, torch.bfloat16);  arg154_1 = None
        permute_287: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_205, [1, 0]);  convert_element_type_205 = None
        clone_79: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
        view_353: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_79, _shape_param_475);  clone_79 = _shape_param_475 = None
        mm_39: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_353, permute_287)
        view_354: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_39, _shape_param_476);  mm_39 = _shape_param_476 = None
        add_112: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_354, convert_element_type_204);  view_354 = convert_element_type_204 = None
        inductor_lookup_seed_28: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 28)
        inductor_random_28: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_477, inductor_lookup_seed_28, 'rand');  _shape_param_477 = inductor_lookup_seed_28 = None
        convert_element_type_206: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_28, torch.bfloat16);  inductor_random_28 = None
        gt_28: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_206, 0.1);  convert_element_type_206 = None
        mul_120: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_28, add_112);  add_112 = None
        mul_121: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_120, 1.1111111111111112);  mul_120 = None
        add_113: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_121, add_107);  mul_121 = add_107 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_113, [2], correction = 0, keepdim = True)
        getitem_36: "f32[8, 1024, 1]" = var_mean_18[0]
        getitem_37: "f32[8, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
        add_114: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        sub_29: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_113, getitem_37);  add_113 = getitem_37 = None
        mul_122: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_18);  sub_29 = None
        mul_123: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_122, arg155_1);  arg155_1 = None
        add_115: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_123, arg156_1);  mul_123 = arg156_1 = None
        convert_element_type_207: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg157_1, torch.bfloat16);  arg157_1 = None
        convert_element_type_208: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg158_1, torch.bfloat16);  arg158_1 = None
        convert_element_type_209: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_115, torch.bfloat16)
        view_355: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_209, _shape_param_478);  convert_element_type_209 = _shape_param_478 = None
        permute_288: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_208, [1, 0]);  convert_element_type_208 = None
        addmm_18: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(convert_element_type_207, view_355, permute_288);  convert_element_type_207 = None
        view_356: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_18, _shape_param_479);  _shape_param_479 = None
        convert_element_type_210: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_356, torch.float32);  view_356 = None
        mul_124: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_210, 0.5)
        mul_125: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_210, 0.7071067811865476);  convert_element_type_210 = None
        erf_9: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_125);  mul_125 = None
        add_116: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_126: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_124, add_116);  mul_124 = add_116 = None
        convert_element_type_211: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16);  mul_126 = None
        convert_element_type_212: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg159_1, torch.bfloat16);  arg159_1 = None
        convert_element_type_213: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg160_1, torch.bfloat16);  arg160_1 = None
        view_357: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_211, _shape_param_480);  convert_element_type_211 = _shape_param_480 = None
        permute_289: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_213, [1, 0]);  convert_element_type_213 = None
        addmm_19: "bf16[8192, 768]" = torch.ops.aten.addmm.default(convert_element_type_212, view_357, permute_289);  convert_element_type_212 = None
        view_358: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_19, _shape_param_481);  addmm_19 = _shape_param_481 = None
        inductor_lookup_seed_29: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 29)
        inductor_random_29: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_482, inductor_lookup_seed_29, 'rand');  _shape_param_482 = inductor_lookup_seed_29 = None
        convert_element_type_214: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_29, torch.bfloat16);  inductor_random_29 = None
        gt_29: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_214, 0.1);  convert_element_type_214 = None
        mul_127: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_29, view_358);  view_358 = None
        mul_128: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_127, 1.1111111111111112);  mul_127 = None
        add_117: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_128, add_115);  mul_128 = add_115 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
        getitem_38: "f32[8, 1024, 1]" = var_mean_19[0]
        getitem_39: "f32[8, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
        add_118: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_30: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_117, getitem_39);  add_117 = getitem_39 = None
        mul_129: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = None
        mul_130: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_129, arg161_1);  arg161_1 = None
        add_119: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_130, arg162_1);  mul_130 = arg162_1 = None
        permute_290: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_119, [1, 0, 2])
        convert_element_type_215: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg163_1, torch.bfloat16);  arg163_1 = None
        convert_element_type_216: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg164_1, torch.bfloat16);  arg164_1 = None
        convert_element_type_217: "bf16[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(permute_290, torch.bfloat16);  permute_290 = None
        permute_291: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_216, [1, 0]);  convert_element_type_216 = None
        clone_80: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(convert_element_type_217, memory_format = torch.contiguous_format);  convert_element_type_217 = None
        view_359: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_80, _shape_param_483);  clone_80 = _shape_param_483 = None
        mm_40: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_359, permute_291)
        view_360: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_40, _shape_param_484);  mm_40 = _shape_param_484 = None
        add_120: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_360, convert_element_type_215);  view_360 = convert_element_type_215 = None
        convert_element_type_218: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg165_1, torch.bfloat16);  arg165_1 = None
        convert_element_type_219: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg166_1, torch.bfloat16);  arg166_1 = None
        permute_292: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_219, [1, 0]);  convert_element_type_219 = None
        mm_41: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_359, permute_292)
        view_361: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_41, _shape_param_485);  mm_41 = _shape_param_485 = None
        add_121: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_361, convert_element_type_218);  view_361 = convert_element_type_218 = None
        convert_element_type_220: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg167_1, torch.bfloat16);  arg167_1 = None
        convert_element_type_221: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg168_1, torch.bfloat16);  arg168_1 = None
        permute_293: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_221, [1, 0]);  convert_element_type_221 = None
        mm_42: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_359, permute_293)
        view_362: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_42, _shape_param_486);  mm_42 = _shape_param_486 = None
        add_122: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_362, convert_element_type_220);  view_362 = convert_element_type_220 = None
        div_20: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_120, 8.0);  add_120 = None
        view_363: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_121, _shape_param_487);  add_121 = _shape_param_487 = None
        permute_294: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_363, [1, 0, 2, 3]);  view_363 = None
        permute_295: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_294, [0, 2, 1, 3]);  permute_294 = None
        view_364: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_295, _shape_param_488);  permute_295 = _shape_param_488 = None
        view_365: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_364, _shape_param_489);  view_364 = _shape_param_489 = None
        as_strided_32: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_365, _shape_param_490, _shape_param_491);  view_365 = _shape_param_490 = _shape_param_491 = None
        unsqueeze_50: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_32, 4);  as_strided_32 = None
        permute_296: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_50, [0, 1, 4, 2, 3]);  unsqueeze_50 = None
        view_366: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_20, _shape_param_492);  div_20 = _shape_param_492 = None
        permute_297: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_366, [1, 0, 2, 3]);  view_366 = None
        permute_298: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_297, [0, 2, 1, 3]);  permute_297 = None
        view_367: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_298, _shape_param_493);  permute_298 = _shape_param_493 = None
        view_368: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_367, _shape_param_494);  view_367 = _shape_param_494 = None
        as_strided_33: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_368, _shape_param_495, _shape_param_496);  view_368 = _shape_param_495 = _shape_param_496 = None
        unsqueeze_51: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_33, 4);  as_strided_33 = None
        clone_81: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_51, memory_format = torch.contiguous_format);  unsqueeze_51 = None
        view_369: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_81, _shape_param_497);  clone_81 = _shape_param_497 = None
        permute_299: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_296, [0, 1, 4, 3, 2]);  permute_296 = None
        clone_82: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
        view_370: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_82, _shape_param_498);  clone_82 = _shape_param_498 = None
        bmm_20: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_369, view_370)
        view_371: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_20, _shape_param_499);  bmm_20 = _shape_param_499 = None
        permute_300: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_371, [0, 1, 2, 4, 3]);  view_371 = None
        view_372: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_300, _shape_param_500);  permute_300 = _shape_param_500 = None
        constant_pad_nd_31: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_372, [0, 0, 0, 1], 0.0);  view_372 = None
        view_373: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_31, _shape_param_501);  constant_pad_nd_31 = _shape_param_501 = None
        slice_212: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_373, 2, 0, 256)
        slice_213: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_212, 3, 0, 257);  slice_212 = None
        copy_66: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_213);  slice_213 = None
        slice_scatter_121: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_66, 3, 256, 9223372036854775807);  copy_66 = None
        slice_scatter_122: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_121, 1, 0, -1);  slice_scatter_121 = None
        select_44: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_373, 1, -1)
        slice_214: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_44, 1, 256, 9223372036854775807);  select_44 = None
        slice_215: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_214, 2, 0, 257);  slice_214 = None
        select_45: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_122, 1, -1)
        slice_216: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_45, 2, 256, 9223372036854775807)
        copy_67: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_216, slice_215);  slice_216 = slice_215 = None
        slice_scatter_123: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_45, copy_67, 2, 256, 9223372036854775807);  select_45 = copy_67 = None
        select_scatter_22: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_122, slice_scatter_123, 1, -1);  slice_scatter_122 = slice_scatter_123 = None
        slice_217: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_373, 2, -257, -1)
        slice_218: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_217, 3, 257, 9223372036854775807);  slice_217 = None
        slice_219: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_22, 1, 1, 9223372036854775807)
        slice_220: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_219, 3, 0, 256)
        copy_68: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_220, slice_218);  slice_220 = slice_218 = None
        slice_scatter_124: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_219, copy_68, 3, 0, 256);  slice_219 = copy_68 = None
        slice_scatter_125: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_22, slice_scatter_124, 1, 1, 9223372036854775807);  select_scatter_22 = slice_scatter_124 = None
        select_46: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_373, 1, 0);  view_373 = None
        slice_221: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_46, 1, 0, 255);  select_46 = None
        slice_222: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_221, 2, -255, 9223372036854775807);  slice_221 = None
        select_47: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_125, 1, 0)
        slice_223: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_47, 1, 1, 256)
        slice_224: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_223, 2, 1, 256)
        copy_69: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_224, slice_222);  slice_224 = slice_222 = None
        slice_scatter_126: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_223, copy_69, 2, 1, 256);  slice_223 = copy_69 = None
        slice_scatter_127: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_47, slice_scatter_126, 1, 1, 256);  select_47 = slice_scatter_126 = None
        select_scatter_23: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_125, slice_scatter_127, 1, 0);  slice_scatter_125 = slice_scatter_127 = None
        view_374: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_23, _shape_param_502);  select_scatter_23 = _shape_param_502 = None
        permute_301: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
        slice_225: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_301, 1, 0, 256)
        slice_226: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_225, 3, 0, 257)
        where_34: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_7, permute_12, slice_226)
        copy_70: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_226, where_34);  slice_226 = where_34 = None
        slice_scatter_128: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_225, copy_70, 3, 0, 257);  slice_225 = copy_70 = None
        slice_scatter_129: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_301, slice_scatter_128, 1, 0, 256);  permute_301 = slice_scatter_128 = None
        permute_302: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_129, [0, 2, 1, 3]);  slice_scatter_129 = None
        view_375: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_302, _shape_param_503);  permute_302 = _shape_param_503 = None
        view_376: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_375, _shape_param_504);  view_375 = _shape_param_504 = None
        permute_303: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        slice_227: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_303, 1, -256, 9223372036854775807)
        slice_228: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_227, 3, -257, 9223372036854775807)
        where_35: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_8, permute_12, slice_228)
        copy_71: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_228, where_35);  slice_228 = where_35 = None
        slice_scatter_130: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_227, copy_71, 3, -257, 9223372036854775807);  slice_227 = copy_71 = None
        slice_scatter_131: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_303, slice_scatter_130, 1, -256, 9223372036854775807);  permute_303 = slice_scatter_130 = None
        permute_304: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_131, [0, 2, 1, 3]);  slice_scatter_131 = None
        permute_305: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_304, [0, 2, 1, 3]);  permute_304 = None
        add_123: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_305, permute_25);  permute_305 = None
        permute_306: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_123, [0, 2, 1, 3]);  add_123 = None
        permute_307: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_306, [0, 2, 1, 3]);  permute_306 = None
        convert_element_type_222: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_307, torch.float32)
        clone_83: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_222, memory_format = torch.contiguous_format);  convert_element_type_222 = None
        amax_10: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_83, [-1], True)
        sub_31: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_83, amax_10);  clone_83 = None
        exp_10: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_21: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = None
        where_36: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_11, full_7, div_21);  div_21 = None
        convert_element_type_223: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_36, torch.bfloat16);  where_36 = None
        inductor_lookup_seed_30: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 30)
        inductor_random_30: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default(_shape_param_505, inductor_lookup_seed_30, 'rand');  _shape_param_505 = inductor_lookup_seed_30 = None
        convert_element_type_224: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(inductor_random_30, torch.bfloat16);  inductor_random_30 = None
        gt_30: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(convert_element_type_224, 0.1);  convert_element_type_224 = None
        mul_131: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_30, convert_element_type_223);  convert_element_type_223 = None
        mul_132: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_131, 1.1111111111111112);  mul_131 = None
        view_377: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_122, _shape_param_506);  add_122 = _shape_param_506 = None
        permute_308: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_377, [1, 0, 2, 3]);  view_377 = None
        permute_309: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_132, [0, 2, 1, 3]);  mul_132 = None
        clone_84: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_309, memory_format = torch.contiguous_format);  permute_309 = None
        view_378: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_84, _shape_param_507);  clone_84 = _shape_param_507 = None
        permute_310: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_308, [0, 2, 1, 3]);  permute_308 = None
        view_379: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_310, _shape_param_508);  permute_310 = _shape_param_508 = None
        constant_pad_nd_32: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_379, _shape_param_509, -1.0);  view_379 = _shape_param_509 = None
        as_strided_34: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_32, _shape_param_510, _shape_param_511);  constant_pad_nd_32 = _shape_param_510 = _shape_param_511 = None
        constant_pad_nd_33: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_378, _shape_param_512, 0.0);  view_378 = _shape_param_512 = None
        view_380: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_33, _shape_param_513);  constant_pad_nd_33 = _shape_param_513 = None
        slice_229: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_380, 2, 0, -256);  view_380 = None
        view_381: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_229, _shape_param_514);  slice_229 = _shape_param_514 = None
        slice_230: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_381, 3, 0, -1);  view_381 = None
        unsqueeze_52: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_230, 4);  slice_230 = None
        unsqueeze_53: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_34, 4);  as_strided_34 = None
        view_382: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_52, _shape_param_515);  unsqueeze_52 = _shape_param_515 = None
        clone_85: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_53, memory_format = torch.contiguous_format);  unsqueeze_53 = None
        view_383: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_85, _shape_param_516);  clone_85 = _shape_param_516 = None
        bmm_21: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_382, view_383)
        view_384: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_21, _shape_param_517);  bmm_21 = _shape_param_517 = None
        permute_311: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_384, [0, 1, 2, 4, 3]);  view_384 = None
        view_385: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_311, _shape_param_518);  permute_311 = _shape_param_518 = None
        view_386: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_385, _shape_param_519);  view_385 = _shape_param_519 = None
        permute_312: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None
        permute_313: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_312, [1, 0, 2, 3]);  permute_312 = None
        clone_86: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_313, memory_format = torch.contiguous_format);  permute_313 = None
        view_387: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_86, _shape_param_520);  clone_86 = _shape_param_520 = None
        permute_314: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_387, [1, 0, 2]);  view_387 = None
        convert_element_type_225: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg169_1, torch.bfloat16);  arg169_1 = None
        convert_element_type_226: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg170_1, torch.bfloat16);  arg170_1 = None
        permute_315: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_226, [1, 0]);  convert_element_type_226 = None
        clone_87: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
        view_388: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_87, _shape_param_521);  clone_87 = _shape_param_521 = None
        mm_43: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_388, permute_315)
        view_389: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_43, _shape_param_522);  mm_43 = _shape_param_522 = None
        add_124: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_389, convert_element_type_225);  view_389 = convert_element_type_225 = None
        inductor_lookup_seed_31: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 31)
        inductor_random_31: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_523, inductor_lookup_seed_31, 'rand');  _shape_param_523 = inductor_lookup_seed_31 = None
        convert_element_type_227: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_31, torch.bfloat16);  inductor_random_31 = None
        gt_31: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_227, 0.1);  convert_element_type_227 = None
        mul_133: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_31, add_124);  add_124 = None
        mul_134: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_133, 1.1111111111111112);  mul_133 = None
        add_125: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_134, add_119);  mul_134 = add_119 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
        getitem_40: "f32[8, 1024, 1]" = var_mean_20[0]
        getitem_41: "f32[8, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
        add_126: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_32: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_125, getitem_41);  add_125 = getitem_41 = None
        mul_135: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_20);  sub_32 = None
        mul_136: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_135, arg171_1);  arg171_1 = None
        add_127: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_136, arg172_1);  mul_136 = arg172_1 = None
        convert_element_type_228: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg173_1, torch.bfloat16);  arg173_1 = None
        convert_element_type_229: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg174_1, torch.bfloat16);  arg174_1 = None
        convert_element_type_230: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_127, torch.bfloat16)
        view_390: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_230, _shape_param_524);  convert_element_type_230 = _shape_param_524 = None
        permute_316: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_229, [1, 0]);  convert_element_type_229 = None
        addmm_20: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(convert_element_type_228, view_390, permute_316);  convert_element_type_228 = None
        view_391: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_20, _shape_param_525);  _shape_param_525 = None
        convert_element_type_231: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_391, torch.float32);  view_391 = None
        mul_137: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_231, 0.5)
        mul_138: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_231, 0.7071067811865476);  convert_element_type_231 = None
        erf_10: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_138);  mul_138 = None
        add_128: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_139: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_137, add_128);  mul_137 = add_128 = None
        convert_element_type_232: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_139, torch.bfloat16);  mul_139 = None
        convert_element_type_233: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg175_1, torch.bfloat16);  arg175_1 = None
        convert_element_type_234: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg176_1, torch.bfloat16);  arg176_1 = None
        view_392: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_232, _shape_param_526);  convert_element_type_232 = _shape_param_526 = None
        permute_317: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_234, [1, 0]);  convert_element_type_234 = None
        addmm_21: "bf16[8192, 768]" = torch.ops.aten.addmm.default(convert_element_type_233, view_392, permute_317);  convert_element_type_233 = None
        view_393: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_21, _shape_param_527);  addmm_21 = _shape_param_527 = None
        inductor_lookup_seed_32: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 32)
        inductor_random_32: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_528, inductor_lookup_seed_32, 'rand');  _shape_param_528 = inductor_lookup_seed_32 = None
        convert_element_type_235: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_32, torch.bfloat16);  inductor_random_32 = None
        gt_32: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_235, 0.1);  convert_element_type_235 = None
        mul_140: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_32, view_393);  view_393 = None
        mul_141: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_140, 1.1111111111111112);  mul_140 = None
        add_129: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_141, add_127);  mul_141 = add_127 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
        getitem_42: "f32[8, 1024, 1]" = var_mean_21[0]
        getitem_43: "f32[8, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
        add_130: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_33: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_129, getitem_43);  add_129 = getitem_43 = None
        mul_142: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
        mul_143: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_142, arg177_1);  arg177_1 = None
        add_131: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_143, arg178_1);  mul_143 = arg178_1 = None
        permute_318: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(add_131, [1, 0, 2])
        convert_element_type_236: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg179_1, torch.bfloat16);  arg179_1 = None
        convert_element_type_237: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg180_1, torch.bfloat16);  arg180_1 = None
        convert_element_type_238: "bf16[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(permute_318, torch.bfloat16);  permute_318 = None
        permute_319: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_237, [1, 0]);  convert_element_type_237 = None
        clone_88: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(convert_element_type_238, memory_format = torch.contiguous_format);  convert_element_type_238 = None
        view_394: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_88, _shape_param_529);  clone_88 = _shape_param_529 = None
        mm_44: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_394, permute_319)
        view_395: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_44, _shape_param_530);  mm_44 = _shape_param_530 = None
        add_132: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_395, convert_element_type_236);  view_395 = convert_element_type_236 = None
        convert_element_type_239: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg181_1, torch.bfloat16);  arg181_1 = None
        convert_element_type_240: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg182_1, torch.bfloat16);  arg182_1 = None
        permute_320: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_240, [1, 0]);  convert_element_type_240 = None
        mm_45: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_394, permute_320)
        view_396: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_45, _shape_param_531);  mm_45 = _shape_param_531 = None
        add_133: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_396, convert_element_type_239);  view_396 = convert_element_type_239 = None
        convert_element_type_241: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg183_1, torch.bfloat16);  arg183_1 = None
        convert_element_type_242: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg184_1, torch.bfloat16);  arg184_1 = None
        permute_321: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_242, [1, 0]);  convert_element_type_242 = None
        mm_46: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_394, permute_321)
        view_397: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_46, _shape_param_532);  mm_46 = _shape_param_532 = None
        add_134: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_397, convert_element_type_241);  view_397 = convert_element_type_241 = None
        div_22: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_132, 8.0);  add_132 = None
        view_398: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_133, _shape_param_533);  add_133 = _shape_param_533 = None
        permute_322: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_398, [1, 0, 2, 3]);  view_398 = None
        permute_323: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_322, [0, 2, 1, 3]);  permute_322 = None
        view_399: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_323, _shape_param_534);  permute_323 = _shape_param_534 = None
        view_400: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_399, _shape_param_535);  view_399 = _shape_param_535 = None
        as_strided_35: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_400, _shape_param_536, _shape_param_537);  view_400 = _shape_param_536 = _shape_param_537 = None
        unsqueeze_54: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_35, 4);  as_strided_35 = None
        permute_324: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_54, [0, 1, 4, 2, 3]);  unsqueeze_54 = None
        view_401: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_22, _shape_param_538);  div_22 = _shape_param_538 = None
        permute_325: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_401, [1, 0, 2, 3]);  view_401 = None
        permute_326: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_325, [0, 2, 1, 3]);  permute_325 = None
        view_402: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_326, _shape_param_539);  permute_326 = _shape_param_539 = None
        view_403: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_402, _shape_param_540);  view_402 = _shape_param_540 = None
        as_strided_36: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_403, _shape_param_541, _shape_param_542);  view_403 = _shape_param_541 = _shape_param_542 = None
        unsqueeze_55: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_36, 4);  as_strided_36 = None
        clone_89: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_55, memory_format = torch.contiguous_format);  unsqueeze_55 = None
        view_404: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_89, _shape_param_543);  clone_89 = _shape_param_543 = None
        permute_327: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_324, [0, 1, 4, 3, 2]);  permute_324 = None
        clone_90: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
        view_405: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_90, _shape_param_544);  clone_90 = _shape_param_544 = None
        bmm_22: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_404, view_405)
        view_406: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_22, _shape_param_545);  bmm_22 = _shape_param_545 = None
        permute_328: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_406, [0, 1, 2, 4, 3]);  view_406 = None
        view_407: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_328, _shape_param_546);  permute_328 = _shape_param_546 = None
        constant_pad_nd_34: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_407, [0, 0, 0, 1], 0.0);  view_407 = None
        view_408: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_34, _shape_param_547);  constant_pad_nd_34 = _shape_param_547 = None
        slice_231: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_408, 2, 0, 256)
        slice_232: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_231, 3, 0, 257);  slice_231 = None
        copy_72: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_4, slice_232);  slice_4 = slice_232 = None
        slice_scatter_132: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_3, copy_72, 3, 256, 9223372036854775807);  slice_3 = copy_72 = None
        slice_scatter_133: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full, slice_scatter_132, 1, 0, -1);  full = slice_scatter_132 = None
        select_48: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_408, 1, -1)
        slice_233: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_48, 1, 256, 9223372036854775807);  select_48 = None
        slice_234: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_233, 2, 0, 257);  slice_233 = None
        select_49: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_133, 1, -1)
        slice_235: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_49, 2, 256, 9223372036854775807)
        copy_73: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_235, slice_234);  slice_235 = slice_234 = None
        slice_scatter_134: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_49, copy_73, 2, 256, 9223372036854775807);  select_49 = copy_73 = None
        select_scatter_24: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_133, slice_scatter_134, 1, -1);  slice_scatter_133 = slice_scatter_134 = None
        slice_236: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_408, 2, -257, -1)
        slice_237: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_236, 3, 257, 9223372036854775807);  slice_236 = None
        slice_238: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_24, 1, 1, 9223372036854775807)
        slice_239: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_238, 3, 0, 256)
        copy_74: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_239, slice_237);  slice_239 = slice_237 = None
        slice_scatter_135: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_238, copy_74, 3, 0, 256);  slice_238 = copy_74 = None
        slice_scatter_136: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_24, slice_scatter_135, 1, 1, 9223372036854775807);  select_scatter_24 = slice_scatter_135 = None
        select_50: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_408, 1, 0);  view_408 = None
        slice_240: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_50, 1, 0, 255);  select_50 = None
        slice_241: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_240, 2, -255, 9223372036854775807);  slice_240 = None
        select_51: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_136, 1, 0)
        slice_242: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_51, 1, 1, 256)
        slice_243: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_242, 2, 1, 256)
        copy_75: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_243, slice_241);  slice_243 = slice_241 = None
        slice_scatter_137: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_242, copy_75, 2, 1, 256);  slice_242 = copy_75 = None
        slice_scatter_138: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_51, slice_scatter_137, 1, 1, 256);  select_51 = slice_scatter_137 = None
        select_scatter_25: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_136, slice_scatter_138, 1, 0);  slice_scatter_136 = slice_scatter_138 = None
        view_409: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_25, _shape_param_548);  select_scatter_25 = _shape_param_548 = None
        permute_329: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None
        slice_244: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_329, 1, 0, 256)
        slice_245: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_244, 3, 0, 257)
        where_37: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_7, permute_12, slice_245);  convert_element_type_7 = None
        copy_76: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_245, where_37);  slice_245 = where_37 = None
        slice_scatter_139: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_244, copy_76, 3, 0, 257);  slice_244 = copy_76 = None
        slice_scatter_140: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_329, slice_scatter_139, 1, 0, 256);  permute_329 = slice_scatter_139 = None
        permute_330: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_140, [0, 2, 1, 3]);  slice_scatter_140 = None
        view_410: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_330, _shape_param_549);  permute_330 = _shape_param_549 = None
        view_411: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_410, _shape_param_550);  view_410 = _shape_param_550 = None
        permute_331: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_411, [0, 2, 1, 3]);  view_411 = None
        slice_246: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_331, 1, -256, 9223372036854775807)
        slice_247: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_246, 3, -257, 9223372036854775807)
        where_38: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_8, permute_12, slice_247);  convert_element_type_8 = permute_12 = None
        copy_77: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_247, where_38);  slice_247 = where_38 = None
        slice_scatter_141: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_246, copy_77, 3, -257, 9223372036854775807);  slice_246 = copy_77 = None
        slice_scatter_142: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_331, slice_scatter_141, 1, -256, 9223372036854775807);  permute_331 = slice_scatter_141 = None
        permute_332: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_142, [0, 2, 1, 3]);  slice_scatter_142 = None
        permute_333: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_332, [0, 2, 1, 3]);  permute_332 = None
        add_135: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_333, permute_25);  permute_333 = permute_25 = None
        permute_334: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_135, [0, 2, 1, 3]);  add_135 = None
        permute_335: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_334, [0, 2, 1, 3]);  permute_334 = None
        convert_element_type_243: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_335, torch.float32)
        clone_91: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_243, memory_format = torch.contiguous_format);  convert_element_type_243 = None
        amax_11: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_91, [-1], True)
        sub_34: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_91, amax_11);  clone_91 = None
        exp_11: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_23: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = None
        where_39: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_11, full_7, div_23);  unsqueeze_11 = full_7 = div_23 = None
        convert_element_type_244: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_39, torch.bfloat16);  where_39 = None
        inductor_lookup_seed_33: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 33)
        inductor_random_33: "f32[8, 1024, 12, 513]" = torch.ops.prims.inductor_random.default(_shape_param_551, inductor_lookup_seed_33, 'rand');  _shape_param_551 = inductor_lookup_seed_33 = None
        convert_element_type_245: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(inductor_random_33, torch.bfloat16);  inductor_random_33 = None
        gt_33: "b8[8, 1024, 12, 513]" = torch.ops.aten.gt.Scalar(convert_element_type_245, 0.1);  convert_element_type_245 = None
        mul_144: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(gt_33, convert_element_type_244);  convert_element_type_244 = None
        mul_145: "bf16[8, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(mul_144, 1.1111111111111112);  mul_144 = None
        view_412: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_134, _shape_param_552);  add_134 = _shape_param_552 = None
        permute_336: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_412, [1, 0, 2, 3]);  view_412 = None
        permute_337: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(mul_145, [0, 2, 1, 3]);  mul_145 = None
        clone_92: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
        view_413: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_92, _shape_param_553);  clone_92 = _shape_param_553 = None
        permute_338: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_336, [0, 2, 1, 3]);  permute_336 = None
        view_414: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_338, _shape_param_554);  permute_338 = _shape_param_554 = None
        constant_pad_nd_35: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_414, _shape_param_555, -1.0);  view_414 = _shape_param_555 = None
        as_strided_37: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_35, _shape_param_556, _shape_param_557);  constant_pad_nd_35 = _shape_param_556 = _shape_param_557 = None
        constant_pad_nd_36: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_413, _shape_param_558, 0.0);  view_413 = _shape_param_558 = None
        view_415: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_36, _shape_param_559);  constant_pad_nd_36 = _shape_param_559 = None
        slice_248: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_415, 2, 0, -256);  view_415 = None
        view_416: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_248, _shape_param_560);  slice_248 = _shape_param_560 = None
        slice_249: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_416, 3, 0, -1);  view_416 = None
        unsqueeze_56: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_249, 4);  slice_249 = None
        unsqueeze_57: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_37, 4);  as_strided_37 = None
        view_417: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_56, _shape_param_561);  unsqueeze_56 = _shape_param_561 = None
        clone_93: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_57, memory_format = torch.contiguous_format);  unsqueeze_57 = None
        view_418: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_93, _shape_param_562);  clone_93 = _shape_param_562 = None
        bmm_23: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_417, view_418)
        view_419: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_563);  bmm_23 = _shape_param_563 = None
        permute_339: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_419, [0, 1, 2, 4, 3]);  view_419 = None
        view_420: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_339, _shape_param_564);  permute_339 = _shape_param_564 = None
        view_421: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_420, _shape_param_565);  view_420 = _shape_param_565 = None
        permute_340: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_421, [0, 2, 1, 3]);  view_421 = None
        permute_341: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_340, [1, 0, 2, 3]);  permute_340 = None
        clone_94: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_341, memory_format = torch.contiguous_format);  permute_341 = None
        view_422: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_94, _shape_param_566);  clone_94 = _shape_param_566 = None
        permute_342: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_422, [1, 0, 2]);  view_422 = None
        convert_element_type_246: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg185_1, torch.bfloat16);  arg185_1 = None
        convert_element_type_247: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg186_1, torch.bfloat16);  arg186_1 = None
        permute_343: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_247, [1, 0]);  convert_element_type_247 = None
        clone_95: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_342, memory_format = torch.contiguous_format);  permute_342 = None
        view_423: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_95, _shape_param_567);  clone_95 = _shape_param_567 = None
        mm_47: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_423, permute_343)
        view_424: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_47, _shape_param_568);  mm_47 = _shape_param_568 = None
        add_136: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_424, convert_element_type_246);  view_424 = convert_element_type_246 = None
        inductor_lookup_seed_34: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 34)
        inductor_random_34: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_569, inductor_lookup_seed_34, 'rand');  _shape_param_569 = inductor_lookup_seed_34 = None
        convert_element_type_248: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_34, torch.bfloat16);  inductor_random_34 = None
        gt_34: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_248, 0.1);  convert_element_type_248 = None
        mul_146: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_34, add_136);  add_136 = None
        mul_147: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_146, 1.1111111111111112);  mul_146 = None
        add_137: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_147, add_131);  mul_147 = add_131 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
        getitem_44: "f32[8, 1024, 1]" = var_mean_22[0]
        getitem_45: "f32[8, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
        add_138: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_35: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_137, getitem_45);  add_137 = getitem_45 = None
        mul_148: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_22);  sub_35 = None
        mul_149: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_148, arg187_1);  arg187_1 = None
        add_139: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_149, arg188_1);  mul_149 = arg188_1 = None
        convert_element_type_249: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg189_1, torch.bfloat16);  arg189_1 = None
        convert_element_type_250: "bf16[3072, 768]" = torch.ops.prims.convert_element_type.default(arg190_1, torch.bfloat16);  arg190_1 = None
        convert_element_type_251: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_139, torch.bfloat16)
        view_425: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_251, _shape_param_570);  convert_element_type_251 = _shape_param_570 = None
        permute_344: "bf16[768, 3072]" = torch.ops.aten.permute.default(convert_element_type_250, [1, 0]);  convert_element_type_250 = None
        addmm_22: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(convert_element_type_249, view_425, permute_344);  convert_element_type_249 = None
        view_426: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_22, _shape_param_571);  _shape_param_571 = None
        convert_element_type_252: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_426, torch.float32);  view_426 = None
        mul_150: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_252, 0.5)
        mul_151: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_252, 0.7071067811865476);  convert_element_type_252 = None
        erf_11: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_151);  mul_151 = None
        add_140: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_152: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_150, add_140);  mul_150 = add_140 = None
        convert_element_type_253: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None
        convert_element_type_254: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg191_1, torch.bfloat16);  arg191_1 = None
        convert_element_type_255: "bf16[768, 3072]" = torch.ops.prims.convert_element_type.default(arg192_1, torch.bfloat16);  arg192_1 = None
        view_427: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_253, _shape_param_572);  convert_element_type_253 = _shape_param_572 = None
        permute_345: "bf16[3072, 768]" = torch.ops.aten.permute.default(convert_element_type_255, [1, 0]);  convert_element_type_255 = None
        addmm_23: "bf16[8192, 768]" = torch.ops.aten.addmm.default(convert_element_type_254, view_427, permute_345);  convert_element_type_254 = None
        view_428: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_23, _shape_param_573);  addmm_23 = _shape_param_573 = None
        inductor_lookup_seed_35: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 35);  inductor_seeds = None
        inductor_random_35: "f32[8, 1024, 768]" = torch.ops.prims.inductor_random.default(_shape_param_574, inductor_lookup_seed_35, 'rand');  _shape_param_574 = inductor_lookup_seed_35 = None
        convert_element_type_256: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(inductor_random_35, torch.bfloat16);  inductor_random_35 = None
        gt_35: "b8[8, 1024, 768]" = torch.ops.aten.gt.Scalar(convert_element_type_256, 0.1);  convert_element_type_256 = None
        mul_153: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_35, view_428);  view_428 = None
        mul_154: "bf16[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_153, 1.1111111111111112);  mul_153 = None
        add_141: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_154, add_139);  mul_154 = add_139 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_141, [2], correction = 0, keepdim = True)
        getitem_46: "f32[8, 1024, 1]" = var_mean_23[0]
        getitem_47: "f32[8, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
        add_142: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        sub_36: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(add_141, getitem_47);  add_141 = getitem_47 = None
        mul_155: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
        mul_156: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_155, arg193_1);  arg193_1 = None
        add_143: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_156, arg194_1);  mul_156 = arg194_1 = None
        div_24: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        permute_346: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_345, [1, 0]);  permute_345 = None
        permute_347: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_344, [1, 0]);  permute_344 = None
        div_25: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        permute_348: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
        permute_349: "bf16[384, 768, 256]" = torch.ops.aten.permute.default(view_417, [0, 2, 1]);  view_417 = None
        permute_350: "bf16[384, 64, 768]" = torch.ops.aten.permute.default(view_418, [0, 2, 1]);  view_418 = None
        permute_351: "bf16[288, 64, 512]" = torch.ops.aten.permute.default(view_404, [0, 2, 1]);  view_404 = None
        permute_352: "bf16[288, 512, 64]" = torch.ops.aten.permute.default(view_405, [0, 2, 1]);  view_405 = None
        permute_353: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_321, [1, 0]);  permute_321 = None
        permute_354: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_320, [1, 0]);  permute_320 = None
        permute_355: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_319, [1, 0]);  permute_319 = None
        div_26: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        permute_356: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
        permute_357: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
        div_27: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        permute_358: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_315, [1, 0]);  permute_315 = None
        permute_359: "bf16[384, 768, 256]" = torch.ops.aten.permute.default(view_382, [0, 2, 1]);  view_382 = None
        permute_360: "bf16[384, 64, 768]" = torch.ops.aten.permute.default(view_383, [0, 2, 1]);  view_383 = None
        permute_361: "bf16[288, 64, 512]" = torch.ops.aten.permute.default(view_369, [0, 2, 1]);  view_369 = None
        permute_362: "bf16[288, 512, 64]" = torch.ops.aten.permute.default(view_370, [0, 2, 1]);  view_370 = None
        permute_363: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
        permute_364: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
        permute_365: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_291, [1, 0]);  permute_291 = None
        div_28: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        permute_366: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_289, [1, 0]);  permute_289 = None
        permute_367: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_288, [1, 0]);  permute_288 = None
        div_29: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        permute_368: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_287, [1, 0]);  permute_287 = None
        permute_369: "bf16[384, 768, 256]" = torch.ops.aten.permute.default(view_347, [0, 2, 1]);  view_347 = None
        permute_370: "bf16[384, 64, 768]" = torch.ops.aten.permute.default(view_348, [0, 2, 1]);  view_348 = None
        permute_371: "bf16[288, 64, 512]" = torch.ops.aten.permute.default(view_334, [0, 2, 1]);  view_334 = None
        permute_372: "bf16[288, 512, 64]" = torch.ops.aten.permute.default(view_335, [0, 2, 1]);  view_335 = None
        permute_373: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
        permute_374: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
        permute_375: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        div_30: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        permute_376: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
        permute_377: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
        div_31: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        permute_378: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_259, [1, 0]);  permute_259 = None
        permute_379: "bf16[384, 768, 256]" = torch.ops.aten.permute.default(view_312, [0, 2, 1]);  view_312 = None
        permute_380: "bf16[384, 64, 768]" = torch.ops.aten.permute.default(view_313, [0, 2, 1]);  view_313 = None
        permute_381: "bf16[288, 64, 512]" = torch.ops.aten.permute.default(view_299, [0, 2, 1]);  view_299 = None
        permute_382: "bf16[288, 512, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1]);  view_300 = None
        permute_383: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None
        permute_384: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
        permute_385: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
        div_32: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        permute_386: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        permute_387: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
        div_33: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        permute_388: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
        permute_389: "bf16[384, 768, 256]" = torch.ops.aten.permute.default(view_277, [0, 2, 1]);  view_277 = None
        permute_390: "bf16[384, 64, 768]" = torch.ops.aten.permute.default(view_278, [0, 2, 1]);  view_278 = None
        permute_391: "bf16[288, 64, 512]" = torch.ops.aten.permute.default(view_264, [0, 2, 1]);  view_264 = None
        permute_392: "bf16[288, 512, 64]" = torch.ops.aten.permute.default(view_265, [0, 2, 1]);  view_265 = None
        permute_393: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
        permute_394: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
        permute_395: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
        div_34: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        permute_396: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
        permute_397: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_204, [1, 0]);  permute_204 = None
        div_35: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        permute_398: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
        permute_399: "bf16[384, 768, 256]" = torch.ops.aten.permute.default(view_242, [0, 2, 1]);  view_242 = None
        permute_400: "bf16[384, 64, 768]" = torch.ops.aten.permute.default(view_243, [0, 2, 1]);  view_243 = None
        permute_401: "bf16[288, 64, 512]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        permute_402: "bf16[288, 512, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
        permute_403: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
        permute_404: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
        permute_405: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
        div_36: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        permute_406: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
        permute_407: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        div_37: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        permute_408: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
        permute_409: "bf16[384, 768, 256]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        permute_410: "bf16[384, 64, 768]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
        permute_411: "bf16[288, 64, 512]" = torch.ops.aten.permute.default(view_194, [0, 2, 1]);  view_194 = None
        permute_412: "bf16[288, 512, 64]" = torch.ops.aten.permute.default(view_195, [0, 2, 1]);  view_195 = None
        permute_413: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        permute_414: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        permute_415: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        div_38: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        permute_416: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
        permute_417: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
        div_39: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        permute_418: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
        permute_419: "bf16[384, 768, 256]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
        permute_420: "bf16[384, 64, 768]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
        permute_421: "bf16[288, 64, 512]" = torch.ops.aten.permute.default(view_159, [0, 2, 1]);  view_159 = None
        permute_422: "bf16[288, 512, 64]" = torch.ops.aten.permute.default(view_160, [0, 2, 1]);  view_160 = None
        permute_423: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        permute_424: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
        permute_425: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        div_40: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        permute_426: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        permute_427: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        div_41: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        permute_428: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        permute_429: "bf16[384, 768, 256]" = torch.ops.aten.permute.default(view_137, [0, 2, 1]);  view_137 = None
        permute_430: "bf16[384, 64, 768]" = torch.ops.aten.permute.default(view_138, [0, 2, 1]);  view_138 = None
        permute_431: "bf16[288, 64, 512]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
        permute_432: "bf16[288, 512, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
        permute_433: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        permute_434: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        permute_435: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
        div_42: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        permute_436: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
        permute_437: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        div_43: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        permute_438: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
        permute_439: "bf16[384, 768, 256]" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
        permute_440: "bf16[384, 64, 768]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
        permute_441: "bf16[288, 64, 512]" = torch.ops.aten.permute.default(view_89, [0, 2, 1]);  view_89 = None
        permute_442: "bf16[288, 512, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1]);  view_90 = None
        permute_443: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        permute_444: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        permute_445: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        div_44: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        permute_446: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        permute_447: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        div_45: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        permute_448: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        permute_449: "bf16[384, 768, 256]" = torch.ops.aten.permute.default(view_67, [0, 2, 1]);  view_67 = None
        permute_450: "bf16[384, 64, 768]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
        permute_451: "bf16[288, 64, 512]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
        permute_452: "bf16[288, 512, 64]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
        permute_453: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        permute_454: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
        permute_455: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
        div_46: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        permute_456: "bf16[768, 3072]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        permute_457: "bf16[3072, 768]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        div_47: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        permute_458: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        permute_459: "bf16[384, 768, 256]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
        permute_460: "bf16[384, 64, 768]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
        permute_461: "bf16[288, 64, 512]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_462: "bf16[288, 512, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        permute_463: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_464: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_465: "bf16[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        return (view, unsqueeze_5, rev_1, slice_scatter_10, slice_scatter_21, amax, sum_1, gt, view_38, gt_1, mul_5, view_40, addmm, view_42, gt_2, mul_12, view_44, permute_55, amax_1, sum_2, gt_3, view_73, gt_4, mul_18, view_75, addmm_2, view_77, gt_5, mul_25, view_79, permute_83, amax_2, sum_3, gt_6, view_108, gt_7, mul_31, view_110, addmm_4, view_112, gt_8, mul_38, view_114, permute_111, amax_3, sum_4, gt_9, view_143, gt_10, mul_44, view_145, addmm_6, view_147, gt_11, mul_51, view_149, permute_139, amax_4, sum_5, gt_12, view_178, gt_13, mul_57, view_180, addmm_8, view_182, gt_14, mul_64, view_184, permute_167, amax_5, sum_6, gt_15, view_213, gt_16, mul_70, view_215, addmm_10, view_217, gt_17, mul_77, view_219, permute_195, amax_6, sum_7, gt_18, view_248, gt_19, mul_83, view_250, addmm_12, view_252, gt_20, mul_90, view_254, permute_223, amax_7, sum_8, gt_21, view_283, gt_22, mul_96, view_285, addmm_14, view_287, gt_23, mul_103, view_289, permute_251, amax_8, sum_9, gt_24, view_318, gt_25, mul_109, view_320, addmm_16, view_322, gt_26, mul_116, view_324, permute_279, amax_9, sum_10, gt_27, view_353, gt_28, mul_122, view_355, addmm_18, view_357, gt_29, mul_129, view_359, permute_307, amax_10, sum_11, gt_30, view_388, gt_31, mul_135, view_390, addmm_20, view_392, gt_32, mul_142, view_394, permute_335, amax_11, sum_12, gt_33, view_423, gt_34, mul_148, view_425, addmm_22, view_427, gt_35, mul_155, add_143, div_24, permute_346, permute_347, div_25, permute_348, permute_349, permute_350, permute_351, permute_352, permute_353, permute_354, permute_355, div_26, permute_356, permute_357, div_27, permute_358, permute_359, permute_360, permute_361, permute_362, permute_363, permute_364, permute_365, div_28, permute_366, permute_367, div_29, permute_368, permute_369, permute_370, permute_371, permute_372, permute_373, permute_374, permute_375, div_30, permute_376, permute_377, div_31, permute_378, permute_379, permute_380, permute_381, permute_382, permute_383, permute_384, permute_385, div_32, permute_386, permute_387, div_33, permute_388, permute_389, permute_390, permute_391, permute_392, permute_393, permute_394, permute_395, div_34, permute_396, permute_397, div_35, permute_398, permute_399, permute_400, permute_401, permute_402, permute_403, permute_404, permute_405, div_36, permute_406, permute_407, div_37, permute_408, permute_409, permute_410, permute_411, permute_412, permute_413, permute_414, permute_415, div_38, permute_416, permute_417, div_39, permute_418, permute_419, permute_420, permute_421, permute_422, permute_423, permute_424, permute_425, div_40, permute_426, permute_427, div_41, permute_428, permute_429, permute_430, permute_431, permute_432, permute_433, permute_434, permute_435, div_42, permute_436, permute_437, div_43, permute_438, permute_439, permute_440, permute_441, permute_442, permute_443, permute_444, permute_445, div_44, permute_446, permute_447, div_45, permute_448, permute_449, permute_450, permute_451, permute_452, permute_453, permute_454, permute_455, div_46, permute_456, permute_457, div_47, permute_458, permute_459, permute_460, permute_461, permute_462, permute_463, permute_464, permute_465)



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
