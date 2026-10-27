"""
Standalone repro captured via capture_hook.
Label: MT5ForConditionalGeneration
Pattern hash: d472ad93675f
Shape hash: b03a4b82
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
    def forward(self, arg0_1: "f32[250112, 512]", arg1_1: "i64[32, 128]", arg2_1: "f32[512]", arg3_1: "f32[384, 512]", arg4_1: "f32[384, 512]", arg5_1: "f32[384, 512]", arg6_1: "f32[32, 6]", arg7_1: "f32[512, 384]", arg8_1: "f32[512]", arg9_1: "f32[1024, 512]", arg10_1: "f32[1024, 512]", arg11_1: "f32[512, 1024]", arg12_1: "f32[512]", arg13_1: "f32[384, 512]", arg14_1: "f32[384, 512]", arg15_1: "f32[384, 512]", arg16_1: "f32[512, 384]", arg17_1: "f32[512]", arg18_1: "f32[1024, 512]", arg19_1: "f32[1024, 512]", arg20_1: "f32[512, 1024]", arg21_1: "f32[512]", arg22_1: "f32[384, 512]", arg23_1: "f32[384, 512]", arg24_1: "f32[384, 512]", arg25_1: "f32[512, 384]", arg26_1: "f32[512]", arg27_1: "f32[1024, 512]", arg28_1: "f32[1024, 512]", arg29_1: "f32[512, 1024]", arg30_1: "f32[512]", arg31_1: "f32[384, 512]", arg32_1: "f32[384, 512]", arg33_1: "f32[384, 512]", arg34_1: "f32[512, 384]", arg35_1: "f32[512]", arg36_1: "f32[1024, 512]", arg37_1: "f32[1024, 512]", arg38_1: "f32[512, 1024]", arg39_1: "f32[512]", arg40_1: "f32[384, 512]", arg41_1: "f32[384, 512]", arg42_1: "f32[384, 512]", arg43_1: "f32[512, 384]", arg44_1: "f32[512]", arg45_1: "f32[1024, 512]", arg46_1: "f32[1024, 512]", arg47_1: "f32[512, 1024]", arg48_1: "f32[512]", arg49_1: "f32[384, 512]", arg50_1: "f32[384, 512]", arg51_1: "f32[384, 512]", arg52_1: "f32[512, 384]", arg53_1: "f32[512]", arg54_1: "f32[1024, 512]", arg55_1: "f32[1024, 512]", arg56_1: "f32[512, 1024]", arg57_1: "f32[512]", arg58_1: "f32[384, 512]", arg59_1: "f32[384, 512]", arg60_1: "f32[384, 512]", arg61_1: "f32[512, 384]", arg62_1: "f32[512]", arg63_1: "f32[1024, 512]", arg64_1: "f32[1024, 512]", arg65_1: "f32[512, 1024]", arg66_1: "f32[512]", arg67_1: "f32[384, 512]", arg68_1: "f32[384, 512]", arg69_1: "f32[384, 512]", arg70_1: "f32[512, 384]", arg71_1: "f32[512]", arg72_1: "f32[1024, 512]", arg73_1: "f32[1024, 512]", arg74_1: "f32[512, 1024]", arg75_1: "f32[512]", arg76_1: "f32[512]", arg77_1: "f32[384, 512]", arg78_1: "f32[384, 512]", arg79_1: "f32[384, 512]", arg80_1: "f32[32, 6]", arg81_1: "f32[512, 384]", arg82_1: "f32[512]", arg83_1: "f32[384, 512]", arg84_1: "f32[384, 512]", arg85_1: "f32[384, 512]", arg86_1: "f32[512, 384]", arg87_1: "f32[512]", arg88_1: "f32[1024, 512]", arg89_1: "f32[1024, 512]", arg90_1: "f32[512, 1024]", arg91_1: "f32[512]", arg92_1: "f32[384, 512]", arg93_1: "f32[384, 512]", arg94_1: "f32[384, 512]", arg95_1: "f32[512, 384]", arg96_1: "f32[512]", arg97_1: "f32[384, 512]", arg98_1: "f32[384, 512]", arg99_1: "f32[384, 512]", arg100_1: "f32[512, 384]", arg101_1: "f32[512]", arg102_1: "f32[1024, 512]", arg103_1: "f32[1024, 512]", arg104_1: "f32[512, 1024]", arg105_1: "f32[512]", arg106_1: "f32[384, 512]", arg107_1: "f32[384, 512]", arg108_1: "f32[384, 512]", arg109_1: "f32[512, 384]", arg110_1: "f32[512]", arg111_1: "f32[384, 512]", arg112_1: "f32[384, 512]", arg113_1: "f32[384, 512]", arg114_1: "f32[512, 384]", arg115_1: "f32[512]", arg116_1: "f32[1024, 512]", arg117_1: "f32[1024, 512]", arg118_1: "f32[512, 1024]", arg119_1: "f32[512]", arg120_1: "f32[384, 512]", arg121_1: "f32[384, 512]", arg122_1: "f32[384, 512]", arg123_1: "f32[512, 384]", arg124_1: "f32[512]", arg125_1: "f32[384, 512]", arg126_1: "f32[384, 512]", arg127_1: "f32[384, 512]", arg128_1: "f32[512, 384]", arg129_1: "f32[512]", arg130_1: "f32[1024, 512]", arg131_1: "f32[1024, 512]", arg132_1: "f32[512, 1024]", arg133_1: "f32[512]", arg134_1: "f32[384, 512]", arg135_1: "f32[384, 512]", arg136_1: "f32[384, 512]", arg137_1: "f32[512, 384]", arg138_1: "f32[512]", arg139_1: "f32[384, 512]", arg140_1: "f32[384, 512]", arg141_1: "f32[384, 512]", arg142_1: "f32[512, 384]", arg143_1: "f32[512]", arg144_1: "f32[1024, 512]", arg145_1: "f32[1024, 512]", arg146_1: "f32[512, 1024]", arg147_1: "f32[512]", arg148_1: "f32[384, 512]", arg149_1: "f32[384, 512]", arg150_1: "f32[384, 512]", arg151_1: "f32[512, 384]", arg152_1: "f32[512]", arg153_1: "f32[384, 512]", arg154_1: "f32[384, 512]", arg155_1: "f32[384, 512]", arg156_1: "f32[512, 384]", arg157_1: "f32[512]", arg158_1: "f32[1024, 512]", arg159_1: "f32[1024, 512]", arg160_1: "f32[512, 1024]", arg161_1: "f32[512]", arg162_1: "f32[384, 512]", arg163_1: "f32[384, 512]", arg164_1: "f32[384, 512]", arg165_1: "f32[512, 384]", arg166_1: "f32[512]", arg167_1: "f32[384, 512]", arg168_1: "f32[384, 512]", arg169_1: "f32[384, 512]", arg170_1: "f32[512, 384]", arg171_1: "f32[512]", arg172_1: "f32[1024, 512]", arg173_1: "f32[1024, 512]", arg174_1: "f32[512, 1024]", arg175_1: "f32[512]", arg176_1: "f32[384, 512]", arg177_1: "f32[384, 512]", arg178_1: "f32[384, 512]", arg179_1: "f32[512, 384]", arg180_1: "f32[512]", arg181_1: "f32[384, 512]", arg182_1: "f32[384, 512]", arg183_1: "f32[384, 512]", arg184_1: "f32[512, 384]", arg185_1: "f32[512]", arg186_1: "f32[1024, 512]", arg187_1: "f32[1024, 512]", arg188_1: "f32[512, 1024]", arg189_1: "f32[512]", arg190_1: "i64[32, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631, _shape_param_632, _shape_param_633, _shape_param_634, _shape_param_635, _shape_param_636, _shape_param_637, _shape_param_638, _shape_param_639, _shape_param_640, _shape_param_641, _shape_param_642, _shape_param_643, _shape_param_644, _shape_param_645, _shape_param_646, _shape_param_647, _shape_param_648, _shape_param_649, _shape_param_650, _shape_param_651, _shape_param_652):
        # No stacktrace found for following nodes
        embedding: "f32[32, 128, 512]" = torch.ops.aten.embedding.default(arg0_1, arg1_1);  arg1_1 = None
        full: "f32[32, 1, 128, 128]" = torch.ops.aten.full.default(_shape_param_0, 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        inductor_seeds: "i64[84]" = torch.ops.prims.inductor_seeds.default(84, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_1, inductor_lookup_seed, 'rand');  _shape_param_1 = inductor_lookup_seed = None
        gt: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt, embedding)
        mul_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        pow_1: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_1, 2)
        mean: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1, rsqrt)
        mul_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg2_1, mul_2);  arg2_1 = mul_2 = None
        convert_element_type: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_1: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        permute: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        view: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_2);  convert_element_type_1 = _shape_param_2 = None
        mm: "bf16[4096, 384]" = torch.ops.aten.mm.default(view, permute)
        view_1: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm, _shape_param_3);  mm = _shape_param_3 = None
        view_2: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_1, _shape_param_4);  view_1 = _shape_param_4 = None
        permute_1: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        convert_element_type_2: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        permute_2: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_2, [1, 0]);  convert_element_type_2 = None
        mm_1: "bf16[4096, 384]" = torch.ops.aten.mm.default(view, permute_2)
        view_3: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_1, _shape_param_5);  mm_1 = _shape_param_5 = None
        view_4: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_3, _shape_param_6);  view_3 = _shape_param_6 = None
        permute_3: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        convert_element_type_3: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        permute_4: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0]);  convert_element_type_3 = None
        mm_2: "bf16[4096, 384]" = torch.ops.aten.mm.default(view, permute_4)
        view_5: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_2, _shape_param_7);  mm_2 = _shape_param_7 = None
        view_6: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_5, _shape_param_8);  view_5 = _shape_param_8 = None
        permute_5: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        permute_6: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_1, _shape_param_9);  permute_1 = _shape_param_9 = None
        clone: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_7: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_10);  clone = _shape_param_10 = None
        expand_1: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_6, _shape_param_11);  permute_6 = _shape_param_11 = None
        clone_1: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_8: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_12);  clone_1 = _shape_param_12 = None
        bmm: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_7, view_8)
        view_9: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_13);  _shape_param_13 = None
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(iota, 1)
        add_1: "i64[128, 1]" = torch.ops.aten.add.Tensor(unsqueeze, 0);  unsqueeze = None
        unsqueeze_1: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota, 0)
        sub: "i64[128, 128]" = torch.ops.aten.sub.Tensor(unsqueeze_1, add_1);  unsqueeze_1 = add_1 = None
        gt_1: "b8[128, 128]" = torch.ops.aten.gt.Scalar(sub, 0)
        convert_element_type_4: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(gt_1, torch.int64);  gt_1 = None
        mul_4: "i64[128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 16);  convert_element_type_4 = None
        add_2: "i64[128, 128]" = torch.ops.aten.add.Tensor(mul_4, 0);  mul_4 = None
        abs_1: "i64[128, 128]" = torch.ops.aten.abs.default(sub)
        lt: "b8[128, 128]" = torch.ops.aten.lt.Scalar(abs_1, 8)
        convert_element_type_5: "f32[128, 128]" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
        div: "f32[128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_5, 8);  convert_element_type_5 = None
        log: "f32[128, 128]" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[128, 128]" = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
        mul_5: "f32[128, 128]" = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None
        convert_element_type_6: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(mul_5, torch.int64);  mul_5 = None
        add_3: "i64[128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_6, 8);  convert_element_type_6 = None
        full_1: "i64[128, 128]" = torch.ops.aten.full.default(_shape_param_14, 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_14 = None
        minimum: "i64[128, 128]" = torch.ops.aten.minimum.default(add_3, full_1);  add_3 = full_1 = None
        where: "i64[128, 128]" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
        add_4: "i64[128, 128]" = torch.ops.aten.add.Tensor(add_2, where);  add_2 = where = None
        embedding_1: "f32[128, 128, 6]" = torch.ops.aten.embedding.default(arg6_1, add_4);  arg6_1 = None
        permute_7: "f32[6, 128, 128]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1])
        unsqueeze_2: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None
        add_5: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_2, full);  unsqueeze_2 = full = None
        add_6: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_9, add_5);  view_9 = None
        convert_element_type_7: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        convert_element_type_8: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
        amax: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_8, [-1], True)
        sub_1: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_8, amax);  convert_element_type_8 = None
        exp: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_9: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1)
        inductor_random_1: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_15, inductor_lookup_seed_1, 'rand');  _shape_param_15 = inductor_lookup_seed_1 = None
        convert_element_type_10: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_1, torch.bfloat16);  inductor_random_1 = None
        gt_2: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_10, 0.1);  convert_element_type_10 = None
        mul_6: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_2, convert_element_type_9);  convert_element_type_9 = None
        mul_7: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_6, 1.1111111111111112);  mul_6 = None
        expand_2: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_7, _shape_param_16);  mul_7 = _shape_param_16 = None
        view_10: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_2, _shape_param_17);  expand_2 = _shape_param_17 = None
        expand_3: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_5, _shape_param_18);  permute_5 = _shape_param_18 = None
        clone_2: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_11: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_19);  clone_2 = _shape_param_19 = None
        bmm_1: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_10, view_11)
        view_12: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_20);  bmm_1 = _shape_param_20 = None
        permute_8: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        clone_3: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_13: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_3, _shape_param_21);  clone_3 = _shape_param_21 = None
        convert_element_type_11: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        permute_9: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_11, [1, 0]);  convert_element_type_11 = None
        view_14: "bf16[4096, 384]" = torch.ops.aten.view.default(view_13, _shape_param_22);  view_13 = _shape_param_22 = None
        mm_3: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_14, permute_9)
        view_15: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_3, _shape_param_23);  mm_3 = _shape_param_23 = None
        inductor_lookup_seed_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 2)
        inductor_random_2: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_24, inductor_lookup_seed_2, 'rand');  _shape_param_24 = inductor_lookup_seed_2 = None
        convert_element_type_12: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_2, torch.bfloat16);  inductor_random_2 = None
        gt_3: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_12, 0.1);  convert_element_type_12 = None
        mul_8: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_3, view_15);  view_15 = None
        mul_9: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_8, 1.1111111111111112);  mul_8 = None
        add_7: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(mul_1, mul_9);  mul_1 = mul_9 = None
        pow_2: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_7, 2)
        mean_1: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_8: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_10: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_7, rsqrt_1)
        mul_11: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg8_1, mul_10);  arg8_1 = mul_10 = None
        convert_element_type_13: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        convert_element_type_14: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        permute_10: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        view_16: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_14, _shape_param_25);  convert_element_type_14 = _shape_param_25 = None
        mm_4: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_16, permute_10)
        view_17: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_4, _shape_param_26);  _shape_param_26 = None
        mul_12: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
        convert_element_type_15: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_17, torch.float32)
        pow_3: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_15, 3.0);  convert_element_type_15 = None
        mul_13: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_9: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_17, mul_13);  view_17 = mul_13 = None
        mul_14: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_9, 0.7978845608028654);  add_9 = None
        tanh: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_10: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_15: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_12, add_10);  mul_12 = add_10 = None
        convert_element_type_16: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        permute_11: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_16, [1, 0]);  convert_element_type_16 = None
        mm_5: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_16, permute_11)
        view_18: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_5, _shape_param_27);  _shape_param_27 = None
        mul_16: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_15, view_18);  mul_15 = view_18 = None
        inductor_lookup_seed_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 3)
        inductor_random_3: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_28, inductor_lookup_seed_3, 'rand');  _shape_param_28 = inductor_lookup_seed_3 = None
        gt_4: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_3, 0.1);  inductor_random_3 = None
        mul_17: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_4, mul_16);  mul_16 = None
        mul_18: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_17, 1.1111111111111112);  mul_17 = None
        convert_element_type_17: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.bfloat16);  arg11_1 = None
        convert_element_type_18: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        permute_12: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_17, [1, 0]);  convert_element_type_17 = None
        view_19: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_18, _shape_param_29);  convert_element_type_18 = _shape_param_29 = None
        mm_6: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_19, permute_12)
        view_20: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_6, _shape_param_30);  mm_6 = _shape_param_30 = None
        inductor_lookup_seed_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 4)
        inductor_random_4: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_31, inductor_lookup_seed_4, 'rand');  _shape_param_31 = inductor_lookup_seed_4 = None
        convert_element_type_19: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_4, torch.bfloat16);  inductor_random_4 = None
        gt_5: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_19, 0.1);  convert_element_type_19 = None
        mul_19: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_5, view_20);  view_20 = None
        mul_20: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_19, 1.1111111111111112);  mul_19 = None
        add_11: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_7, mul_20);  mul_20 = None
        pow_4: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_11, 2)
        mean_2: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
        add_12: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_21: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_11, rsqrt_2)
        mul_22: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg12_1, mul_21);  arg12_1 = mul_21 = None
        convert_element_type_20: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        convert_element_type_21: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None
        permute_13: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_20, [1, 0]);  convert_element_type_20 = None
        view_21: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_21, _shape_param_32);  convert_element_type_21 = _shape_param_32 = None
        mm_7: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_21, permute_13)
        view_22: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_7, _shape_param_33);  mm_7 = _shape_param_33 = None
        view_23: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_22, _shape_param_34);  view_22 = _shape_param_34 = None
        permute_14: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
        convert_element_type_22: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.bfloat16);  arg14_1 = None
        permute_15: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_22, [1, 0]);  convert_element_type_22 = None
        mm_8: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_21, permute_15)
        view_24: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_8, _shape_param_35);  mm_8 = _shape_param_35 = None
        view_25: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_24, _shape_param_36);  view_24 = _shape_param_36 = None
        permute_16: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        convert_element_type_23: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg15_1, torch.bfloat16);  arg15_1 = None
        permute_17: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_23, [1, 0]);  convert_element_type_23 = None
        mm_9: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_21, permute_17)
        view_26: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_9, _shape_param_37);  mm_9 = _shape_param_37 = None
        view_27: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_26, _shape_param_38);  view_26 = _shape_param_38 = None
        permute_18: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        permute_19: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2]);  permute_16 = None
        expand_4: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_14, _shape_param_39);  permute_14 = _shape_param_39 = None
        clone_4: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_28: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_40);  clone_4 = _shape_param_40 = None
        expand_5: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_19, _shape_param_41);  permute_19 = _shape_param_41 = None
        clone_5: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_29: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_5, _shape_param_42);  clone_5 = _shape_param_42 = None
        bmm_2: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_28, view_29)
        view_30: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_43);  bmm_2 = _shape_param_43 = None
        add_13: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_30, add_5);  view_30 = None
        convert_element_type_24: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None
        convert_element_type_25: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_24, torch.float32)
        amax_1: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_25, [-1], True)
        sub_2: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_25, amax_1);  convert_element_type_25 = None
        exp_1: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_2: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None
        convert_element_type_26: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        inductor_lookup_seed_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 5)
        inductor_random_5: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_44, inductor_lookup_seed_5, 'rand');  _shape_param_44 = inductor_lookup_seed_5 = None
        convert_element_type_27: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_5, torch.bfloat16);  inductor_random_5 = None
        gt_6: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_27, 0.1);  convert_element_type_27 = None
        mul_23: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_6, convert_element_type_26);  convert_element_type_26 = None
        mul_24: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_23, 1.1111111111111112);  mul_23 = None
        expand_6: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_24, _shape_param_45);  mul_24 = _shape_param_45 = None
        view_31: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_6, _shape_param_46);  expand_6 = _shape_param_46 = None
        expand_7: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_18, _shape_param_47);  permute_18 = _shape_param_47 = None
        clone_6: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_32: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_6, _shape_param_48);  clone_6 = _shape_param_48 = None
        bmm_3: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_31, view_32)
        view_33: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_49);  bmm_3 = _shape_param_49 = None
        permute_20: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        clone_7: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
        view_34: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_7, _shape_param_50);  clone_7 = _shape_param_50 = None
        convert_element_type_28: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.bfloat16);  arg16_1 = None
        permute_21: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_28, [1, 0]);  convert_element_type_28 = None
        view_35: "bf16[4096, 384]" = torch.ops.aten.view.default(view_34, _shape_param_51);  view_34 = _shape_param_51 = None
        mm_10: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_35, permute_21)
        view_36: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_10, _shape_param_52);  mm_10 = _shape_param_52 = None
        inductor_lookup_seed_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 6)
        inductor_random_6: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_53, inductor_lookup_seed_6, 'rand');  _shape_param_53 = inductor_lookup_seed_6 = None
        convert_element_type_29: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_6, torch.bfloat16);  inductor_random_6 = None
        gt_7: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_29, 0.1);  convert_element_type_29 = None
        mul_25: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_7, view_36);  view_36 = None
        mul_26: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_25, 1.1111111111111112);  mul_25 = None
        add_14: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_11, mul_26);  mul_26 = None
        pow_5: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_14, 2)
        mean_3: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
        add_15: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_27: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_14, rsqrt_3)
        mul_28: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg17_1, mul_27);  arg17_1 = mul_27 = None
        convert_element_type_30: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.bfloat16);  arg18_1 = None
        convert_element_type_31: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_28, torch.bfloat16);  mul_28 = None
        permute_22: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_30, [1, 0]);  convert_element_type_30 = None
        view_37: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_31, _shape_param_54);  convert_element_type_31 = _shape_param_54 = None
        mm_11: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_37, permute_22)
        view_38: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_11, _shape_param_55);  _shape_param_55 = None
        mul_29: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
        convert_element_type_32: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_38, torch.float32)
        pow_6: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_32, 3.0);  convert_element_type_32 = None
        mul_30: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_16: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_38, mul_30);  view_38 = mul_30 = None
        mul_31: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_16, 0.7978845608028654);  add_16 = None
        tanh_1: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_31);  mul_31 = None
        add_17: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_32: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_29, add_17);  mul_29 = add_17 = None
        convert_element_type_33: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg19_1, torch.bfloat16);  arg19_1 = None
        permute_23: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_33, [1, 0]);  convert_element_type_33 = None
        mm_12: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_37, permute_23)
        view_39: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_12, _shape_param_56);  _shape_param_56 = None
        mul_33: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_32, view_39);  mul_32 = view_39 = None
        inductor_lookup_seed_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 7)
        inductor_random_7: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_57, inductor_lookup_seed_7, 'rand');  _shape_param_57 = inductor_lookup_seed_7 = None
        gt_8: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_7, 0.1);  inductor_random_7 = None
        mul_34: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_8, mul_33);  mul_33 = None
        mul_35: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_34, 1.1111111111111112);  mul_34 = None
        convert_element_type_34: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.bfloat16);  arg20_1 = None
        convert_element_type_35: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None
        permute_24: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        view_40: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_35, _shape_param_58);  convert_element_type_35 = _shape_param_58 = None
        mm_13: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_40, permute_24)
        view_41: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_13, _shape_param_59);  mm_13 = _shape_param_59 = None
        inductor_lookup_seed_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 8)
        inductor_random_8: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_60, inductor_lookup_seed_8, 'rand');  _shape_param_60 = inductor_lookup_seed_8 = None
        convert_element_type_36: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_8, torch.bfloat16);  inductor_random_8 = None
        gt_9: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_36, 0.1);  convert_element_type_36 = None
        mul_36: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_9, view_41);  view_41 = None
        mul_37: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_36, 1.1111111111111112);  mul_36 = None
        add_18: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_14, mul_37);  mul_37 = None
        pow_7: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_18, 2)
        mean_4: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
        add_19: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_38: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_18, rsqrt_4)
        mul_39: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg21_1, mul_38);  arg21_1 = mul_38 = None
        convert_element_type_37: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg22_1, torch.bfloat16);  arg22_1 = None
        convert_element_type_38: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None
        permute_25: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_37, [1, 0]);  convert_element_type_37 = None
        view_42: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_38, _shape_param_61);  convert_element_type_38 = _shape_param_61 = None
        mm_14: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_42, permute_25)
        view_43: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_14, _shape_param_62);  mm_14 = _shape_param_62 = None
        view_44: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_43, _shape_param_63);  view_43 = _shape_param_63 = None
        permute_26: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None
        convert_element_type_39: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg23_1, torch.bfloat16);  arg23_1 = None
        permute_27: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_39, [1, 0]);  convert_element_type_39 = None
        mm_15: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_42, permute_27)
        view_45: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_15, _shape_param_64);  mm_15 = _shape_param_64 = None
        view_46: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_45, _shape_param_65);  view_45 = _shape_param_65 = None
        permute_28: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        convert_element_type_40: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg24_1, torch.bfloat16);  arg24_1 = None
        permute_29: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_40, [1, 0]);  convert_element_type_40 = None
        mm_16: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_42, permute_29)
        view_47: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_16, _shape_param_66);  mm_16 = _shape_param_66 = None
        view_48: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_47, _shape_param_67);  view_47 = _shape_param_67 = None
        permute_30: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        permute_31: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_28, [0, 1, 3, 2]);  permute_28 = None
        expand_8: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_26, _shape_param_68);  permute_26 = _shape_param_68 = None
        clone_8: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_49: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_69);  clone_8 = _shape_param_69 = None
        expand_9: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_31, _shape_param_70);  permute_31 = _shape_param_70 = None
        clone_9: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_50: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_9, _shape_param_71);  clone_9 = _shape_param_71 = None
        bmm_4: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_49, view_50)
        view_51: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_4, _shape_param_72);  bmm_4 = _shape_param_72 = None
        add_20: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_51, add_5);  view_51 = None
        convert_element_type_41: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_20, torch.bfloat16);  add_20 = None
        convert_element_type_42: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_41, torch.float32)
        amax_2: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_42, [-1], True)
        sub_3: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_42, amax_2);  convert_element_type_42 = None
        exp_2: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_3: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = None
        convert_element_type_43: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        inductor_lookup_seed_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 9)
        inductor_random_9: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_73, inductor_lookup_seed_9, 'rand');  _shape_param_73 = inductor_lookup_seed_9 = None
        convert_element_type_44: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_9, torch.bfloat16);  inductor_random_9 = None
        gt_10: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_44, 0.1);  convert_element_type_44 = None
        mul_40: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_10, convert_element_type_43);  convert_element_type_43 = None
        mul_41: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_40, 1.1111111111111112);  mul_40 = None
        expand_10: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_41, _shape_param_74);  mul_41 = _shape_param_74 = None
        view_52: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_10, _shape_param_75);  expand_10 = _shape_param_75 = None
        expand_11: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_30, _shape_param_76);  permute_30 = _shape_param_76 = None
        clone_10: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_53: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_10, _shape_param_77);  clone_10 = _shape_param_77 = None
        bmm_5: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_52, view_53)
        view_54: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_5, _shape_param_78);  bmm_5 = _shape_param_78 = None
        permute_32: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        clone_11: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_55: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_11, _shape_param_79);  clone_11 = _shape_param_79 = None
        convert_element_type_45: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg25_1, torch.bfloat16);  arg25_1 = None
        permute_33: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_45, [1, 0]);  convert_element_type_45 = None
        view_56: "bf16[4096, 384]" = torch.ops.aten.view.default(view_55, _shape_param_80);  view_55 = _shape_param_80 = None
        mm_17: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_56, permute_33)
        view_57: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_17, _shape_param_81);  mm_17 = _shape_param_81 = None
        inductor_lookup_seed_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 10)
        inductor_random_10: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_82, inductor_lookup_seed_10, 'rand');  _shape_param_82 = inductor_lookup_seed_10 = None
        convert_element_type_46: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_10, torch.bfloat16);  inductor_random_10 = None
        gt_11: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_46, 0.1);  convert_element_type_46 = None
        mul_42: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_11, view_57);  view_57 = None
        mul_43: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_42, 1.1111111111111112);  mul_42 = None
        add_21: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_18, mul_43);  mul_43 = None
        pow_8: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_21, 2)
        mean_5: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
        add_22: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_44: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_21, rsqrt_5)
        mul_45: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg26_1, mul_44);  arg26_1 = mul_44 = None
        convert_element_type_47: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg27_1, torch.bfloat16);  arg27_1 = None
        convert_element_type_48: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        permute_34: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_47, [1, 0]);  convert_element_type_47 = None
        view_58: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_48, _shape_param_83);  convert_element_type_48 = _shape_param_83 = None
        mm_18: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_58, permute_34)
        view_59: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_18, _shape_param_84);  _shape_param_84 = None
        mul_46: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_59, 0.5)
        convert_element_type_49: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_59, torch.float32)
        pow_9: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_49, 3.0);  convert_element_type_49 = None
        mul_47: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_23: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_59, mul_47);  view_59 = mul_47 = None
        mul_48: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        tanh_2: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
        add_24: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_49: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_46, add_24);  mul_46 = add_24 = None
        convert_element_type_50: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg28_1, torch.bfloat16);  arg28_1 = None
        permute_35: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_50, [1, 0]);  convert_element_type_50 = None
        mm_19: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_58, permute_35)
        view_60: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_19, _shape_param_85);  _shape_param_85 = None
        mul_50: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_49, view_60);  mul_49 = view_60 = None
        inductor_lookup_seed_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 11)
        inductor_random_11: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_86, inductor_lookup_seed_11, 'rand');  _shape_param_86 = inductor_lookup_seed_11 = None
        gt_12: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_11, 0.1);  inductor_random_11 = None
        mul_51: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_12, mul_50);  mul_50 = None
        mul_52: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_51, 1.1111111111111112);  mul_51 = None
        convert_element_type_51: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg29_1, torch.bfloat16);  arg29_1 = None
        convert_element_type_52: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None
        permute_36: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_51, [1, 0]);  convert_element_type_51 = None
        view_61: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_52, _shape_param_87);  convert_element_type_52 = _shape_param_87 = None
        mm_20: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_61, permute_36)
        view_62: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_20, _shape_param_88);  mm_20 = _shape_param_88 = None
        inductor_lookup_seed_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 12)
        inductor_random_12: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_89, inductor_lookup_seed_12, 'rand');  _shape_param_89 = inductor_lookup_seed_12 = None
        convert_element_type_53: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_12, torch.bfloat16);  inductor_random_12 = None
        gt_13: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_53, 0.1);  convert_element_type_53 = None
        mul_53: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_13, view_62);  view_62 = None
        mul_54: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_53, 1.1111111111111112);  mul_53 = None
        add_25: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_21, mul_54);  mul_54 = None
        pow_10: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_25, 2)
        mean_6: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
        add_26: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        mul_55: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_25, rsqrt_6)
        mul_56: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg30_1, mul_55);  arg30_1 = mul_55 = None
        convert_element_type_54: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg31_1, torch.bfloat16);  arg31_1 = None
        convert_element_type_55: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_56, torch.bfloat16);  mul_56 = None
        permute_37: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_54, [1, 0]);  convert_element_type_54 = None
        view_63: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_55, _shape_param_90);  convert_element_type_55 = _shape_param_90 = None
        mm_21: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_63, permute_37)
        view_64: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_21, _shape_param_91);  mm_21 = _shape_param_91 = None
        view_65: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_64, _shape_param_92);  view_64 = _shape_param_92 = None
        permute_38: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        convert_element_type_56: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg32_1, torch.bfloat16);  arg32_1 = None
        permute_39: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_56, [1, 0]);  convert_element_type_56 = None
        mm_22: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_63, permute_39)
        view_66: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_22, _shape_param_93);  mm_22 = _shape_param_93 = None
        view_67: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_66, _shape_param_94);  view_66 = _shape_param_94 = None
        permute_40: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_67, [0, 2, 1, 3]);  view_67 = None
        convert_element_type_57: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg33_1, torch.bfloat16);  arg33_1 = None
        permute_41: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_57, [1, 0]);  convert_element_type_57 = None
        mm_23: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_63, permute_41)
        view_68: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_23, _shape_param_95);  mm_23 = _shape_param_95 = None
        view_69: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_68, _shape_param_96);  view_68 = _shape_param_96 = None
        permute_42: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        permute_43: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_40, [0, 1, 3, 2]);  permute_40 = None
        expand_12: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_38, _shape_param_97);  permute_38 = _shape_param_97 = None
        clone_12: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_70: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_98);  clone_12 = _shape_param_98 = None
        expand_13: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_43, _shape_param_99);  permute_43 = _shape_param_99 = None
        clone_13: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_71: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_13, _shape_param_100);  clone_13 = _shape_param_100 = None
        bmm_6: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_70, view_71)
        view_72: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_101);  bmm_6 = _shape_param_101 = None
        add_27: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_72, add_5);  view_72 = None
        convert_element_type_58: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16);  add_27 = None
        convert_element_type_59: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_58, torch.float32)
        amax_3: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_59, [-1], True)
        sub_4: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_59, amax_3);  convert_element_type_59 = None
        exp_3: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_4: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_5: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = None
        convert_element_type_60: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        inductor_lookup_seed_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 13)
        inductor_random_13: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_102, inductor_lookup_seed_13, 'rand');  _shape_param_102 = inductor_lookup_seed_13 = None
        convert_element_type_61: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_13, torch.bfloat16);  inductor_random_13 = None
        gt_14: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_61, 0.1);  convert_element_type_61 = None
        mul_57: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_14, convert_element_type_60);  convert_element_type_60 = None
        mul_58: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_57, 1.1111111111111112);  mul_57 = None
        expand_14: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_58, _shape_param_103);  mul_58 = _shape_param_103 = None
        view_73: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_14, _shape_param_104);  expand_14 = _shape_param_104 = None
        expand_15: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_42, _shape_param_105);  permute_42 = _shape_param_105 = None
        clone_14: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_74: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_14, _shape_param_106);  clone_14 = _shape_param_106 = None
        bmm_7: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_73, view_74)
        view_75: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_107);  bmm_7 = _shape_param_107 = None
        permute_44: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        clone_15: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_76: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_15, _shape_param_108);  clone_15 = _shape_param_108 = None
        convert_element_type_62: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg34_1, torch.bfloat16);  arg34_1 = None
        permute_45: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_62, [1, 0]);  convert_element_type_62 = None
        view_77: "bf16[4096, 384]" = torch.ops.aten.view.default(view_76, _shape_param_109);  view_76 = _shape_param_109 = None
        mm_24: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_77, permute_45)
        view_78: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_24, _shape_param_110);  mm_24 = _shape_param_110 = None
        inductor_lookup_seed_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 14)
        inductor_random_14: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_111, inductor_lookup_seed_14, 'rand');  _shape_param_111 = inductor_lookup_seed_14 = None
        convert_element_type_63: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_14, torch.bfloat16);  inductor_random_14 = None
        gt_15: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_63, 0.1);  convert_element_type_63 = None
        mul_59: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_15, view_78);  view_78 = None
        mul_60: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_59, 1.1111111111111112);  mul_59 = None
        add_28: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_25, mul_60);  mul_60 = None
        pow_11: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_28, 2)
        mean_7: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
        add_29: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        mul_61: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_28, rsqrt_7)
        mul_62: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg35_1, mul_61);  arg35_1 = mul_61 = None
        convert_element_type_64: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg36_1, torch.bfloat16);  arg36_1 = None
        convert_element_type_65: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        permute_46: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_64, [1, 0]);  convert_element_type_64 = None
        view_79: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_65, _shape_param_112);  convert_element_type_65 = _shape_param_112 = None
        mm_25: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_79, permute_46)
        view_80: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_25, _shape_param_113);  _shape_param_113 = None
        mul_63: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_80, 0.5)
        convert_element_type_66: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_80, torch.float32)
        pow_12: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_66, 3.0);  convert_element_type_66 = None
        mul_64: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_30: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_80, mul_64);  view_80 = mul_64 = None
        mul_65: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
        tanh_3: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_65);  mul_65 = None
        add_31: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_66: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_63, add_31);  mul_63 = add_31 = None
        convert_element_type_67: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg37_1, torch.bfloat16);  arg37_1 = None
        permute_47: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_67, [1, 0]);  convert_element_type_67 = None
        mm_26: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_79, permute_47)
        view_81: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_26, _shape_param_114);  _shape_param_114 = None
        mul_67: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_66, view_81);  mul_66 = view_81 = None
        inductor_lookup_seed_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 15)
        inductor_random_15: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_115, inductor_lookup_seed_15, 'rand');  _shape_param_115 = inductor_lookup_seed_15 = None
        gt_16: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_15, 0.1);  inductor_random_15 = None
        mul_68: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_16, mul_67);  mul_67 = None
        mul_69: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_68, 1.1111111111111112);  mul_68 = None
        convert_element_type_68: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg38_1, torch.bfloat16);  arg38_1 = None
        convert_element_type_69: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        permute_48: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_68, [1, 0]);  convert_element_type_68 = None
        view_82: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_69, _shape_param_116);  convert_element_type_69 = _shape_param_116 = None
        mm_27: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_82, permute_48)
        view_83: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_27, _shape_param_117);  mm_27 = _shape_param_117 = None
        inductor_lookup_seed_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 16)
        inductor_random_16: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_118, inductor_lookup_seed_16, 'rand');  _shape_param_118 = inductor_lookup_seed_16 = None
        convert_element_type_70: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_16, torch.bfloat16);  inductor_random_16 = None
        gt_17: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_70, 0.1);  convert_element_type_70 = None
        mul_70: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_17, view_83);  view_83 = None
        mul_71: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_70, 1.1111111111111112);  mul_70 = None
        add_32: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_28, mul_71);  mul_71 = None
        pow_13: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_32, 2)
        mean_8: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
        add_33: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        mul_72: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_32, rsqrt_8)
        mul_73: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg39_1, mul_72);  arg39_1 = mul_72 = None
        convert_element_type_71: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg40_1, torch.bfloat16);  arg40_1 = None
        convert_element_type_72: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_73, torch.bfloat16);  mul_73 = None
        permute_49: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_71, [1, 0]);  convert_element_type_71 = None
        view_84: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_72, _shape_param_119);  convert_element_type_72 = _shape_param_119 = None
        mm_28: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_84, permute_49)
        view_85: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_28, _shape_param_120);  mm_28 = _shape_param_120 = None
        view_86: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_85, _shape_param_121);  view_85 = _shape_param_121 = None
        permute_50: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        convert_element_type_73: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg41_1, torch.bfloat16);  arg41_1 = None
        permute_51: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_73, [1, 0]);  convert_element_type_73 = None
        mm_29: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_84, permute_51)
        view_87: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_29, _shape_param_122);  mm_29 = _shape_param_122 = None
        view_88: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_87, _shape_param_123);  view_87 = _shape_param_123 = None
        permute_52: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        convert_element_type_74: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg42_1, torch.bfloat16);  arg42_1 = None
        permute_53: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_74, [1, 0]);  convert_element_type_74 = None
        mm_30: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_84, permute_53)
        view_89: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_30, _shape_param_124);  mm_30 = _shape_param_124 = None
        view_90: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_89, _shape_param_125);  view_89 = _shape_param_125 = None
        permute_54: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        permute_55: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_52, [0, 1, 3, 2]);  permute_52 = None
        expand_16: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_50, _shape_param_126);  permute_50 = _shape_param_126 = None
        clone_16: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_91: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_127);  clone_16 = _shape_param_127 = None
        expand_17: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_55, _shape_param_128);  permute_55 = _shape_param_128 = None
        clone_17: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_92: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_17, _shape_param_129);  clone_17 = _shape_param_129 = None
        bmm_8: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_91, view_92)
        view_93: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_8, _shape_param_130);  bmm_8 = _shape_param_130 = None
        add_34: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_93, add_5);  view_93 = None
        convert_element_type_75: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16);  add_34 = None
        convert_element_type_76: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_75, torch.float32)
        amax_4: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_76, [-1], True)
        sub_5: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_76, amax_4);  convert_element_type_76 = None
        exp_4: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_5: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_6: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = None
        convert_element_type_77: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        inductor_lookup_seed_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 17)
        inductor_random_17: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_131, inductor_lookup_seed_17, 'rand');  _shape_param_131 = inductor_lookup_seed_17 = None
        convert_element_type_78: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_17, torch.bfloat16);  inductor_random_17 = None
        gt_18: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_78, 0.1);  convert_element_type_78 = None
        mul_74: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_18, convert_element_type_77);  convert_element_type_77 = None
        mul_75: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_74, 1.1111111111111112);  mul_74 = None
        expand_18: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_75, _shape_param_132);  mul_75 = _shape_param_132 = None
        view_94: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_18, _shape_param_133);  expand_18 = _shape_param_133 = None
        expand_19: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_54, _shape_param_134);  permute_54 = _shape_param_134 = None
        clone_18: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_95: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_18, _shape_param_135);  clone_18 = _shape_param_135 = None
        bmm_9: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_94, view_95)
        view_96: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_9, _shape_param_136);  bmm_9 = _shape_param_136 = None
        permute_56: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        clone_19: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_97: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_19, _shape_param_137);  clone_19 = _shape_param_137 = None
        convert_element_type_79: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg43_1, torch.bfloat16);  arg43_1 = None
        permute_57: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_79, [1, 0]);  convert_element_type_79 = None
        view_98: "bf16[4096, 384]" = torch.ops.aten.view.default(view_97, _shape_param_138);  view_97 = _shape_param_138 = None
        mm_31: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_98, permute_57)
        view_99: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_31, _shape_param_139);  mm_31 = _shape_param_139 = None
        inductor_lookup_seed_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 18)
        inductor_random_18: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_140, inductor_lookup_seed_18, 'rand');  _shape_param_140 = inductor_lookup_seed_18 = None
        convert_element_type_80: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_18, torch.bfloat16);  inductor_random_18 = None
        gt_19: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_80, 0.1);  convert_element_type_80 = None
        mul_76: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_19, view_99);  view_99 = None
        mul_77: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_76, 1.1111111111111112);  mul_76 = None
        add_35: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_32, mul_77);  mul_77 = None
        pow_14: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_35, 2)
        mean_9: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
        add_36: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_78: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_35, rsqrt_9)
        mul_79: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg44_1, mul_78);  arg44_1 = mul_78 = None
        convert_element_type_81: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg45_1, torch.bfloat16);  arg45_1 = None
        convert_element_type_82: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None
        permute_58: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_81, [1, 0]);  convert_element_type_81 = None
        view_100: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_82, _shape_param_141);  convert_element_type_82 = _shape_param_141 = None
        mm_32: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_100, permute_58)
        view_101: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_32, _shape_param_142);  _shape_param_142 = None
        mul_80: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_101, 0.5)
        convert_element_type_83: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_101, torch.float32)
        pow_15: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_83, 3.0);  convert_element_type_83 = None
        mul_81: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_37: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_101, mul_81);  view_101 = mul_81 = None
        mul_82: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_37, 0.7978845608028654);  add_37 = None
        tanh_4: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_82);  mul_82 = None
        add_38: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_83: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_80, add_38);  mul_80 = add_38 = None
        convert_element_type_84: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg46_1, torch.bfloat16);  arg46_1 = None
        permute_59: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_84, [1, 0]);  convert_element_type_84 = None
        mm_33: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_100, permute_59)
        view_102: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_33, _shape_param_143);  _shape_param_143 = None
        mul_84: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_83, view_102);  mul_83 = view_102 = None
        inductor_lookup_seed_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 19)
        inductor_random_19: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_144, inductor_lookup_seed_19, 'rand');  _shape_param_144 = inductor_lookup_seed_19 = None
        gt_20: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_19, 0.1);  inductor_random_19 = None
        mul_85: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_20, mul_84);  mul_84 = None
        mul_86: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_85, 1.1111111111111112);  mul_85 = None
        convert_element_type_85: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg47_1, torch.bfloat16);  arg47_1 = None
        convert_element_type_86: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_86, torch.bfloat16);  mul_86 = None
        permute_60: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_85, [1, 0]);  convert_element_type_85 = None
        view_103: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_86, _shape_param_145);  convert_element_type_86 = _shape_param_145 = None
        mm_34: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_103, permute_60)
        view_104: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_34, _shape_param_146);  mm_34 = _shape_param_146 = None
        inductor_lookup_seed_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 20)
        inductor_random_20: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_147, inductor_lookup_seed_20, 'rand');  _shape_param_147 = inductor_lookup_seed_20 = None
        convert_element_type_87: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_20, torch.bfloat16);  inductor_random_20 = None
        gt_21: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_87, 0.1);  convert_element_type_87 = None
        mul_87: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_21, view_104);  view_104 = None
        mul_88: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_87, 1.1111111111111112);  mul_87 = None
        add_39: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_35, mul_88);  mul_88 = None
        pow_16: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_39, 2)
        mean_10: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
        add_40: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_89: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_39, rsqrt_10)
        mul_90: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg48_1, mul_89);  arg48_1 = mul_89 = None
        convert_element_type_88: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg49_1, torch.bfloat16);  arg49_1 = None
        convert_element_type_89: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16);  mul_90 = None
        permute_61: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_88, [1, 0]);  convert_element_type_88 = None
        view_105: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_89, _shape_param_148);  convert_element_type_89 = _shape_param_148 = None
        mm_35: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_105, permute_61)
        view_106: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_35, _shape_param_149);  mm_35 = _shape_param_149 = None
        view_107: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_106, _shape_param_150);  view_106 = _shape_param_150 = None
        permute_62: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
        convert_element_type_90: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg50_1, torch.bfloat16);  arg50_1 = None
        permute_63: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_90, [1, 0]);  convert_element_type_90 = None
        mm_36: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_105, permute_63)
        view_108: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_36, _shape_param_151);  mm_36 = _shape_param_151 = None
        view_109: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_108, _shape_param_152);  view_108 = _shape_param_152 = None
        permute_64: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
        convert_element_type_91: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg51_1, torch.bfloat16);  arg51_1 = None
        permute_65: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_91, [1, 0]);  convert_element_type_91 = None
        mm_37: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_105, permute_65)
        view_110: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_37, _shape_param_153);  mm_37 = _shape_param_153 = None
        view_111: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_110, _shape_param_154);  view_110 = _shape_param_154 = None
        permute_66: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
        permute_67: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_64, [0, 1, 3, 2]);  permute_64 = None
        expand_20: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_62, _shape_param_155);  permute_62 = _shape_param_155 = None
        clone_20: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_112: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_156);  clone_20 = _shape_param_156 = None
        expand_21: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_67, _shape_param_157);  permute_67 = _shape_param_157 = None
        clone_21: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_113: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_21, _shape_param_158);  clone_21 = _shape_param_158 = None
        bmm_10: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_112, view_113)
        view_114: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_159);  bmm_10 = _shape_param_159 = None
        add_41: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_114, add_5);  view_114 = None
        convert_element_type_92: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None
        convert_element_type_93: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_92, torch.float32)
        amax_5: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_93, [-1], True)
        sub_6: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_93, amax_5);  convert_element_type_93 = None
        exp_5: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_6: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_7: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = None
        convert_element_type_94: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        inductor_lookup_seed_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 21)
        inductor_random_21: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_160, inductor_lookup_seed_21, 'rand');  _shape_param_160 = inductor_lookup_seed_21 = None
        convert_element_type_95: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_21, torch.bfloat16);  inductor_random_21 = None
        gt_22: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_95, 0.1);  convert_element_type_95 = None
        mul_91: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_22, convert_element_type_94);  convert_element_type_94 = None
        mul_92: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_91, 1.1111111111111112);  mul_91 = None
        expand_22: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_92, _shape_param_161);  mul_92 = _shape_param_161 = None
        view_115: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_22, _shape_param_162);  expand_22 = _shape_param_162 = None
        expand_23: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_66, _shape_param_163);  permute_66 = _shape_param_163 = None
        clone_22: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_116: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_22, _shape_param_164);  clone_22 = _shape_param_164 = None
        bmm_11: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_115, view_116)
        view_117: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_165);  bmm_11 = _shape_param_165 = None
        permute_68: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        clone_23: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_118: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_23, _shape_param_166);  clone_23 = _shape_param_166 = None
        convert_element_type_96: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg52_1, torch.bfloat16);  arg52_1 = None
        permute_69: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_96, [1, 0]);  convert_element_type_96 = None
        view_119: "bf16[4096, 384]" = torch.ops.aten.view.default(view_118, _shape_param_167);  view_118 = _shape_param_167 = None
        mm_38: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_119, permute_69)
        view_120: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_38, _shape_param_168);  mm_38 = _shape_param_168 = None
        inductor_lookup_seed_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 22)
        inductor_random_22: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_169, inductor_lookup_seed_22, 'rand');  _shape_param_169 = inductor_lookup_seed_22 = None
        convert_element_type_97: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_22, torch.bfloat16);  inductor_random_22 = None
        gt_23: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_97, 0.1);  convert_element_type_97 = None
        mul_93: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_23, view_120);  view_120 = None
        mul_94: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_93, 1.1111111111111112);  mul_93 = None
        add_42: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_39, mul_94);  mul_94 = None
        pow_17: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_42, 2)
        mean_11: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
        add_43: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_95: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_42, rsqrt_11)
        mul_96: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg53_1, mul_95);  arg53_1 = mul_95 = None
        convert_element_type_98: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg54_1, torch.bfloat16);  arg54_1 = None
        convert_element_type_99: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_96, torch.bfloat16);  mul_96 = None
        permute_70: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_98, [1, 0]);  convert_element_type_98 = None
        view_121: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_99, _shape_param_170);  convert_element_type_99 = _shape_param_170 = None
        mm_39: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_121, permute_70)
        view_122: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_39, _shape_param_171);  _shape_param_171 = None
        mul_97: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_122, 0.5)
        convert_element_type_100: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_122, torch.float32)
        pow_18: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_100, 3.0);  convert_element_type_100 = None
        mul_98: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_44: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_122, mul_98);  view_122 = mul_98 = None
        mul_99: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_44, 0.7978845608028654);  add_44 = None
        tanh_5: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_99);  mul_99 = None
        add_45: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_100: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_97, add_45);  mul_97 = add_45 = None
        convert_element_type_101: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg55_1, torch.bfloat16);  arg55_1 = None
        permute_71: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_101, [1, 0]);  convert_element_type_101 = None
        mm_40: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_121, permute_71)
        view_123: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_40, _shape_param_172);  _shape_param_172 = None
        mul_101: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_100, view_123);  mul_100 = view_123 = None
        inductor_lookup_seed_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 23)
        inductor_random_23: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_173, inductor_lookup_seed_23, 'rand');  _shape_param_173 = inductor_lookup_seed_23 = None
        gt_24: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_23, 0.1);  inductor_random_23 = None
        mul_102: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_24, mul_101);  mul_101 = None
        mul_103: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_102, 1.1111111111111112);  mul_102 = None
        convert_element_type_102: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg56_1, torch.bfloat16);  arg56_1 = None
        convert_element_type_103: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_103, torch.bfloat16);  mul_103 = None
        permute_72: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_102, [1, 0]);  convert_element_type_102 = None
        view_124: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_103, _shape_param_174);  convert_element_type_103 = _shape_param_174 = None
        mm_41: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_124, permute_72)
        view_125: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_41, _shape_param_175);  mm_41 = _shape_param_175 = None
        inductor_lookup_seed_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 24)
        inductor_random_24: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_176, inductor_lookup_seed_24, 'rand');  _shape_param_176 = inductor_lookup_seed_24 = None
        convert_element_type_104: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_24, torch.bfloat16);  inductor_random_24 = None
        gt_25: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_104, 0.1);  convert_element_type_104 = None
        mul_104: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_25, view_125);  view_125 = None
        mul_105: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_104, 1.1111111111111112);  mul_104 = None
        add_46: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_42, mul_105);  mul_105 = None
        pow_19: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_46, 2)
        mean_12: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
        add_47: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_106: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_46, rsqrt_12)
        mul_107: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg57_1, mul_106);  arg57_1 = mul_106 = None
        convert_element_type_105: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg58_1, torch.bfloat16);  arg58_1 = None
        convert_element_type_106: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_107, torch.bfloat16);  mul_107 = None
        permute_73: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_105, [1, 0]);  convert_element_type_105 = None
        view_126: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_106, _shape_param_177);  convert_element_type_106 = _shape_param_177 = None
        mm_42: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_126, permute_73)
        view_127: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_42, _shape_param_178);  mm_42 = _shape_param_178 = None
        view_128: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_127, _shape_param_179);  view_127 = _shape_param_179 = None
        permute_74: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
        convert_element_type_107: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg59_1, torch.bfloat16);  arg59_1 = None
        permute_75: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_107, [1, 0]);  convert_element_type_107 = None
        mm_43: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_126, permute_75)
        view_129: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_43, _shape_param_180);  mm_43 = _shape_param_180 = None
        view_130: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_129, _shape_param_181);  view_129 = _shape_param_181 = None
        permute_76: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        convert_element_type_108: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg60_1, torch.bfloat16);  arg60_1 = None
        permute_77: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_108, [1, 0]);  convert_element_type_108 = None
        mm_44: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_126, permute_77)
        view_131: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_44, _shape_param_182);  mm_44 = _shape_param_182 = None
        view_132: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_131, _shape_param_183);  view_131 = _shape_param_183 = None
        permute_78: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
        permute_79: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_76, [0, 1, 3, 2]);  permute_76 = None
        expand_24: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_74, _shape_param_184);  permute_74 = _shape_param_184 = None
        clone_24: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_133: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_185);  clone_24 = _shape_param_185 = None
        expand_25: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_79, _shape_param_186);  permute_79 = _shape_param_186 = None
        clone_25: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_134: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_25, _shape_param_187);  clone_25 = _shape_param_187 = None
        bmm_12: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_133, view_134)
        view_135: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_12, _shape_param_188);  bmm_12 = _shape_param_188 = None
        add_48: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_135, add_5);  view_135 = None
        convert_element_type_109: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_48, torch.bfloat16);  add_48 = None
        convert_element_type_110: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_109, torch.float32)
        amax_6: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_110, [-1], True)
        sub_7: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_110, amax_6);  convert_element_type_110 = None
        exp_6: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_7: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_8: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = None
        convert_element_type_111: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        inductor_lookup_seed_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 25)
        inductor_random_25: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_189, inductor_lookup_seed_25, 'rand');  _shape_param_189 = inductor_lookup_seed_25 = None
        convert_element_type_112: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_25, torch.bfloat16);  inductor_random_25 = None
        gt_26: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_112, 0.1);  convert_element_type_112 = None
        mul_108: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_26, convert_element_type_111);  convert_element_type_111 = None
        mul_109: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_108, 1.1111111111111112);  mul_108 = None
        expand_26: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_109, _shape_param_190);  mul_109 = _shape_param_190 = None
        view_136: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_26, _shape_param_191);  expand_26 = _shape_param_191 = None
        expand_27: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_78, _shape_param_192);  permute_78 = _shape_param_192 = None
        clone_26: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_137: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_26, _shape_param_193);  clone_26 = _shape_param_193 = None
        bmm_13: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_136, view_137)
        view_138: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_13, _shape_param_194);  bmm_13 = _shape_param_194 = None
        permute_80: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        clone_27: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_139: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_27, _shape_param_195);  clone_27 = _shape_param_195 = None
        convert_element_type_113: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg61_1, torch.bfloat16);  arg61_1 = None
        permute_81: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_113, [1, 0]);  convert_element_type_113 = None
        view_140: "bf16[4096, 384]" = torch.ops.aten.view.default(view_139, _shape_param_196);  view_139 = _shape_param_196 = None
        mm_45: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_140, permute_81)
        view_141: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_45, _shape_param_197);  mm_45 = _shape_param_197 = None
        inductor_lookup_seed_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 26)
        inductor_random_26: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_198, inductor_lookup_seed_26, 'rand');  _shape_param_198 = inductor_lookup_seed_26 = None
        convert_element_type_114: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_26, torch.bfloat16);  inductor_random_26 = None
        gt_27: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_114, 0.1);  convert_element_type_114 = None
        mul_110: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_27, view_141);  view_141 = None
        mul_111: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_110, 1.1111111111111112);  mul_110 = None
        add_49: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_46, mul_111);  mul_111 = None
        pow_20: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_49, 2)
        mean_13: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
        add_50: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_112: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_49, rsqrt_13)
        mul_113: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg62_1, mul_112);  arg62_1 = mul_112 = None
        convert_element_type_115: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg63_1, torch.bfloat16);  arg63_1 = None
        convert_element_type_116: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_113, torch.bfloat16);  mul_113 = None
        permute_82: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_115, [1, 0]);  convert_element_type_115 = None
        view_142: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_116, _shape_param_199);  convert_element_type_116 = _shape_param_199 = None
        mm_46: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_142, permute_82)
        view_143: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_46, _shape_param_200);  _shape_param_200 = None
        mul_114: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_143, 0.5)
        convert_element_type_117: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_143, torch.float32)
        pow_21: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_117, 3.0);  convert_element_type_117 = None
        mul_115: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_51: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_143, mul_115);  view_143 = mul_115 = None
        mul_116: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_51, 0.7978845608028654);  add_51 = None
        tanh_6: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_116);  mul_116 = None
        add_52: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_117: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_114, add_52);  mul_114 = add_52 = None
        convert_element_type_118: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg64_1, torch.bfloat16);  arg64_1 = None
        permute_83: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_118, [1, 0]);  convert_element_type_118 = None
        mm_47: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_142, permute_83)
        view_144: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_47, _shape_param_201);  _shape_param_201 = None
        mul_118: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_117, view_144);  mul_117 = view_144 = None
        inductor_lookup_seed_27: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 27)
        inductor_random_27: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_202, inductor_lookup_seed_27, 'rand');  _shape_param_202 = inductor_lookup_seed_27 = None
        gt_28: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_27, 0.1);  inductor_random_27 = None
        mul_119: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_28, mul_118);  mul_118 = None
        mul_120: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_119, 1.1111111111111112);  mul_119 = None
        convert_element_type_119: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg65_1, torch.bfloat16);  arg65_1 = None
        convert_element_type_120: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None
        permute_84: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_119, [1, 0]);  convert_element_type_119 = None
        view_145: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_120, _shape_param_203);  convert_element_type_120 = _shape_param_203 = None
        mm_48: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_145, permute_84)
        view_146: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_48, _shape_param_204);  mm_48 = _shape_param_204 = None
        inductor_lookup_seed_28: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 28)
        inductor_random_28: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_205, inductor_lookup_seed_28, 'rand');  _shape_param_205 = inductor_lookup_seed_28 = None
        convert_element_type_121: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_28, torch.bfloat16);  inductor_random_28 = None
        gt_29: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_121, 0.1);  convert_element_type_121 = None
        mul_121: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_29, view_146);  view_146 = None
        mul_122: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_121, 1.1111111111111112);  mul_121 = None
        add_53: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_49, mul_122);  mul_122 = None
        pow_22: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_53, 2)
        mean_14: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
        add_54: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_123: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_53, rsqrt_14)
        mul_124: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg66_1, mul_123);  arg66_1 = mul_123 = None
        convert_element_type_122: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg67_1, torch.bfloat16);  arg67_1 = None
        convert_element_type_123: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_124, torch.bfloat16);  mul_124 = None
        permute_85: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_122, [1, 0]);  convert_element_type_122 = None
        view_147: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_123, _shape_param_206);  convert_element_type_123 = _shape_param_206 = None
        mm_49: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_147, permute_85)
        view_148: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_49, _shape_param_207);  mm_49 = _shape_param_207 = None
        view_149: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_148, _shape_param_208);  view_148 = _shape_param_208 = None
        permute_86: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
        convert_element_type_124: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg68_1, torch.bfloat16);  arg68_1 = None
        permute_87: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_124, [1, 0]);  convert_element_type_124 = None
        mm_50: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_147, permute_87)
        view_150: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_50, _shape_param_209);  mm_50 = _shape_param_209 = None
        view_151: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_150, _shape_param_210);  view_150 = _shape_param_210 = None
        permute_88: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
        convert_element_type_125: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg69_1, torch.bfloat16);  arg69_1 = None
        permute_89: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_125, [1, 0]);  convert_element_type_125 = None
        mm_51: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_147, permute_89)
        view_152: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_51, _shape_param_211);  mm_51 = _shape_param_211 = None
        view_153: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_152, _shape_param_212);  view_152 = _shape_param_212 = None
        permute_90: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
        permute_91: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_88, [0, 1, 3, 2]);  permute_88 = None
        expand_28: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_86, _shape_param_213);  permute_86 = _shape_param_213 = None
        clone_28: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_154: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_214);  clone_28 = _shape_param_214 = None
        expand_29: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_91, _shape_param_215);  permute_91 = _shape_param_215 = None
        clone_29: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_155: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_29, _shape_param_216);  clone_29 = _shape_param_216 = None
        bmm_14: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_154, view_155)
        view_156: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_14, _shape_param_217);  bmm_14 = _shape_param_217 = None
        add_55: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_156, add_5);  view_156 = add_5 = None
        convert_element_type_126: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None
        convert_element_type_127: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_126, torch.float32)
        amax_7: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_127, [-1], True)
        sub_8: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_127, amax_7);  convert_element_type_127 = None
        exp_7: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_8: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_9: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = None
        convert_element_type_128: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        inductor_lookup_seed_29: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 29)
        inductor_random_29: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_218, inductor_lookup_seed_29, 'rand');  _shape_param_218 = inductor_lookup_seed_29 = None
        convert_element_type_129: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_29, torch.bfloat16);  inductor_random_29 = None
        gt_30: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_129, 0.1);  convert_element_type_129 = None
        mul_125: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_30, convert_element_type_128);  convert_element_type_128 = None
        mul_126: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_125, 1.1111111111111112);  mul_125 = None
        expand_30: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_126, _shape_param_219);  mul_126 = _shape_param_219 = None
        view_157: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_30, _shape_param_220);  expand_30 = _shape_param_220 = None
        expand_31: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_90, _shape_param_221);  permute_90 = _shape_param_221 = None
        clone_30: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_158: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_30, _shape_param_222);  clone_30 = _shape_param_222 = None
        bmm_15: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_157, view_158)
        view_159: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_223);  bmm_15 = _shape_param_223 = None
        permute_92: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
        clone_31: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_160: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_31, _shape_param_224);  clone_31 = _shape_param_224 = None
        convert_element_type_130: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg70_1, torch.bfloat16);  arg70_1 = None
        permute_93: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_130, [1, 0]);  convert_element_type_130 = None
        view_161: "bf16[4096, 384]" = torch.ops.aten.view.default(view_160, _shape_param_225);  view_160 = _shape_param_225 = None
        mm_52: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_161, permute_93)
        view_162: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_52, _shape_param_226);  mm_52 = _shape_param_226 = None
        inductor_lookup_seed_30: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 30)
        inductor_random_30: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_227, inductor_lookup_seed_30, 'rand');  _shape_param_227 = inductor_lookup_seed_30 = None
        convert_element_type_131: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_30, torch.bfloat16);  inductor_random_30 = None
        gt_31: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_131, 0.1);  convert_element_type_131 = None
        mul_127: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_31, view_162);  view_162 = None
        mul_128: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_127, 1.1111111111111112);  mul_127 = None
        add_56: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_53, mul_128);  mul_128 = None
        pow_23: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_56, 2)
        mean_15: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
        add_57: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        mul_129: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_56, rsqrt_15)
        mul_130: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg71_1, mul_129);  arg71_1 = mul_129 = None
        convert_element_type_132: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg72_1, torch.bfloat16);  arg72_1 = None
        convert_element_type_133: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_130, torch.bfloat16);  mul_130 = None
        permute_94: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_132, [1, 0]);  convert_element_type_132 = None
        view_163: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_133, _shape_param_228);  convert_element_type_133 = _shape_param_228 = None
        mm_53: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_163, permute_94)
        view_164: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_53, _shape_param_229);  _shape_param_229 = None
        mul_131: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_164, 0.5)
        convert_element_type_134: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_164, torch.float32)
        pow_24: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_134, 3.0);  convert_element_type_134 = None
        mul_132: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_58: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_164, mul_132);  view_164 = mul_132 = None
        mul_133: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_58, 0.7978845608028654);  add_58 = None
        tanh_7: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_133);  mul_133 = None
        add_59: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_134: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_131, add_59);  mul_131 = add_59 = None
        convert_element_type_135: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg73_1, torch.bfloat16);  arg73_1 = None
        permute_95: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_135, [1, 0]);  convert_element_type_135 = None
        mm_54: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_163, permute_95)
        view_165: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_54, _shape_param_230);  _shape_param_230 = None
        mul_135: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_134, view_165);  mul_134 = view_165 = None
        inductor_lookup_seed_31: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 31)
        inductor_random_31: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_231, inductor_lookup_seed_31, 'rand');  _shape_param_231 = inductor_lookup_seed_31 = None
        gt_32: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_31, 0.1);  inductor_random_31 = None
        mul_136: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_32, mul_135);  mul_135 = None
        mul_137: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_136, 1.1111111111111112);  mul_136 = None
        convert_element_type_136: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg74_1, torch.bfloat16);  arg74_1 = None
        convert_element_type_137: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_137, torch.bfloat16);  mul_137 = None
        permute_96: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_136, [1, 0]);  convert_element_type_136 = None
        view_166: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_137, _shape_param_232);  convert_element_type_137 = _shape_param_232 = None
        mm_55: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_166, permute_96)
        view_167: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_55, _shape_param_233);  mm_55 = _shape_param_233 = None
        inductor_lookup_seed_32: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 32)
        inductor_random_32: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_234, inductor_lookup_seed_32, 'rand');  _shape_param_234 = inductor_lookup_seed_32 = None
        convert_element_type_138: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_32, torch.bfloat16);  inductor_random_32 = None
        gt_33: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_138, 0.1);  convert_element_type_138 = None
        mul_138: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_33, view_167);  view_167 = None
        mul_139: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_138, 1.1111111111111112);  mul_138 = None
        add_60: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_56, mul_139);  mul_139 = None
        pow_25: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_60, 2)
        mean_16: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
        add_61: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_140: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_60, rsqrt_16)
        mul_141: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg75_1, mul_140);  arg75_1 = mul_140 = None
        inductor_lookup_seed_33: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 33)
        inductor_random_33: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_235, inductor_lookup_seed_33, 'rand');  _shape_param_235 = inductor_lookup_seed_33 = None
        gt_34: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(inductor_random_33, 0.1);  inductor_random_33 = None
        mul_142: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_34, mul_141);  mul_141 = None
        mul_143: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_142, 1.1111111111111112);  mul_142 = None
        add_62: "i64[128]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze_3: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_62, 0);  add_62 = None
        unsqueeze_4: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3)
        unsqueeze_6: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2)
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_6, unsqueeze_5);  unsqueeze_6 = unsqueeze_5 = None
        expand_32: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(le, _shape_param_236);  le = _shape_param_236 = None
        full_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_3: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_32, full_2, full_3);  expand_32 = full_3 = None
        inductor_lookup_seed_34: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 34)
        inductor_random_34: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_237, inductor_lookup_seed_34, 'rand');  _shape_param_237 = inductor_lookup_seed_34 = None
        gt_35: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(inductor_random_34, 0.1);  inductor_random_34 = None
        mul_144: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_35, embedding)
        mul_145: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_144, 1.1111111111111112);  mul_144 = None
        pow_26: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_145, 2)
        mean_17: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None
        add_63: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_146: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_145, rsqrt_17)
        mul_147: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg76_1, mul_146);  arg76_1 = mul_146 = None
        convert_element_type_139: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg77_1, torch.bfloat16);  arg77_1 = None
        convert_element_type_140: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_147, torch.bfloat16);  mul_147 = None
        permute_97: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_139, [1, 0]);  convert_element_type_139 = None
        view_168: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_140, _shape_param_238);  convert_element_type_140 = _shape_param_238 = None
        mm_56: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_168, permute_97)
        view_169: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_56, _shape_param_239);  mm_56 = _shape_param_239 = None
        view_170: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_169, _shape_param_240);  view_169 = _shape_param_240 = None
        permute_98: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        convert_element_type_141: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg78_1, torch.bfloat16);  arg78_1 = None
        permute_99: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_141, [1, 0]);  convert_element_type_141 = None
        mm_57: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_168, permute_99)
        view_171: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_57, _shape_param_241);  mm_57 = _shape_param_241 = None
        view_172: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_171, _shape_param_242);  view_171 = _shape_param_242 = None
        permute_100: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
        convert_element_type_142: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg79_1, torch.bfloat16);  arg79_1 = None
        permute_101: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_142, [1, 0]);  convert_element_type_142 = None
        mm_58: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_168, permute_101)
        view_173: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_58, _shape_param_243);  mm_58 = _shape_param_243 = None
        view_174: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_173, _shape_param_244);  view_173 = _shape_param_244 = None
        permute_102: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
        permute_103: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2]);  permute_100 = None
        expand_33: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_98, _shape_param_245);  permute_98 = _shape_param_245 = None
        clone_32: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_175: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_32, _shape_param_246);  clone_32 = _shape_param_246 = None
        expand_34: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_103, _shape_param_247);  permute_103 = _shape_param_247 = None
        clone_33: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_176: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_33, _shape_param_248);  clone_33 = _shape_param_248 = None
        bmm_16: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_175, view_176)
        view_177: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_16, _shape_param_249);  _shape_param_249 = None
        full_4: "i64[128, 128]" = torch.ops.aten.full.default(_shape_param_250, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_250 = None
        minimum_1: "i64[128, 128]" = torch.ops.aten.minimum.default(sub, full_4);  sub = full_4 = None
        neg: "i64[128, 128]" = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None
        lt_1: "b8[128, 128]" = torch.ops.aten.lt.Scalar(neg, 16)
        convert_element_type_143: "f32[128, 128]" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
        div_10: "f32[128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_143, 16);  convert_element_type_143 = None
        log_1: "f32[128, 128]" = torch.ops.aten.log.default(div_10);  div_10 = None
        div_11: "f32[128, 128]" = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
        mul_148: "f32[128, 128]" = torch.ops.aten.mul.Tensor(div_11, 16);  div_11 = None
        convert_element_type_144: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(mul_148, torch.int64);  mul_148 = None
        add_64: "i64[128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_144, 16);  convert_element_type_144 = None
        full_5: "i64[128, 128]" = torch.ops.aten.full.default(_shape_param_251, 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_251 = None
        minimum_2: "i64[128, 128]" = torch.ops.aten.minimum.default(add_64, full_5);  add_64 = full_5 = None
        where_2: "i64[128, 128]" = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
        add_65: "i64[128, 128]" = torch.ops.aten.add.Tensor(where_2, 0);  where_2 = None
        embedding_2: "f32[128, 128, 6]" = torch.ops.aten.embedding.default(arg80_1, add_65);  arg80_1 = None
        permute_104: "f32[6, 128, 128]" = torch.ops.aten.permute.default(embedding_2, [2, 0, 1])
        unsqueeze_7: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_104, 0);  permute_104 = None
        add_66: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_7, where_1);  unsqueeze_7 = where_1 = None
        add_67: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_177, add_66);  view_177 = None
        convert_element_type_145: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_67, torch.bfloat16);  add_67 = None
        convert_element_type_146: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_145, torch.float32);  convert_element_type_145 = None
        amax_8: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_146, [-1], True)
        sub_9: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_146, amax_8);  convert_element_type_146 = None
        exp_8: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_9: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_12: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = None
        convert_element_type_147: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None
        inductor_lookup_seed_35: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 35)
        inductor_random_35: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_252, inductor_lookup_seed_35, 'rand');  _shape_param_252 = inductor_lookup_seed_35 = None
        convert_element_type_148: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_35, torch.bfloat16);  inductor_random_35 = None
        gt_36: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_148, 0.1);  convert_element_type_148 = None
        mul_149: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_36, convert_element_type_147);  convert_element_type_147 = None
        mul_150: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_149, 1.1111111111111112);  mul_149 = None
        expand_35: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_150, _shape_param_253);  mul_150 = _shape_param_253 = None
        view_178: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_35, _shape_param_254);  expand_35 = _shape_param_254 = None
        expand_36: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_102, _shape_param_255);  permute_102 = _shape_param_255 = None
        clone_34: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_179: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_34, _shape_param_256);  clone_34 = _shape_param_256 = None
        bmm_17: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_178, view_179)
        view_180: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_17, _shape_param_257);  bmm_17 = _shape_param_257 = None
        permute_105: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
        clone_35: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        view_181: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_35, _shape_param_258);  clone_35 = _shape_param_258 = None
        convert_element_type_149: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg81_1, torch.bfloat16);  arg81_1 = None
        permute_106: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_149, [1, 0]);  convert_element_type_149 = None
        view_182: "bf16[4096, 384]" = torch.ops.aten.view.default(view_181, _shape_param_259);  view_181 = _shape_param_259 = None
        mm_59: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_182, permute_106)
        view_183: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_59, _shape_param_260);  mm_59 = _shape_param_260 = None
        inductor_lookup_seed_36: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 36)
        inductor_random_36: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_261, inductor_lookup_seed_36, 'rand');  _shape_param_261 = inductor_lookup_seed_36 = None
        convert_element_type_150: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_36, torch.bfloat16);  inductor_random_36 = None
        gt_37: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_150, 0.1);  convert_element_type_150 = None
        mul_151: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_37, view_183);  view_183 = None
        mul_152: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_151, 1.1111111111111112);  mul_151 = None
        add_68: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(mul_145, mul_152);  mul_145 = mul_152 = None
        pow_27: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_68, 2)
        mean_18: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
        add_69: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_153: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_68, rsqrt_18)
        mul_154: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg82_1, mul_153);  arg82_1 = mul_153 = None
        convert_element_type_151: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg83_1, torch.bfloat16);  arg83_1 = None
        convert_element_type_152: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_154, torch.bfloat16);  mul_154 = None
        permute_107: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_151, [1, 0]);  convert_element_type_151 = None
        view_184: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_152, _shape_param_262);  convert_element_type_152 = _shape_param_262 = None
        mm_60: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_184, permute_107)
        view_185: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_60, _shape_param_263);  mm_60 = _shape_param_263 = None
        view_186: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_185, _shape_param_264);  view_185 = _shape_param_264 = None
        permute_108: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        convert_element_type_153: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg84_1, torch.bfloat16);  arg84_1 = None
        convert_element_type_154: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_143, torch.bfloat16)
        permute_109: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_153, [1, 0]);  convert_element_type_153 = None
        view_187: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_154, _shape_param_265);  convert_element_type_154 = _shape_param_265 = None
        mm_61: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_187, permute_109)
        view_188: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_61, _shape_param_266);  mm_61 = _shape_param_266 = None
        view_189: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_188, _shape_param_267);  view_188 = _shape_param_267 = None
        permute_110: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None
        convert_element_type_155: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg85_1, torch.bfloat16);  arg85_1 = None
        permute_111: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_155, [1, 0]);  convert_element_type_155 = None
        mm_62: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_187, permute_111)
        view_190: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_62, _shape_param_268);  mm_62 = _shape_param_268 = None
        view_191: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_190, _shape_param_269);  view_190 = _shape_param_269 = None
        permute_112: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
        permute_113: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_110, [0, 1, 3, 2]);  permute_110 = None
        expand_37: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_108, _shape_param_270);  permute_108 = _shape_param_270 = None
        clone_36: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_192: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_271);  clone_36 = _shape_param_271 = None
        expand_38: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_113, _shape_param_272);  permute_113 = _shape_param_272 = None
        clone_37: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_193: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_37, _shape_param_273);  clone_37 = _shape_param_273 = None
        bmm_18: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_192, view_193)
        view_194: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_18, _shape_param_274);  _shape_param_274 = None
        convert_element_type_156: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_194, torch.bfloat16);  view_194 = None
        convert_element_type_157: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_156, torch.float32);  convert_element_type_156 = None
        amax_9: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_157, [-1], True)
        sub_10: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_157, amax_9);  convert_element_type_157 = None
        exp_9: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_10: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_13: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = None
        convert_element_type_158: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None
        inductor_lookup_seed_37: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 37)
        inductor_random_37: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_275, inductor_lookup_seed_37, 'rand');  _shape_param_275 = inductor_lookup_seed_37 = None
        convert_element_type_159: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_37, torch.bfloat16);  inductor_random_37 = None
        gt_38: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_159, 0.1);  convert_element_type_159 = None
        mul_155: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_38, convert_element_type_158);  convert_element_type_158 = None
        mul_156: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_155, 1.1111111111111112);  mul_155 = None
        expand_39: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_156, _shape_param_276);  mul_156 = _shape_param_276 = None
        view_195: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_39, _shape_param_277);  expand_39 = _shape_param_277 = None
        expand_40: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_112, _shape_param_278);  permute_112 = _shape_param_278 = None
        clone_38: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_196: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_38, _shape_param_279);  clone_38 = _shape_param_279 = None
        bmm_19: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_195, view_196)
        view_197: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_280);  bmm_19 = _shape_param_280 = None
        permute_114: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_197, [0, 2, 1, 3]);  view_197 = None
        clone_39: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        view_198: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_39, _shape_param_281);  clone_39 = _shape_param_281 = None
        convert_element_type_160: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg86_1, torch.bfloat16);  arg86_1 = None
        permute_115: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_160, [1, 0]);  convert_element_type_160 = None
        view_199: "bf16[4096, 384]" = torch.ops.aten.view.default(view_198, _shape_param_282);  view_198 = _shape_param_282 = None
        mm_63: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_199, permute_115)
        view_200: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_63, _shape_param_283);  mm_63 = _shape_param_283 = None
        inductor_lookup_seed_38: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 38)
        inductor_random_38: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_284, inductor_lookup_seed_38, 'rand');  _shape_param_284 = inductor_lookup_seed_38 = None
        convert_element_type_161: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_38, torch.bfloat16);  inductor_random_38 = None
        gt_39: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_161, 0.1);  convert_element_type_161 = None
        mul_157: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_39, view_200);  view_200 = None
        mul_158: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_157, 1.1111111111111112);  mul_157 = None
        add_70: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_68, mul_158);  mul_158 = None
        pow_28: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_70, 2)
        mean_19: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None
        add_71: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_159: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_70, rsqrt_19)
        mul_160: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg87_1, mul_159);  arg87_1 = mul_159 = None
        convert_element_type_162: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg88_1, torch.bfloat16);  arg88_1 = None
        convert_element_type_163: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_160, torch.bfloat16);  mul_160 = None
        permute_116: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_162, [1, 0]);  convert_element_type_162 = None
        view_201: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_163, _shape_param_285);  convert_element_type_163 = _shape_param_285 = None
        mm_64: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_201, permute_116)
        view_202: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_64, _shape_param_286);  _shape_param_286 = None
        mul_161: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_202, 0.5)
        convert_element_type_164: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_202, torch.float32)
        pow_29: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_164, 3.0);  convert_element_type_164 = None
        mul_162: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
        add_72: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_202, mul_162);  view_202 = mul_162 = None
        mul_163: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_72, 0.7978845608028654);  add_72 = None
        tanh_8: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_163);  mul_163 = None
        add_73: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_164: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_161, add_73);  mul_161 = add_73 = None
        convert_element_type_165: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg89_1, torch.bfloat16);  arg89_1 = None
        permute_117: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_165, [1, 0]);  convert_element_type_165 = None
        mm_65: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_201, permute_117)
        view_203: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_65, _shape_param_287);  _shape_param_287 = None
        mul_165: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_164, view_203);  mul_164 = view_203 = None
        inductor_lookup_seed_39: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 39)
        inductor_random_39: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_288, inductor_lookup_seed_39, 'rand');  _shape_param_288 = inductor_lookup_seed_39 = None
        gt_40: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_39, 0.1);  inductor_random_39 = None
        mul_166: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_40, mul_165);  mul_165 = None
        mul_167: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_166, 1.1111111111111112);  mul_166 = None
        convert_element_type_166: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg90_1, torch.bfloat16);  arg90_1 = None
        convert_element_type_167: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_167, torch.bfloat16);  mul_167 = None
        permute_118: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_166, [1, 0]);  convert_element_type_166 = None
        view_204: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_167, _shape_param_289);  convert_element_type_167 = _shape_param_289 = None
        mm_66: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_204, permute_118)
        view_205: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_66, _shape_param_290);  mm_66 = _shape_param_290 = None
        inductor_lookup_seed_40: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 40)
        inductor_random_40: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_291, inductor_lookup_seed_40, 'rand');  _shape_param_291 = inductor_lookup_seed_40 = None
        convert_element_type_168: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_40, torch.bfloat16);  inductor_random_40 = None
        gt_41: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_168, 0.1);  convert_element_type_168 = None
        mul_168: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_41, view_205);  view_205 = None
        mul_169: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_168, 1.1111111111111112);  mul_168 = None
        add_74: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_70, mul_169);  mul_169 = None
        pow_30: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_74, 2)
        mean_20: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None
        add_75: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_170: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_74, rsqrt_20)
        mul_171: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg91_1, mul_170);  arg91_1 = mul_170 = None
        convert_element_type_169: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg92_1, torch.bfloat16);  arg92_1 = None
        convert_element_type_170: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_171, torch.bfloat16);  mul_171 = None
        permute_119: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_169, [1, 0]);  convert_element_type_169 = None
        view_206: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_170, _shape_param_292);  convert_element_type_170 = _shape_param_292 = None
        mm_67: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_206, permute_119)
        view_207: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_67, _shape_param_293);  mm_67 = _shape_param_293 = None
        view_208: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_207, _shape_param_294);  view_207 = _shape_param_294 = None
        permute_120: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        convert_element_type_171: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg93_1, torch.bfloat16);  arg93_1 = None
        permute_121: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_171, [1, 0]);  convert_element_type_171 = None
        mm_68: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_206, permute_121)
        view_209: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_68, _shape_param_295);  mm_68 = _shape_param_295 = None
        view_210: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_209, _shape_param_296);  view_209 = _shape_param_296 = None
        permute_122: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
        convert_element_type_172: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg94_1, torch.bfloat16);  arg94_1 = None
        permute_123: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_172, [1, 0]);  convert_element_type_172 = None
        mm_69: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_206, permute_123)
        view_211: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_69, _shape_param_297);  mm_69 = _shape_param_297 = None
        view_212: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_211, _shape_param_298);  view_211 = _shape_param_298 = None
        permute_124: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        permute_125: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_122, [0, 1, 3, 2]);  permute_122 = None
        expand_41: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_120, _shape_param_299);  permute_120 = _shape_param_299 = None
        clone_40: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_213: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_300);  clone_40 = _shape_param_300 = None
        expand_42: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_125, _shape_param_301);  permute_125 = _shape_param_301 = None
        clone_41: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_214: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_41, _shape_param_302);  clone_41 = _shape_param_302 = None
        bmm_20: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_213, view_214)
        view_215: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_20, _shape_param_303);  bmm_20 = _shape_param_303 = None
        add_76: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_215, add_66);  view_215 = None
        convert_element_type_173: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_76, torch.bfloat16);  add_76 = None
        convert_element_type_174: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_173, torch.float32)
        amax_10: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_174, [-1], True)
        sub_11: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_174, amax_10);  convert_element_type_174 = None
        exp_10: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_11: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_14: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = None
        convert_element_type_175: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None
        inductor_lookup_seed_41: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 41)
        inductor_random_41: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_304, inductor_lookup_seed_41, 'rand');  _shape_param_304 = inductor_lookup_seed_41 = None
        convert_element_type_176: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_41, torch.bfloat16);  inductor_random_41 = None
        gt_42: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_176, 0.1);  convert_element_type_176 = None
        mul_172: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_42, convert_element_type_175);  convert_element_type_175 = None
        mul_173: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_172, 1.1111111111111112);  mul_172 = None
        expand_43: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_173, _shape_param_305);  mul_173 = _shape_param_305 = None
        view_216: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_43, _shape_param_306);  expand_43 = _shape_param_306 = None
        expand_44: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_124, _shape_param_307);  permute_124 = _shape_param_307 = None
        clone_42: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_217: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_42, _shape_param_308);  clone_42 = _shape_param_308 = None
        bmm_21: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_216, view_217)
        view_218: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_21, _shape_param_309);  bmm_21 = _shape_param_309 = None
        permute_126: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
        clone_43: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_219: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_43, _shape_param_310);  clone_43 = _shape_param_310 = None
        convert_element_type_177: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg95_1, torch.bfloat16);  arg95_1 = None
        permute_127: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_177, [1, 0]);  convert_element_type_177 = None
        view_220: "bf16[4096, 384]" = torch.ops.aten.view.default(view_219, _shape_param_311);  view_219 = _shape_param_311 = None
        mm_70: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_220, permute_127)
        view_221: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_70, _shape_param_312);  mm_70 = _shape_param_312 = None
        inductor_lookup_seed_42: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 42)
        inductor_random_42: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_313, inductor_lookup_seed_42, 'rand');  _shape_param_313 = inductor_lookup_seed_42 = None
        convert_element_type_178: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_42, torch.bfloat16);  inductor_random_42 = None
        gt_43: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_178, 0.1);  convert_element_type_178 = None
        mul_174: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_43, view_221);  view_221 = None
        mul_175: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_174, 1.1111111111111112);  mul_174 = None
        add_77: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_74, mul_175);  mul_175 = None
        pow_31: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_77, 2)
        mean_21: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
        add_78: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_176: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_77, rsqrt_21)
        mul_177: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg96_1, mul_176);  arg96_1 = mul_176 = None
        convert_element_type_179: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg97_1, torch.bfloat16);  arg97_1 = None
        convert_element_type_180: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_177, torch.bfloat16);  mul_177 = None
        permute_128: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_179, [1, 0]);  convert_element_type_179 = None
        view_222: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_180, _shape_param_314);  convert_element_type_180 = _shape_param_314 = None
        mm_71: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_222, permute_128)
        view_223: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_71, _shape_param_315);  mm_71 = _shape_param_315 = None
        view_224: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_223, _shape_param_316);  view_223 = _shape_param_316 = None
        permute_129: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
        convert_element_type_181: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg98_1, torch.bfloat16);  arg98_1 = None
        permute_130: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_181, [1, 0]);  convert_element_type_181 = None
        mm_72: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_187, permute_130)
        view_225: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_72, _shape_param_317);  mm_72 = _shape_param_317 = None
        view_226: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_225, _shape_param_318);  view_225 = _shape_param_318 = None
        permute_131: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        convert_element_type_182: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg99_1, torch.bfloat16);  arg99_1 = None
        permute_132: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_182, [1, 0]);  convert_element_type_182 = None
        mm_73: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_187, permute_132)
        view_227: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_73, _shape_param_319);  mm_73 = _shape_param_319 = None
        view_228: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_227, _shape_param_320);  view_227 = _shape_param_320 = None
        permute_133: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
        permute_134: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2]);  permute_131 = None
        expand_45: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_129, _shape_param_321);  permute_129 = _shape_param_321 = None
        clone_44: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_229: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_44, _shape_param_322);  clone_44 = _shape_param_322 = None
        expand_46: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_134, _shape_param_323);  permute_134 = _shape_param_323 = None
        clone_45: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_230: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_45, _shape_param_324);  clone_45 = _shape_param_324 = None
        bmm_22: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_229, view_230)
        view_231: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_325);  _shape_param_325 = None
        convert_element_type_183: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_231, torch.bfloat16);  view_231 = None
        convert_element_type_184: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_183, torch.float32);  convert_element_type_183 = None
        amax_11: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_184, [-1], True)
        sub_12: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_184, amax_11);  convert_element_type_184 = None
        exp_11: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_12: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_15: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = None
        convert_element_type_185: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None
        inductor_lookup_seed_43: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 43)
        inductor_random_43: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_326, inductor_lookup_seed_43, 'rand');  _shape_param_326 = inductor_lookup_seed_43 = None
        convert_element_type_186: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_43, torch.bfloat16);  inductor_random_43 = None
        gt_44: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_186, 0.1);  convert_element_type_186 = None
        mul_178: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_44, convert_element_type_185);  convert_element_type_185 = None
        mul_179: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_178, 1.1111111111111112);  mul_178 = None
        expand_47: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_179, _shape_param_327);  mul_179 = _shape_param_327 = None
        view_232: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_47, _shape_param_328);  expand_47 = _shape_param_328 = None
        expand_48: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_133, _shape_param_329);  permute_133 = _shape_param_329 = None
        clone_46: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_233: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_46, _shape_param_330);  clone_46 = _shape_param_330 = None
        bmm_23: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_232, view_233)
        view_234: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_331);  bmm_23 = _shape_param_331 = None
        permute_135: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_47: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_235: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_47, _shape_param_332);  clone_47 = _shape_param_332 = None
        convert_element_type_187: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg100_1, torch.bfloat16);  arg100_1 = None
        permute_136: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_187, [1, 0]);  convert_element_type_187 = None
        view_236: "bf16[4096, 384]" = torch.ops.aten.view.default(view_235, _shape_param_333);  view_235 = _shape_param_333 = None
        mm_74: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_236, permute_136)
        view_237: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_74, _shape_param_334);  mm_74 = _shape_param_334 = None
        inductor_lookup_seed_44: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 44)
        inductor_random_44: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_335, inductor_lookup_seed_44, 'rand');  _shape_param_335 = inductor_lookup_seed_44 = None
        convert_element_type_188: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_44, torch.bfloat16);  inductor_random_44 = None
        gt_45: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_188, 0.1);  convert_element_type_188 = None
        mul_180: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_45, view_237);  view_237 = None
        mul_181: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_180, 1.1111111111111112);  mul_180 = None
        add_79: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_77, mul_181);  mul_181 = None
        pow_32: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_79, 2)
        mean_22: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None
        add_80: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_182: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_79, rsqrt_22)
        mul_183: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg101_1, mul_182);  arg101_1 = mul_182 = None
        convert_element_type_189: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg102_1, torch.bfloat16);  arg102_1 = None
        convert_element_type_190: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_183, torch.bfloat16);  mul_183 = None
        permute_137: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_189, [1, 0]);  convert_element_type_189 = None
        view_238: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_190, _shape_param_336);  convert_element_type_190 = _shape_param_336 = None
        mm_75: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_238, permute_137)
        view_239: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_75, _shape_param_337);  _shape_param_337 = None
        mul_184: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        convert_element_type_191: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_239, torch.float32)
        pow_33: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_191, 3.0);  convert_element_type_191 = None
        mul_185: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
        add_81: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_239, mul_185);  view_239 = mul_185 = None
        mul_186: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_81, 0.7978845608028654);  add_81 = None
        tanh_9: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_186);  mul_186 = None
        add_82: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_187: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_184, add_82);  mul_184 = add_82 = None
        convert_element_type_192: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg103_1, torch.bfloat16);  arg103_1 = None
        permute_138: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_192, [1, 0]);  convert_element_type_192 = None
        mm_76: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_238, permute_138)
        view_240: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_76, _shape_param_338);  _shape_param_338 = None
        mul_188: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_187, view_240);  mul_187 = view_240 = None
        inductor_lookup_seed_45: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 45)
        inductor_random_45: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_339, inductor_lookup_seed_45, 'rand');  _shape_param_339 = inductor_lookup_seed_45 = None
        gt_46: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_45, 0.1);  inductor_random_45 = None
        mul_189: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_46, mul_188);  mul_188 = None
        mul_190: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_189, 1.1111111111111112);  mul_189 = None
        convert_element_type_193: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg104_1, torch.bfloat16);  arg104_1 = None
        convert_element_type_194: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_190, torch.bfloat16);  mul_190 = None
        permute_139: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_193, [1, 0]);  convert_element_type_193 = None
        view_241: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_194, _shape_param_340);  convert_element_type_194 = _shape_param_340 = None
        mm_77: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_241, permute_139)
        view_242: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_77, _shape_param_341);  mm_77 = _shape_param_341 = None
        inductor_lookup_seed_46: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 46)
        inductor_random_46: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_342, inductor_lookup_seed_46, 'rand');  _shape_param_342 = inductor_lookup_seed_46 = None
        convert_element_type_195: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_46, torch.bfloat16);  inductor_random_46 = None
        gt_47: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_195, 0.1);  convert_element_type_195 = None
        mul_191: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_47, view_242);  view_242 = None
        mul_192: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_191, 1.1111111111111112);  mul_191 = None
        add_83: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_79, mul_192);  mul_192 = None
        pow_34: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_83, 2)
        mean_23: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None
        add_84: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_193: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_83, rsqrt_23)
        mul_194: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg105_1, mul_193);  arg105_1 = mul_193 = None
        convert_element_type_196: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg106_1, torch.bfloat16);  arg106_1 = None
        convert_element_type_197: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_194, torch.bfloat16);  mul_194 = None
        permute_140: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_196, [1, 0]);  convert_element_type_196 = None
        view_243: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_197, _shape_param_343);  convert_element_type_197 = _shape_param_343 = None
        mm_78: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_243, permute_140)
        view_244: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_78, _shape_param_344);  mm_78 = _shape_param_344 = None
        view_245: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_244, _shape_param_345);  view_244 = _shape_param_345 = None
        permute_141: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None
        convert_element_type_198: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg107_1, torch.bfloat16);  arg107_1 = None
        permute_142: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_198, [1, 0]);  convert_element_type_198 = None
        mm_79: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_243, permute_142)
        view_246: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_79, _shape_param_346);  mm_79 = _shape_param_346 = None
        view_247: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_246, _shape_param_347);  view_246 = _shape_param_347 = None
        permute_143: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
        convert_element_type_199: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg108_1, torch.bfloat16);  arg108_1 = None
        permute_144: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_199, [1, 0]);  convert_element_type_199 = None
        mm_80: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_243, permute_144)
        view_248: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_80, _shape_param_348);  mm_80 = _shape_param_348 = None
        view_249: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_248, _shape_param_349);  view_248 = _shape_param_349 = None
        permute_145: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
        permute_146: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_143, [0, 1, 3, 2]);  permute_143 = None
        expand_49: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_141, _shape_param_350);  permute_141 = _shape_param_350 = None
        clone_48: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_250: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_48, _shape_param_351);  clone_48 = _shape_param_351 = None
        expand_50: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_146, _shape_param_352);  permute_146 = _shape_param_352 = None
        clone_49: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_251: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_49, _shape_param_353);  clone_49 = _shape_param_353 = None
        bmm_24: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_250, view_251)
        view_252: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_24, _shape_param_354);  bmm_24 = _shape_param_354 = None
        add_85: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_252, add_66);  view_252 = None
        convert_element_type_200: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16);  add_85 = None
        convert_element_type_201: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_200, torch.float32)
        amax_12: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_201, [-1], True)
        sub_13: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_201, amax_12);  convert_element_type_201 = None
        exp_12: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_13: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_16: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = None
        convert_element_type_202: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None
        inductor_lookup_seed_47: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 47)
        inductor_random_47: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_355, inductor_lookup_seed_47, 'rand');  _shape_param_355 = inductor_lookup_seed_47 = None
        convert_element_type_203: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_47, torch.bfloat16);  inductor_random_47 = None
        gt_48: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_203, 0.1);  convert_element_type_203 = None
        mul_195: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_48, convert_element_type_202);  convert_element_type_202 = None
        mul_196: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_195, 1.1111111111111112);  mul_195 = None
        expand_51: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_196, _shape_param_356);  mul_196 = _shape_param_356 = None
        view_253: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_51, _shape_param_357);  expand_51 = _shape_param_357 = None
        expand_52: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_145, _shape_param_358);  permute_145 = _shape_param_358 = None
        clone_50: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_254: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_50, _shape_param_359);  clone_50 = _shape_param_359 = None
        bmm_25: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_253, view_254)
        view_255: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_25, _shape_param_360);  bmm_25 = _shape_param_360 = None
        permute_147: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
        clone_51: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
        view_256: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_51, _shape_param_361);  clone_51 = _shape_param_361 = None
        convert_element_type_204: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg109_1, torch.bfloat16);  arg109_1 = None
        permute_148: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_204, [1, 0]);  convert_element_type_204 = None
        view_257: "bf16[4096, 384]" = torch.ops.aten.view.default(view_256, _shape_param_362);  view_256 = _shape_param_362 = None
        mm_81: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_257, permute_148)
        view_258: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_81, _shape_param_363);  mm_81 = _shape_param_363 = None
        inductor_lookup_seed_48: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 48)
        inductor_random_48: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_364, inductor_lookup_seed_48, 'rand');  _shape_param_364 = inductor_lookup_seed_48 = None
        convert_element_type_205: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_48, torch.bfloat16);  inductor_random_48 = None
        gt_49: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_205, 0.1);  convert_element_type_205 = None
        mul_197: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_49, view_258);  view_258 = None
        mul_198: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_197, 1.1111111111111112);  mul_197 = None
        add_86: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_83, mul_198);  mul_198 = None
        pow_35: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_86, 2)
        mean_24: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None
        add_87: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        mul_199: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_86, rsqrt_24)
        mul_200: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg110_1, mul_199);  arg110_1 = mul_199 = None
        convert_element_type_206: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg111_1, torch.bfloat16);  arg111_1 = None
        convert_element_type_207: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_200, torch.bfloat16);  mul_200 = None
        permute_149: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_206, [1, 0]);  convert_element_type_206 = None
        view_259: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_207, _shape_param_365);  convert_element_type_207 = _shape_param_365 = None
        mm_82: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_259, permute_149)
        view_260: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_82, _shape_param_366);  mm_82 = _shape_param_366 = None
        view_261: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_260, _shape_param_367);  view_260 = _shape_param_367 = None
        permute_150: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
        convert_element_type_208: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg112_1, torch.bfloat16);  arg112_1 = None
        permute_151: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_208, [1, 0]);  convert_element_type_208 = None
        mm_83: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_187, permute_151)
        view_262: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_83, _shape_param_368);  mm_83 = _shape_param_368 = None
        view_263: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_262, _shape_param_369);  view_262 = _shape_param_369 = None
        permute_152: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_263, [0, 2, 1, 3]);  view_263 = None
        convert_element_type_209: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg113_1, torch.bfloat16);  arg113_1 = None
        permute_153: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_209, [1, 0]);  convert_element_type_209 = None
        mm_84: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_187, permute_153)
        view_264: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_84, _shape_param_370);  mm_84 = _shape_param_370 = None
        view_265: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_264, _shape_param_371);  view_264 = _shape_param_371 = None
        permute_154: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
        permute_155: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_152, [0, 1, 3, 2]);  permute_152 = None
        expand_53: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_150, _shape_param_372);  permute_150 = _shape_param_372 = None
        clone_52: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_266: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_52, _shape_param_373);  clone_52 = _shape_param_373 = None
        expand_54: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_155, _shape_param_374);  permute_155 = _shape_param_374 = None
        clone_53: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_267: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_53, _shape_param_375);  clone_53 = _shape_param_375 = None
        bmm_26: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_266, view_267)
        view_268: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_26, _shape_param_376);  _shape_param_376 = None
        convert_element_type_210: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_268, torch.bfloat16);  view_268 = None
        convert_element_type_211: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_210, torch.float32);  convert_element_type_210 = None
        amax_13: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_211, [-1], True)
        sub_14: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_211, amax_13);  convert_element_type_211 = None
        exp_13: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_14: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_17: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = None
        convert_element_type_212: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None
        inductor_lookup_seed_49: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 49)
        inductor_random_49: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_377, inductor_lookup_seed_49, 'rand');  _shape_param_377 = inductor_lookup_seed_49 = None
        convert_element_type_213: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_49, torch.bfloat16);  inductor_random_49 = None
        gt_50: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_213, 0.1);  convert_element_type_213 = None
        mul_201: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_50, convert_element_type_212);  convert_element_type_212 = None
        mul_202: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_201, 1.1111111111111112);  mul_201 = None
        expand_55: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_202, _shape_param_378);  mul_202 = _shape_param_378 = None
        view_269: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_55, _shape_param_379);  expand_55 = _shape_param_379 = None
        expand_56: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_154, _shape_param_380);  permute_154 = _shape_param_380 = None
        clone_54: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_270: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_54, _shape_param_381);  clone_54 = _shape_param_381 = None
        bmm_27: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_269, view_270)
        view_271: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_27, _shape_param_382);  bmm_27 = _shape_param_382 = None
        permute_156: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_271, [0, 2, 1, 3]);  view_271 = None
        clone_55: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
        view_272: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_55, _shape_param_383);  clone_55 = _shape_param_383 = None
        convert_element_type_214: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg114_1, torch.bfloat16);  arg114_1 = None
        permute_157: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_214, [1, 0]);  convert_element_type_214 = None
        view_273: "bf16[4096, 384]" = torch.ops.aten.view.default(view_272, _shape_param_384);  view_272 = _shape_param_384 = None
        mm_85: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_273, permute_157)
        view_274: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_85, _shape_param_385);  mm_85 = _shape_param_385 = None
        inductor_lookup_seed_50: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 50)
        inductor_random_50: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_386, inductor_lookup_seed_50, 'rand');  _shape_param_386 = inductor_lookup_seed_50 = None
        convert_element_type_215: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_50, torch.bfloat16);  inductor_random_50 = None
        gt_51: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_215, 0.1);  convert_element_type_215 = None
        mul_203: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_51, view_274);  view_274 = None
        mul_204: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_203, 1.1111111111111112);  mul_203 = None
        add_88: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_86, mul_204);  mul_204 = None
        pow_36: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_88, 2)
        mean_25: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None
        add_89: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_205: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_88, rsqrt_25)
        mul_206: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg115_1, mul_205);  arg115_1 = mul_205 = None
        convert_element_type_216: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg116_1, torch.bfloat16);  arg116_1 = None
        convert_element_type_217: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_206, torch.bfloat16);  mul_206 = None
        permute_158: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_216, [1, 0]);  convert_element_type_216 = None
        view_275: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_217, _shape_param_387);  convert_element_type_217 = _shape_param_387 = None
        mm_86: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_275, permute_158)
        view_276: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_86, _shape_param_388);  _shape_param_388 = None
        mul_207: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_276, 0.5)
        convert_element_type_218: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_276, torch.float32)
        pow_37: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_218, 3.0);  convert_element_type_218 = None
        mul_208: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_37, 0.044715);  pow_37 = None
        add_90: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_276, mul_208);  view_276 = mul_208 = None
        mul_209: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_90, 0.7978845608028654);  add_90 = None
        tanh_10: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_209);  mul_209 = None
        add_91: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_210: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_207, add_91);  mul_207 = add_91 = None
        convert_element_type_219: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg117_1, torch.bfloat16);  arg117_1 = None
        permute_159: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_219, [1, 0]);  convert_element_type_219 = None
        mm_87: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_275, permute_159)
        view_277: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_87, _shape_param_389);  _shape_param_389 = None
        mul_211: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_210, view_277);  mul_210 = view_277 = None
        inductor_lookup_seed_51: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 51)
        inductor_random_51: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_390, inductor_lookup_seed_51, 'rand');  _shape_param_390 = inductor_lookup_seed_51 = None
        gt_52: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_51, 0.1);  inductor_random_51 = None
        mul_212: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_52, mul_211);  mul_211 = None
        mul_213: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_212, 1.1111111111111112);  mul_212 = None
        convert_element_type_220: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg118_1, torch.bfloat16);  arg118_1 = None
        convert_element_type_221: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_213, torch.bfloat16);  mul_213 = None
        permute_160: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_220, [1, 0]);  convert_element_type_220 = None
        view_278: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_221, _shape_param_391);  convert_element_type_221 = _shape_param_391 = None
        mm_88: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_278, permute_160)
        view_279: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_88, _shape_param_392);  mm_88 = _shape_param_392 = None
        inductor_lookup_seed_52: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 52)
        inductor_random_52: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_393, inductor_lookup_seed_52, 'rand');  _shape_param_393 = inductor_lookup_seed_52 = None
        convert_element_type_222: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_52, torch.bfloat16);  inductor_random_52 = None
        gt_53: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_222, 0.1);  convert_element_type_222 = None
        mul_214: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_53, view_279);  view_279 = None
        mul_215: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_214, 1.1111111111111112);  mul_214 = None
        add_92: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_88, mul_215);  mul_215 = None
        pow_38: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_92, 2)
        mean_26: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None
        add_93: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_216: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_92, rsqrt_26)
        mul_217: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg119_1, mul_216);  arg119_1 = mul_216 = None
        convert_element_type_223: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg120_1, torch.bfloat16);  arg120_1 = None
        convert_element_type_224: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_217, torch.bfloat16);  mul_217 = None
        permute_161: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_223, [1, 0]);  convert_element_type_223 = None
        view_280: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_224, _shape_param_394);  convert_element_type_224 = _shape_param_394 = None
        mm_89: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_280, permute_161)
        view_281: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_89, _shape_param_395);  mm_89 = _shape_param_395 = None
        view_282: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_281, _shape_param_396);  view_281 = _shape_param_396 = None
        permute_162: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
        convert_element_type_225: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg121_1, torch.bfloat16);  arg121_1 = None
        permute_163: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_225, [1, 0]);  convert_element_type_225 = None
        mm_90: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_280, permute_163)
        view_283: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_90, _shape_param_397);  mm_90 = _shape_param_397 = None
        view_284: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_283, _shape_param_398);  view_283 = _shape_param_398 = None
        permute_164: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None
        convert_element_type_226: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg122_1, torch.bfloat16);  arg122_1 = None
        permute_165: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_226, [1, 0]);  convert_element_type_226 = None
        mm_91: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_280, permute_165)
        view_285: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_91, _shape_param_399);  mm_91 = _shape_param_399 = None
        view_286: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_285, _shape_param_400);  view_285 = _shape_param_400 = None
        permute_166: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
        permute_167: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_164, [0, 1, 3, 2]);  permute_164 = None
        expand_57: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_162, _shape_param_401);  permute_162 = _shape_param_401 = None
        clone_56: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_287: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_56, _shape_param_402);  clone_56 = _shape_param_402 = None
        expand_58: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_167, _shape_param_403);  permute_167 = _shape_param_403 = None
        clone_57: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_288: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_57, _shape_param_404);  clone_57 = _shape_param_404 = None
        bmm_28: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_287, view_288)
        view_289: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_28, _shape_param_405);  bmm_28 = _shape_param_405 = None
        add_94: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_289, add_66);  view_289 = None
        convert_element_type_227: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16);  add_94 = None
        convert_element_type_228: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_227, torch.float32)
        amax_14: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_228, [-1], True)
        sub_15: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_228, amax_14);  convert_element_type_228 = None
        exp_14: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_15: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_18: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = None
        convert_element_type_229: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None
        inductor_lookup_seed_53: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 53)
        inductor_random_53: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_406, inductor_lookup_seed_53, 'rand');  _shape_param_406 = inductor_lookup_seed_53 = None
        convert_element_type_230: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_53, torch.bfloat16);  inductor_random_53 = None
        gt_54: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_230, 0.1);  convert_element_type_230 = None
        mul_218: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_54, convert_element_type_229);  convert_element_type_229 = None
        mul_219: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_218, 1.1111111111111112);  mul_218 = None
        expand_59: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_219, _shape_param_407);  mul_219 = _shape_param_407 = None
        view_290: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_59, _shape_param_408);  expand_59 = _shape_param_408 = None
        expand_60: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_166, _shape_param_409);  permute_166 = _shape_param_409 = None
        clone_58: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_291: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_58, _shape_param_410);  clone_58 = _shape_param_410 = None
        bmm_29: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_290, view_291)
        view_292: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_29, _shape_param_411);  bmm_29 = _shape_param_411 = None
        permute_168: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
        clone_59: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
        view_293: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_59, _shape_param_412);  clone_59 = _shape_param_412 = None
        convert_element_type_231: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg123_1, torch.bfloat16);  arg123_1 = None
        permute_169: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_231, [1, 0]);  convert_element_type_231 = None
        view_294: "bf16[4096, 384]" = torch.ops.aten.view.default(view_293, _shape_param_413);  view_293 = _shape_param_413 = None
        mm_92: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_294, permute_169)
        view_295: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_92, _shape_param_414);  mm_92 = _shape_param_414 = None
        inductor_lookup_seed_54: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 54)
        inductor_random_54: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_415, inductor_lookup_seed_54, 'rand');  _shape_param_415 = inductor_lookup_seed_54 = None
        convert_element_type_232: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_54, torch.bfloat16);  inductor_random_54 = None
        gt_55: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_232, 0.1);  convert_element_type_232 = None
        mul_220: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_55, view_295);  view_295 = None
        mul_221: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_220, 1.1111111111111112);  mul_220 = None
        add_95: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_92, mul_221);  mul_221 = None
        pow_39: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_95, 2)
        mean_27: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None
        add_96: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_222: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_95, rsqrt_27)
        mul_223: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg124_1, mul_222);  arg124_1 = mul_222 = None
        convert_element_type_233: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg125_1, torch.bfloat16);  arg125_1 = None
        convert_element_type_234: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_223, torch.bfloat16);  mul_223 = None
        permute_170: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_233, [1, 0]);  convert_element_type_233 = None
        view_296: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_234, _shape_param_416);  convert_element_type_234 = _shape_param_416 = None
        mm_93: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_296, permute_170)
        view_297: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_93, _shape_param_417);  mm_93 = _shape_param_417 = None
        view_298: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_297, _shape_param_418);  view_297 = _shape_param_418 = None
        permute_171: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_298, [0, 2, 1, 3]);  view_298 = None
        convert_element_type_235: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg126_1, torch.bfloat16);  arg126_1 = None
        permute_172: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_235, [1, 0]);  convert_element_type_235 = None
        mm_94: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_187, permute_172)
        view_299: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_94, _shape_param_419);  mm_94 = _shape_param_419 = None
        view_300: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_299, _shape_param_420);  view_299 = _shape_param_420 = None
        permute_173: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
        convert_element_type_236: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg127_1, torch.bfloat16);  arg127_1 = None
        permute_174: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_236, [1, 0]);  convert_element_type_236 = None
        mm_95: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_187, permute_174)
        view_301: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_95, _shape_param_421);  mm_95 = _shape_param_421 = None
        view_302: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_301, _shape_param_422);  view_301 = _shape_param_422 = None
        permute_175: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
        permute_176: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_173, [0, 1, 3, 2]);  permute_173 = None
        expand_61: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_171, _shape_param_423);  permute_171 = _shape_param_423 = None
        clone_60: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_303: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_60, _shape_param_424);  clone_60 = _shape_param_424 = None
        expand_62: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_176, _shape_param_425);  permute_176 = _shape_param_425 = None
        clone_61: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_304: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_61, _shape_param_426);  clone_61 = _shape_param_426 = None
        bmm_30: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_303, view_304)
        view_305: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_30, _shape_param_427);  _shape_param_427 = None
        convert_element_type_237: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_305, torch.bfloat16);  view_305 = None
        convert_element_type_238: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_237, torch.float32);  convert_element_type_237 = None
        amax_15: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_238, [-1], True)
        sub_16: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_238, amax_15);  convert_element_type_238 = None
        exp_15: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_16: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_19: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = None
        convert_element_type_239: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None
        inductor_lookup_seed_55: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 55)
        inductor_random_55: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_428, inductor_lookup_seed_55, 'rand');  _shape_param_428 = inductor_lookup_seed_55 = None
        convert_element_type_240: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_55, torch.bfloat16);  inductor_random_55 = None
        gt_56: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_240, 0.1);  convert_element_type_240 = None
        mul_224: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_56, convert_element_type_239);  convert_element_type_239 = None
        mul_225: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_224, 1.1111111111111112);  mul_224 = None
        expand_63: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_225, _shape_param_429);  mul_225 = _shape_param_429 = None
        view_306: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_63, _shape_param_430);  expand_63 = _shape_param_430 = None
        expand_64: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_175, _shape_param_431);  permute_175 = _shape_param_431 = None
        clone_62: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_307: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_62, _shape_param_432);  clone_62 = _shape_param_432 = None
        bmm_31: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_306, view_307)
        view_308: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_31, _shape_param_433);  bmm_31 = _shape_param_433 = None
        permute_177: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
        clone_63: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_309: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_63, _shape_param_434);  clone_63 = _shape_param_434 = None
        convert_element_type_241: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg128_1, torch.bfloat16);  arg128_1 = None
        permute_178: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_241, [1, 0]);  convert_element_type_241 = None
        view_310: "bf16[4096, 384]" = torch.ops.aten.view.default(view_309, _shape_param_435);  view_309 = _shape_param_435 = None
        mm_96: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_310, permute_178)
        view_311: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_96, _shape_param_436);  mm_96 = _shape_param_436 = None
        inductor_lookup_seed_56: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 56)
        inductor_random_56: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_437, inductor_lookup_seed_56, 'rand');  _shape_param_437 = inductor_lookup_seed_56 = None
        convert_element_type_242: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_56, torch.bfloat16);  inductor_random_56 = None
        gt_57: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_242, 0.1);  convert_element_type_242 = None
        mul_226: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_57, view_311);  view_311 = None
        mul_227: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_226, 1.1111111111111112);  mul_226 = None
        add_97: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_95, mul_227);  mul_227 = None
        pow_40: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_97, 2)
        mean_28: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None
        add_98: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_228: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_97, rsqrt_28)
        mul_229: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg129_1, mul_228);  arg129_1 = mul_228 = None
        convert_element_type_243: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg130_1, torch.bfloat16);  arg130_1 = None
        convert_element_type_244: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_229, torch.bfloat16);  mul_229 = None
        permute_179: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_243, [1, 0]);  convert_element_type_243 = None
        view_312: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_244, _shape_param_438);  convert_element_type_244 = _shape_param_438 = None
        mm_97: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_312, permute_179)
        view_313: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_97, _shape_param_439);  _shape_param_439 = None
        mul_230: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_313, 0.5)
        convert_element_type_245: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_313, torch.float32)
        pow_41: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_245, 3.0);  convert_element_type_245 = None
        mul_231: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_41, 0.044715);  pow_41 = None
        add_99: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_313, mul_231);  view_313 = mul_231 = None
        mul_232: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_99, 0.7978845608028654);  add_99 = None
        tanh_11: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_232);  mul_232 = None
        add_100: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_233: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_230, add_100);  mul_230 = add_100 = None
        convert_element_type_246: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg131_1, torch.bfloat16);  arg131_1 = None
        permute_180: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_246, [1, 0]);  convert_element_type_246 = None
        mm_98: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_312, permute_180)
        view_314: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_98, _shape_param_440);  _shape_param_440 = None
        mul_234: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_233, view_314);  mul_233 = view_314 = None
        inductor_lookup_seed_57: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 57)
        inductor_random_57: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_441, inductor_lookup_seed_57, 'rand');  _shape_param_441 = inductor_lookup_seed_57 = None
        gt_58: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_57, 0.1);  inductor_random_57 = None
        mul_235: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_58, mul_234);  mul_234 = None
        mul_236: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_235, 1.1111111111111112);  mul_235 = None
        convert_element_type_247: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg132_1, torch.bfloat16);  arg132_1 = None
        convert_element_type_248: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_236, torch.bfloat16);  mul_236 = None
        permute_181: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_247, [1, 0]);  convert_element_type_247 = None
        view_315: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_248, _shape_param_442);  convert_element_type_248 = _shape_param_442 = None
        mm_99: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_315, permute_181)
        view_316: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_99, _shape_param_443);  mm_99 = _shape_param_443 = None
        inductor_lookup_seed_58: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 58)
        inductor_random_58: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_444, inductor_lookup_seed_58, 'rand');  _shape_param_444 = inductor_lookup_seed_58 = None
        convert_element_type_249: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_58, torch.bfloat16);  inductor_random_58 = None
        gt_59: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_249, 0.1);  convert_element_type_249 = None
        mul_237: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_59, view_316);  view_316 = None
        mul_238: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_237, 1.1111111111111112);  mul_237 = None
        add_101: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_97, mul_238);  mul_238 = None
        pow_42: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_101, 2)
        mean_29: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None
        add_102: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_239: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_101, rsqrt_29)
        mul_240: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg133_1, mul_239);  arg133_1 = mul_239 = None
        convert_element_type_250: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg134_1, torch.bfloat16);  arg134_1 = None
        convert_element_type_251: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_240, torch.bfloat16);  mul_240 = None
        permute_182: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_250, [1, 0]);  convert_element_type_250 = None
        view_317: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_251, _shape_param_445);  convert_element_type_251 = _shape_param_445 = None
        mm_100: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_317, permute_182)
        view_318: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_100, _shape_param_446);  mm_100 = _shape_param_446 = None
        view_319: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_318, _shape_param_447);  view_318 = _shape_param_447 = None
        permute_183: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None
        convert_element_type_252: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg135_1, torch.bfloat16);  arg135_1 = None
        permute_184: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_252, [1, 0]);  convert_element_type_252 = None
        mm_101: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_317, permute_184)
        view_320: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_101, _shape_param_448);  mm_101 = _shape_param_448 = None
        view_321: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_320, _shape_param_449);  view_320 = _shape_param_449 = None
        permute_185: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_321, [0, 2, 1, 3]);  view_321 = None
        convert_element_type_253: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg136_1, torch.bfloat16);  arg136_1 = None
        permute_186: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_253, [1, 0]);  convert_element_type_253 = None
        mm_102: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_317, permute_186)
        view_322: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_102, _shape_param_450);  mm_102 = _shape_param_450 = None
        view_323: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_322, _shape_param_451);  view_322 = _shape_param_451 = None
        permute_187: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None
        permute_188: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_185, [0, 1, 3, 2]);  permute_185 = None
        expand_65: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_183, _shape_param_452);  permute_183 = _shape_param_452 = None
        clone_64: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_324: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_64, _shape_param_453);  clone_64 = _shape_param_453 = None
        expand_66: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_188, _shape_param_454);  permute_188 = _shape_param_454 = None
        clone_65: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_325: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_65, _shape_param_455);  clone_65 = _shape_param_455 = None
        bmm_32: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_324, view_325)
        view_326: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_32, _shape_param_456);  bmm_32 = _shape_param_456 = None
        add_103: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_326, add_66);  view_326 = None
        convert_element_type_254: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16);  add_103 = None
        convert_element_type_255: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_254, torch.float32)
        amax_16: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_255, [-1], True)
        sub_17: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_255, amax_16);  convert_element_type_255 = None
        exp_16: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_17: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_20: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = None
        convert_element_type_256: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None
        inductor_lookup_seed_59: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 59)
        inductor_random_59: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_457, inductor_lookup_seed_59, 'rand');  _shape_param_457 = inductor_lookup_seed_59 = None
        convert_element_type_257: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_59, torch.bfloat16);  inductor_random_59 = None
        gt_60: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_257, 0.1);  convert_element_type_257 = None
        mul_241: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_60, convert_element_type_256);  convert_element_type_256 = None
        mul_242: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_241, 1.1111111111111112);  mul_241 = None
        expand_67: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_242, _shape_param_458);  mul_242 = _shape_param_458 = None
        view_327: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_67, _shape_param_459);  expand_67 = _shape_param_459 = None
        expand_68: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_187, _shape_param_460);  permute_187 = _shape_param_460 = None
        clone_66: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_328: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_66, _shape_param_461);  clone_66 = _shape_param_461 = None
        bmm_33: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_327, view_328)
        view_329: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_33, _shape_param_462);  bmm_33 = _shape_param_462 = None
        permute_189: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_329, [0, 2, 1, 3]);  view_329 = None
        clone_67: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_330: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_67, _shape_param_463);  clone_67 = _shape_param_463 = None
        convert_element_type_258: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg137_1, torch.bfloat16);  arg137_1 = None
        permute_190: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_258, [1, 0]);  convert_element_type_258 = None
        view_331: "bf16[4096, 384]" = torch.ops.aten.view.default(view_330, _shape_param_464);  view_330 = _shape_param_464 = None
        mm_103: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_331, permute_190)
        view_332: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_103, _shape_param_465);  mm_103 = _shape_param_465 = None
        inductor_lookup_seed_60: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 60)
        inductor_random_60: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_466, inductor_lookup_seed_60, 'rand');  _shape_param_466 = inductor_lookup_seed_60 = None
        convert_element_type_259: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_60, torch.bfloat16);  inductor_random_60 = None
        gt_61: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_259, 0.1);  convert_element_type_259 = None
        mul_243: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_61, view_332);  view_332 = None
        mul_244: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_243, 1.1111111111111112);  mul_243 = None
        add_104: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_101, mul_244);  mul_244 = None
        pow_43: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_104, 2)
        mean_30: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None
        add_105: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        mul_245: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_104, rsqrt_30)
        mul_246: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg138_1, mul_245);  arg138_1 = mul_245 = None
        convert_element_type_260: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg139_1, torch.bfloat16);  arg139_1 = None
        convert_element_type_261: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_246, torch.bfloat16);  mul_246 = None
        permute_191: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_260, [1, 0]);  convert_element_type_260 = None
        view_333: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_261, _shape_param_467);  convert_element_type_261 = _shape_param_467 = None
        mm_104: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_333, permute_191)
        view_334: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_104, _shape_param_468);  mm_104 = _shape_param_468 = None
        view_335: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_334, _shape_param_469);  view_334 = _shape_param_469 = None
        permute_192: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_335, [0, 2, 1, 3]);  view_335 = None
        convert_element_type_262: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg140_1, torch.bfloat16);  arg140_1 = None
        permute_193: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_262, [1, 0]);  convert_element_type_262 = None
        mm_105: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_187, permute_193)
        view_336: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_105, _shape_param_470);  mm_105 = _shape_param_470 = None
        view_337: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_336, _shape_param_471);  view_336 = _shape_param_471 = None
        permute_194: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None
        convert_element_type_263: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg141_1, torch.bfloat16);  arg141_1 = None
        permute_195: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_263, [1, 0]);  convert_element_type_263 = None
        mm_106: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_187, permute_195)
        view_338: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_106, _shape_param_472);  mm_106 = _shape_param_472 = None
        view_339: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_338, _shape_param_473);  view_338 = _shape_param_473 = None
        permute_196: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
        permute_197: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_194, [0, 1, 3, 2]);  permute_194 = None
        expand_69: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_192, _shape_param_474);  permute_192 = _shape_param_474 = None
        clone_68: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_340: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_68, _shape_param_475);  clone_68 = _shape_param_475 = None
        expand_70: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_197, _shape_param_476);  permute_197 = _shape_param_476 = None
        clone_69: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_341: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_69, _shape_param_477);  clone_69 = _shape_param_477 = None
        bmm_34: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_340, view_341)
        view_342: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_34, _shape_param_478);  _shape_param_478 = None
        convert_element_type_264: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_342, torch.bfloat16);  view_342 = None
        convert_element_type_265: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_264, torch.float32);  convert_element_type_264 = None
        amax_17: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_265, [-1], True)
        sub_18: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_265, amax_17);  convert_element_type_265 = None
        exp_17: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_18: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_21: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = None
        convert_element_type_266: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None
        inductor_lookup_seed_61: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 61)
        inductor_random_61: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_479, inductor_lookup_seed_61, 'rand');  _shape_param_479 = inductor_lookup_seed_61 = None
        convert_element_type_267: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_61, torch.bfloat16);  inductor_random_61 = None
        gt_62: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_267, 0.1);  convert_element_type_267 = None
        mul_247: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_62, convert_element_type_266);  convert_element_type_266 = None
        mul_248: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_247, 1.1111111111111112);  mul_247 = None
        expand_71: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_248, _shape_param_480);  mul_248 = _shape_param_480 = None
        view_343: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_71, _shape_param_481);  expand_71 = _shape_param_481 = None
        expand_72: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_196, _shape_param_482);  permute_196 = _shape_param_482 = None
        clone_70: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_344: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_70, _shape_param_483);  clone_70 = _shape_param_483 = None
        bmm_35: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_343, view_344)
        view_345: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_35, _shape_param_484);  bmm_35 = _shape_param_484 = None
        permute_198: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
        clone_71: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
        view_346: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_71, _shape_param_485);  clone_71 = _shape_param_485 = None
        convert_element_type_268: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg142_1, torch.bfloat16);  arg142_1 = None
        permute_199: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_268, [1, 0]);  convert_element_type_268 = None
        view_347: "bf16[4096, 384]" = torch.ops.aten.view.default(view_346, _shape_param_486);  view_346 = _shape_param_486 = None
        mm_107: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_347, permute_199)
        view_348: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_107, _shape_param_487);  mm_107 = _shape_param_487 = None
        inductor_lookup_seed_62: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 62)
        inductor_random_62: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_488, inductor_lookup_seed_62, 'rand');  _shape_param_488 = inductor_lookup_seed_62 = None
        convert_element_type_269: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_62, torch.bfloat16);  inductor_random_62 = None
        gt_63: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_269, 0.1);  convert_element_type_269 = None
        mul_249: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_63, view_348);  view_348 = None
        mul_250: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_249, 1.1111111111111112);  mul_249 = None
        add_106: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_104, mul_250);  mul_250 = None
        pow_44: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_106, 2)
        mean_31: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None
        add_107: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        mul_251: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_106, rsqrt_31)
        mul_252: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg143_1, mul_251);  arg143_1 = mul_251 = None
        convert_element_type_270: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg144_1, torch.bfloat16);  arg144_1 = None
        convert_element_type_271: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_252, torch.bfloat16);  mul_252 = None
        permute_200: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_270, [1, 0]);  convert_element_type_270 = None
        view_349: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_271, _shape_param_489);  convert_element_type_271 = _shape_param_489 = None
        mm_108: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_349, permute_200)
        view_350: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_108, _shape_param_490);  _shape_param_490 = None
        mul_253: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_350, 0.5)
        convert_element_type_272: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_350, torch.float32)
        pow_45: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_272, 3.0);  convert_element_type_272 = None
        mul_254: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_45, 0.044715);  pow_45 = None
        add_108: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_350, mul_254);  view_350 = mul_254 = None
        mul_255: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_108, 0.7978845608028654);  add_108 = None
        tanh_12: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_255);  mul_255 = None
        add_109: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_256: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_253, add_109);  mul_253 = add_109 = None
        convert_element_type_273: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg145_1, torch.bfloat16);  arg145_1 = None
        permute_201: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_273, [1, 0]);  convert_element_type_273 = None
        mm_109: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_349, permute_201)
        view_351: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_109, _shape_param_491);  _shape_param_491 = None
        mul_257: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_256, view_351);  mul_256 = view_351 = None
        inductor_lookup_seed_63: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 63)
        inductor_random_63: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_492, inductor_lookup_seed_63, 'rand');  _shape_param_492 = inductor_lookup_seed_63 = None
        gt_64: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_63, 0.1);  inductor_random_63 = None
        mul_258: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_64, mul_257);  mul_257 = None
        mul_259: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_258, 1.1111111111111112);  mul_258 = None
        convert_element_type_274: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg146_1, torch.bfloat16);  arg146_1 = None
        convert_element_type_275: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_259, torch.bfloat16);  mul_259 = None
        permute_202: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_274, [1, 0]);  convert_element_type_274 = None
        view_352: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_275, _shape_param_493);  convert_element_type_275 = _shape_param_493 = None
        mm_110: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_352, permute_202)
        view_353: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_110, _shape_param_494);  mm_110 = _shape_param_494 = None
        inductor_lookup_seed_64: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 64)
        inductor_random_64: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_495, inductor_lookup_seed_64, 'rand');  _shape_param_495 = inductor_lookup_seed_64 = None
        convert_element_type_276: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_64, torch.bfloat16);  inductor_random_64 = None
        gt_65: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_276, 0.1);  convert_element_type_276 = None
        mul_260: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_65, view_353);  view_353 = None
        mul_261: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_260, 1.1111111111111112);  mul_260 = None
        add_110: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_106, mul_261);  mul_261 = None
        pow_46: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_110, 2)
        mean_32: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None
        add_111: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None
        rsqrt_32: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        mul_262: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_110, rsqrt_32)
        mul_263: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg147_1, mul_262);  arg147_1 = mul_262 = None
        convert_element_type_277: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg148_1, torch.bfloat16);  arg148_1 = None
        convert_element_type_278: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_263, torch.bfloat16);  mul_263 = None
        permute_203: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_277, [1, 0]);  convert_element_type_277 = None
        view_354: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_278, _shape_param_496);  convert_element_type_278 = _shape_param_496 = None
        mm_111: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_354, permute_203)
        view_355: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_111, _shape_param_497);  mm_111 = _shape_param_497 = None
        view_356: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_355, _shape_param_498);  view_355 = _shape_param_498 = None
        permute_204: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None
        convert_element_type_279: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg149_1, torch.bfloat16);  arg149_1 = None
        permute_205: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_279, [1, 0]);  convert_element_type_279 = None
        mm_112: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_354, permute_205)
        view_357: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_112, _shape_param_499);  mm_112 = _shape_param_499 = None
        view_358: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_357, _shape_param_500);  view_357 = _shape_param_500 = None
        permute_206: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
        convert_element_type_280: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg150_1, torch.bfloat16);  arg150_1 = None
        permute_207: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_280, [1, 0]);  convert_element_type_280 = None
        mm_113: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_354, permute_207)
        view_359: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_113, _shape_param_501);  mm_113 = _shape_param_501 = None
        view_360: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_359, _shape_param_502);  view_359 = _shape_param_502 = None
        permute_208: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
        permute_209: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_206, [0, 1, 3, 2]);  permute_206 = None
        expand_73: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_204, _shape_param_503);  permute_204 = _shape_param_503 = None
        clone_72: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_361: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_72, _shape_param_504);  clone_72 = _shape_param_504 = None
        expand_74: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_209, _shape_param_505);  permute_209 = _shape_param_505 = None
        clone_73: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_362: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_73, _shape_param_506);  clone_73 = _shape_param_506 = None
        bmm_36: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_361, view_362)
        view_363: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_36, _shape_param_507);  bmm_36 = _shape_param_507 = None
        add_112: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_363, add_66);  view_363 = None
        convert_element_type_281: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16);  add_112 = None
        convert_element_type_282: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_281, torch.float32)
        amax_18: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_282, [-1], True)
        sub_19: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_282, amax_18);  convert_element_type_282 = None
        exp_18: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_19: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_22: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = None
        convert_element_type_283: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None
        inductor_lookup_seed_65: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 65)
        inductor_random_65: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_508, inductor_lookup_seed_65, 'rand');  _shape_param_508 = inductor_lookup_seed_65 = None
        convert_element_type_284: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_65, torch.bfloat16);  inductor_random_65 = None
        gt_66: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_284, 0.1);  convert_element_type_284 = None
        mul_264: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_66, convert_element_type_283);  convert_element_type_283 = None
        mul_265: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_264, 1.1111111111111112);  mul_264 = None
        expand_75: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_265, _shape_param_509);  mul_265 = _shape_param_509 = None
        view_364: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_75, _shape_param_510);  expand_75 = _shape_param_510 = None
        expand_76: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_208, _shape_param_511);  permute_208 = _shape_param_511 = None
        clone_74: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_365: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_74, _shape_param_512);  clone_74 = _shape_param_512 = None
        bmm_37: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_364, view_365)
        view_366: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_37, _shape_param_513);  bmm_37 = _shape_param_513 = None
        permute_210: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
        clone_75: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_367: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_75, _shape_param_514);  clone_75 = _shape_param_514 = None
        convert_element_type_285: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg151_1, torch.bfloat16);  arg151_1 = None
        permute_211: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_285, [1, 0]);  convert_element_type_285 = None
        view_368: "bf16[4096, 384]" = torch.ops.aten.view.default(view_367, _shape_param_515);  view_367 = _shape_param_515 = None
        mm_114: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_368, permute_211)
        view_369: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_114, _shape_param_516);  mm_114 = _shape_param_516 = None
        inductor_lookup_seed_66: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 66)
        inductor_random_66: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_517, inductor_lookup_seed_66, 'rand');  _shape_param_517 = inductor_lookup_seed_66 = None
        convert_element_type_286: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_66, torch.bfloat16);  inductor_random_66 = None
        gt_67: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_286, 0.1);  convert_element_type_286 = None
        mul_266: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_67, view_369);  view_369 = None
        mul_267: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_266, 1.1111111111111112);  mul_266 = None
        add_113: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_110, mul_267);  mul_267 = None
        pow_47: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_113, 2)
        mean_33: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None
        add_114: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None
        rsqrt_33: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        mul_268: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_113, rsqrt_33)
        mul_269: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg152_1, mul_268);  arg152_1 = mul_268 = None
        convert_element_type_287: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg153_1, torch.bfloat16);  arg153_1 = None
        convert_element_type_288: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_269, torch.bfloat16);  mul_269 = None
        permute_212: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_287, [1, 0]);  convert_element_type_287 = None
        view_370: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_288, _shape_param_518);  convert_element_type_288 = _shape_param_518 = None
        mm_115: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_370, permute_212)
        view_371: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_115, _shape_param_519);  mm_115 = _shape_param_519 = None
        view_372: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_371, _shape_param_520);  view_371 = _shape_param_520 = None
        permute_213: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
        convert_element_type_289: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg154_1, torch.bfloat16);  arg154_1 = None
        permute_214: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_289, [1, 0]);  convert_element_type_289 = None
        mm_116: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_187, permute_214)
        view_373: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_116, _shape_param_521);  mm_116 = _shape_param_521 = None
        view_374: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_373, _shape_param_522);  view_373 = _shape_param_522 = None
        permute_215: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
        convert_element_type_290: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg155_1, torch.bfloat16);  arg155_1 = None
        permute_216: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_290, [1, 0]);  convert_element_type_290 = None
        mm_117: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_187, permute_216)
        view_375: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_117, _shape_param_523);  mm_117 = _shape_param_523 = None
        view_376: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_375, _shape_param_524);  view_375 = _shape_param_524 = None
        permute_217: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        permute_218: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_215, [0, 1, 3, 2]);  permute_215 = None
        expand_77: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_213, _shape_param_525);  permute_213 = _shape_param_525 = None
        clone_76: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_377: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_76, _shape_param_526);  clone_76 = _shape_param_526 = None
        expand_78: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_218, _shape_param_527);  permute_218 = _shape_param_527 = None
        clone_77: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
        view_378: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_77, _shape_param_528);  clone_77 = _shape_param_528 = None
        bmm_38: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_377, view_378)
        view_379: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_38, _shape_param_529);  _shape_param_529 = None
        convert_element_type_291: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_379, torch.bfloat16);  view_379 = None
        convert_element_type_292: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_291, torch.float32);  convert_element_type_291 = None
        amax_19: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_292, [-1], True)
        sub_20: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_292, amax_19);  convert_element_type_292 = None
        exp_19: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_20: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_23: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = None
        convert_element_type_293: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None
        inductor_lookup_seed_67: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 67)
        inductor_random_67: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_530, inductor_lookup_seed_67, 'rand');  _shape_param_530 = inductor_lookup_seed_67 = None
        convert_element_type_294: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_67, torch.bfloat16);  inductor_random_67 = None
        gt_68: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_294, 0.1);  convert_element_type_294 = None
        mul_270: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_68, convert_element_type_293);  convert_element_type_293 = None
        mul_271: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_270, 1.1111111111111112);  mul_270 = None
        expand_79: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_271, _shape_param_531);  mul_271 = _shape_param_531 = None
        view_380: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_79, _shape_param_532);  expand_79 = _shape_param_532 = None
        expand_80: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_217, _shape_param_533);  permute_217 = _shape_param_533 = None
        clone_78: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_381: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_78, _shape_param_534);  clone_78 = _shape_param_534 = None
        bmm_39: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_380, view_381)
        view_382: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_39, _shape_param_535);  bmm_39 = _shape_param_535 = None
        permute_219: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
        clone_79: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
        view_383: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_79, _shape_param_536);  clone_79 = _shape_param_536 = None
        convert_element_type_295: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg156_1, torch.bfloat16);  arg156_1 = None
        permute_220: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_295, [1, 0]);  convert_element_type_295 = None
        view_384: "bf16[4096, 384]" = torch.ops.aten.view.default(view_383, _shape_param_537);  view_383 = _shape_param_537 = None
        mm_118: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_384, permute_220)
        view_385: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_118, _shape_param_538);  mm_118 = _shape_param_538 = None
        inductor_lookup_seed_68: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 68)
        inductor_random_68: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_539, inductor_lookup_seed_68, 'rand');  _shape_param_539 = inductor_lookup_seed_68 = None
        convert_element_type_296: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_68, torch.bfloat16);  inductor_random_68 = None
        gt_69: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_296, 0.1);  convert_element_type_296 = None
        mul_272: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_69, view_385);  view_385 = None
        mul_273: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_272, 1.1111111111111112);  mul_272 = None
        add_115: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_113, mul_273);  mul_273 = None
        pow_48: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_115, 2)
        mean_34: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None
        add_116: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None
        rsqrt_34: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_274: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_115, rsqrt_34)
        mul_275: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg157_1, mul_274);  arg157_1 = mul_274 = None
        convert_element_type_297: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg158_1, torch.bfloat16);  arg158_1 = None
        convert_element_type_298: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_275, torch.bfloat16);  mul_275 = None
        permute_221: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_297, [1, 0]);  convert_element_type_297 = None
        view_386: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_298, _shape_param_540);  convert_element_type_298 = _shape_param_540 = None
        mm_119: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_386, permute_221)
        view_387: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_119, _shape_param_541);  _shape_param_541 = None
        mul_276: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_387, 0.5)
        convert_element_type_299: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_387, torch.float32)
        pow_49: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_299, 3.0);  convert_element_type_299 = None
        mul_277: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_49, 0.044715);  pow_49 = None
        add_117: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_387, mul_277);  view_387 = mul_277 = None
        mul_278: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_117, 0.7978845608028654);  add_117 = None
        tanh_13: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_278);  mul_278 = None
        add_118: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_279: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_276, add_118);  mul_276 = add_118 = None
        convert_element_type_300: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg159_1, torch.bfloat16);  arg159_1 = None
        permute_222: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_300, [1, 0]);  convert_element_type_300 = None
        mm_120: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_386, permute_222)
        view_388: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_120, _shape_param_542);  _shape_param_542 = None
        mul_280: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_279, view_388);  mul_279 = view_388 = None
        inductor_lookup_seed_69: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 69)
        inductor_random_69: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_543, inductor_lookup_seed_69, 'rand');  _shape_param_543 = inductor_lookup_seed_69 = None
        gt_70: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_69, 0.1);  inductor_random_69 = None
        mul_281: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_70, mul_280);  mul_280 = None
        mul_282: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_281, 1.1111111111111112);  mul_281 = None
        convert_element_type_301: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg160_1, torch.bfloat16);  arg160_1 = None
        convert_element_type_302: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_282, torch.bfloat16);  mul_282 = None
        permute_223: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_301, [1, 0]);  convert_element_type_301 = None
        view_389: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_302, _shape_param_544);  convert_element_type_302 = _shape_param_544 = None
        mm_121: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_389, permute_223)
        view_390: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_121, _shape_param_545);  mm_121 = _shape_param_545 = None
        inductor_lookup_seed_70: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 70)
        inductor_random_70: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_546, inductor_lookup_seed_70, 'rand');  _shape_param_546 = inductor_lookup_seed_70 = None
        convert_element_type_303: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_70, torch.bfloat16);  inductor_random_70 = None
        gt_71: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_303, 0.1);  convert_element_type_303 = None
        mul_283: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_71, view_390);  view_390 = None
        mul_284: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_283, 1.1111111111111112);  mul_283 = None
        add_119: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_115, mul_284);  mul_284 = None
        pow_50: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_119, 2)
        mean_35: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None
        add_120: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None
        rsqrt_35: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        mul_285: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_119, rsqrt_35)
        mul_286: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg161_1, mul_285);  arg161_1 = mul_285 = None
        convert_element_type_304: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg162_1, torch.bfloat16);  arg162_1 = None
        convert_element_type_305: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_286, torch.bfloat16);  mul_286 = None
        permute_224: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_304, [1, 0]);  convert_element_type_304 = None
        view_391: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_305, _shape_param_547);  convert_element_type_305 = _shape_param_547 = None
        mm_122: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_391, permute_224)
        view_392: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_122, _shape_param_548);  mm_122 = _shape_param_548 = None
        view_393: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_392, _shape_param_549);  view_392 = _shape_param_549 = None
        permute_225: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
        convert_element_type_306: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg163_1, torch.bfloat16);  arg163_1 = None
        permute_226: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_306, [1, 0]);  convert_element_type_306 = None
        mm_123: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_391, permute_226)
        view_394: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_123, _shape_param_550);  mm_123 = _shape_param_550 = None
        view_395: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_394, _shape_param_551);  view_394 = _shape_param_551 = None
        permute_227: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_395, [0, 2, 1, 3]);  view_395 = None
        convert_element_type_307: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg164_1, torch.bfloat16);  arg164_1 = None
        permute_228: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_307, [1, 0]);  convert_element_type_307 = None
        mm_124: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_391, permute_228)
        view_396: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_124, _shape_param_552);  mm_124 = _shape_param_552 = None
        view_397: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_396, _shape_param_553);  view_396 = _shape_param_553 = None
        permute_229: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_397, [0, 2, 1, 3]);  view_397 = None
        permute_230: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_227, [0, 1, 3, 2]);  permute_227 = None
        expand_81: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_225, _shape_param_554);  permute_225 = _shape_param_554 = None
        clone_80: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_398: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_80, _shape_param_555);  clone_80 = _shape_param_555 = None
        expand_82: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_230, _shape_param_556);  permute_230 = _shape_param_556 = None
        clone_81: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_399: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_81, _shape_param_557);  clone_81 = _shape_param_557 = None
        bmm_40: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_398, view_399)
        view_400: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_40, _shape_param_558);  bmm_40 = _shape_param_558 = None
        add_121: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_400, add_66);  view_400 = None
        convert_element_type_308: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_121, torch.bfloat16);  add_121 = None
        convert_element_type_309: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_308, torch.float32)
        amax_20: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_309, [-1], True)
        sub_21: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_309, amax_20);  convert_element_type_309 = None
        exp_20: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_21: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_24: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = None
        convert_element_type_310: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None
        inductor_lookup_seed_71: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 71)
        inductor_random_71: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_559, inductor_lookup_seed_71, 'rand');  _shape_param_559 = inductor_lookup_seed_71 = None
        convert_element_type_311: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_71, torch.bfloat16);  inductor_random_71 = None
        gt_72: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_311, 0.1);  convert_element_type_311 = None
        mul_287: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_72, convert_element_type_310);  convert_element_type_310 = None
        mul_288: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_287, 1.1111111111111112);  mul_287 = None
        expand_83: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_288, _shape_param_560);  mul_288 = _shape_param_560 = None
        view_401: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_83, _shape_param_561);  expand_83 = _shape_param_561 = None
        expand_84: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_229, _shape_param_562);  permute_229 = _shape_param_562 = None
        clone_82: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_402: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_82, _shape_param_563);  clone_82 = _shape_param_563 = None
        bmm_41: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_401, view_402)
        view_403: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_41, _shape_param_564);  bmm_41 = _shape_param_564 = None
        permute_231: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
        clone_83: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
        view_404: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_83, _shape_param_565);  clone_83 = _shape_param_565 = None
        convert_element_type_312: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg165_1, torch.bfloat16);  arg165_1 = None
        permute_232: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_312, [1, 0]);  convert_element_type_312 = None
        view_405: "bf16[4096, 384]" = torch.ops.aten.view.default(view_404, _shape_param_566);  view_404 = _shape_param_566 = None
        mm_125: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_405, permute_232)
        view_406: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_125, _shape_param_567);  mm_125 = _shape_param_567 = None
        inductor_lookup_seed_72: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 72)
        inductor_random_72: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_568, inductor_lookup_seed_72, 'rand');  _shape_param_568 = inductor_lookup_seed_72 = None
        convert_element_type_313: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_72, torch.bfloat16);  inductor_random_72 = None
        gt_73: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_313, 0.1);  convert_element_type_313 = None
        mul_289: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_73, view_406);  view_406 = None
        mul_290: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_289, 1.1111111111111112);  mul_289 = None
        add_122: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_119, mul_290);  mul_290 = None
        pow_51: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_122, 2)
        mean_36: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None
        add_123: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None
        rsqrt_36: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        mul_291: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_122, rsqrt_36)
        mul_292: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg166_1, mul_291);  arg166_1 = mul_291 = None
        convert_element_type_314: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg167_1, torch.bfloat16);  arg167_1 = None
        convert_element_type_315: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_292, torch.bfloat16);  mul_292 = None
        permute_233: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_314, [1, 0]);  convert_element_type_314 = None
        view_407: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_315, _shape_param_569);  convert_element_type_315 = _shape_param_569 = None
        mm_126: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_407, permute_233)
        view_408: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_126, _shape_param_570);  mm_126 = _shape_param_570 = None
        view_409: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_408, _shape_param_571);  view_408 = _shape_param_571 = None
        permute_234: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None
        convert_element_type_316: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg168_1, torch.bfloat16);  arg168_1 = None
        permute_235: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_316, [1, 0]);  convert_element_type_316 = None
        mm_127: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_187, permute_235)
        view_410: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_127, _shape_param_572);  mm_127 = _shape_param_572 = None
        view_411: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_410, _shape_param_573);  view_410 = _shape_param_573 = None
        permute_236: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_411, [0, 2, 1, 3]);  view_411 = None
        convert_element_type_317: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg169_1, torch.bfloat16);  arg169_1 = None
        permute_237: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_317, [1, 0]);  convert_element_type_317 = None
        mm_128: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_187, permute_237)
        view_412: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_128, _shape_param_574);  mm_128 = _shape_param_574 = None
        view_413: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_412, _shape_param_575);  view_412 = _shape_param_575 = None
        permute_238: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_413, [0, 2, 1, 3]);  view_413 = None
        permute_239: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_236, [0, 1, 3, 2]);  permute_236 = None
        expand_85: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_234, _shape_param_576);  permute_234 = _shape_param_576 = None
        clone_84: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_414: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_84, _shape_param_577);  clone_84 = _shape_param_577 = None
        expand_86: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_239, _shape_param_578);  permute_239 = _shape_param_578 = None
        clone_85: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
        view_415: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_85, _shape_param_579);  clone_85 = _shape_param_579 = None
        bmm_42: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_414, view_415)
        view_416: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_42, _shape_param_580);  _shape_param_580 = None
        convert_element_type_318: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_416, torch.bfloat16);  view_416 = None
        convert_element_type_319: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_318, torch.float32);  convert_element_type_318 = None
        amax_21: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_319, [-1], True)
        sub_22: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_319, amax_21);  convert_element_type_319 = None
        exp_21: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_22: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_25: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = None
        convert_element_type_320: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None
        inductor_lookup_seed_73: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 73)
        inductor_random_73: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_581, inductor_lookup_seed_73, 'rand');  _shape_param_581 = inductor_lookup_seed_73 = None
        convert_element_type_321: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_73, torch.bfloat16);  inductor_random_73 = None
        gt_74: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_321, 0.1);  convert_element_type_321 = None
        mul_293: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_74, convert_element_type_320);  convert_element_type_320 = None
        mul_294: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_293, 1.1111111111111112);  mul_293 = None
        expand_87: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_294, _shape_param_582);  mul_294 = _shape_param_582 = None
        view_417: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_87, _shape_param_583);  expand_87 = _shape_param_583 = None
        expand_88: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_238, _shape_param_584);  permute_238 = _shape_param_584 = None
        clone_86: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_418: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_86, _shape_param_585);  clone_86 = _shape_param_585 = None
        bmm_43: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_417, view_418)
        view_419: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_43, _shape_param_586);  bmm_43 = _shape_param_586 = None
        permute_240: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_419, [0, 2, 1, 3]);  view_419 = None
        clone_87: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
        view_420: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_87, _shape_param_587);  clone_87 = _shape_param_587 = None
        convert_element_type_322: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg170_1, torch.bfloat16);  arg170_1 = None
        permute_241: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_322, [1, 0]);  convert_element_type_322 = None
        view_421: "bf16[4096, 384]" = torch.ops.aten.view.default(view_420, _shape_param_588);  view_420 = _shape_param_588 = None
        mm_129: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_421, permute_241)
        view_422: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_129, _shape_param_589);  mm_129 = _shape_param_589 = None
        inductor_lookup_seed_74: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 74)
        inductor_random_74: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_590, inductor_lookup_seed_74, 'rand');  _shape_param_590 = inductor_lookup_seed_74 = None
        convert_element_type_323: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_74, torch.bfloat16);  inductor_random_74 = None
        gt_75: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_323, 0.1);  convert_element_type_323 = None
        mul_295: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_75, view_422);  view_422 = None
        mul_296: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_295, 1.1111111111111112);  mul_295 = None
        add_124: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_122, mul_296);  mul_296 = None
        pow_52: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_124, 2)
        mean_37: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None
        add_125: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None
        rsqrt_37: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
        mul_297: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_124, rsqrt_37)
        mul_298: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg171_1, mul_297);  arg171_1 = mul_297 = None
        convert_element_type_324: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg172_1, torch.bfloat16);  arg172_1 = None
        convert_element_type_325: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_298, torch.bfloat16);  mul_298 = None
        permute_242: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_324, [1, 0]);  convert_element_type_324 = None
        view_423: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_325, _shape_param_591);  convert_element_type_325 = _shape_param_591 = None
        mm_130: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_423, permute_242)
        view_424: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_130, _shape_param_592);  _shape_param_592 = None
        mul_299: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_424, 0.5)
        convert_element_type_326: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_424, torch.float32)
        pow_53: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_326, 3.0);  convert_element_type_326 = None
        mul_300: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_53, 0.044715);  pow_53 = None
        add_126: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_424, mul_300);  view_424 = mul_300 = None
        mul_301: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_126, 0.7978845608028654);  add_126 = None
        tanh_14: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_301);  mul_301 = None
        add_127: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_302: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_299, add_127);  mul_299 = add_127 = None
        convert_element_type_327: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg173_1, torch.bfloat16);  arg173_1 = None
        permute_243: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_327, [1, 0]);  convert_element_type_327 = None
        mm_131: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_423, permute_243)
        view_425: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_131, _shape_param_593);  _shape_param_593 = None
        mul_303: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_302, view_425);  mul_302 = view_425 = None
        inductor_lookup_seed_75: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 75)
        inductor_random_75: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_594, inductor_lookup_seed_75, 'rand');  _shape_param_594 = inductor_lookup_seed_75 = None
        gt_76: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_75, 0.1);  inductor_random_75 = None
        mul_304: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_76, mul_303);  mul_303 = None
        mul_305: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_304, 1.1111111111111112);  mul_304 = None
        convert_element_type_328: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg174_1, torch.bfloat16);  arg174_1 = None
        convert_element_type_329: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_305, torch.bfloat16);  mul_305 = None
        permute_244: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_328, [1, 0]);  convert_element_type_328 = None
        view_426: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_329, _shape_param_595);  convert_element_type_329 = _shape_param_595 = None
        mm_132: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_426, permute_244)
        view_427: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_132, _shape_param_596);  mm_132 = _shape_param_596 = None
        inductor_lookup_seed_76: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 76)
        inductor_random_76: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_597, inductor_lookup_seed_76, 'rand');  _shape_param_597 = inductor_lookup_seed_76 = None
        convert_element_type_330: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_76, torch.bfloat16);  inductor_random_76 = None
        gt_77: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_330, 0.1);  convert_element_type_330 = None
        mul_306: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_77, view_427);  view_427 = None
        mul_307: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_306, 1.1111111111111112);  mul_306 = None
        add_128: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_124, mul_307);  mul_307 = None
        pow_54: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_128, 2)
        mean_38: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None
        add_129: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None
        rsqrt_38: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_308: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_128, rsqrt_38)
        mul_309: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg175_1, mul_308);  arg175_1 = mul_308 = None
        convert_element_type_331: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg176_1, torch.bfloat16);  arg176_1 = None
        convert_element_type_332: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_309, torch.bfloat16);  mul_309 = None
        permute_245: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_331, [1, 0]);  convert_element_type_331 = None
        view_428: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_332, _shape_param_598);  convert_element_type_332 = _shape_param_598 = None
        mm_133: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_428, permute_245)
        view_429: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_133, _shape_param_599);  mm_133 = _shape_param_599 = None
        view_430: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_429, _shape_param_600);  view_429 = _shape_param_600 = None
        permute_246: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
        convert_element_type_333: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg177_1, torch.bfloat16);  arg177_1 = None
        permute_247: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_333, [1, 0]);  convert_element_type_333 = None
        mm_134: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_428, permute_247)
        view_431: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_134, _shape_param_601);  mm_134 = _shape_param_601 = None
        view_432: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_431, _shape_param_602);  view_431 = _shape_param_602 = None
        permute_248: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
        convert_element_type_334: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg178_1, torch.bfloat16);  arg178_1 = None
        permute_249: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_334, [1, 0]);  convert_element_type_334 = None
        mm_135: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_428, permute_249)
        view_433: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_135, _shape_param_603);  mm_135 = _shape_param_603 = None
        view_434: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_433, _shape_param_604);  view_433 = _shape_param_604 = None
        permute_250: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_434, [0, 2, 1, 3]);  view_434 = None
        permute_251: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_248, [0, 1, 3, 2]);  permute_248 = None
        expand_89: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_246, _shape_param_605);  permute_246 = _shape_param_605 = None
        clone_88: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_435: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_88, _shape_param_606);  clone_88 = _shape_param_606 = None
        expand_90: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_251, _shape_param_607);  permute_251 = _shape_param_607 = None
        clone_89: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_436: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_89, _shape_param_608);  clone_89 = _shape_param_608 = None
        bmm_44: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_435, view_436)
        view_437: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_44, _shape_param_609);  bmm_44 = _shape_param_609 = None
        add_130: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_437, add_66);  view_437 = add_66 = None
        convert_element_type_335: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_130, torch.bfloat16);  add_130 = None
        convert_element_type_336: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_335, torch.float32)
        amax_22: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_336, [-1], True)
        sub_23: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_336, amax_22);  convert_element_type_336 = None
        exp_22: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_23: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_26: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = None
        convert_element_type_337: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_26, torch.bfloat16);  div_26 = None
        inductor_lookup_seed_77: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 77)
        inductor_random_77: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_610, inductor_lookup_seed_77, 'rand');  _shape_param_610 = inductor_lookup_seed_77 = None
        convert_element_type_338: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_77, torch.bfloat16);  inductor_random_77 = None
        gt_78: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_338, 0.1);  convert_element_type_338 = None
        mul_310: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_78, convert_element_type_337);  convert_element_type_337 = None
        mul_311: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_310, 1.1111111111111112);  mul_310 = None
        expand_91: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_311, _shape_param_611);  mul_311 = _shape_param_611 = None
        view_438: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_91, _shape_param_612);  expand_91 = _shape_param_612 = None
        expand_92: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_250, _shape_param_613);  permute_250 = _shape_param_613 = None
        clone_90: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_439: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_90, _shape_param_614);  clone_90 = _shape_param_614 = None
        bmm_45: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_438, view_439)
        view_440: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_45, _shape_param_615);  bmm_45 = _shape_param_615 = None
        permute_252: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_440, [0, 2, 1, 3]);  view_440 = None
        clone_91: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None
        view_441: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_91, _shape_param_616);  clone_91 = _shape_param_616 = None
        convert_element_type_339: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg179_1, torch.bfloat16);  arg179_1 = None
        permute_253: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_339, [1, 0]);  convert_element_type_339 = None
        view_442: "bf16[4096, 384]" = torch.ops.aten.view.default(view_441, _shape_param_617);  view_441 = _shape_param_617 = None
        mm_136: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_442, permute_253)
        view_443: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_136, _shape_param_618);  mm_136 = _shape_param_618 = None
        inductor_lookup_seed_78: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 78)
        inductor_random_78: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_619, inductor_lookup_seed_78, 'rand');  _shape_param_619 = inductor_lookup_seed_78 = None
        convert_element_type_340: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_78, torch.bfloat16);  inductor_random_78 = None
        gt_79: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_340, 0.1);  convert_element_type_340 = None
        mul_312: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_79, view_443);  view_443 = None
        mul_313: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_312, 1.1111111111111112);  mul_312 = None
        add_131: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_128, mul_313);  mul_313 = None
        pow_55: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_131, 2)
        mean_39: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None
        add_132: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None
        rsqrt_39: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        mul_314: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_131, rsqrt_39)
        mul_315: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg180_1, mul_314);  arg180_1 = mul_314 = None
        convert_element_type_341: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg181_1, torch.bfloat16);  arg181_1 = None
        convert_element_type_342: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_315, torch.bfloat16);  mul_315 = None
        permute_254: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_341, [1, 0]);  convert_element_type_341 = None
        view_444: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_342, _shape_param_620);  convert_element_type_342 = _shape_param_620 = None
        mm_137: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_444, permute_254)
        view_445: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_137, _shape_param_621);  mm_137 = _shape_param_621 = None
        view_446: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_445, _shape_param_622);  view_445 = _shape_param_622 = None
        permute_255: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_446, [0, 2, 1, 3]);  view_446 = None
        convert_element_type_343: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg182_1, torch.bfloat16);  arg182_1 = None
        permute_256: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_343, [1, 0]);  convert_element_type_343 = None
        mm_138: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_187, permute_256)
        view_447: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_138, _shape_param_623);  mm_138 = _shape_param_623 = None
        view_448: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_447, _shape_param_624);  view_447 = _shape_param_624 = None
        permute_257: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
        convert_element_type_344: "bf16[384, 512]" = torch.ops.prims.convert_element_type.default(arg183_1, torch.bfloat16);  arg183_1 = None
        permute_258: "bf16[512, 384]" = torch.ops.aten.permute.default(convert_element_type_344, [1, 0]);  convert_element_type_344 = None
        mm_139: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_187, permute_258)
        view_449: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_139, _shape_param_625);  mm_139 = _shape_param_625 = None
        view_450: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_449, _shape_param_626);  view_449 = _shape_param_626 = None
        permute_259: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None
        permute_260: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_257, [0, 1, 3, 2]);  permute_257 = None
        expand_93: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_255, _shape_param_627);  permute_255 = _shape_param_627 = None
        clone_92: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_451: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_92, _shape_param_628);  clone_92 = _shape_param_628 = None
        expand_94: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_260, _shape_param_629);  permute_260 = _shape_param_629 = None
        clone_93: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
        view_452: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_93, _shape_param_630);  clone_93 = _shape_param_630 = None
        bmm_46: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_451, view_452)
        view_453: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_46, _shape_param_631);  _shape_param_631 = None
        convert_element_type_345: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_453, torch.bfloat16);  view_453 = None
        convert_element_type_346: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_345, torch.float32);  convert_element_type_345 = None
        amax_23: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_346, [-1], True)
        sub_24: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_346, amax_23);  convert_element_type_346 = None
        exp_23: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
        sum_24: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_27: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = None
        convert_element_type_347: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None
        inductor_lookup_seed_79: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 79)
        inductor_random_79: "f32[32, 6, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_632, inductor_lookup_seed_79, 'rand');  _shape_param_632 = inductor_lookup_seed_79 = None
        convert_element_type_348: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_79, torch.bfloat16);  inductor_random_79 = None
        gt_80: "b8[32, 6, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_348, 0.1);  convert_element_type_348 = None
        mul_316: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(gt_80, convert_element_type_347);  convert_element_type_347 = None
        mul_317: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_316, 1.1111111111111112);  mul_316 = None
        expand_95: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(mul_317, _shape_param_633);  mul_317 = _shape_param_633 = None
        view_454: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_95, _shape_param_634);  expand_95 = _shape_param_634 = None
        expand_96: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_259, _shape_param_635);  permute_259 = _shape_param_635 = None
        clone_94: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_455: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_94, _shape_param_636);  clone_94 = _shape_param_636 = None
        bmm_47: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_454, view_455)
        view_456: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_47, _shape_param_637);  bmm_47 = _shape_param_637 = None
        permute_261: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
        clone_95: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
        view_457: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_95, _shape_param_638);  clone_95 = _shape_param_638 = None
        convert_element_type_349: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg184_1, torch.bfloat16);  arg184_1 = None
        permute_262: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_349, [1, 0]);  convert_element_type_349 = None
        view_458: "bf16[4096, 384]" = torch.ops.aten.view.default(view_457, _shape_param_639);  view_457 = _shape_param_639 = None
        mm_140: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_458, permute_262)
        view_459: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_140, _shape_param_640);  mm_140 = _shape_param_640 = None
        inductor_lookup_seed_80: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 80)
        inductor_random_80: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_641, inductor_lookup_seed_80, 'rand');  _shape_param_641 = inductor_lookup_seed_80 = None
        convert_element_type_350: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_80, torch.bfloat16);  inductor_random_80 = None
        gt_81: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_350, 0.1);  convert_element_type_350 = None
        mul_318: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_81, view_459);  view_459 = None
        mul_319: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_318, 1.1111111111111112);  mul_318 = None
        add_133: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_131, mul_319);  mul_319 = None
        pow_56: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_133, 2)
        mean_40: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None
        add_134: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None
        rsqrt_40: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        mul_320: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_133, rsqrt_40)
        mul_321: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg185_1, mul_320);  arg185_1 = mul_320 = None
        convert_element_type_351: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg186_1, torch.bfloat16);  arg186_1 = None
        convert_element_type_352: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_321, torch.bfloat16);  mul_321 = None
        permute_263: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_351, [1, 0]);  convert_element_type_351 = None
        view_460: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_352, _shape_param_642);  convert_element_type_352 = _shape_param_642 = None
        mm_141: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_460, permute_263)
        view_461: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_141, _shape_param_643);  _shape_param_643 = None
        mul_322: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_461, 0.5)
        convert_element_type_353: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_461, torch.float32)
        pow_57: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_353, 3.0);  convert_element_type_353 = None
        mul_323: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_57, 0.044715);  pow_57 = None
        add_135: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_461, mul_323);  view_461 = mul_323 = None
        mul_324: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_135, 0.7978845608028654);  add_135 = None
        tanh_15: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_324);  mul_324 = None
        add_136: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_325: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_322, add_136);  mul_322 = add_136 = None
        convert_element_type_354: "bf16[1024, 512]" = torch.ops.prims.convert_element_type.default(arg187_1, torch.bfloat16);  arg187_1 = None
        permute_264: "bf16[512, 1024]" = torch.ops.aten.permute.default(convert_element_type_354, [1, 0]);  convert_element_type_354 = None
        mm_142: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_460, permute_264)
        view_462: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_142, _shape_param_644);  _shape_param_644 = None
        mul_326: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_325, view_462);  mul_325 = view_462 = None
        inductor_lookup_seed_81: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 81)
        inductor_random_81: "f32[32, 128, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_645, inductor_lookup_seed_81, 'rand');  _shape_param_645 = inductor_lookup_seed_81 = None
        gt_82: "b8[32, 128, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_81, 0.1);  inductor_random_81 = None
        mul_327: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(gt_82, mul_326);  mul_326 = None
        mul_328: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_327, 1.1111111111111112);  mul_327 = None
        convert_element_type_355: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg188_1, torch.bfloat16);  arg188_1 = None
        convert_element_type_356: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_328, torch.bfloat16);  mul_328 = None
        permute_265: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_355, [1, 0]);  convert_element_type_355 = None
        view_463: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_356, _shape_param_646);  convert_element_type_356 = _shape_param_646 = None
        mm_143: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_463, permute_265)
        view_464: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_143, _shape_param_647);  mm_143 = _shape_param_647 = None
        inductor_lookup_seed_82: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 82)
        inductor_random_82: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_648, inductor_lookup_seed_82, 'rand');  _shape_param_648 = inductor_lookup_seed_82 = None
        convert_element_type_357: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(inductor_random_82, torch.bfloat16);  inductor_random_82 = None
        gt_83: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(convert_element_type_357, 0.1);  convert_element_type_357 = None
        mul_329: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_83, view_464);  view_464 = None
        mul_330: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_329, 1.1111111111111112);  mul_329 = None
        add_137: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_133, mul_330);  mul_330 = None
        pow_58: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_137, 2)
        mean_41: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None
        add_138: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None
        rsqrt_41: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        mul_331: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_137, rsqrt_41)
        mul_332: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg189_1, mul_331);  arg189_1 = mul_331 = None
        inductor_lookup_seed_83: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 83);  inductor_seeds = None
        inductor_random_83: "f32[32, 128, 512]" = torch.ops.prims.inductor_random.default(_shape_param_649, inductor_lookup_seed_83, 'rand');  _shape_param_649 = inductor_lookup_seed_83 = None
        gt_84: "b8[32, 128, 512]" = torch.ops.aten.gt.Scalar(inductor_random_83, 0.1);  inductor_random_83 = None
        mul_333: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_84, mul_332);  mul_332 = None
        mul_334: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_333, 1.1111111111111112);  mul_333 = None
        convert_element_type_358: "bf16[250112, 512]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_359: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_334, torch.bfloat16);  mul_334 = None
        permute_266: "bf16[512, 250112]" = torch.ops.aten.permute.default(convert_element_type_358, [1, 0]);  convert_element_type_358 = None
        view_465: "bf16[4096, 512]" = torch.ops.aten.view.default(convert_element_type_359, _shape_param_650);  convert_element_type_359 = _shape_param_650 = None
        mm_144: "bf16[4096, 250112]" = torch.ops.aten.mm.default(view_465, permute_266)
        view_466: "bf16[32, 128, 250112]" = torch.ops.aten.view.default(mm_144, _shape_param_651);  mm_144 = _shape_param_651 = None
        view_467: "bf16[4096, 250112]" = torch.ops.aten.view.default(view_466, _shape_param_652);  _shape_param_652 = None
        view_468: "i64[4096]" = torch.ops.aten.view.default(arg190_1, [-1]);  arg190_1 = None
        convert_element_type_360: "f32[4096, 250112]" = torch.ops.prims.convert_element_type.default(view_467, torch.float32);  view_467 = None
        amax_24: "f32[4096, 1]" = torch.ops.aten.amax.default(convert_element_type_360, [1], True)
        sub_25: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(convert_element_type_360, amax_24);  convert_element_type_360 = None
        exp_24: "f32[4096, 250112]" = torch.ops.aten.exp.default(sub_25)
        sum_25: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log_2: "f32[4096, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_26: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(sub_25, log_2);  sub_25 = None
        convert_element_type_361: "bf16[4096, 250112]" = torch.ops.prims.convert_element_type.default(sub_26, torch.bfloat16);  sub_26 = None
        convert_element_type_362: "f32[4096, 250112]" = torch.ops.prims.convert_element_type.default(convert_element_type_361, torch.float32);  convert_element_type_361 = None
        ne: "b8[4096]" = torch.ops.aten.ne.Scalar(view_468, -100)
        full_6: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "i64[4096]" = torch.ops.aten.where.self(ne, view_468, full_6);  view_468 = full_6 = None
        unsqueeze_8: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_3, 1);  where_3 = None
        gather: "f32[4096, 1]" = torch.ops.aten.gather.default(convert_element_type_362, 1, unsqueeze_8);  convert_element_type_362 = unsqueeze_8 = None
        squeeze: "f32[4096]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_1: "f32[4096]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_4: "f32[4096]" = torch.ops.aten.where.self(ne, neg_1, full_2);  neg_1 = full_2 = None
        sum_26: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_363: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27: "f32[]" = torch.ops.aten.sum.default(where_4);  where_4 = None
        div_28: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_363);  sum_27 = None
        permute_267: "bf16[250112, 512]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
        permute_268: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
        permute_269: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
        permute_270: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        permute_271: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
        permute_272: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_454, [0, 2, 1]);  view_454 = None
        permute_273: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_455, [0, 2, 1]);  view_455 = None
        permute_274: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_451, [0, 2, 1]);  view_451 = None
        permute_275: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_452, [0, 2, 1]);  view_452 = None
        permute_276: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
        permute_277: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
        permute_278: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
        permute_279: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
        permute_280: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_438, [0, 2, 1]);  view_438 = None
        permute_281: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_439, [0, 2, 1]);  view_439 = None
        permute_282: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_435, [0, 2, 1]);  view_435 = None
        permute_283: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_436, [0, 2, 1]);  view_436 = None
        permute_284: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_249, [1, 0]);  permute_249 = None
        permute_285: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
        permute_286: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_245, [1, 0]);  permute_245 = None
        permute_287: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
        permute_288: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
        permute_289: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        permute_290: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        permute_291: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_417, [0, 2, 1]);  view_417 = None
        permute_292: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_418, [0, 2, 1]);  view_418 = None
        permute_293: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_414, [0, 2, 1]);  view_414 = None
        permute_294: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_415, [0, 2, 1]);  view_415 = None
        permute_295: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None
        permute_296: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
        permute_297: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        permute_298: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
        permute_299: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_401, [0, 2, 1]);  view_401 = None
        permute_300: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_402, [0, 2, 1]);  view_402 = None
        permute_301: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_398, [0, 2, 1]);  view_398 = None
        permute_302: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_399, [0, 2, 1]);  view_399 = None
        permute_303: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
        permute_304: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
        permute_305: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
        permute_306: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
        permute_307: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
        permute_308: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
        permute_309: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
        permute_310: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_380, [0, 2, 1]);  view_380 = None
        permute_311: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_381, [0, 2, 1]);  view_381 = None
        permute_312: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_377, [0, 2, 1]);  view_377 = None
        permute_313: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_378, [0, 2, 1]);  view_378 = None
        permute_314: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
        permute_315: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
        permute_316: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
        permute_317: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        permute_318: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_364, [0, 2, 1]);  view_364 = None
        permute_319: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None
        permute_320: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_361, [0, 2, 1]);  view_361 = None
        permute_321: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_362, [0, 2, 1]);  view_362 = None
        permute_322: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
        permute_323: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
        permute_324: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
        permute_325: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
        permute_326: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
        permute_327: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
        permute_328: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
        permute_329: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_343, [0, 2, 1]);  view_343 = None
        permute_330: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_344, [0, 2, 1]);  view_344 = None
        permute_331: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_340, [0, 2, 1]);  view_340 = None
        permute_332: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_341, [0, 2, 1]);  view_341 = None
        permute_333: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        permute_334: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
        permute_335: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
        permute_336: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
        permute_337: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_327, [0, 2, 1]);  view_327 = None
        permute_338: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_328, [0, 2, 1]);  view_328 = None
        permute_339: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_324, [0, 2, 1]);  view_324 = None
        permute_340: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_325, [0, 2, 1]);  view_325 = None
        permute_341: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        permute_342: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
        permute_343: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
        permute_344: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
        permute_345: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
        permute_346: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
        permute_347: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        permute_348: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_306, [0, 2, 1]);  view_306 = None
        permute_349: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_307, [0, 2, 1]);  view_307 = None
        permute_350: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_303, [0, 2, 1]);  view_303 = None
        permute_351: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_304, [0, 2, 1]);  view_304 = None
        permute_352: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        permute_353: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
        permute_354: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
        permute_355: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
        permute_356: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_290, [0, 2, 1]);  view_290 = None
        permute_357: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_291, [0, 2, 1]);  view_291 = None
        permute_358: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None
        permute_359: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_288, [0, 2, 1]);  view_288 = None
        permute_360: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        permute_361: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        permute_362: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
        permute_363: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
        permute_364: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
        permute_365: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
        permute_366: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
        permute_367: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_269, [0, 2, 1]);  view_269 = None
        permute_368: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_270, [0, 2, 1]);  view_270 = None
        permute_369: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_266, [0, 2, 1]);  view_266 = None
        permute_370: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
        permute_371: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        permute_372: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        permute_373: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
        permute_374: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
        permute_375: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
        permute_376: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        permute_377: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_250, [0, 2, 1]);  view_250 = None
        permute_378: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
        permute_379: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        permute_380: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        permute_381: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        permute_382: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
        permute_383: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
        permute_384: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
        permute_385: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        permute_386: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        permute_387: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
        permute_388: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        permute_389: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
        permute_390: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        permute_391: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        permute_392: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
        permute_393: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
        permute_394: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_216, [0, 2, 1]);  view_216 = None
        permute_395: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_217, [0, 2, 1]);  view_217 = None
        permute_396: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_213, [0, 2, 1]);  view_213 = None
        permute_397: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1]);  view_214 = None
        permute_398: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        permute_399: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        permute_400: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        permute_401: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        permute_402: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
        permute_403: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
        permute_404: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
        permute_405: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_195, [0, 2, 1]);  view_195 = None
        permute_406: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_196, [0, 2, 1]);  view_196 = None
        permute_407: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_192, [0, 2, 1]);  view_192 = None
        permute_408: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
        permute_409: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        permute_410: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        permute_411: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        permute_412: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
        permute_413: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_178, [0, 2, 1]);  view_178 = None
        permute_414: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None
        permute_415: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_175, [0, 2, 1]);  view_175 = None
        permute_416: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None
        permute_417: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        permute_418: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        permute_419: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        permute_420: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        permute_421: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
        permute_422: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
        permute_423: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
        permute_424: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_157, [0, 2, 1]);  view_157 = None
        permute_425: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_158, [0, 2, 1]);  view_158 = None
        permute_426: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_154, [0, 2, 1]);  view_154 = None
        permute_427: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
        permute_428: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        permute_429: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        permute_430: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        permute_431: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
        permute_432: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
        permute_433: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
        permute_434: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        permute_435: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None
        permute_436: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_137, [0, 2, 1]);  view_137 = None
        permute_437: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
        permute_438: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
        permute_439: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        permute_440: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        permute_441: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
        permute_442: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
        permute_443: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        permute_444: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        permute_445: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        permute_446: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_115, [0, 2, 1]);  view_115 = None
        permute_447: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_116, [0, 2, 1]);  view_116 = None
        permute_448: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_112, [0, 2, 1]);  view_112 = None
        permute_449: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
        permute_450: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        permute_451: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        permute_452: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
        permute_453: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        permute_454: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        permute_455: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        permute_456: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        permute_457: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
        permute_458: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
        permute_459: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_460: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None
        permute_461: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        permute_462: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
        permute_463: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        permute_464: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        permute_465: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        permute_466: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        permute_467: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        permute_468: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_73, [0, 2, 1]);  view_73 = None
        permute_469: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_74, [0, 2, 1]);  view_74 = None
        permute_470: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None
        permute_471: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None
        permute_472: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        permute_473: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
        permute_474: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        permute_475: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        permute_476: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        permute_477: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        permute_478: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        permute_479: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
        permute_480: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_481: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_49, [0, 2, 1]);  view_49 = None
        permute_482: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
        permute_483: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        permute_484: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        permute_485: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        permute_486: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        permute_487: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        permute_488: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        permute_489: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        permute_490: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_491: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
        permute_492: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
        permute_493: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
        permute_494: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
        permute_495: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        permute_496: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        permute_497: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_498: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        permute_499: "bf16[1024, 512]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_500: "bf16[512, 384]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        permute_501: "bf16[192, 128, 128]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_502: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        permute_503: "bf16[192, 64, 128]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        permute_504: "bf16[192, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
        permute_505: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        permute_506: "bf16[384, 512]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_507: "bf16[384, 512]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (embedding, gt, rsqrt, view, bmm, add_4, embedding_1, amax, sum_1, gt_2, view_14, gt_3, add_7, rsqrt_1, view_16, mm_4, mm_5, gt_4, view_19, gt_5, add_11, rsqrt_2, view_21, convert_element_type_24, amax_1, sum_2, gt_6, view_35, gt_7, add_14, rsqrt_3, view_37, mm_11, mm_12, gt_8, view_40, gt_9, add_18, rsqrt_4, view_42, convert_element_type_41, amax_2, sum_3, gt_10, view_56, gt_11, add_21, rsqrt_5, view_58, mm_18, mm_19, gt_12, view_61, gt_13, add_25, rsqrt_6, view_63, convert_element_type_58, amax_3, sum_4, gt_14, view_77, gt_15, add_28, rsqrt_7, view_79, mm_25, mm_26, gt_16, view_82, gt_17, add_32, rsqrt_8, view_84, convert_element_type_75, amax_4, sum_5, gt_18, view_98, gt_19, add_35, rsqrt_9, view_100, mm_32, mm_33, gt_20, view_103, gt_21, add_39, rsqrt_10, view_105, convert_element_type_92, amax_5, sum_6, gt_22, view_119, gt_23, add_42, rsqrt_11, view_121, mm_39, mm_40, gt_24, view_124, gt_25, add_46, rsqrt_12, view_126, convert_element_type_109, amax_6, sum_7, gt_26, view_140, gt_27, add_49, rsqrt_13, view_142, mm_46, mm_47, gt_28, view_145, gt_29, add_53, rsqrt_14, view_147, convert_element_type_126, amax_7, sum_8, gt_30, view_161, gt_31, add_56, rsqrt_15, view_163, mm_53, mm_54, gt_32, view_166, gt_33, add_60, rsqrt_16, gt_34, mul_143, unsqueeze_4, gt_35, rsqrt_17, view_168, bmm_16, add_65, embedding_2, amax_8, sum_9, gt_36, view_182, gt_37, add_68, rsqrt_18, view_184, view_187, bmm_18, amax_9, sum_10, gt_38, view_199, gt_39, add_70, rsqrt_19, view_201, mm_64, mm_65, gt_40, view_204, gt_41, add_74, rsqrt_20, view_206, convert_element_type_173, amax_10, sum_11, gt_42, view_220, gt_43, add_77, rsqrt_21, view_222, bmm_22, amax_11, sum_12, gt_44, view_236, gt_45, add_79, rsqrt_22, view_238, mm_75, mm_76, gt_46, view_241, gt_47, add_83, rsqrt_23, view_243, convert_element_type_200, amax_12, sum_13, gt_48, view_257, gt_49, add_86, rsqrt_24, view_259, bmm_26, amax_13, sum_14, gt_50, view_273, gt_51, add_88, rsqrt_25, view_275, mm_86, mm_87, gt_52, view_278, gt_53, add_92, rsqrt_26, view_280, convert_element_type_227, amax_14, sum_15, gt_54, view_294, gt_55, add_95, rsqrt_27, view_296, bmm_30, amax_15, sum_16, gt_56, view_310, gt_57, add_97, rsqrt_28, view_312, mm_97, mm_98, gt_58, view_315, gt_59, add_101, rsqrt_29, view_317, convert_element_type_254, amax_16, sum_17, gt_60, view_331, gt_61, add_104, rsqrt_30, view_333, bmm_34, amax_17, sum_18, gt_62, view_347, gt_63, add_106, rsqrt_31, view_349, mm_108, mm_109, gt_64, view_352, gt_65, add_110, rsqrt_32, view_354, convert_element_type_281, amax_18, sum_19, gt_66, view_368, gt_67, add_113, rsqrt_33, view_370, bmm_38, amax_19, sum_20, gt_68, view_384, gt_69, add_115, rsqrt_34, view_386, mm_119, mm_120, gt_70, view_389, gt_71, add_119, rsqrt_35, view_391, convert_element_type_308, amax_20, sum_21, gt_72, view_405, gt_73, add_122, rsqrt_36, view_407, bmm_42, amax_21, sum_22, gt_74, view_421, gt_75, add_124, rsqrt_37, view_423, mm_130, mm_131, gt_76, view_426, gt_77, add_128, rsqrt_38, view_428, convert_element_type_335, amax_22, sum_23, gt_78, view_442, gt_79, add_131, rsqrt_39, view_444, bmm_46, amax_23, sum_24, gt_80, view_458, gt_81, add_133, rsqrt_40, view_460, mm_141, mm_142, gt_82, view_463, gt_83, add_137, rsqrt_41, gt_84, view_465, view_466, amax_24, log_2, convert_element_type_363, div_28, permute_267, permute_268, permute_269, permute_270, permute_271, permute_272, permute_273, permute_274, permute_275, permute_276, permute_277, permute_278, permute_279, permute_280, permute_281, permute_282, permute_283, permute_284, permute_285, permute_286, permute_287, permute_288, permute_289, permute_290, permute_291, permute_292, permute_293, permute_294, permute_295, permute_296, permute_297, permute_298, permute_299, permute_300, permute_301, permute_302, permute_303, permute_304, permute_305, permute_306, permute_307, permute_308, permute_309, permute_310, permute_311, permute_312, permute_313, permute_314, permute_315, permute_316, permute_317, permute_318, permute_319, permute_320, permute_321, permute_322, permute_323, permute_324, permute_325, permute_326, permute_327, permute_328, permute_329, permute_330, permute_331, permute_332, permute_333, permute_334, permute_335, permute_336, permute_337, permute_338, permute_339, permute_340, permute_341, permute_342, permute_343, permute_344, permute_345, permute_346, permute_347, permute_348, permute_349, permute_350, permute_351, permute_352, permute_353, permute_354, permute_355, permute_356, permute_357, permute_358, permute_359, permute_360, permute_361, permute_362, permute_363, permute_364, permute_365, permute_366, permute_367, permute_368, permute_369, permute_370, permute_371, permute_372, permute_373, permute_374, permute_375, permute_376, permute_377, permute_378, permute_379, permute_380, permute_381, permute_382, permute_383, permute_384, permute_385, permute_386, permute_387, permute_388, permute_389, permute_390, permute_391, permute_392, permute_393, permute_394, permute_395, permute_396, permute_397, permute_398, permute_399, permute_400, permute_401, permute_402, permute_403, permute_404, permute_405, permute_406, permute_407, permute_408, permute_409, permute_410, permute_411, permute_412, permute_413, permute_414, permute_415, permute_416, permute_417, permute_418, permute_419, permute_420, permute_421, permute_422, permute_423, permute_424, permute_425, permute_426, permute_427, permute_428, permute_429, permute_430, permute_431, permute_432, permute_433, permute_434, permute_435, permute_436, permute_437, permute_438, permute_439, permute_440, permute_441, permute_442, permute_443, permute_444, permute_445, permute_446, permute_447, permute_448, permute_449, permute_450, permute_451, permute_452, permute_453, permute_454, permute_455, permute_456, permute_457, permute_458, permute_459, permute_460, permute_461, permute_462, permute_463, permute_464, permute_465, permute_466, permute_467, permute_468, permute_469, permute_470, permute_471, permute_472, permute_473, permute_474, permute_475, permute_476, permute_477, permute_478, permute_479, permute_480, permute_481, permute_482, permute_483, permute_484, permute_485, permute_486, permute_487, permute_488, permute_489, permute_490, permute_491, permute_492, permute_493, permute_494, permute_495, permute_496, permute_497, permute_498, permute_499, permute_500, permute_501, permute_502, permute_503, permute_504, permute_505, permute_506, permute_507)



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
