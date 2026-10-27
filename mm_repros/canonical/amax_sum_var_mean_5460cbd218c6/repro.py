"""
Standalone repro captured via capture_hook.
Label: AllenaiLongformerBase
Pattern hash: 5460cbd218c6
Shape hash: ea3b7739
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
    def forward(self, arg0_1: "b8[8, 1024]", arg1_1: "bf16[8, 1024, 768]", arg2_1: "bf16[768, 768]", arg3_1: "bf16[768]", arg4_1: "bf16[768, 768]", arg5_1: "bf16[768]", arg6_1: "bf16[8, 1024]", arg7_1: "bf16[768, 768]", arg8_1: "bf16[768]", arg9_1: "bf16[768, 768]", arg10_1: "bf16[768]", arg11_1: "bf16[768]", arg12_1: "bf16[768]", arg13_1: "bf16[3072, 768]", arg14_1: "bf16[3072]", arg15_1: "bf16[768, 3072]", arg16_1: "bf16[768]", arg17_1: "bf16[768]", arg18_1: "bf16[768]", arg19_1: "bf16[768, 768]", arg20_1: "bf16[768]", arg21_1: "bf16[768, 768]", arg22_1: "bf16[768]", arg23_1: "bf16[768, 768]", arg24_1: "bf16[768]", arg25_1: "bf16[768, 768]", arg26_1: "bf16[768]", arg27_1: "bf16[768]", arg28_1: "bf16[768]", arg29_1: "bf16[3072, 768]", arg30_1: "bf16[3072]", arg31_1: "bf16[768, 3072]", arg32_1: "bf16[768]", arg33_1: "bf16[768]", arg34_1: "bf16[768]", arg35_1: "bf16[768, 768]", arg36_1: "bf16[768]", arg37_1: "bf16[768, 768]", arg38_1: "bf16[768]", arg39_1: "bf16[768, 768]", arg40_1: "bf16[768]", arg41_1: "bf16[768, 768]", arg42_1: "bf16[768]", arg43_1: "bf16[768]", arg44_1: "bf16[768]", arg45_1: "bf16[3072, 768]", arg46_1: "bf16[3072]", arg47_1: "bf16[768, 3072]", arg48_1: "bf16[768]", arg49_1: "bf16[768]", arg50_1: "bf16[768]", arg51_1: "bf16[768, 768]", arg52_1: "bf16[768]", arg53_1: "bf16[768, 768]", arg54_1: "bf16[768]", arg55_1: "bf16[768, 768]", arg56_1: "bf16[768]", arg57_1: "bf16[768, 768]", arg58_1: "bf16[768]", arg59_1: "bf16[768]", arg60_1: "bf16[768]", arg61_1: "bf16[3072, 768]", arg62_1: "bf16[3072]", arg63_1: "bf16[768, 3072]", arg64_1: "bf16[768]", arg65_1: "bf16[768]", arg66_1: "bf16[768]", arg67_1: "bf16[768, 768]", arg68_1: "bf16[768]", arg69_1: "bf16[768, 768]", arg70_1: "bf16[768]", arg71_1: "bf16[768, 768]", arg72_1: "bf16[768]", arg73_1: "bf16[768, 768]", arg74_1: "bf16[768]", arg75_1: "bf16[768]", arg76_1: "bf16[768]", arg77_1: "bf16[3072, 768]", arg78_1: "bf16[3072]", arg79_1: "bf16[768, 3072]", arg80_1: "bf16[768]", arg81_1: "bf16[768]", arg82_1: "bf16[768]", arg83_1: "bf16[768, 768]", arg84_1: "bf16[768]", arg85_1: "bf16[768, 768]", arg86_1: "bf16[768]", arg87_1: "bf16[768, 768]", arg88_1: "bf16[768]", arg89_1: "bf16[768, 768]", arg90_1: "bf16[768]", arg91_1: "bf16[768]", arg92_1: "bf16[768]", arg93_1: "bf16[3072, 768]", arg94_1: "bf16[3072]", arg95_1: "bf16[768, 3072]", arg96_1: "bf16[768]", arg97_1: "bf16[768]", arg98_1: "bf16[768]", arg99_1: "bf16[768, 768]", arg100_1: "bf16[768]", arg101_1: "bf16[768, 768]", arg102_1: "bf16[768]", arg103_1: "bf16[768, 768]", arg104_1: "bf16[768]", arg105_1: "bf16[768, 768]", arg106_1: "bf16[768]", arg107_1: "bf16[768]", arg108_1: "bf16[768]", arg109_1: "bf16[3072, 768]", arg110_1: "bf16[3072]", arg111_1: "bf16[768, 3072]", arg112_1: "bf16[768]", arg113_1: "bf16[768]", arg114_1: "bf16[768]", arg115_1: "bf16[768, 768]", arg116_1: "bf16[768]", arg117_1: "bf16[768, 768]", arg118_1: "bf16[768]", arg119_1: "bf16[768, 768]", arg120_1: "bf16[768]", arg121_1: "bf16[768, 768]", arg122_1: "bf16[768]", arg123_1: "bf16[768]", arg124_1: "bf16[768]", arg125_1: "bf16[3072, 768]", arg126_1: "bf16[3072]", arg127_1: "bf16[768, 3072]", arg128_1: "bf16[768]", arg129_1: "bf16[768]", arg130_1: "bf16[768]", arg131_1: "bf16[768, 768]", arg132_1: "bf16[768]", arg133_1: "bf16[768, 768]", arg134_1: "bf16[768]", arg135_1: "bf16[768, 768]", arg136_1: "bf16[768]", arg137_1: "bf16[768, 768]", arg138_1: "bf16[768]", arg139_1: "bf16[768]", arg140_1: "bf16[768]", arg141_1: "bf16[3072, 768]", arg142_1: "bf16[3072]", arg143_1: "bf16[768, 3072]", arg144_1: "bf16[768]", arg145_1: "bf16[768]", arg146_1: "bf16[768]", arg147_1: "bf16[768, 768]", arg148_1: "bf16[768]", arg149_1: "bf16[768, 768]", arg150_1: "bf16[768]", arg151_1: "bf16[768, 768]", arg152_1: "bf16[768]", arg153_1: "bf16[768, 768]", arg154_1: "bf16[768]", arg155_1: "bf16[768]", arg156_1: "bf16[768]", arg157_1: "bf16[3072, 768]", arg158_1: "bf16[3072]", arg159_1: "bf16[768, 3072]", arg160_1: "bf16[768]", arg161_1: "bf16[768]", arg162_1: "bf16[768]", arg163_1: "bf16[768, 768]", arg164_1: "bf16[768]", arg165_1: "bf16[768, 768]", arg166_1: "bf16[768]", arg167_1: "bf16[768, 768]", arg168_1: "bf16[768]", arg169_1: "bf16[768, 768]", arg170_1: "bf16[768]", arg171_1: "bf16[768]", arg172_1: "bf16[768]", arg173_1: "bf16[3072, 768]", arg174_1: "bf16[3072]", arg175_1: "bf16[768, 3072]", arg176_1: "bf16[768]", arg177_1: "bf16[768]", arg178_1: "bf16[768]", arg179_1: "bf16[768, 768]", arg180_1: "bf16[768]", arg181_1: "bf16[768, 768]", arg182_1: "bf16[768]", arg183_1: "bf16[768, 768]", arg184_1: "bf16[768]", arg185_1: "bf16[768, 768]", arg186_1: "bf16[768]", arg187_1: "bf16[768]", arg188_1: "bf16[768]", arg189_1: "bf16[3072, 768]", arg190_1: "bf16[3072]", arg191_1: "bf16[768, 3072]", arg192_1: "bf16[768]", arg193_1: "bf16[768]", arg194_1: "bf16[768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631, _shape_param_632, _shape_param_633, _shape_param_634, _shape_param_635, _shape_param_636, _shape_param_637, _shape_param_638, _shape_param_639, _shape_param_640, _shape_param_641, _shape_param_642, _shape_param_643, _shape_param_644, _shape_param_645, _shape_param_646, _shape_param_647, _shape_param_648, _shape_param_649, _shape_param_650, _shape_param_651, _shape_param_652, _shape_param_653, _shape_param_654, _shape_param_655, _shape_param_656, _shape_param_657, _shape_param_658, _shape_param_659, _shape_param_660, _shape_param_661, _shape_param_662, _shape_param_663, _shape_param_664, _shape_param_665, _shape_param_666, _shape_param_667, _shape_param_668, _shape_param_669, _shape_param_670, _shape_param_671, _shape_param_672, _shape_param_673, _shape_param_674, _shape_param_675, _shape_param_676, _shape_param_677, _shape_param_678, _shape_param_679, _shape_param_680, _shape_param_681, _shape_param_682, _shape_param_683, _shape_param_684, _shape_param_685, _shape_param_686, _shape_param_687, _shape_param_688, _shape_param_689, _shape_param_690, _shape_param_691, _shape_param_692, _shape_param_693, _shape_param_694, _shape_param_695, _shape_param_696, _shape_param_697, _shape_param_698, _shape_param_699, _shape_param_700, _shape_param_701, _shape_param_702, _shape_param_703, _shape_param_704, _shape_param_705, _shape_param_706, _shape_param_707, _shape_param_708, _shape_param_709, _shape_param_710, _shape_param_711, _shape_param_712, _shape_param_713, _shape_param_714, _shape_param_715, _shape_param_716, _shape_param_717, _shape_param_718, _shape_param_719, _shape_param_720, _shape_param_721, _shape_param_722, _shape_param_723, _shape_param_724, _shape_param_725, _shape_param_726, _shape_param_727, _shape_param_728, _shape_param_729, _shape_param_730, _shape_param_731, _shape_param_732, _shape_param_733, _shape_param_734, _shape_param_735, _shape_param_736, _shape_param_737, _shape_param_738, _shape_param_739, _shape_param_740, _shape_param_741, _shape_param_742, _shape_param_743, _shape_param_744, _shape_param_745, _shape_param_746, _shape_param_747, _shape_param_748, _shape_param_749, _shape_param_750, _shape_param_751, _shape_param_752, _shape_param_753, _shape_param_754, _shape_param_755, _shape_param_756, _shape_param_757, _shape_param_758, _shape_param_759, _shape_param_760, _shape_param_761, _shape_param_762, _shape_param_763, _shape_param_764, _shape_param_765, _shape_param_766, _shape_param_767, _shape_param_768, _shape_param_769, _shape_param_770, _shape_param_771, _shape_param_772, _shape_param_773, _shape_param_774, _shape_param_775, _shape_param_776, _shape_param_777, _shape_param_778, _shape_param_779, _shape_param_780, _shape_param_781, _shape_param_782, _shape_param_783, _shape_param_784, _shape_param_785, _shape_param_786, _shape_param_787, _shape_param_788, _shape_param_789, _shape_param_790, _shape_param_791, _shape_param_792, _shape_param_793, _shape_param_794, _shape_param_795, _shape_param_796, _shape_param_797, _shape_param_798, _shape_param_799, _shape_param_800, _shape_param_801, _shape_param_802, _shape_param_803, _shape_param_804, _shape_param_805, _shape_param_806, _shape_param_807, _shape_param_808, _shape_param_809, _shape_param_810, _shape_param_811, _shape_param_812, _shape_param_813, _shape_param_814, _shape_param_815, _shape_param_816, _shape_param_817, _shape_param_818, _shape_param_819, _shape_param_820, _shape_param_821, _shape_param_822, _shape_param_823, _shape_param_824, _shape_param_825, _shape_param_826, _shape_param_827, _shape_param_828, _shape_param_829, _shape_param_830, _shape_param_831, _shape_param_832, _shape_param_833, _shape_param_834, _shape_param_835, _shape_param_836, _shape_param_837, _shape_param_838, _shape_param_839, _shape_param_840, _shape_param_841, _shape_param_842, _shape_param_843, _shape_param_844, _shape_param_845, _shape_param_846, _shape_param_847, _shape_param_848, _shape_param_849, _shape_param_850, _shape_param_851, _shape_param_852, _shape_param_853, _shape_param_854, _shape_param_855, _shape_param_856, _shape_param_857, _shape_param_858, _shape_param_859, _shape_param_860, _shape_param_861, _shape_param_862, _shape_param_863, _shape_param_864, _shape_param_865, _shape_param_866, _shape_param_867, _shape_param_868, _shape_param_869, _shape_param_870, _shape_param_871, _shape_param_872, _shape_param_873, _shape_param_874, _shape_param_875, _shape_param_876, _shape_param_877, _shape_param_878, _shape_param_879, _shape_param_880, _shape_param_881, _shape_param_882, _shape_param_883, _shape_param_884, _shape_param_885, _shape_param_886, _shape_param_887, _shape_param_888, _shape_param_889, _shape_param_890, _shape_param_891, _shape_param_892, _shape_param_893, _shape_param_894, _shape_param_895, _shape_param_896, _shape_param_897, _shape_param_898, _shape_param_899, _shape_param_900, _shape_param_901, _shape_param_902, _shape_param_903, _shape_param_904, _shape_param_905, _shape_param_906, _shape_param_907, _shape_param_908, _shape_param_909, _shape_param_910, _shape_param_911, _shape_param_912, _shape_param_913, _shape_param_914, _shape_param_915, _shape_param_916, _shape_param_917, _shape_param_918, _shape_param_919, _shape_param_920, _shape_param_921, _shape_param_922, _shape_param_923, _shape_param_924, _shape_param_925, _shape_param_926, _shape_param_927, _shape_param_928, _shape_param_929, _shape_param_930, _shape_param_931, _shape_param_932, _shape_param_933, _shape_param_934, _shape_param_935, _shape_param_936, _shape_param_937, _shape_param_938, _shape_param_939, _shape_param_940, _shape_param_941, _shape_param_942, _shape_param_943, _shape_param_944, _shape_param_945, _shape_param_946, _shape_param_947):
        # No stacktrace found for following nodes
        unsqueeze: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 2)
        unsqueeze_1: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 3);  unsqueeze = None
        full: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        slice_1: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_1, 1, 0, -1)
        slice_2: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_1, 1, 0, -1)
        slice_3: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2, 3, 256, 9223372036854775807);  slice_2 = None
        permute: "bf16[1024, 8, 768]" = torch.ops.aten.permute.default(arg1_1, [1, 0, 2])
        clone: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
        view: "bf16[8192, 768]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        permute_1: "bf16[768, 768]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(view, permute_1);  view = permute_1 = None
        view_1: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        add: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_1, arg3_1);  view_1 = arg3_1 = None
        div: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add, 8.0);  add = None
        view_2: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div, _shape_param_3);  div = _shape_param_3 = None
        permute_2: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2, [1, 0, 2, 3]);  view_2 = None
        permute_3: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_2, [0, 2, 1, 3]);  permute_2 = None
        view_3: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_3, _shape_param_4);  permute_3 = _shape_param_4 = None
        view_4: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_3, _shape_param_5);  view_3 = _shape_param_5 = None
        as_strided: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_4, _shape_param_6, _shape_param_7);  view_4 = _shape_param_6 = _shape_param_7 = None
        unsqueeze_2: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided, 4);  as_strided = None
        clone_1: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_2, memory_format = torch.contiguous_format);  unsqueeze_2 = None
        view_5: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_8);  clone_1 = _shape_param_8 = None
        clone_2: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format)
        view_6: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_2, _shape_param_9);  clone_2 = _shape_param_9 = None
        permute_4: "bf16[768, 768]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm_1: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_6, permute_4);  view_6 = permute_4 = None
        view_7: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_1, _shape_param_10);  mm_1 = _shape_param_10 = None
        add_1: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_7, arg5_1);  view_7 = arg5_1 = None
        view_8: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_1, _shape_param_11);  add_1 = _shape_param_11 = None
        permute_5: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_8, [1, 0, 2, 3]);  view_8 = None
        permute_6: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_5, [0, 2, 1, 3]);  permute_5 = None
        view_9: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_6, _shape_param_12);  permute_6 = _shape_param_12 = None
        view_10: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_9, _shape_param_13);  view_9 = _shape_param_13 = None
        as_strided_1: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_10, _shape_param_14, _shape_param_15);  view_10 = _shape_param_14 = _shape_param_15 = None
        unsqueeze_3: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_1, 4);  as_strided_1 = None
        permute_7: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_3, [0, 1, 4, 2, 3]);  unsqueeze_3 = None
        permute_8: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_7, [0, 1, 4, 3, 2]);  permute_7 = None
        clone_3: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_11: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_3, _shape_param_16);  clone_3 = _shape_param_16 = None
        bmm: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_5, view_11);  view_5 = view_11 = None
        view_12: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm, _shape_param_17);  bmm = _shape_param_17 = None
        permute_9: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_12, [0, 1, 2, 4, 3]);  view_12 = None
        view_13: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_9, _shape_param_18);  permute_9 = _shape_param_18 = None
        constant_pad_nd: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_13, [0, 0, 0, 1], 0.0);  view_13 = None
        view_14: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd, _shape_param_19);  constant_pad_nd = _shape_param_19 = None
        slice_4: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_14, 2, 0, 256)
        slice_5: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_4, 3, 0, 257);  slice_4 = None
        copy: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_3, slice_5);  slice_3 = slice_5 = None
        slice_scatter: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1, copy, 3, 256, 9223372036854775807);  slice_1 = copy = None
        slice_scatter_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_1, slice_scatter, 1, 0, -1);  full_1 = slice_scatter = None
        select: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
        select_1: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_1, 1, -1)
        slice_6: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_1, 2, 256, 9223372036854775807);  select_1 = None
        select_2: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_14, 1, -1)
        slice_7: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_2, 1, 256, 9223372036854775807);  select_2 = None
        slice_8: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_7, 2, 0, 257);  slice_7 = None
        copy_1: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_6, slice_8);  slice_6 = slice_8 = None
        slice_scatter_2: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select, copy_1, 2, 256, 9223372036854775807);  select = copy_1 = None
        select_scatter: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_1, slice_scatter_2, 1, -1);  slice_scatter_1 = slice_scatter_2 = None
        slice_9: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_10: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter, 1, 1, 9223372036854775807)
        slice_11: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_10, 3, 0, 256);  slice_10 = None
        slice_12: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_14, 2, -257, -1)
        slice_13: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_12, 3, 257, 9223372036854775807);  slice_12 = None
        copy_2: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_11, slice_13);  slice_11 = slice_13 = None
        slice_scatter_3: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_9, copy_2, 3, 0, 256);  slice_9 = copy_2 = None
        slice_scatter_4: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter, slice_scatter_3, 1, 1, 9223372036854775807);  select_scatter = slice_scatter_3 = None
        select_3: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_4, 1, 0)
        slice_14: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_3, 1, 1, 256)
        select_4: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_4, 1, 0)
        slice_15: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_4, 1, 1, 256);  select_4 = None
        slice_16: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_15, 2, 1, 256);  slice_15 = None
        select_5: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_14, 1, 0);  view_14 = None
        slice_17: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_5, 1, 0, 255);  select_5 = None
        slice_18: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_17, 2, -255, 9223372036854775807);  slice_17 = None
        copy_3: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_16, slice_18);  slice_16 = slice_18 = None
        slice_scatter_5: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_14, copy_3, 2, 1, 256);  slice_14 = copy_3 = None
        slice_scatter_6: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_3, slice_scatter_5, 1, 1, 256);  select_3 = slice_scatter_5 = None
        select_scatter_1: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_4, slice_scatter_6, 1, 0);  slice_scatter_4 = slice_scatter_6 = None
        view_15: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_1, _shape_param_20);  _shape_param_20 = None
        permute_10: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        slice_19: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_10, 1, 0, 256)
        view_16: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_1, _shape_param_21);  _shape_param_21 = None
        permute_11: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        slice_20: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_11, 1, 0, 256);  permute_11 = None
        slice_21: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_20, 3, 0, 257);  slice_20 = None
        iota: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_4: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota, -2);  iota = None
        iota_1: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_5: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
        sub: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_4, unsqueeze_5);  unsqueeze_4 = unsqueeze_5 = None
        le: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub, 0);  sub = None
        full_2: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_22, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_22 = None
        scalar_tensor: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where: "bf16[256, 257]" = torch.ops.aten.where.self(le, full_2, scalar_tensor);  le = full_2 = scalar_tensor = None
        rev: "bf16[256, 257]" = torch.ops.prims.rev.default(where, [0]);  where = None
        unsqueeze_6: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev, 0);  rev = None
        unsqueeze_7: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        expand: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_7, _shape_param_23);  _shape_param_23 = None
        convert_element_type: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand, torch.bool);  expand = None
        full_3: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_24, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_24 = None
        permute_12: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_3, [0, 2, 1, 3]);  full_3 = None
        view_17: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_1, _shape_param_25);  select_scatter_1 = _shape_param_25 = None
        permute_13: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        slice_22: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_13, 1, 0, 256);  permute_13 = None
        slice_23: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_22, 3, 0, 257);  slice_22 = None
        where_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, permute_12, slice_23);  convert_element_type = permute_12 = slice_23 = None
        copy_4: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_21, where_1);  slice_21 = where_1 = None
        slice_scatter_7: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_19, copy_4, 3, 0, 257);  slice_19 = copy_4 = None
        slice_scatter_8: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_10, slice_scatter_7, 1, 0, 256);  permute_10 = slice_scatter_7 = None
        permute_14: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_8, [0, 2, 1, 3]);  slice_scatter_8 = None
        view_18: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_14, _shape_param_26);  permute_14 = _shape_param_26 = None
        view_19: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_18, _shape_param_27);  _shape_param_27 = None
        permute_15: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None
        slice_24: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_15, 1, -256, 9223372036854775807)
        view_20: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_18, _shape_param_28);  _shape_param_28 = None
        permute_16: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
        slice_25: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_16, 1, -256, 9223372036854775807);  permute_16 = None
        slice_26: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_25, 3, -257, 9223372036854775807);  slice_25 = None
        rev_1: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_7, [1, 3]);  unsqueeze_7 = None
        expand_1: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_1, _shape_param_29);  rev_1 = _shape_param_29 = None
        convert_element_type_1: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_1, torch.bool);  expand_1 = None
        full_4: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_30, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_30 = None
        permute_17: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_4, [0, 2, 1, 3]);  full_4 = None
        view_21: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_18, _shape_param_31);  view_18 = _shape_param_31 = None
        permute_18: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        slice_27: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_18, 1, -256, 9223372036854775807);  permute_18 = None
        slice_28: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_27, 3, -257, 9223372036854775807);  slice_27 = None
        where_2: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, permute_17, slice_28);  convert_element_type_1 = permute_17 = slice_28 = None
        copy_5: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_26, where_2);  slice_26 = where_2 = None
        slice_scatter_9: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_24, copy_5, 3, -257, 9223372036854775807);  slice_24 = copy_5 = None
        slice_scatter_10: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_15, slice_scatter_9, 1, -256, 9223372036854775807);  permute_15 = slice_scatter_9 = None
        permute_19: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_10, [0, 2, 1, 3]);  slice_scatter_10 = None
        permute_20: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_19, [0, 2, 1, 3]);  permute_19 = None
        full_5: "bf16[8, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_32, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_32 = None
        slice_29: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_5, 1, 0, -1)
        slice_30: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_5, 1, 0, -1)
        slice_31: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_30, 3, 256, 9223372036854775807);  slice_30 = None
        full_6: "bf16[8, 1024, 1, 1]" = torch.ops.aten.full.default(_shape_param_33, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_33 = None
        permute_21: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(full_6, [0, 2, 1, 3]);  full_6 = None
        view_22: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_21, _shape_param_34);  permute_21 = _shape_param_34 = None
        view_23: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_22, _shape_param_35);  view_22 = _shape_param_35 = None
        as_strided_2: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_23, _shape_param_36, _shape_param_37);  view_23 = _shape_param_36 = _shape_param_37 = None
        unsqueeze_8: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_2, 4);  as_strided_2 = None
        permute_22: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_8, [0, 1, 2, 4, 3]);  unsqueeze_8 = None
        full_7: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ne: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg6_1, 0)
        unsqueeze_9: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne, 2);  ne = None
        unsqueeze_10: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 3);  unsqueeze_9 = None
        convert_element_type_2: "bf16[8, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_10, torch.bfloat16)
        where_3: "bf16[8, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_10, full_7, convert_element_type_2);  unsqueeze_10 = full_7 = convert_element_type_2 = None
        permute_23: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(where_3, [0, 2, 1, 3]);  where_3 = None
        view_24: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_23, _shape_param_38);  permute_23 = _shape_param_38 = None
        view_25: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_24, _shape_param_39);  view_24 = _shape_param_39 = None
        as_strided_3: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_25, _shape_param_40, _shape_param_41);  view_25 = _shape_param_40 = _shape_param_41 = None
        unsqueeze_11: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_3, 4);  as_strided_3 = None
        permute_24: "bf16[8, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_11, [0, 1, 4, 2, 3]);  unsqueeze_11 = None
        mul: "bf16[8, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_22, permute_24);  permute_22 = permute_24 = None
        view_26: "bf16[8, 3, 512, 512]" = torch.ops.aten.view.default(mul, _shape_param_42);  mul = _shape_param_42 = None
        constant_pad_nd_1: "bf16[8, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_26, [0, 0, 0, 1], 0.0);  view_26 = None
        view_27: "bf16[8, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_1, _shape_param_43);  constant_pad_nd_1 = _shape_param_43 = None
        slice_32: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_27, 2, 0, 256)
        slice_33: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_32, 3, 0, 257);  slice_32 = None
        copy_6: "bf16[8, 3, 256, 257]" = torch.ops.aten.copy.default(slice_31, slice_33);  slice_31 = slice_33 = None
        slice_scatter_11: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_29, copy_6, 3, 256, 9223372036854775807);  slice_29 = copy_6 = None
        slice_scatter_12: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_11, 1, 0, -1);  full_5 = slice_scatter_11 = None
        select_6: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_12, 1, -1)
        select_7: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_12, 1, -1)
        slice_34: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(select_7, 2, 256, 9223372036854775807);  select_7 = None
        select_8: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_27, 1, -1)
        slice_35: "bf16[8, 256, 513]" = torch.ops.aten.slice.Tensor(select_8, 1, 256, 9223372036854775807);  select_8 = None
        slice_36: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(slice_35, 2, 0, 257);  slice_35 = None
        copy_7: "bf16[8, 256, 257]" = torch.ops.aten.copy.default(slice_34, slice_36);  slice_34 = slice_36 = None
        slice_scatter_13: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_6, copy_7, 2, 256, 9223372036854775807);  select_6 = copy_7 = None
        select_scatter_2: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_12, slice_scatter_13, 1, -1);  slice_scatter_12 = slice_scatter_13 = None
        slice_37: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_2, 1, 1, 9223372036854775807)
        slice_38: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_2, 1, 1, 9223372036854775807)
        slice_39: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_38, 3, 0, 256);  slice_38 = None
        slice_40: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_27, 2, -257, -1)
        slice_41: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_40, 3, 257, 9223372036854775807);  slice_40 = None
        copy_8: "bf16[8, 3, 256, 256]" = torch.ops.aten.copy.default(slice_39, slice_41);  slice_39 = slice_41 = None
        slice_scatter_14: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_37, copy_8, 3, 0, 256);  slice_37 = copy_8 = None
        slice_scatter_15: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_2, slice_scatter_14, 1, 1, 9223372036854775807);  select_scatter_2 = slice_scatter_14 = None
        select_9: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_15, 1, 0)
        slice_42: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_9, 1, 1, 256)
        select_10: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_15, 1, 0)
        slice_43: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_10, 1, 1, 256);  select_10 = None
        slice_44: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_43, 2, 1, 256);  slice_43 = None
        select_11: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_27, 1, 0);  view_27 = None
        slice_45: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_11, 1, 0, 255);  select_11 = None
        slice_46: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_45, 2, -255, 9223372036854775807);  slice_45 = None
        copy_9: "bf16[8, 255, 255]" = torch.ops.aten.copy.default(slice_44, slice_46);  slice_44 = slice_46 = None
        slice_scatter_16: "bf16[8, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_42, copy_9, 2, 1, 256);  slice_42 = copy_9 = None
        slice_scatter_17: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_9, slice_scatter_16, 1, 1, 256);  select_9 = slice_scatter_16 = None
        select_scatter_3: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_15, slice_scatter_17, 1, 0);  slice_scatter_15 = slice_scatter_17 = None
        view_28: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_3, _shape_param_44);  _shape_param_44 = None
        permute_25: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        slice_47: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_25, 1, 0, 256)
        view_29: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_3, _shape_param_45);  _shape_param_45 = None
        permute_26: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        slice_48: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_26, 1, 0, 256);  permute_26 = None
        slice_49: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_48, 3, 0, 257);  slice_48 = None
        iota_2: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_12: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_2, -2);  iota_2 = None
        iota_3: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_13: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_3, -1);  iota_3 = None
        sub_1: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_12, unsqueeze_13);  unsqueeze_12 = unsqueeze_13 = None
        le_1: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_1, 0);  sub_1 = None
        full_8: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_46, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_46 = None
        scalar_tensor_1: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_4: "bf16[256, 257]" = torch.ops.aten.where.self(le_1, full_8, scalar_tensor_1);  le_1 = full_8 = scalar_tensor_1 = None
        rev_2: "bf16[256, 257]" = torch.ops.prims.rev.default(where_4, [0]);  where_4 = None
        unsqueeze_14: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_2, 0);  rev_2 = None
        unsqueeze_15: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 2);  unsqueeze_14 = None
        expand_2: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_15, _shape_param_47);  _shape_param_47 = None
        convert_element_type_3: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_2, torch.bool);  expand_2 = None
        full_9: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_48, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_48 = None
        view_30: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_3, _shape_param_49);  select_scatter_3 = _shape_param_49 = None
        permute_27: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        slice_50: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_27, 1, 0, 256);  permute_27 = None
        slice_51: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_50, 3, 0, 257);  slice_50 = None
        where_5: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_3, full_9, slice_51);  convert_element_type_3 = full_9 = slice_51 = None
        copy_10: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_49, where_5);  slice_49 = where_5 = None
        slice_scatter_18: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_47, copy_10, 3, 0, 257);  slice_47 = copy_10 = None
        slice_scatter_19: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_25, slice_scatter_18, 1, 0, 256);  permute_25 = slice_scatter_18 = None
        permute_28: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_19, [0, 2, 1, 3]);  slice_scatter_19 = None
        view_31: "bf16[8, 4, 256, 513]" = torch.ops.aten.view.default(permute_28, _shape_param_50);  permute_28 = _shape_param_50 = None
        view_32: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_31, _shape_param_51);  _shape_param_51 = None
        permute_29: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        slice_52: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_29, 1, -256, 9223372036854775807)
        view_33: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_31, _shape_param_52);  _shape_param_52 = None
        permute_30: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
        slice_53: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_30, 1, -256, 9223372036854775807);  permute_30 = None
        slice_54: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_53, 3, -257, 9223372036854775807);  slice_53 = None
        rev_3: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_15, [1, 3]);  unsqueeze_15 = None
        expand_3: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(rev_3, _shape_param_53);  rev_3 = _shape_param_53 = None
        convert_element_type_4: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_3, torch.bool);  expand_3 = None
        full_10: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_54, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_54 = None
        view_34: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_31, _shape_param_55);  view_31 = _shape_param_55 = None
        permute_31: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        slice_55: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_31, 1, -256, 9223372036854775807);  permute_31 = None
        slice_56: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_55, 3, -257, 9223372036854775807);  slice_55 = None
        where_6: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_4, full_10, slice_56);  convert_element_type_4 = full_10 = slice_56 = None
        copy_11: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_54, where_6);  slice_54 = where_6 = None
        slice_scatter_20: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_52, copy_11, 3, -257, 9223372036854775807);  slice_52 = copy_11 = None
        slice_scatter_21: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_29, slice_scatter_20, 1, -256, 9223372036854775807);  permute_29 = slice_scatter_20 = None
        permute_32: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_21, [0, 2, 1, 3]);  slice_scatter_21 = None
        permute_33: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_32, [0, 2, 1, 3]);  permute_32 = None
        add_2: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_20, permute_33);  permute_20 = permute_33 = None
        permute_34: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_2, [0, 2, 1, 3]);  add_2 = None
        permute_35: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_34, [0, 2, 1, 3]);  permute_34 = None
        convert_element_type_5: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_35, torch.float32);  permute_35 = None
        clone_4: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_5, memory_format = torch.contiguous_format);  convert_element_type_5 = None
        amax: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_4, [-1], True)
        sub_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_4, amax);  clone_4 = amax = None
        exp: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        where_7: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_1, full, div_1);  unsqueeze_1 = full = div_1 = None
        convert_element_type_6: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_7, torch.bfloat16);  where_7 = None
        permute_36: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_6, [0, 2, 1, 3]);  convert_element_type_6 = None
        clone_5: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        view_35: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_5, _shape_param_56);  clone_5 = _shape_param_56 = None
        constant_pad_nd_2: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_35, _shape_param_57, 0.0);  view_35 = _shape_param_57 = None
        view_36: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_2, _shape_param_58);  constant_pad_nd_2 = _shape_param_58 = None
        slice_57: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_36, 2, 0, -256);  view_36 = None
        view_37: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_57, _shape_param_59);  slice_57 = _shape_param_59 = None
        slice_58: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_37, 3, 0, -1);  view_37 = None
        unsqueeze_16: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_58, 4);  slice_58 = None
        view_38: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_16, _shape_param_60);  unsqueeze_16 = _shape_param_60 = None
        clone_6: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_39: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_6, _shape_param_61);  clone_6 = _shape_param_61 = None
        permute_37: "bf16[768, 768]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_2: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_39, permute_37);  view_39 = permute_37 = None
        view_40: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_2, _shape_param_62);  mm_2 = _shape_param_62 = None
        add_3: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_40, arg8_1);  view_40 = arg8_1 = None
        view_41: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_3, _shape_param_63);  add_3 = _shape_param_63 = None
        permute_38: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_41, [1, 0, 2, 3]);  view_41 = None
        permute_39: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_38, [0, 2, 1, 3]);  permute_38 = None
        view_42: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_39, _shape_param_64);  permute_39 = _shape_param_64 = None
        constant_pad_nd_3: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_42, _shape_param_65, -1.0);  view_42 = _shape_param_65 = None
        as_strided_4: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_3, _shape_param_66, _shape_param_67);  constant_pad_nd_3 = _shape_param_66 = _shape_param_67 = None
        unsqueeze_17: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_4, 4);  as_strided_4 = None
        clone_7: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_17, memory_format = torch.contiguous_format);  unsqueeze_17 = None
        view_43: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_7, _shape_param_68);  clone_7 = _shape_param_68 = None
        bmm_1: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_38, view_43);  view_38 = view_43 = None
        view_44: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_69);  bmm_1 = _shape_param_69 = None
        permute_40: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_44, [0, 1, 2, 4, 3]);  view_44 = None
        view_45: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_40, _shape_param_70);  permute_40 = _shape_param_70 = None
        view_46: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_45, _shape_param_71);  view_45 = _shape_param_71 = None
        permute_41: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        permute_42: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_41, [1, 0, 2, 3]);  permute_41 = None
        clone_8: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
        view_47: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_8, _shape_param_72);  clone_8 = _shape_param_72 = None
        permute_43: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_47, [1, 0, 2]);  view_47 = None
        clone_9: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
        view_48: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_9, _shape_param_73);  clone_9 = _shape_param_73 = None
        permute_44: "bf16[768, 768]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_3: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_48, permute_44);  view_48 = permute_44 = None
        view_49: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_3, _shape_param_74);  mm_3 = _shape_param_74 = None
        add_4: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_49, arg10_1);  view_49 = arg10_1 = None
        add_5: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_4, arg1_1);  add_4 = arg1_1 = None
        convert_element_type_7: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_5, torch.float32);  add_5 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_7, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1024, 1]" = var_mean[0]
        getitem_1: "f32[8, 1024, 1]" = var_mean[1];  var_mean = None
        sub_3: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_7, getitem_1);  convert_element_type_7 = getitem_1 = None
        add_6: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt);  sub_3 = rsqrt = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1, arg11_1);  mul_1 = arg11_1 = None
        add_7: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_2, arg12_1);  mul_2 = arg12_1 = None
        convert_element_type_8: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None
        view_50: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_8, _shape_param_75);  _shape_param_75 = None
        permute_45: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(arg14_1, view_50, permute_45);  arg14_1 = view_50 = permute_45 = None
        view_51: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm, _shape_param_76);  addmm = _shape_param_76 = None
        convert_element_type_9: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_51, torch.float32);  view_51 = None
        mul_3: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 0.5)
        mul_4: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 0.7071067811865476);  convert_element_type_9 = None
        erf: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_8: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_5: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_3, add_8);  mul_3 = add_8 = None
        convert_element_type_10: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        view_52: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_10, _shape_param_77);  convert_element_type_10 = _shape_param_77 = None
        permute_46: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_1: "bf16[8192, 768]" = torch.ops.aten.addmm.default(arg16_1, view_52, permute_46);  arg16_1 = view_52 = permute_46 = None
        view_53: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_1, _shape_param_78);  addmm_1 = _shape_param_78 = None
        add_9: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_53, convert_element_type_8);  view_53 = convert_element_type_8 = None
        convert_element_type_11: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_9, torch.float32);  add_9 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_11, [2], correction = 0, keepdim = True)
        getitem_2: "f32[8, 1024, 1]" = var_mean_1[0]
        getitem_3: "f32[8, 1024, 1]" = var_mean_1[1];  var_mean_1 = None
        unsqueeze_18: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 2)
        unsqueeze_19: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 3);  unsqueeze_18 = None
        full_11: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_12: "bf16[96, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_79, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_79 = None
        slice_59: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_12, 1, 0, -1)
        slice_60: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_12, 1, 0, -1)
        slice_61: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_60, 3, 256, 9223372036854775807);  slice_60 = None
        sub_4: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_11, getitem_3);  convert_element_type_11 = getitem_3 = None
        add_10: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_6: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = rsqrt_1 = None
        mul_7: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_6, arg17_1);  mul_6 = arg17_1 = None
        add_11: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_7, arg18_1);  mul_7 = arg18_1 = None
        convert_element_type_12: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None
        permute_47: "bf16[1024, 8, 768]" = torch.ops.aten.permute.default(convert_element_type_12, [1, 0, 2])
        clone_10: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format)
        view_54: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_10, _shape_param_80);  clone_10 = _shape_param_80 = None
        permute_48: "bf16[768, 768]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_4: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_54, permute_48);  view_54 = permute_48 = None
        view_55: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_4, _shape_param_81);  mm_4 = _shape_param_81 = None
        add_12: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_55, arg20_1);  view_55 = arg20_1 = None
        div_2: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_12, 8.0);  add_12 = None
        view_56: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_2, _shape_param_82);  div_2 = _shape_param_82 = None
        permute_49: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_56, [1, 0, 2, 3]);  view_56 = None
        permute_50: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_49, [0, 2, 1, 3]);  permute_49 = None
        view_57: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_50, _shape_param_83);  permute_50 = _shape_param_83 = None
        view_58: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_57, _shape_param_84);  view_57 = _shape_param_84 = None
        as_strided_5: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_58, _shape_param_85, _shape_param_86);  view_58 = _shape_param_85 = _shape_param_86 = None
        unsqueeze_20: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_5, 4);  as_strided_5 = None
        clone_11: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_20, memory_format = torch.contiguous_format);  unsqueeze_20 = None
        view_59: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_11, _shape_param_87);  clone_11 = _shape_param_87 = None
        clone_12: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format)
        view_60: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_12, _shape_param_88);  clone_12 = _shape_param_88 = None
        permute_51: "bf16[768, 768]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        mm_5: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_60, permute_51);  view_60 = permute_51 = None
        view_61: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_5, _shape_param_89);  mm_5 = _shape_param_89 = None
        add_13: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_61, arg22_1);  view_61 = arg22_1 = None
        view_62: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_13, _shape_param_90);  add_13 = _shape_param_90 = None
        permute_52: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_62, [1, 0, 2, 3]);  view_62 = None
        permute_53: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_52, [0, 2, 1, 3]);  permute_52 = None
        view_63: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_53, _shape_param_91);  permute_53 = _shape_param_91 = None
        view_64: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_63, _shape_param_92);  view_63 = _shape_param_92 = None
        as_strided_6: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_64, _shape_param_93, _shape_param_94);  view_64 = _shape_param_93 = _shape_param_94 = None
        unsqueeze_21: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_6, 4);  as_strided_6 = None
        permute_54: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_21, [0, 1, 4, 2, 3]);  unsqueeze_21 = None
        permute_55: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_54, [0, 1, 4, 3, 2]);  permute_54 = None
        clone_13: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_65: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_13, _shape_param_95);  clone_13 = _shape_param_95 = None
        bmm_2: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_59, view_65);  view_59 = view_65 = None
        view_66: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_2, _shape_param_96);  bmm_2 = _shape_param_96 = None
        permute_56: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_66, [0, 1, 2, 4, 3]);  view_66 = None
        view_67: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_56, _shape_param_97);  permute_56 = _shape_param_97 = None
        constant_pad_nd_4: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_67, [0, 0, 0, 1], 0.0);  view_67 = None
        view_68: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_4, _shape_param_98);  constant_pad_nd_4 = _shape_param_98 = None
        slice_62: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_68, 2, 0, 256)
        slice_63: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_62, 3, 0, 257);  slice_62 = None
        copy_12: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_61, slice_63);  slice_61 = slice_63 = None
        slice_scatter_22: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_59, copy_12, 3, 256, 9223372036854775807);  slice_59 = copy_12 = None
        slice_scatter_23: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_12, slice_scatter_22, 1, 0, -1);  full_12 = slice_scatter_22 = None
        select_12: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_23, 1, -1)
        select_13: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_23, 1, -1)
        slice_64: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_13, 2, 256, 9223372036854775807);  select_13 = None
        select_14: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_68, 1, -1)
        slice_65: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_14, 1, 256, 9223372036854775807);  select_14 = None
        slice_66: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_65, 2, 0, 257);  slice_65 = None
        copy_13: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_64, slice_66);  slice_64 = slice_66 = None
        slice_scatter_24: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_12, copy_13, 2, 256, 9223372036854775807);  select_12 = copy_13 = None
        select_scatter_4: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_23, slice_scatter_24, 1, -1);  slice_scatter_23 = slice_scatter_24 = None
        slice_67: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_4, 1, 1, 9223372036854775807)
        slice_68: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_4, 1, 1, 9223372036854775807)
        slice_69: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_68, 3, 0, 256);  slice_68 = None
        slice_70: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_68, 2, -257, -1)
        slice_71: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_70, 3, 257, 9223372036854775807);  slice_70 = None
        copy_14: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_69, slice_71);  slice_69 = slice_71 = None
        slice_scatter_25: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_67, copy_14, 3, 0, 256);  slice_67 = copy_14 = None
        slice_scatter_26: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_4, slice_scatter_25, 1, 1, 9223372036854775807);  select_scatter_4 = slice_scatter_25 = None
        select_15: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_26, 1, 0)
        slice_72: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_15, 1, 1, 256)
        select_16: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_26, 1, 0)
        slice_73: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_16, 1, 1, 256);  select_16 = None
        slice_74: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_73, 2, 1, 256);  slice_73 = None
        select_17: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_68, 1, 0);  view_68 = None
        slice_75: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_17, 1, 0, 255);  select_17 = None
        slice_76: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_75, 2, -255, 9223372036854775807);  slice_75 = None
        copy_15: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_74, slice_76);  slice_74 = slice_76 = None
        slice_scatter_27: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_72, copy_15, 2, 1, 256);  slice_72 = copy_15 = None
        slice_scatter_28: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_15, slice_scatter_27, 1, 1, 256);  select_15 = slice_scatter_27 = None
        select_scatter_5: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_26, slice_scatter_28, 1, 0);  slice_scatter_26 = slice_scatter_28 = None
        view_69: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_5, _shape_param_99);  _shape_param_99 = None
        permute_57: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        slice_77: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_57, 1, 0, 256)
        view_70: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_5, _shape_param_100);  _shape_param_100 = None
        permute_58: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        slice_78: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_58, 1, 0, 256);  permute_58 = None
        slice_79: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_78, 3, 0, 257);  slice_78 = None
        iota_4: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_22: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_4, -2);  iota_4 = None
        iota_5: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_23: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_5, -1);  iota_5 = None
        sub_5: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_22, unsqueeze_23);  unsqueeze_22 = unsqueeze_23 = None
        le_2: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_5, 0);  sub_5 = None
        full_13: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_101, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_101 = None
        scalar_tensor_2: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_8: "bf16[256, 257]" = torch.ops.aten.where.self(le_2, full_13, scalar_tensor_2);  le_2 = full_13 = scalar_tensor_2 = None
        rev_4: "bf16[256, 257]" = torch.ops.prims.rev.default(where_8, [0]);  where_8 = None
        unsqueeze_24: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_4, 0);  rev_4 = None
        unsqueeze_25: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, 2);  unsqueeze_24 = None
        expand_4: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_25, _shape_param_102);  _shape_param_102 = None
        convert_element_type_13: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_4, torch.bool);  expand_4 = None
        full_14: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_103, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_103 = None
        permute_59: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_14, [0, 2, 1, 3]);  full_14 = None
        view_71: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_5, _shape_param_104);  select_scatter_5 = _shape_param_104 = None
        permute_60: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        slice_80: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_60, 1, 0, 256);  permute_60 = None
        slice_81: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_80, 3, 0, 257);  slice_80 = None
        where_9: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_13, permute_59, slice_81);  convert_element_type_13 = permute_59 = slice_81 = None
        copy_16: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_79, where_9);  slice_79 = where_9 = None
        slice_scatter_29: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_77, copy_16, 3, 0, 257);  slice_77 = copy_16 = None
        slice_scatter_30: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_57, slice_scatter_29, 1, 0, 256);  permute_57 = slice_scatter_29 = None
        permute_61: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_30, [0, 2, 1, 3]);  slice_scatter_30 = None
        view_72: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_61, _shape_param_105);  permute_61 = _shape_param_105 = None
        view_73: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_72, _shape_param_106);  _shape_param_106 = None
        permute_62: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
        slice_82: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_62, 1, -256, 9223372036854775807)
        view_74: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_72, _shape_param_107);  _shape_param_107 = None
        permute_63: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        slice_83: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_63, 1, -256, 9223372036854775807);  permute_63 = None
        slice_84: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_83, 3, -257, 9223372036854775807);  slice_83 = None
        rev_5: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_25, [1, 3]);  unsqueeze_25 = None
        expand_5: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_5, _shape_param_108);  rev_5 = _shape_param_108 = None
        convert_element_type_14: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_5, torch.bool);  expand_5 = None
        full_15: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_109, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_109 = None
        permute_64: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_15, [0, 2, 1, 3]);  full_15 = None
        view_75: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_72, _shape_param_110);  view_72 = _shape_param_110 = None
        permute_65: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        slice_85: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_65, 1, -256, 9223372036854775807);  permute_65 = None
        slice_86: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_85, 3, -257, 9223372036854775807);  slice_85 = None
        where_10: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_14, permute_64, slice_86);  convert_element_type_14 = permute_64 = slice_86 = None
        copy_17: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_84, where_10);  slice_84 = where_10 = None
        slice_scatter_31: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_82, copy_17, 3, -257, 9223372036854775807);  slice_82 = copy_17 = None
        slice_scatter_32: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_62, slice_scatter_31, 1, -256, 9223372036854775807);  permute_62 = slice_scatter_31 = None
        permute_66: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_32, [0, 2, 1, 3]);  slice_scatter_32 = None
        permute_67: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_66, [0, 2, 1, 3]);  permute_66 = None
        full_16: "bf16[8, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_111, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_111 = None
        slice_87: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_16, 1, 0, -1)
        slice_88: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_16, 1, 0, -1)
        slice_89: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_88, 3, 256, 9223372036854775807);  slice_88 = None
        full_17: "bf16[8, 1024, 1, 1]" = torch.ops.aten.full.default(_shape_param_112, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_112 = None
        permute_68: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(full_17, [0, 2, 1, 3]);  full_17 = None
        view_76: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_68, _shape_param_113);  permute_68 = _shape_param_113 = None
        view_77: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_76, _shape_param_114);  view_76 = _shape_param_114 = None
        as_strided_7: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_77, _shape_param_115, _shape_param_116);  view_77 = _shape_param_115 = _shape_param_116 = None
        unsqueeze_26: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_7, 4);  as_strided_7 = None
        permute_69: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_26, [0, 1, 2, 4, 3]);  unsqueeze_26 = None
        full_18: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ne_1: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg6_1, 0)
        unsqueeze_27: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_1, 2);  ne_1 = None
        unsqueeze_28: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_27, 3);  unsqueeze_27 = None
        convert_element_type_15: "bf16[8, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_28, torch.bfloat16)
        where_11: "bf16[8, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_28, full_18, convert_element_type_15);  unsqueeze_28 = full_18 = convert_element_type_15 = None
        permute_70: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(where_11, [0, 2, 1, 3]);  where_11 = None
        view_78: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_70, _shape_param_117);  permute_70 = _shape_param_117 = None
        view_79: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_78, _shape_param_118);  view_78 = _shape_param_118 = None
        as_strided_8: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_79, _shape_param_119, _shape_param_120);  view_79 = _shape_param_119 = _shape_param_120 = None
        unsqueeze_29: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_8, 4);  as_strided_8 = None
        permute_71: "bf16[8, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_29, [0, 1, 4, 2, 3]);  unsqueeze_29 = None
        mul_8: "bf16[8, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_69, permute_71);  permute_69 = permute_71 = None
        view_80: "bf16[8, 3, 512, 512]" = torch.ops.aten.view.default(mul_8, _shape_param_121);  mul_8 = _shape_param_121 = None
        constant_pad_nd_5: "bf16[8, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_80, [0, 0, 0, 1], 0.0);  view_80 = None
        view_81: "bf16[8, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_5, _shape_param_122);  constant_pad_nd_5 = _shape_param_122 = None
        slice_90: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_81, 2, 0, 256)
        slice_91: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_90, 3, 0, 257);  slice_90 = None
        copy_18: "bf16[8, 3, 256, 257]" = torch.ops.aten.copy.default(slice_89, slice_91);  slice_89 = slice_91 = None
        slice_scatter_33: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_87, copy_18, 3, 256, 9223372036854775807);  slice_87 = copy_18 = None
        slice_scatter_34: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_16, slice_scatter_33, 1, 0, -1);  full_16 = slice_scatter_33 = None
        select_18: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_34, 1, -1)
        select_19: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_34, 1, -1)
        slice_92: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(select_19, 2, 256, 9223372036854775807);  select_19 = None
        select_20: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_81, 1, -1)
        slice_93: "bf16[8, 256, 513]" = torch.ops.aten.slice.Tensor(select_20, 1, 256, 9223372036854775807);  select_20 = None
        slice_94: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(slice_93, 2, 0, 257);  slice_93 = None
        copy_19: "bf16[8, 256, 257]" = torch.ops.aten.copy.default(slice_92, slice_94);  slice_92 = slice_94 = None
        slice_scatter_35: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_18, copy_19, 2, 256, 9223372036854775807);  select_18 = copy_19 = None
        select_scatter_6: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_34, slice_scatter_35, 1, -1);  slice_scatter_34 = slice_scatter_35 = None
        slice_95: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_6, 1, 1, 9223372036854775807)
        slice_96: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_6, 1, 1, 9223372036854775807)
        slice_97: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_96, 3, 0, 256);  slice_96 = None
        slice_98: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_81, 2, -257, -1)
        slice_99: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_98, 3, 257, 9223372036854775807);  slice_98 = None
        copy_20: "bf16[8, 3, 256, 256]" = torch.ops.aten.copy.default(slice_97, slice_99);  slice_97 = slice_99 = None
        slice_scatter_36: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_95, copy_20, 3, 0, 256);  slice_95 = copy_20 = None
        slice_scatter_37: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_6, slice_scatter_36, 1, 1, 9223372036854775807);  select_scatter_6 = slice_scatter_36 = None
        select_21: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_37, 1, 0)
        slice_100: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_21, 1, 1, 256)
        select_22: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_37, 1, 0)
        slice_101: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_22, 1, 1, 256);  select_22 = None
        slice_102: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_101, 2, 1, 256);  slice_101 = None
        select_23: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_81, 1, 0);  view_81 = None
        slice_103: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_23, 1, 0, 255);  select_23 = None
        slice_104: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_103, 2, -255, 9223372036854775807);  slice_103 = None
        copy_21: "bf16[8, 255, 255]" = torch.ops.aten.copy.default(slice_102, slice_104);  slice_102 = slice_104 = None
        slice_scatter_38: "bf16[8, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_100, copy_21, 2, 1, 256);  slice_100 = copy_21 = None
        slice_scatter_39: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_21, slice_scatter_38, 1, 1, 256);  select_21 = slice_scatter_38 = None
        select_scatter_7: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_37, slice_scatter_39, 1, 0);  slice_scatter_37 = slice_scatter_39 = None
        view_82: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_7, _shape_param_123);  _shape_param_123 = None
        permute_72: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        slice_105: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_72, 1, 0, 256)
        view_83: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_7, _shape_param_124);  _shape_param_124 = None
        permute_73: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
        slice_106: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_73, 1, 0, 256);  permute_73 = None
        slice_107: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_106, 3, 0, 257);  slice_106 = None
        iota_6: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_30: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_6, -2);  iota_6 = None
        iota_7: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_31: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_7, -1);  iota_7 = None
        sub_6: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_30, unsqueeze_31);  unsqueeze_30 = unsqueeze_31 = None
        le_3: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_6, 0);  sub_6 = None
        full_19: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_125, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_125 = None
        scalar_tensor_3: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_12: "bf16[256, 257]" = torch.ops.aten.where.self(le_3, full_19, scalar_tensor_3);  le_3 = full_19 = scalar_tensor_3 = None
        rev_6: "bf16[256, 257]" = torch.ops.prims.rev.default(where_12, [0]);  where_12 = None
        unsqueeze_32: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_6, 0);  rev_6 = None
        unsqueeze_33: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, 2);  unsqueeze_32 = None
        expand_6: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_33, _shape_param_126);  _shape_param_126 = None
        convert_element_type_16: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_6, torch.bool);  expand_6 = None
        full_20: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_127, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_127 = None
        view_84: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_7, _shape_param_128);  select_scatter_7 = _shape_param_128 = None
        permute_74: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        slice_108: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_74, 1, 0, 256);  permute_74 = None
        slice_109: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_108, 3, 0, 257);  slice_108 = None
        where_13: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_16, full_20, slice_109);  convert_element_type_16 = full_20 = slice_109 = None
        copy_22: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_107, where_13);  slice_107 = where_13 = None
        slice_scatter_40: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_105, copy_22, 3, 0, 257);  slice_105 = copy_22 = None
        slice_scatter_41: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_72, slice_scatter_40, 1, 0, 256);  permute_72 = slice_scatter_40 = None
        permute_75: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_41, [0, 2, 1, 3]);  slice_scatter_41 = None
        view_85: "bf16[8, 4, 256, 513]" = torch.ops.aten.view.default(permute_75, _shape_param_129);  permute_75 = _shape_param_129 = None
        view_86: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_85, _shape_param_130);  _shape_param_130 = None
        permute_76: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        slice_110: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_76, 1, -256, 9223372036854775807)
        view_87: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_85, _shape_param_131);  _shape_param_131 = None
        permute_77: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
        slice_111: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_77, 1, -256, 9223372036854775807);  permute_77 = None
        slice_112: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_111, 3, -257, 9223372036854775807);  slice_111 = None
        rev_7: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_33, [1, 3]);  unsqueeze_33 = None
        expand_7: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(rev_7, _shape_param_132);  rev_7 = _shape_param_132 = None
        convert_element_type_17: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_7, torch.bool);  expand_7 = None
        full_21: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_133, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_133 = None
        view_88: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_85, _shape_param_134);  view_85 = _shape_param_134 = None
        permute_78: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        slice_113: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_78, 1, -256, 9223372036854775807);  permute_78 = None
        slice_114: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_113, 3, -257, 9223372036854775807);  slice_113 = None
        where_14: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_17, full_21, slice_114);  convert_element_type_17 = full_21 = slice_114 = None
        copy_23: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_112, where_14);  slice_112 = where_14 = None
        slice_scatter_42: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_110, copy_23, 3, -257, 9223372036854775807);  slice_110 = copy_23 = None
        slice_scatter_43: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_76, slice_scatter_42, 1, -256, 9223372036854775807);  permute_76 = slice_scatter_42 = None
        permute_79: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_43, [0, 2, 1, 3]);  slice_scatter_43 = None
        permute_80: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_79, [0, 2, 1, 3]);  permute_79 = None
        add_14: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_67, permute_80);  permute_67 = permute_80 = None
        permute_81: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_14, [0, 2, 1, 3]);  add_14 = None
        permute_82: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_81, [0, 2, 1, 3]);  permute_81 = None
        convert_element_type_18: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_82, torch.float32);  permute_82 = None
        clone_14: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_18, memory_format = torch.contiguous_format);  convert_element_type_18 = None
        amax_1: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_14, [-1], True)
        sub_7: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_14, amax_1);  clone_14 = amax_1 = None
        exp_1: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_2: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        where_15: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_19, full_11, div_3);  unsqueeze_19 = full_11 = div_3 = None
        convert_element_type_19: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_15, torch.bfloat16);  where_15 = None
        permute_83: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_19, [0, 2, 1, 3]);  convert_element_type_19 = None
        clone_15: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
        view_89: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_15, _shape_param_135);  clone_15 = _shape_param_135 = None
        constant_pad_nd_6: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_89, _shape_param_136, 0.0);  view_89 = _shape_param_136 = None
        view_90: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_6, _shape_param_137);  constant_pad_nd_6 = _shape_param_137 = None
        slice_115: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_90, 2, 0, -256);  view_90 = None
        view_91: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_115, _shape_param_138);  slice_115 = _shape_param_138 = None
        slice_116: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_91, 3, 0, -1);  view_91 = None
        unsqueeze_34: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_116, 4);  slice_116 = None
        view_92: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_34, _shape_param_139);  unsqueeze_34 = _shape_param_139 = None
        clone_16: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_93: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_16, _shape_param_140);  clone_16 = _shape_param_140 = None
        permute_84: "bf16[768, 768]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_6: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_93, permute_84);  view_93 = permute_84 = None
        view_94: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_6, _shape_param_141);  mm_6 = _shape_param_141 = None
        add_15: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_94, arg24_1);  view_94 = arg24_1 = None
        view_95: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_15, _shape_param_142);  add_15 = _shape_param_142 = None
        permute_85: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_95, [1, 0, 2, 3]);  view_95 = None
        permute_86: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_85, [0, 2, 1, 3]);  permute_85 = None
        view_96: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_86, _shape_param_143);  permute_86 = _shape_param_143 = None
        constant_pad_nd_7: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_96, _shape_param_144, -1.0);  view_96 = _shape_param_144 = None
        as_strided_9: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_7, _shape_param_145, _shape_param_146);  constant_pad_nd_7 = _shape_param_145 = _shape_param_146 = None
        unsqueeze_35: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_9, 4);  as_strided_9 = None
        clone_17: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_35, memory_format = torch.contiguous_format);  unsqueeze_35 = None
        view_97: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_17, _shape_param_147);  clone_17 = _shape_param_147 = None
        bmm_3: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_92, view_97);  view_92 = view_97 = None
        view_98: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_148);  bmm_3 = _shape_param_148 = None
        permute_87: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_98, [0, 1, 2, 4, 3]);  view_98 = None
        view_99: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_87, _shape_param_149);  permute_87 = _shape_param_149 = None
        view_100: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_99, _shape_param_150);  view_99 = _shape_param_150 = None
        permute_88: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        permute_89: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_88, [1, 0, 2, 3]);  permute_88 = None
        clone_18: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
        view_101: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_18, _shape_param_151);  clone_18 = _shape_param_151 = None
        permute_90: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_101, [1, 0, 2]);  view_101 = None
        clone_19: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        view_102: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_19, _shape_param_152);  clone_19 = _shape_param_152 = None
        permute_91: "bf16[768, 768]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        mm_7: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_102, permute_91);  view_102 = permute_91 = None
        view_103: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_7, _shape_param_153);  mm_7 = _shape_param_153 = None
        add_16: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_103, arg26_1);  view_103 = arg26_1 = None
        add_17: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_16, convert_element_type_12);  add_16 = convert_element_type_12 = None
        convert_element_type_20: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_17, torch.float32);  add_17 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_20, [2], correction = 0, keepdim = True)
        getitem_4: "f32[8, 1024, 1]" = var_mean_2[0]
        getitem_5: "f32[8, 1024, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_8: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_20, getitem_5);  convert_element_type_20 = getitem_5 = None
        add_18: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_9: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_2);  sub_8 = rsqrt_2 = None
        mul_10: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_9, arg27_1);  mul_9 = arg27_1 = None
        add_19: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_10, arg28_1);  mul_10 = arg28_1 = None
        convert_element_type_21: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None
        view_104: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_21, _shape_param_154);  _shape_param_154 = None
        permute_92: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_2: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(arg30_1, view_104, permute_92);  arg30_1 = view_104 = permute_92 = None
        view_105: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_2, _shape_param_155);  addmm_2 = _shape_param_155 = None
        convert_element_type_22: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        mul_11: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.5)
        mul_12: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.7071067811865476);  convert_element_type_22 = None
        erf_1: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_20: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_20);  mul_11 = add_20 = None
        convert_element_type_23: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        view_106: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_23, _shape_param_156);  convert_element_type_23 = _shape_param_156 = None
        permute_93: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_3: "bf16[8192, 768]" = torch.ops.aten.addmm.default(arg32_1, view_106, permute_93);  arg32_1 = view_106 = permute_93 = None
        view_107: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_3, _shape_param_157);  addmm_3 = _shape_param_157 = None
        add_21: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_107, convert_element_type_21);  view_107 = convert_element_type_21 = None
        convert_element_type_24: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_21, torch.float32);  add_21 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_24, [2], correction = 0, keepdim = True)
        getitem_6: "f32[8, 1024, 1]" = var_mean_3[0]
        getitem_7: "f32[8, 1024, 1]" = var_mean_3[1];  var_mean_3 = None
        unsqueeze_36: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 2)
        unsqueeze_37: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 3);  unsqueeze_36 = None
        full_22: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_23: "bf16[96, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_158, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_158 = None
        slice_117: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_23, 1, 0, -1)
        slice_118: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_23, 1, 0, -1)
        slice_119: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_118, 3, 256, 9223372036854775807);  slice_118 = None
        sub_9: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_24, getitem_7);  convert_element_type_24 = getitem_7 = None
        add_22: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_14: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_3);  sub_9 = rsqrt_3 = None
        mul_15: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_14, arg33_1);  mul_14 = arg33_1 = None
        add_23: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_15, arg34_1);  mul_15 = arg34_1 = None
        convert_element_type_25: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None
        permute_94: "bf16[1024, 8, 768]" = torch.ops.aten.permute.default(convert_element_type_25, [1, 0, 2])
        clone_20: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format)
        view_108: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_20, _shape_param_159);  clone_20 = _shape_param_159 = None
        permute_95: "bf16[768, 768]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        mm_8: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_108, permute_95);  view_108 = permute_95 = None
        view_109: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_8, _shape_param_160);  mm_8 = _shape_param_160 = None
        add_24: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_109, arg36_1);  view_109 = arg36_1 = None
        div_4: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_24, 8.0);  add_24 = None
        view_110: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_4, _shape_param_161);  div_4 = _shape_param_161 = None
        permute_96: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_110, [1, 0, 2, 3]);  view_110 = None
        permute_97: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_96, [0, 2, 1, 3]);  permute_96 = None
        view_111: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_97, _shape_param_162);  permute_97 = _shape_param_162 = None
        view_112: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_111, _shape_param_163);  view_111 = _shape_param_163 = None
        as_strided_10: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_112, _shape_param_164, _shape_param_165);  view_112 = _shape_param_164 = _shape_param_165 = None
        unsqueeze_38: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_10, 4);  as_strided_10 = None
        clone_21: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_38, memory_format = torch.contiguous_format);  unsqueeze_38 = None
        view_113: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_21, _shape_param_166);  clone_21 = _shape_param_166 = None
        clone_22: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format)
        view_114: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_22, _shape_param_167);  clone_22 = _shape_param_167 = None
        permute_98: "bf16[768, 768]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_9: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_114, permute_98);  view_114 = permute_98 = None
        view_115: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_9, _shape_param_168);  mm_9 = _shape_param_168 = None
        add_25: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_115, arg38_1);  view_115 = arg38_1 = None
        view_116: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_25, _shape_param_169);  add_25 = _shape_param_169 = None
        permute_99: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_116, [1, 0, 2, 3]);  view_116 = None
        permute_100: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_99, [0, 2, 1, 3]);  permute_99 = None
        view_117: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_100, _shape_param_170);  permute_100 = _shape_param_170 = None
        view_118: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_117, _shape_param_171);  view_117 = _shape_param_171 = None
        as_strided_11: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_118, _shape_param_172, _shape_param_173);  view_118 = _shape_param_172 = _shape_param_173 = None
        unsqueeze_39: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_11, 4);  as_strided_11 = None
        permute_101: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_39, [0, 1, 4, 2, 3]);  unsqueeze_39 = None
        permute_102: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_101, [0, 1, 4, 3, 2]);  permute_101 = None
        clone_23: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        view_119: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_23, _shape_param_174);  clone_23 = _shape_param_174 = None
        bmm_4: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_113, view_119);  view_113 = view_119 = None
        view_120: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_4, _shape_param_175);  bmm_4 = _shape_param_175 = None
        permute_103: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_120, [0, 1, 2, 4, 3]);  view_120 = None
        view_121: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_103, _shape_param_176);  permute_103 = _shape_param_176 = None
        constant_pad_nd_8: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_121, [0, 0, 0, 1], 0.0);  view_121 = None
        view_122: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_8, _shape_param_177);  constant_pad_nd_8 = _shape_param_177 = None
        slice_120: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_122, 2, 0, 256)
        slice_121: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_120, 3, 0, 257);  slice_120 = None
        copy_24: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_119, slice_121);  slice_119 = slice_121 = None
        slice_scatter_44: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_117, copy_24, 3, 256, 9223372036854775807);  slice_117 = copy_24 = None
        slice_scatter_45: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_23, slice_scatter_44, 1, 0, -1);  full_23 = slice_scatter_44 = None
        select_24: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_45, 1, -1)
        select_25: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_45, 1, -1)
        slice_122: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_25, 2, 256, 9223372036854775807);  select_25 = None
        select_26: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_122, 1, -1)
        slice_123: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_26, 1, 256, 9223372036854775807);  select_26 = None
        slice_124: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_123, 2, 0, 257);  slice_123 = None
        copy_25: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_122, slice_124);  slice_122 = slice_124 = None
        slice_scatter_46: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_24, copy_25, 2, 256, 9223372036854775807);  select_24 = copy_25 = None
        select_scatter_8: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_45, slice_scatter_46, 1, -1);  slice_scatter_45 = slice_scatter_46 = None
        slice_125: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_8, 1, 1, 9223372036854775807)
        slice_126: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_8, 1, 1, 9223372036854775807)
        slice_127: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_126, 3, 0, 256);  slice_126 = None
        slice_128: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_122, 2, -257, -1)
        slice_129: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_128, 3, 257, 9223372036854775807);  slice_128 = None
        copy_26: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_127, slice_129);  slice_127 = slice_129 = None
        slice_scatter_47: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_125, copy_26, 3, 0, 256);  slice_125 = copy_26 = None
        slice_scatter_48: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_8, slice_scatter_47, 1, 1, 9223372036854775807);  select_scatter_8 = slice_scatter_47 = None
        select_27: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_48, 1, 0)
        slice_130: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_27, 1, 1, 256)
        select_28: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_48, 1, 0)
        slice_131: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_28, 1, 1, 256);  select_28 = None
        slice_132: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_131, 2, 1, 256);  slice_131 = None
        select_29: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_122, 1, 0);  view_122 = None
        slice_133: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_29, 1, 0, 255);  select_29 = None
        slice_134: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_133, 2, -255, 9223372036854775807);  slice_133 = None
        copy_27: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_132, slice_134);  slice_132 = slice_134 = None
        slice_scatter_49: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_130, copy_27, 2, 1, 256);  slice_130 = copy_27 = None
        slice_scatter_50: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_27, slice_scatter_49, 1, 1, 256);  select_27 = slice_scatter_49 = None
        select_scatter_9: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_48, slice_scatter_50, 1, 0);  slice_scatter_48 = slice_scatter_50 = None
        view_123: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_9, _shape_param_178);  _shape_param_178 = None
        permute_104: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
        slice_135: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_104, 1, 0, 256)
        view_124: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_9, _shape_param_179);  _shape_param_179 = None
        permute_105: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        slice_136: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_105, 1, 0, 256);  permute_105 = None
        slice_137: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_136, 3, 0, 257);  slice_136 = None
        iota_8: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_40: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_8, -2);  iota_8 = None
        iota_9: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_41: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_9, -1);  iota_9 = None
        sub_10: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_40, unsqueeze_41);  unsqueeze_40 = unsqueeze_41 = None
        le_4: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_10, 0);  sub_10 = None
        full_24: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_180, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_180 = None
        scalar_tensor_4: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_16: "bf16[256, 257]" = torch.ops.aten.where.self(le_4, full_24, scalar_tensor_4);  le_4 = full_24 = scalar_tensor_4 = None
        rev_8: "bf16[256, 257]" = torch.ops.prims.rev.default(where_16, [0]);  where_16 = None
        unsqueeze_42: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_8, 0);  rev_8 = None
        unsqueeze_43: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 2);  unsqueeze_42 = None
        expand_8: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_43, _shape_param_181);  _shape_param_181 = None
        convert_element_type_26: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_8, torch.bool);  expand_8 = None
        full_25: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_182, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_182 = None
        permute_106: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_25, [0, 2, 1, 3]);  full_25 = None
        view_125: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_9, _shape_param_183);  select_scatter_9 = _shape_param_183 = None
        permute_107: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
        slice_138: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_107, 1, 0, 256);  permute_107 = None
        slice_139: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_138, 3, 0, 257);  slice_138 = None
        where_17: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_26, permute_106, slice_139);  convert_element_type_26 = permute_106 = slice_139 = None
        copy_28: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_137, where_17);  slice_137 = where_17 = None
        slice_scatter_51: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_135, copy_28, 3, 0, 257);  slice_135 = copy_28 = None
        slice_scatter_52: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_104, slice_scatter_51, 1, 0, 256);  permute_104 = slice_scatter_51 = None
        permute_108: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_52, [0, 2, 1, 3]);  slice_scatter_52 = None
        view_126: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_108, _shape_param_184);  permute_108 = _shape_param_184 = None
        view_127: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_126, _shape_param_185);  _shape_param_185 = None
        permute_109: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_127, [0, 2, 1, 3]);  view_127 = None
        slice_140: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_109, 1, -256, 9223372036854775807)
        view_128: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_126, _shape_param_186);  _shape_param_186 = None
        permute_110: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
        slice_141: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_110, 1, -256, 9223372036854775807);  permute_110 = None
        slice_142: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_141, 3, -257, 9223372036854775807);  slice_141 = None
        rev_9: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_43, [1, 3]);  unsqueeze_43 = None
        expand_9: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_9, _shape_param_187);  rev_9 = _shape_param_187 = None
        convert_element_type_27: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_9, torch.bool);  expand_9 = None
        full_26: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_188, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_188 = None
        permute_111: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_26, [0, 2, 1, 3]);  full_26 = None
        view_129: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_126, _shape_param_189);  view_126 = _shape_param_189 = None
        permute_112: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        slice_143: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_112, 1, -256, 9223372036854775807);  permute_112 = None
        slice_144: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_143, 3, -257, 9223372036854775807);  slice_143 = None
        where_18: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_27, permute_111, slice_144);  convert_element_type_27 = permute_111 = slice_144 = None
        copy_29: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_142, where_18);  slice_142 = where_18 = None
        slice_scatter_53: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_140, copy_29, 3, -257, 9223372036854775807);  slice_140 = copy_29 = None
        slice_scatter_54: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_109, slice_scatter_53, 1, -256, 9223372036854775807);  permute_109 = slice_scatter_53 = None
        permute_113: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_54, [0, 2, 1, 3]);  slice_scatter_54 = None
        permute_114: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_113, [0, 2, 1, 3]);  permute_113 = None
        full_27: "bf16[8, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_190, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_190 = None
        slice_145: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_27, 1, 0, -1)
        slice_146: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_27, 1, 0, -1)
        slice_147: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_146, 3, 256, 9223372036854775807);  slice_146 = None
        full_28: "bf16[8, 1024, 1, 1]" = torch.ops.aten.full.default(_shape_param_191, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_191 = None
        permute_115: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(full_28, [0, 2, 1, 3]);  full_28 = None
        view_130: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_115, _shape_param_192);  permute_115 = _shape_param_192 = None
        view_131: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_130, _shape_param_193);  view_130 = _shape_param_193 = None
        as_strided_12: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_131, _shape_param_194, _shape_param_195);  view_131 = _shape_param_194 = _shape_param_195 = None
        unsqueeze_44: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_12, 4);  as_strided_12 = None
        permute_116: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_44, [0, 1, 2, 4, 3]);  unsqueeze_44 = None
        full_29: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ne_2: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg6_1, 0)
        unsqueeze_45: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_2, 2);  ne_2 = None
        unsqueeze_46: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_45, 3);  unsqueeze_45 = None
        convert_element_type_28: "bf16[8, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_46, torch.bfloat16)
        where_19: "bf16[8, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_46, full_29, convert_element_type_28);  unsqueeze_46 = full_29 = convert_element_type_28 = None
        permute_117: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(where_19, [0, 2, 1, 3]);  where_19 = None
        view_132: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_117, _shape_param_196);  permute_117 = _shape_param_196 = None
        view_133: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_132, _shape_param_197);  view_132 = _shape_param_197 = None
        as_strided_13: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_133, _shape_param_198, _shape_param_199);  view_133 = _shape_param_198 = _shape_param_199 = None
        unsqueeze_47: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_13, 4);  as_strided_13 = None
        permute_118: "bf16[8, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_47, [0, 1, 4, 2, 3]);  unsqueeze_47 = None
        mul_16: "bf16[8, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_116, permute_118);  permute_116 = permute_118 = None
        view_134: "bf16[8, 3, 512, 512]" = torch.ops.aten.view.default(mul_16, _shape_param_200);  mul_16 = _shape_param_200 = None
        constant_pad_nd_9: "bf16[8, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_134, [0, 0, 0, 1], 0.0);  view_134 = None
        view_135: "bf16[8, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_9, _shape_param_201);  constant_pad_nd_9 = _shape_param_201 = None
        slice_148: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_135, 2, 0, 256)
        slice_149: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_148, 3, 0, 257);  slice_148 = None
        copy_30: "bf16[8, 3, 256, 257]" = torch.ops.aten.copy.default(slice_147, slice_149);  slice_147 = slice_149 = None
        slice_scatter_55: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_145, copy_30, 3, 256, 9223372036854775807);  slice_145 = copy_30 = None
        slice_scatter_56: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_27, slice_scatter_55, 1, 0, -1);  full_27 = slice_scatter_55 = None
        select_30: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_56, 1, -1)
        select_31: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_56, 1, -1)
        slice_150: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(select_31, 2, 256, 9223372036854775807);  select_31 = None
        select_32: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_135, 1, -1)
        slice_151: "bf16[8, 256, 513]" = torch.ops.aten.slice.Tensor(select_32, 1, 256, 9223372036854775807);  select_32 = None
        slice_152: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(slice_151, 2, 0, 257);  slice_151 = None
        copy_31: "bf16[8, 256, 257]" = torch.ops.aten.copy.default(slice_150, slice_152);  slice_150 = slice_152 = None
        slice_scatter_57: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_30, copy_31, 2, 256, 9223372036854775807);  select_30 = copy_31 = None
        select_scatter_10: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_56, slice_scatter_57, 1, -1);  slice_scatter_56 = slice_scatter_57 = None
        slice_153: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_10, 1, 1, 9223372036854775807)
        slice_154: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_10, 1, 1, 9223372036854775807)
        slice_155: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_154, 3, 0, 256);  slice_154 = None
        slice_156: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_135, 2, -257, -1)
        slice_157: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_156, 3, 257, 9223372036854775807);  slice_156 = None
        copy_32: "bf16[8, 3, 256, 256]" = torch.ops.aten.copy.default(slice_155, slice_157);  slice_155 = slice_157 = None
        slice_scatter_58: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_153, copy_32, 3, 0, 256);  slice_153 = copy_32 = None
        slice_scatter_59: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_10, slice_scatter_58, 1, 1, 9223372036854775807);  select_scatter_10 = slice_scatter_58 = None
        select_33: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_59, 1, 0)
        slice_158: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_33, 1, 1, 256)
        select_34: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_59, 1, 0)
        slice_159: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_34, 1, 1, 256);  select_34 = None
        slice_160: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_159, 2, 1, 256);  slice_159 = None
        select_35: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_135, 1, 0);  view_135 = None
        slice_161: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_35, 1, 0, 255);  select_35 = None
        slice_162: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_161, 2, -255, 9223372036854775807);  slice_161 = None
        copy_33: "bf16[8, 255, 255]" = torch.ops.aten.copy.default(slice_160, slice_162);  slice_160 = slice_162 = None
        slice_scatter_60: "bf16[8, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_158, copy_33, 2, 1, 256);  slice_158 = copy_33 = None
        slice_scatter_61: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_33, slice_scatter_60, 1, 1, 256);  select_33 = slice_scatter_60 = None
        select_scatter_11: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_59, slice_scatter_61, 1, 0);  slice_scatter_59 = slice_scatter_61 = None
        view_136: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_11, _shape_param_202);  _shape_param_202 = None
        permute_119: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        slice_163: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_119, 1, 0, 256)
        view_137: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_11, _shape_param_203);  _shape_param_203 = None
        permute_120: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        slice_164: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_120, 1, 0, 256);  permute_120 = None
        slice_165: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_164, 3, 0, 257);  slice_164 = None
        iota_10: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_48: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_10, -2);  iota_10 = None
        iota_11: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_49: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_11, -1);  iota_11 = None
        sub_11: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_48, unsqueeze_49);  unsqueeze_48 = unsqueeze_49 = None
        le_5: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_11, 0);  sub_11 = None
        full_30: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_204, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_204 = None
        scalar_tensor_5: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_20: "bf16[256, 257]" = torch.ops.aten.where.self(le_5, full_30, scalar_tensor_5);  le_5 = full_30 = scalar_tensor_5 = None
        rev_10: "bf16[256, 257]" = torch.ops.prims.rev.default(where_20, [0]);  where_20 = None
        unsqueeze_50: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_10, 0);  rev_10 = None
        unsqueeze_51: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, 2);  unsqueeze_50 = None
        expand_10: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_51, _shape_param_205);  _shape_param_205 = None
        convert_element_type_29: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_10, torch.bool);  expand_10 = None
        full_31: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_206, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_206 = None
        view_138: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_11, _shape_param_207);  select_scatter_11 = _shape_param_207 = None
        permute_121: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        slice_166: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_121, 1, 0, 256);  permute_121 = None
        slice_167: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_166, 3, 0, 257);  slice_166 = None
        where_21: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_29, full_31, slice_167);  convert_element_type_29 = full_31 = slice_167 = None
        copy_34: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_165, where_21);  slice_165 = where_21 = None
        slice_scatter_62: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_163, copy_34, 3, 0, 257);  slice_163 = copy_34 = None
        slice_scatter_63: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_119, slice_scatter_62, 1, 0, 256);  permute_119 = slice_scatter_62 = None
        permute_122: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_63, [0, 2, 1, 3]);  slice_scatter_63 = None
        view_139: "bf16[8, 4, 256, 513]" = torch.ops.aten.view.default(permute_122, _shape_param_208);  permute_122 = _shape_param_208 = None
        view_140: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_139, _shape_param_209);  _shape_param_209 = None
        permute_123: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        slice_168: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_123, 1, -256, 9223372036854775807)
        view_141: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_139, _shape_param_210);  _shape_param_210 = None
        permute_124: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
        slice_169: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_124, 1, -256, 9223372036854775807);  permute_124 = None
        slice_170: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_169, 3, -257, 9223372036854775807);  slice_169 = None
        rev_11: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_51, [1, 3]);  unsqueeze_51 = None
        expand_11: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(rev_11, _shape_param_211);  rev_11 = _shape_param_211 = None
        convert_element_type_30: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_11, torch.bool);  expand_11 = None
        full_32: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_212, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_212 = None
        view_142: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_139, _shape_param_213);  view_139 = _shape_param_213 = None
        permute_125: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
        slice_171: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_125, 1, -256, 9223372036854775807);  permute_125 = None
        slice_172: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_171, 3, -257, 9223372036854775807);  slice_171 = None
        where_22: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_30, full_32, slice_172);  convert_element_type_30 = full_32 = slice_172 = None
        copy_35: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_170, where_22);  slice_170 = where_22 = None
        slice_scatter_64: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_168, copy_35, 3, -257, 9223372036854775807);  slice_168 = copy_35 = None
        slice_scatter_65: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_123, slice_scatter_64, 1, -256, 9223372036854775807);  permute_123 = slice_scatter_64 = None
        permute_126: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_65, [0, 2, 1, 3]);  slice_scatter_65 = None
        permute_127: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_126, [0, 2, 1, 3]);  permute_126 = None
        add_26: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_114, permute_127);  permute_114 = permute_127 = None
        permute_128: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_26, [0, 2, 1, 3]);  add_26 = None
        permute_129: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_128, [0, 2, 1, 3]);  permute_128 = None
        convert_element_type_31: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_129, torch.float32);  permute_129 = None
        clone_24: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_31, memory_format = torch.contiguous_format);  convert_element_type_31 = None
        amax_2: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_24, [-1], True)
        sub_12: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_24, amax_2);  clone_24 = amax_2 = None
        exp_2: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_3: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_5: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        where_23: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_37, full_22, div_5);  unsqueeze_37 = full_22 = div_5 = None
        convert_element_type_32: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_23, torch.bfloat16);  where_23 = None
        permute_130: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_32, [0, 2, 1, 3]);  convert_element_type_32 = None
        clone_25: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
        view_143: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_25, _shape_param_214);  clone_25 = _shape_param_214 = None
        constant_pad_nd_10: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_143, _shape_param_215, 0.0);  view_143 = _shape_param_215 = None
        view_144: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_10, _shape_param_216);  constant_pad_nd_10 = _shape_param_216 = None
        slice_173: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_144, 2, 0, -256);  view_144 = None
        view_145: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_173, _shape_param_217);  slice_173 = _shape_param_217 = None
        slice_174: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_145, 3, 0, -1);  view_145 = None
        unsqueeze_52: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_174, 4);  slice_174 = None
        view_146: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_52, _shape_param_218);  unsqueeze_52 = _shape_param_218 = None
        clone_26: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
        view_147: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_26, _shape_param_219);  clone_26 = _shape_param_219 = None
        permute_131: "bf16[768, 768]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        mm_10: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_147, permute_131);  view_147 = permute_131 = None
        view_148: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_10, _shape_param_220);  mm_10 = _shape_param_220 = None
        add_27: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_148, arg40_1);  view_148 = arg40_1 = None
        view_149: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_27, _shape_param_221);  add_27 = _shape_param_221 = None
        permute_132: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_149, [1, 0, 2, 3]);  view_149 = None
        permute_133: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_132, [0, 2, 1, 3]);  permute_132 = None
        view_150: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_133, _shape_param_222);  permute_133 = _shape_param_222 = None
        constant_pad_nd_11: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_150, _shape_param_223, -1.0);  view_150 = _shape_param_223 = None
        as_strided_14: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_11, _shape_param_224, _shape_param_225);  constant_pad_nd_11 = _shape_param_224 = _shape_param_225 = None
        unsqueeze_53: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_14, 4);  as_strided_14 = None
        clone_27: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_53, memory_format = torch.contiguous_format);  unsqueeze_53 = None
        view_151: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_27, _shape_param_226);  clone_27 = _shape_param_226 = None
        bmm_5: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_146, view_151);  view_146 = view_151 = None
        view_152: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_5, _shape_param_227);  bmm_5 = _shape_param_227 = None
        permute_134: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_152, [0, 1, 2, 4, 3]);  view_152 = None
        view_153: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_134, _shape_param_228);  permute_134 = _shape_param_228 = None
        view_154: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_153, _shape_param_229);  view_153 = _shape_param_229 = None
        permute_135: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
        permute_136: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_135, [1, 0, 2, 3]);  permute_135 = None
        clone_28: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
        view_155: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_28, _shape_param_230);  clone_28 = _shape_param_230 = None
        permute_137: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_155, [1, 0, 2]);  view_155 = None
        clone_29: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
        view_156: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_29, _shape_param_231);  clone_29 = _shape_param_231 = None
        permute_138: "bf16[768, 768]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_11: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_156, permute_138);  view_156 = permute_138 = None
        view_157: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_11, _shape_param_232);  mm_11 = _shape_param_232 = None
        add_28: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_157, arg42_1);  view_157 = arg42_1 = None
        add_29: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_28, convert_element_type_25);  add_28 = convert_element_type_25 = None
        convert_element_type_33: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_29, torch.float32);  add_29 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_33, [2], correction = 0, keepdim = True)
        getitem_8: "f32[8, 1024, 1]" = var_mean_4[0]
        getitem_9: "f32[8, 1024, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_13: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_33, getitem_9);  convert_element_type_33 = getitem_9 = None
        add_30: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_17: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_4);  sub_13 = rsqrt_4 = None
        mul_18: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_17, arg43_1);  mul_17 = arg43_1 = None
        add_31: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_18, arg44_1);  mul_18 = arg44_1 = None
        convert_element_type_34: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None
        view_158: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_34, _shape_param_233);  _shape_param_233 = None
        permute_139: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_4: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(arg46_1, view_158, permute_139);  arg46_1 = view_158 = permute_139 = None
        view_159: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_4, _shape_param_234);  addmm_4 = _shape_param_234 = None
        convert_element_type_35: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_159, torch.float32);  view_159 = None
        mul_19: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 0.5)
        mul_20: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 0.7071067811865476);  convert_element_type_35 = None
        erf_2: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_32: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_21: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_19, add_32);  mul_19 = add_32 = None
        convert_element_type_36: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        view_160: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_36, _shape_param_235);  convert_element_type_36 = _shape_param_235 = None
        permute_140: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_5: "bf16[8192, 768]" = torch.ops.aten.addmm.default(arg48_1, view_160, permute_140);  arg48_1 = view_160 = permute_140 = None
        view_161: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_5, _shape_param_236);  addmm_5 = _shape_param_236 = None
        add_33: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_161, convert_element_type_34);  view_161 = convert_element_type_34 = None
        convert_element_type_37: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_33, torch.float32);  add_33 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_37, [2], correction = 0, keepdim = True)
        getitem_10: "f32[8, 1024, 1]" = var_mean_5[0]
        getitem_11: "f32[8, 1024, 1]" = var_mean_5[1];  var_mean_5 = None
        unsqueeze_54: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 2)
        unsqueeze_55: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, 3);  unsqueeze_54 = None
        full_33: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_34: "bf16[96, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_237, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_237 = None
        slice_175: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_34, 1, 0, -1)
        slice_176: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_34, 1, 0, -1)
        slice_177: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_176, 3, 256, 9223372036854775807);  slice_176 = None
        sub_14: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_37, getitem_11);  convert_element_type_37 = getitem_11 = None
        add_34: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_22: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_5);  sub_14 = rsqrt_5 = None
        mul_23: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_22, arg49_1);  mul_22 = arg49_1 = None
        add_35: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_23, arg50_1);  mul_23 = arg50_1 = None
        convert_element_type_38: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None
        permute_141: "bf16[1024, 8, 768]" = torch.ops.aten.permute.default(convert_element_type_38, [1, 0, 2])
        clone_30: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format)
        view_162: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_30, _shape_param_238);  clone_30 = _shape_param_238 = None
        permute_142: "bf16[768, 768]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        mm_12: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_162, permute_142);  view_162 = permute_142 = None
        view_163: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_12, _shape_param_239);  mm_12 = _shape_param_239 = None
        add_36: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_163, arg52_1);  view_163 = arg52_1 = None
        div_6: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_36, 8.0);  add_36 = None
        view_164: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_6, _shape_param_240);  div_6 = _shape_param_240 = None
        permute_143: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_164, [1, 0, 2, 3]);  view_164 = None
        permute_144: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_143, [0, 2, 1, 3]);  permute_143 = None
        view_165: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_144, _shape_param_241);  permute_144 = _shape_param_241 = None
        view_166: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_165, _shape_param_242);  view_165 = _shape_param_242 = None
        as_strided_15: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_166, _shape_param_243, _shape_param_244);  view_166 = _shape_param_243 = _shape_param_244 = None
        unsqueeze_56: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_15, 4);  as_strided_15 = None
        clone_31: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_56, memory_format = torch.contiguous_format);  unsqueeze_56 = None
        view_167: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_31, _shape_param_245);  clone_31 = _shape_param_245 = None
        clone_32: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format)
        view_168: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_32, _shape_param_246);  clone_32 = _shape_param_246 = None
        permute_145: "bf16[768, 768]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        mm_13: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_168, permute_145);  view_168 = permute_145 = None
        view_169: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_13, _shape_param_247);  mm_13 = _shape_param_247 = None
        add_37: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_169, arg54_1);  view_169 = arg54_1 = None
        view_170: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_37, _shape_param_248);  add_37 = _shape_param_248 = None
        permute_146: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_170, [1, 0, 2, 3]);  view_170 = None
        permute_147: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_146, [0, 2, 1, 3]);  permute_146 = None
        view_171: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_147, _shape_param_249);  permute_147 = _shape_param_249 = None
        view_172: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_171, _shape_param_250);  view_171 = _shape_param_250 = None
        as_strided_16: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_172, _shape_param_251, _shape_param_252);  view_172 = _shape_param_251 = _shape_param_252 = None
        unsqueeze_57: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_16, 4);  as_strided_16 = None
        permute_148: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_57, [0, 1, 4, 2, 3]);  unsqueeze_57 = None
        permute_149: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_148, [0, 1, 4, 3, 2]);  permute_148 = None
        clone_33: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
        view_173: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_33, _shape_param_253);  clone_33 = _shape_param_253 = None
        bmm_6: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_167, view_173);  view_167 = view_173 = None
        view_174: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_6, _shape_param_254);  bmm_6 = _shape_param_254 = None
        permute_150: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_174, [0, 1, 2, 4, 3]);  view_174 = None
        view_175: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_150, _shape_param_255);  permute_150 = _shape_param_255 = None
        constant_pad_nd_12: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_175, [0, 0, 0, 1], 0.0);  view_175 = None
        view_176: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_12, _shape_param_256);  constant_pad_nd_12 = _shape_param_256 = None
        slice_178: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_176, 2, 0, 256)
        slice_179: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_178, 3, 0, 257);  slice_178 = None
        copy_36: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_177, slice_179);  slice_177 = slice_179 = None
        slice_scatter_66: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_175, copy_36, 3, 256, 9223372036854775807);  slice_175 = copy_36 = None
        slice_scatter_67: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_34, slice_scatter_66, 1, 0, -1);  full_34 = slice_scatter_66 = None
        select_36: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_67, 1, -1)
        select_37: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_67, 1, -1)
        slice_180: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_37, 2, 256, 9223372036854775807);  select_37 = None
        select_38: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_176, 1, -1)
        slice_181: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_38, 1, 256, 9223372036854775807);  select_38 = None
        slice_182: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_181, 2, 0, 257);  slice_181 = None
        copy_37: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_180, slice_182);  slice_180 = slice_182 = None
        slice_scatter_68: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_36, copy_37, 2, 256, 9223372036854775807);  select_36 = copy_37 = None
        select_scatter_12: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_67, slice_scatter_68, 1, -1);  slice_scatter_67 = slice_scatter_68 = None
        slice_183: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_12, 1, 1, 9223372036854775807)
        slice_184: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_12, 1, 1, 9223372036854775807)
        slice_185: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_184, 3, 0, 256);  slice_184 = None
        slice_186: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_176, 2, -257, -1)
        slice_187: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_186, 3, 257, 9223372036854775807);  slice_186 = None
        copy_38: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_185, slice_187);  slice_185 = slice_187 = None
        slice_scatter_69: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_183, copy_38, 3, 0, 256);  slice_183 = copy_38 = None
        slice_scatter_70: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_12, slice_scatter_69, 1, 1, 9223372036854775807);  select_scatter_12 = slice_scatter_69 = None
        select_39: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_70, 1, 0)
        slice_188: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_39, 1, 1, 256)
        select_40: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_70, 1, 0)
        slice_189: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_40, 1, 1, 256);  select_40 = None
        slice_190: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_189, 2, 1, 256);  slice_189 = None
        select_41: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_176, 1, 0);  view_176 = None
        slice_191: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_41, 1, 0, 255);  select_41 = None
        slice_192: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_191, 2, -255, 9223372036854775807);  slice_191 = None
        copy_39: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_190, slice_192);  slice_190 = slice_192 = None
        slice_scatter_71: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_188, copy_39, 2, 1, 256);  slice_188 = copy_39 = None
        slice_scatter_72: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_39, slice_scatter_71, 1, 1, 256);  select_39 = slice_scatter_71 = None
        select_scatter_13: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_70, slice_scatter_72, 1, 0);  slice_scatter_70 = slice_scatter_72 = None
        view_177: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_13, _shape_param_257);  _shape_param_257 = None
        permute_151: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None
        slice_193: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_151, 1, 0, 256)
        view_178: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_13, _shape_param_258);  _shape_param_258 = None
        permute_152: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        slice_194: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_152, 1, 0, 256);  permute_152 = None
        slice_195: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_194, 3, 0, 257);  slice_194 = None
        iota_12: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_58: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_12, -2);  iota_12 = None
        iota_13: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_59: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_13, -1);  iota_13 = None
        sub_15: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_58, unsqueeze_59);  unsqueeze_58 = unsqueeze_59 = None
        le_6: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_15, 0);  sub_15 = None
        full_35: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_259, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_259 = None
        scalar_tensor_6: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_24: "bf16[256, 257]" = torch.ops.aten.where.self(le_6, full_35, scalar_tensor_6);  le_6 = full_35 = scalar_tensor_6 = None
        rev_12: "bf16[256, 257]" = torch.ops.prims.rev.default(where_24, [0]);  where_24 = None
        unsqueeze_60: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_12, 0);  rev_12 = None
        unsqueeze_61: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, 2);  unsqueeze_60 = None
        expand_12: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_61, _shape_param_260);  _shape_param_260 = None
        convert_element_type_39: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_12, torch.bool);  expand_12 = None
        full_36: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_261, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_261 = None
        permute_153: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_36, [0, 2, 1, 3]);  full_36 = None
        view_179: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_13, _shape_param_262);  select_scatter_13 = _shape_param_262 = None
        permute_154: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_179, [0, 2, 1, 3]);  view_179 = None
        slice_196: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_154, 1, 0, 256);  permute_154 = None
        slice_197: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_196, 3, 0, 257);  slice_196 = None
        where_25: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_39, permute_153, slice_197);  convert_element_type_39 = permute_153 = slice_197 = None
        copy_40: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_195, where_25);  slice_195 = where_25 = None
        slice_scatter_73: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_193, copy_40, 3, 0, 257);  slice_193 = copy_40 = None
        slice_scatter_74: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_151, slice_scatter_73, 1, 0, 256);  permute_151 = slice_scatter_73 = None
        permute_155: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_74, [0, 2, 1, 3]);  slice_scatter_74 = None
        view_180: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_155, _shape_param_263);  permute_155 = _shape_param_263 = None
        view_181: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_180, _shape_param_264);  _shape_param_264 = None
        permute_156: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        slice_198: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_156, 1, -256, 9223372036854775807)
        view_182: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_180, _shape_param_265);  _shape_param_265 = None
        permute_157: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        slice_199: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_157, 1, -256, 9223372036854775807);  permute_157 = None
        slice_200: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_199, 3, -257, 9223372036854775807);  slice_199 = None
        rev_13: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_61, [1, 3]);  unsqueeze_61 = None
        expand_13: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_13, _shape_param_266);  rev_13 = _shape_param_266 = None
        convert_element_type_40: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_13, torch.bool);  expand_13 = None
        full_37: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_267, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_267 = None
        permute_158: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_37, [0, 2, 1, 3]);  full_37 = None
        view_183: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_180, _shape_param_268);  view_180 = _shape_param_268 = None
        permute_159: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        slice_201: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_159, 1, -256, 9223372036854775807);  permute_159 = None
        slice_202: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_201, 3, -257, 9223372036854775807);  slice_201 = None
        where_26: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_40, permute_158, slice_202);  convert_element_type_40 = permute_158 = slice_202 = None
        copy_41: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_200, where_26);  slice_200 = where_26 = None
        slice_scatter_75: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_198, copy_41, 3, -257, 9223372036854775807);  slice_198 = copy_41 = None
        slice_scatter_76: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_156, slice_scatter_75, 1, -256, 9223372036854775807);  permute_156 = slice_scatter_75 = None
        permute_160: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_76, [0, 2, 1, 3]);  slice_scatter_76 = None
        permute_161: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_160, [0, 2, 1, 3]);  permute_160 = None
        full_38: "bf16[8, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_269, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_269 = None
        slice_203: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_38, 1, 0, -1)
        slice_204: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_38, 1, 0, -1)
        slice_205: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_204, 3, 256, 9223372036854775807);  slice_204 = None
        full_39: "bf16[8, 1024, 1, 1]" = torch.ops.aten.full.default(_shape_param_270, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_270 = None
        permute_162: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(full_39, [0, 2, 1, 3]);  full_39 = None
        view_184: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_162, _shape_param_271);  permute_162 = _shape_param_271 = None
        view_185: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_184, _shape_param_272);  view_184 = _shape_param_272 = None
        as_strided_17: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_185, _shape_param_273, _shape_param_274);  view_185 = _shape_param_273 = _shape_param_274 = None
        unsqueeze_62: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_17, 4);  as_strided_17 = None
        permute_163: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_62, [0, 1, 2, 4, 3]);  unsqueeze_62 = None
        full_40: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ne_3: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg6_1, 0)
        unsqueeze_63: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_3, 2);  ne_3 = None
        unsqueeze_64: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 3);  unsqueeze_63 = None
        convert_element_type_41: "bf16[8, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_64, torch.bfloat16)
        where_27: "bf16[8, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_64, full_40, convert_element_type_41);  unsqueeze_64 = full_40 = convert_element_type_41 = None
        permute_164: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(where_27, [0, 2, 1, 3]);  where_27 = None
        view_186: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_164, _shape_param_275);  permute_164 = _shape_param_275 = None
        view_187: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_186, _shape_param_276);  view_186 = _shape_param_276 = None
        as_strided_18: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_187, _shape_param_277, _shape_param_278);  view_187 = _shape_param_277 = _shape_param_278 = None
        unsqueeze_65: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_18, 4);  as_strided_18 = None
        permute_165: "bf16[8, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_65, [0, 1, 4, 2, 3]);  unsqueeze_65 = None
        mul_24: "bf16[8, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_163, permute_165);  permute_163 = permute_165 = None
        view_188: "bf16[8, 3, 512, 512]" = torch.ops.aten.view.default(mul_24, _shape_param_279);  mul_24 = _shape_param_279 = None
        constant_pad_nd_13: "bf16[8, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_188, [0, 0, 0, 1], 0.0);  view_188 = None
        view_189: "bf16[8, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_13, _shape_param_280);  constant_pad_nd_13 = _shape_param_280 = None
        slice_206: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_189, 2, 0, 256)
        slice_207: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_206, 3, 0, 257);  slice_206 = None
        copy_42: "bf16[8, 3, 256, 257]" = torch.ops.aten.copy.default(slice_205, slice_207);  slice_205 = slice_207 = None
        slice_scatter_77: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_203, copy_42, 3, 256, 9223372036854775807);  slice_203 = copy_42 = None
        slice_scatter_78: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_38, slice_scatter_77, 1, 0, -1);  full_38 = slice_scatter_77 = None
        select_42: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_78, 1, -1)
        select_43: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_78, 1, -1)
        slice_208: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(select_43, 2, 256, 9223372036854775807);  select_43 = None
        select_44: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_189, 1, -1)
        slice_209: "bf16[8, 256, 513]" = torch.ops.aten.slice.Tensor(select_44, 1, 256, 9223372036854775807);  select_44 = None
        slice_210: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(slice_209, 2, 0, 257);  slice_209 = None
        copy_43: "bf16[8, 256, 257]" = torch.ops.aten.copy.default(slice_208, slice_210);  slice_208 = slice_210 = None
        slice_scatter_79: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_42, copy_43, 2, 256, 9223372036854775807);  select_42 = copy_43 = None
        select_scatter_14: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_78, slice_scatter_79, 1, -1);  slice_scatter_78 = slice_scatter_79 = None
        slice_211: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_14, 1, 1, 9223372036854775807)
        slice_212: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_14, 1, 1, 9223372036854775807)
        slice_213: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_212, 3, 0, 256);  slice_212 = None
        slice_214: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_189, 2, -257, -1)
        slice_215: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_214, 3, 257, 9223372036854775807);  slice_214 = None
        copy_44: "bf16[8, 3, 256, 256]" = torch.ops.aten.copy.default(slice_213, slice_215);  slice_213 = slice_215 = None
        slice_scatter_80: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_211, copy_44, 3, 0, 256);  slice_211 = copy_44 = None
        slice_scatter_81: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_14, slice_scatter_80, 1, 1, 9223372036854775807);  select_scatter_14 = slice_scatter_80 = None
        select_45: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_81, 1, 0)
        slice_216: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_45, 1, 1, 256)
        select_46: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_81, 1, 0)
        slice_217: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_46, 1, 1, 256);  select_46 = None
        slice_218: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_217, 2, 1, 256);  slice_217 = None
        select_47: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_189, 1, 0);  view_189 = None
        slice_219: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_47, 1, 0, 255);  select_47 = None
        slice_220: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_219, 2, -255, 9223372036854775807);  slice_219 = None
        copy_45: "bf16[8, 255, 255]" = torch.ops.aten.copy.default(slice_218, slice_220);  slice_218 = slice_220 = None
        slice_scatter_82: "bf16[8, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_216, copy_45, 2, 1, 256);  slice_216 = copy_45 = None
        slice_scatter_83: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_45, slice_scatter_82, 1, 1, 256);  select_45 = slice_scatter_82 = None
        select_scatter_15: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_81, slice_scatter_83, 1, 0);  slice_scatter_81 = slice_scatter_83 = None
        view_190: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_15, _shape_param_281);  _shape_param_281 = None
        permute_166: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        slice_221: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_166, 1, 0, 256)
        view_191: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_15, _shape_param_282);  _shape_param_282 = None
        permute_167: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
        slice_222: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_167, 1, 0, 256);  permute_167 = None
        slice_223: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_222, 3, 0, 257);  slice_222 = None
        iota_14: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_66: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_14, -2);  iota_14 = None
        iota_15: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_67: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_15, -1);  iota_15 = None
        sub_16: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_66, unsqueeze_67);  unsqueeze_66 = unsqueeze_67 = None
        le_7: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_16, 0);  sub_16 = None
        full_41: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_283, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_283 = None
        scalar_tensor_7: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_28: "bf16[256, 257]" = torch.ops.aten.where.self(le_7, full_41, scalar_tensor_7);  le_7 = full_41 = scalar_tensor_7 = None
        rev_14: "bf16[256, 257]" = torch.ops.prims.rev.default(where_28, [0]);  where_28 = None
        unsqueeze_68: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_14, 0);  rev_14 = None
        unsqueeze_69: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, 2);  unsqueeze_68 = None
        expand_14: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_69, _shape_param_284);  _shape_param_284 = None
        convert_element_type_42: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_14, torch.bool);  expand_14 = None
        full_42: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_285, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_285 = None
        view_192: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_15, _shape_param_286);  select_scatter_15 = _shape_param_286 = None
        permute_168: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        slice_224: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_168, 1, 0, 256);  permute_168 = None
        slice_225: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_224, 3, 0, 257);  slice_224 = None
        where_29: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_42, full_42, slice_225);  convert_element_type_42 = full_42 = slice_225 = None
        copy_46: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_223, where_29);  slice_223 = where_29 = None
        slice_scatter_84: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_221, copy_46, 3, 0, 257);  slice_221 = copy_46 = None
        slice_scatter_85: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_166, slice_scatter_84, 1, 0, 256);  permute_166 = slice_scatter_84 = None
        permute_169: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_85, [0, 2, 1, 3]);  slice_scatter_85 = None
        view_193: "bf16[8, 4, 256, 513]" = torch.ops.aten.view.default(permute_169, _shape_param_287);  permute_169 = _shape_param_287 = None
        view_194: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_193, _shape_param_288);  _shape_param_288 = None
        permute_170: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
        slice_226: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_170, 1, -256, 9223372036854775807)
        view_195: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_193, _shape_param_289);  _shape_param_289 = None
        permute_171: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None
        slice_227: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_171, 1, -256, 9223372036854775807);  permute_171 = None
        slice_228: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_227, 3, -257, 9223372036854775807);  slice_227 = None
        rev_15: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_69, [1, 3]);  unsqueeze_69 = None
        expand_15: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(rev_15, _shape_param_290);  rev_15 = _shape_param_290 = None
        convert_element_type_43: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_15, torch.bool);  expand_15 = None
        full_43: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_291, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_291 = None
        view_196: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_193, _shape_param_292);  view_193 = _shape_param_292 = None
        permute_172: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_196, [0, 2, 1, 3]);  view_196 = None
        slice_229: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_172, 1, -256, 9223372036854775807);  permute_172 = None
        slice_230: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_229, 3, -257, 9223372036854775807);  slice_229 = None
        where_30: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_43, full_43, slice_230);  convert_element_type_43 = full_43 = slice_230 = None
        copy_47: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_228, where_30);  slice_228 = where_30 = None
        slice_scatter_86: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_226, copy_47, 3, -257, 9223372036854775807);  slice_226 = copy_47 = None
        slice_scatter_87: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_170, slice_scatter_86, 1, -256, 9223372036854775807);  permute_170 = slice_scatter_86 = None
        permute_173: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_87, [0, 2, 1, 3]);  slice_scatter_87 = None
        permute_174: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_173, [0, 2, 1, 3]);  permute_173 = None
        add_38: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_161, permute_174);  permute_161 = permute_174 = None
        permute_175: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_38, [0, 2, 1, 3]);  add_38 = None
        permute_176: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_175, [0, 2, 1, 3]);  permute_175 = None
        convert_element_type_44: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_176, torch.float32);  permute_176 = None
        clone_34: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_44, memory_format = torch.contiguous_format);  convert_element_type_44 = None
        amax_3: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_34, [-1], True)
        sub_17: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_34, amax_3);  clone_34 = amax_3 = None
        exp_3: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_4: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        where_31: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_55, full_33, div_7);  unsqueeze_55 = full_33 = div_7 = None
        convert_element_type_45: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_31, torch.bfloat16);  where_31 = None
        permute_177: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_45, [0, 2, 1, 3]);  convert_element_type_45 = None
        clone_35: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_197: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_35, _shape_param_293);  clone_35 = _shape_param_293 = None
        constant_pad_nd_14: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_197, _shape_param_294, 0.0);  view_197 = _shape_param_294 = None
        view_198: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_14, _shape_param_295);  constant_pad_nd_14 = _shape_param_295 = None
        slice_231: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_198, 2, 0, -256);  view_198 = None
        view_199: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_231, _shape_param_296);  slice_231 = _shape_param_296 = None
        slice_232: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_199, 3, 0, -1);  view_199 = None
        unsqueeze_70: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_232, 4);  slice_232 = None
        view_200: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_70, _shape_param_297);  unsqueeze_70 = _shape_param_297 = None
        clone_36: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None
        view_201: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_36, _shape_param_298);  clone_36 = _shape_param_298 = None
        permute_178: "bf16[768, 768]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_14: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_201, permute_178);  view_201 = permute_178 = None
        view_202: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_14, _shape_param_299);  mm_14 = _shape_param_299 = None
        add_39: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_202, arg56_1);  view_202 = arg56_1 = None
        view_203: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_39, _shape_param_300);  add_39 = _shape_param_300 = None
        permute_179: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_203, [1, 0, 2, 3]);  view_203 = None
        permute_180: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_179, [0, 2, 1, 3]);  permute_179 = None
        view_204: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_180, _shape_param_301);  permute_180 = _shape_param_301 = None
        constant_pad_nd_15: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_204, _shape_param_302, -1.0);  view_204 = _shape_param_302 = None
        as_strided_19: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_15, _shape_param_303, _shape_param_304);  constant_pad_nd_15 = _shape_param_303 = _shape_param_304 = None
        unsqueeze_71: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_19, 4);  as_strided_19 = None
        clone_37: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_71, memory_format = torch.contiguous_format);  unsqueeze_71 = None
        view_205: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_37, _shape_param_305);  clone_37 = _shape_param_305 = None
        bmm_7: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_200, view_205);  view_200 = view_205 = None
        view_206: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_306);  bmm_7 = _shape_param_306 = None
        permute_181: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_206, [0, 1, 2, 4, 3]);  view_206 = None
        view_207: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_181, _shape_param_307);  permute_181 = _shape_param_307 = None
        view_208: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_207, _shape_param_308);  view_207 = _shape_param_308 = None
        permute_182: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        permute_183: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_182, [1, 0, 2, 3]);  permute_182 = None
        clone_38: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        view_209: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_38, _shape_param_309);  clone_38 = _shape_param_309 = None
        permute_184: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_209, [1, 0, 2]);  view_209 = None
        clone_39: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
        view_210: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_39, _shape_param_310);  clone_39 = _shape_param_310 = None
        permute_185: "bf16[768, 768]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        mm_15: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_210, permute_185);  view_210 = permute_185 = None
        view_211: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_15, _shape_param_311);  mm_15 = _shape_param_311 = None
        add_40: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_211, arg58_1);  view_211 = arg58_1 = None
        add_41: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_40, convert_element_type_38);  add_40 = convert_element_type_38 = None
        convert_element_type_46: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_41, torch.float32);  add_41 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_46, [2], correction = 0, keepdim = True)
        getitem_12: "f32[8, 1024, 1]" = var_mean_6[0]
        getitem_13: "f32[8, 1024, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_18: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_46, getitem_13);  convert_element_type_46 = getitem_13 = None
        add_42: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_25: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_6);  sub_18 = rsqrt_6 = None
        mul_26: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_25, arg59_1);  mul_25 = arg59_1 = None
        add_43: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_26, arg60_1);  mul_26 = arg60_1 = None
        convert_element_type_47: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None
        view_212: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_47, _shape_param_312);  _shape_param_312 = None
        permute_186: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_6: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(arg62_1, view_212, permute_186);  arg62_1 = view_212 = permute_186 = None
        view_213: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_6, _shape_param_313);  addmm_6 = _shape_param_313 = None
        convert_element_type_48: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_213, torch.float32);  view_213 = None
        mul_27: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 0.5)
        mul_28: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 0.7071067811865476);  convert_element_type_48 = None
        erf_3: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_44: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_29: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_27, add_44);  mul_27 = add_44 = None
        convert_element_type_49: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        view_214: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_49, _shape_param_314);  convert_element_type_49 = _shape_param_314 = None
        permute_187: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_7: "bf16[8192, 768]" = torch.ops.aten.addmm.default(arg64_1, view_214, permute_187);  arg64_1 = view_214 = permute_187 = None
        view_215: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_7, _shape_param_315);  addmm_7 = _shape_param_315 = None
        add_45: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_215, convert_element_type_47);  view_215 = convert_element_type_47 = None
        convert_element_type_50: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_45, torch.float32);  add_45 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_50, [2], correction = 0, keepdim = True)
        getitem_14: "f32[8, 1024, 1]" = var_mean_7[0]
        getitem_15: "f32[8, 1024, 1]" = var_mean_7[1];  var_mean_7 = None
        unsqueeze_72: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 2)
        unsqueeze_73: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, 3);  unsqueeze_72 = None
        full_44: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_45: "bf16[96, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_316, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_316 = None
        slice_233: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_45, 1, 0, -1)
        slice_234: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_45, 1, 0, -1)
        slice_235: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_234, 3, 256, 9223372036854775807);  slice_234 = None
        sub_19: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_50, getitem_15);  convert_element_type_50 = getitem_15 = None
        add_46: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_30: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_7);  sub_19 = rsqrt_7 = None
        mul_31: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_30, arg65_1);  mul_30 = arg65_1 = None
        add_47: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_31, arg66_1);  mul_31 = arg66_1 = None
        convert_element_type_51: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None
        permute_188: "bf16[1024, 8, 768]" = torch.ops.aten.permute.default(convert_element_type_51, [1, 0, 2])
        clone_40: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format)
        view_216: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_40, _shape_param_317);  clone_40 = _shape_param_317 = None
        permute_189: "bf16[768, 768]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        mm_16: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_216, permute_189);  view_216 = permute_189 = None
        view_217: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_16, _shape_param_318);  mm_16 = _shape_param_318 = None
        add_48: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_217, arg68_1);  view_217 = arg68_1 = None
        div_8: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_48, 8.0);  add_48 = None
        view_218: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_8, _shape_param_319);  div_8 = _shape_param_319 = None
        permute_190: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_218, [1, 0, 2, 3]);  view_218 = None
        permute_191: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_190, [0, 2, 1, 3]);  permute_190 = None
        view_219: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_191, _shape_param_320);  permute_191 = _shape_param_320 = None
        view_220: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_219, _shape_param_321);  view_219 = _shape_param_321 = None
        as_strided_20: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_220, _shape_param_322, _shape_param_323);  view_220 = _shape_param_322 = _shape_param_323 = None
        unsqueeze_74: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_20, 4);  as_strided_20 = None
        clone_41: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_74, memory_format = torch.contiguous_format);  unsqueeze_74 = None
        view_221: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_41, _shape_param_324);  clone_41 = _shape_param_324 = None
        clone_42: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format)
        view_222: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_42, _shape_param_325);  clone_42 = _shape_param_325 = None
        permute_192: "bf16[768, 768]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_17: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_222, permute_192);  view_222 = permute_192 = None
        view_223: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_17, _shape_param_326);  mm_17 = _shape_param_326 = None
        add_49: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_223, arg70_1);  view_223 = arg70_1 = None
        view_224: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_49, _shape_param_327);  add_49 = _shape_param_327 = None
        permute_193: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_224, [1, 0, 2, 3]);  view_224 = None
        permute_194: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_193, [0, 2, 1, 3]);  permute_193 = None
        view_225: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_194, _shape_param_328);  permute_194 = _shape_param_328 = None
        view_226: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_225, _shape_param_329);  view_225 = _shape_param_329 = None
        as_strided_21: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_226, _shape_param_330, _shape_param_331);  view_226 = _shape_param_330 = _shape_param_331 = None
        unsqueeze_75: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_21, 4);  as_strided_21 = None
        permute_195: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_75, [0, 1, 4, 2, 3]);  unsqueeze_75 = None
        permute_196: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_195, [0, 1, 4, 3, 2]);  permute_195 = None
        clone_43: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
        view_227: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_43, _shape_param_332);  clone_43 = _shape_param_332 = None
        bmm_8: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_221, view_227);  view_221 = view_227 = None
        view_228: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_8, _shape_param_333);  bmm_8 = _shape_param_333 = None
        permute_197: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_228, [0, 1, 2, 4, 3]);  view_228 = None
        view_229: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_197, _shape_param_334);  permute_197 = _shape_param_334 = None
        constant_pad_nd_16: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_229, [0, 0, 0, 1], 0.0);  view_229 = None
        view_230: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_16, _shape_param_335);  constant_pad_nd_16 = _shape_param_335 = None
        slice_236: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_230, 2, 0, 256)
        slice_237: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_236, 3, 0, 257);  slice_236 = None
        copy_48: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_235, slice_237);  slice_235 = slice_237 = None
        slice_scatter_88: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_233, copy_48, 3, 256, 9223372036854775807);  slice_233 = copy_48 = None
        slice_scatter_89: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_45, slice_scatter_88, 1, 0, -1);  full_45 = slice_scatter_88 = None
        select_48: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_89, 1, -1)
        select_49: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_89, 1, -1)
        slice_238: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_49, 2, 256, 9223372036854775807);  select_49 = None
        select_50: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_230, 1, -1)
        slice_239: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_50, 1, 256, 9223372036854775807);  select_50 = None
        slice_240: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_239, 2, 0, 257);  slice_239 = None
        copy_49: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_238, slice_240);  slice_238 = slice_240 = None
        slice_scatter_90: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_48, copy_49, 2, 256, 9223372036854775807);  select_48 = copy_49 = None
        select_scatter_16: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_89, slice_scatter_90, 1, -1);  slice_scatter_89 = slice_scatter_90 = None
        slice_241: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_16, 1, 1, 9223372036854775807)
        slice_242: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_16, 1, 1, 9223372036854775807)
        slice_243: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_242, 3, 0, 256);  slice_242 = None
        slice_244: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_230, 2, -257, -1)
        slice_245: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_244, 3, 257, 9223372036854775807);  slice_244 = None
        copy_50: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_243, slice_245);  slice_243 = slice_245 = None
        slice_scatter_91: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_241, copy_50, 3, 0, 256);  slice_241 = copy_50 = None
        slice_scatter_92: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_16, slice_scatter_91, 1, 1, 9223372036854775807);  select_scatter_16 = slice_scatter_91 = None
        select_51: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_92, 1, 0)
        slice_246: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_51, 1, 1, 256)
        select_52: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_92, 1, 0)
        slice_247: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_52, 1, 1, 256);  select_52 = None
        slice_248: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_247, 2, 1, 256);  slice_247 = None
        select_53: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_230, 1, 0);  view_230 = None
        slice_249: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_53, 1, 0, 255);  select_53 = None
        slice_250: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_249, 2, -255, 9223372036854775807);  slice_249 = None
        copy_51: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_248, slice_250);  slice_248 = slice_250 = None
        slice_scatter_93: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_246, copy_51, 2, 1, 256);  slice_246 = copy_51 = None
        slice_scatter_94: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_51, slice_scatter_93, 1, 1, 256);  select_51 = slice_scatter_93 = None
        select_scatter_17: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_92, slice_scatter_94, 1, 0);  slice_scatter_92 = slice_scatter_94 = None
        view_231: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_17, _shape_param_336);  _shape_param_336 = None
        permute_198: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_231, [0, 2, 1, 3]);  view_231 = None
        slice_251: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_198, 1, 0, 256)
        view_232: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_17, _shape_param_337);  _shape_param_337 = None
        permute_199: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None
        slice_252: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_199, 1, 0, 256);  permute_199 = None
        slice_253: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_252, 3, 0, 257);  slice_252 = None
        iota_16: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_76: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_16, -2);  iota_16 = None
        iota_17: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_77: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_17, -1);  iota_17 = None
        sub_20: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_76, unsqueeze_77);  unsqueeze_76 = unsqueeze_77 = None
        le_8: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_20, 0);  sub_20 = None
        full_46: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_338, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_338 = None
        scalar_tensor_8: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_32: "bf16[256, 257]" = torch.ops.aten.where.self(le_8, full_46, scalar_tensor_8);  le_8 = full_46 = scalar_tensor_8 = None
        rev_16: "bf16[256, 257]" = torch.ops.prims.rev.default(where_32, [0]);  where_32 = None
        unsqueeze_78: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_16, 0);  rev_16 = None
        unsqueeze_79: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 2);  unsqueeze_78 = None
        expand_16: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_79, _shape_param_339);  _shape_param_339 = None
        convert_element_type_52: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_16, torch.bool);  expand_16 = None
        full_47: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_340, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_340 = None
        permute_200: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_47, [0, 2, 1, 3]);  full_47 = None
        view_233: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_17, _shape_param_341);  select_scatter_17 = _shape_param_341 = None
        permute_201: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None
        slice_254: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_201, 1, 0, 256);  permute_201 = None
        slice_255: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_254, 3, 0, 257);  slice_254 = None
        where_33: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_52, permute_200, slice_255);  convert_element_type_52 = permute_200 = slice_255 = None
        copy_52: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_253, where_33);  slice_253 = where_33 = None
        slice_scatter_95: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_251, copy_52, 3, 0, 257);  slice_251 = copy_52 = None
        slice_scatter_96: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_198, slice_scatter_95, 1, 0, 256);  permute_198 = slice_scatter_95 = None
        permute_202: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_96, [0, 2, 1, 3]);  slice_scatter_96 = None
        view_234: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_202, _shape_param_342);  permute_202 = _shape_param_342 = None
        view_235: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_234, _shape_param_343);  _shape_param_343 = None
        permute_203: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_235, [0, 2, 1, 3]);  view_235 = None
        slice_256: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_203, 1, -256, 9223372036854775807)
        view_236: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_234, _shape_param_344);  _shape_param_344 = None
        permute_204: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        slice_257: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_204, 1, -256, 9223372036854775807);  permute_204 = None
        slice_258: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_257, 3, -257, 9223372036854775807);  slice_257 = None
        rev_17: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_79, [1, 3]);  unsqueeze_79 = None
        expand_17: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_17, _shape_param_345);  rev_17 = _shape_param_345 = None
        convert_element_type_53: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_17, torch.bool);  expand_17 = None
        full_48: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_346, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_346 = None
        permute_205: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_48, [0, 2, 1, 3]);  full_48 = None
        view_237: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_234, _shape_param_347);  view_234 = _shape_param_347 = None
        permute_206: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None
        slice_259: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_206, 1, -256, 9223372036854775807);  permute_206 = None
        slice_260: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_259, 3, -257, 9223372036854775807);  slice_259 = None
        where_34: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_53, permute_205, slice_260);  convert_element_type_53 = permute_205 = slice_260 = None
        copy_53: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_258, where_34);  slice_258 = where_34 = None
        slice_scatter_97: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_256, copy_53, 3, -257, 9223372036854775807);  slice_256 = copy_53 = None
        slice_scatter_98: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_203, slice_scatter_97, 1, -256, 9223372036854775807);  permute_203 = slice_scatter_97 = None
        permute_207: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_98, [0, 2, 1, 3]);  slice_scatter_98 = None
        permute_208: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_207, [0, 2, 1, 3]);  permute_207 = None
        full_49: "bf16[8, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_348, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_348 = None
        slice_261: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_49, 1, 0, -1)
        slice_262: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_49, 1, 0, -1)
        slice_263: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_262, 3, 256, 9223372036854775807);  slice_262 = None
        full_50: "bf16[8, 1024, 1, 1]" = torch.ops.aten.full.default(_shape_param_349, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_349 = None
        permute_209: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(full_50, [0, 2, 1, 3]);  full_50 = None
        view_238: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_209, _shape_param_350);  permute_209 = _shape_param_350 = None
        view_239: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_238, _shape_param_351);  view_238 = _shape_param_351 = None
        as_strided_22: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_239, _shape_param_352, _shape_param_353);  view_239 = _shape_param_352 = _shape_param_353 = None
        unsqueeze_80: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_22, 4);  as_strided_22 = None
        permute_210: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_80, [0, 1, 2, 4, 3]);  unsqueeze_80 = None
        full_51: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ne_4: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg6_1, 0)
        unsqueeze_81: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_4, 2);  ne_4 = None
        unsqueeze_82: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_81, 3);  unsqueeze_81 = None
        convert_element_type_54: "bf16[8, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_82, torch.bfloat16)
        where_35: "bf16[8, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_82, full_51, convert_element_type_54);  unsqueeze_82 = full_51 = convert_element_type_54 = None
        permute_211: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(where_35, [0, 2, 1, 3]);  where_35 = None
        view_240: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_211, _shape_param_354);  permute_211 = _shape_param_354 = None
        view_241: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_240, _shape_param_355);  view_240 = _shape_param_355 = None
        as_strided_23: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_241, _shape_param_356, _shape_param_357);  view_241 = _shape_param_356 = _shape_param_357 = None
        unsqueeze_83: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_23, 4);  as_strided_23 = None
        permute_212: "bf16[8, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_83, [0, 1, 4, 2, 3]);  unsqueeze_83 = None
        mul_32: "bf16[8, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_210, permute_212);  permute_210 = permute_212 = None
        view_242: "bf16[8, 3, 512, 512]" = torch.ops.aten.view.default(mul_32, _shape_param_358);  mul_32 = _shape_param_358 = None
        constant_pad_nd_17: "bf16[8, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_242, [0, 0, 0, 1], 0.0);  view_242 = None
        view_243: "bf16[8, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_17, _shape_param_359);  constant_pad_nd_17 = _shape_param_359 = None
        slice_264: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_243, 2, 0, 256)
        slice_265: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_264, 3, 0, 257);  slice_264 = None
        copy_54: "bf16[8, 3, 256, 257]" = torch.ops.aten.copy.default(slice_263, slice_265);  slice_263 = slice_265 = None
        slice_scatter_99: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_261, copy_54, 3, 256, 9223372036854775807);  slice_261 = copy_54 = None
        slice_scatter_100: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_49, slice_scatter_99, 1, 0, -1);  full_49 = slice_scatter_99 = None
        select_54: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_100, 1, -1)
        select_55: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_100, 1, -1)
        slice_266: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(select_55, 2, 256, 9223372036854775807);  select_55 = None
        select_56: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_243, 1, -1)
        slice_267: "bf16[8, 256, 513]" = torch.ops.aten.slice.Tensor(select_56, 1, 256, 9223372036854775807);  select_56 = None
        slice_268: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(slice_267, 2, 0, 257);  slice_267 = None
        copy_55: "bf16[8, 256, 257]" = torch.ops.aten.copy.default(slice_266, slice_268);  slice_266 = slice_268 = None
        slice_scatter_101: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_54, copy_55, 2, 256, 9223372036854775807);  select_54 = copy_55 = None
        select_scatter_18: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_100, slice_scatter_101, 1, -1);  slice_scatter_100 = slice_scatter_101 = None
        slice_269: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_18, 1, 1, 9223372036854775807)
        slice_270: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_18, 1, 1, 9223372036854775807)
        slice_271: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_270, 3, 0, 256);  slice_270 = None
        slice_272: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_243, 2, -257, -1)
        slice_273: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_272, 3, 257, 9223372036854775807);  slice_272 = None
        copy_56: "bf16[8, 3, 256, 256]" = torch.ops.aten.copy.default(slice_271, slice_273);  slice_271 = slice_273 = None
        slice_scatter_102: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_269, copy_56, 3, 0, 256);  slice_269 = copy_56 = None
        slice_scatter_103: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_18, slice_scatter_102, 1, 1, 9223372036854775807);  select_scatter_18 = slice_scatter_102 = None
        select_57: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_103, 1, 0)
        slice_274: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_57, 1, 1, 256)
        select_58: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_103, 1, 0)
        slice_275: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_58, 1, 1, 256);  select_58 = None
        slice_276: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_275, 2, 1, 256);  slice_275 = None
        select_59: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_243, 1, 0);  view_243 = None
        slice_277: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_59, 1, 0, 255);  select_59 = None
        slice_278: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_277, 2, -255, 9223372036854775807);  slice_277 = None
        copy_57: "bf16[8, 255, 255]" = torch.ops.aten.copy.default(slice_276, slice_278);  slice_276 = slice_278 = None
        slice_scatter_104: "bf16[8, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_274, copy_57, 2, 1, 256);  slice_274 = copy_57 = None
        slice_scatter_105: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_57, slice_scatter_104, 1, 1, 256);  select_57 = slice_scatter_104 = None
        select_scatter_19: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_103, slice_scatter_105, 1, 0);  slice_scatter_103 = slice_scatter_105 = None
        view_244: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_19, _shape_param_360);  _shape_param_360 = None
        permute_213: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
        slice_279: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_213, 1, 0, 256)
        view_245: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_19, _shape_param_361);  _shape_param_361 = None
        permute_214: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None
        slice_280: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_214, 1, 0, 256);  permute_214 = None
        slice_281: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_280, 3, 0, 257);  slice_280 = None
        iota_18: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_84: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_18, -2);  iota_18 = None
        iota_19: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_85: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_19, -1);  iota_19 = None
        sub_21: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_84, unsqueeze_85);  unsqueeze_84 = unsqueeze_85 = None
        le_9: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_21, 0);  sub_21 = None
        full_52: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_362, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_362 = None
        scalar_tensor_9: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_36: "bf16[256, 257]" = torch.ops.aten.where.self(le_9, full_52, scalar_tensor_9);  le_9 = full_52 = scalar_tensor_9 = None
        rev_18: "bf16[256, 257]" = torch.ops.prims.rev.default(where_36, [0]);  where_36 = None
        unsqueeze_86: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_18, 0);  rev_18 = None
        unsqueeze_87: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, 2);  unsqueeze_86 = None
        expand_18: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_87, _shape_param_363);  _shape_param_363 = None
        convert_element_type_55: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_18, torch.bool);  expand_18 = None
        full_53: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_364, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_364 = None
        view_246: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_19, _shape_param_365);  select_scatter_19 = _shape_param_365 = None
        permute_215: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        slice_282: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_215, 1, 0, 256);  permute_215 = None
        slice_283: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_282, 3, 0, 257);  slice_282 = None
        where_37: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_55, full_53, slice_283);  convert_element_type_55 = full_53 = slice_283 = None
        copy_58: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_281, where_37);  slice_281 = where_37 = None
        slice_scatter_106: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_279, copy_58, 3, 0, 257);  slice_279 = copy_58 = None
        slice_scatter_107: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_213, slice_scatter_106, 1, 0, 256);  permute_213 = slice_scatter_106 = None
        permute_216: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_107, [0, 2, 1, 3]);  slice_scatter_107 = None
        view_247: "bf16[8, 4, 256, 513]" = torch.ops.aten.view.default(permute_216, _shape_param_366);  permute_216 = _shape_param_366 = None
        view_248: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_247, _shape_param_367);  _shape_param_367 = None
        permute_217: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
        slice_284: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_217, 1, -256, 9223372036854775807)
        view_249: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_247, _shape_param_368);  _shape_param_368 = None
        permute_218: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
        slice_285: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_218, 1, -256, 9223372036854775807);  permute_218 = None
        slice_286: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_285, 3, -257, 9223372036854775807);  slice_285 = None
        rev_19: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_87, [1, 3]);  unsqueeze_87 = None
        expand_19: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(rev_19, _shape_param_369);  rev_19 = _shape_param_369 = None
        convert_element_type_56: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_19, torch.bool);  expand_19 = None
        full_54: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_370, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_370 = None
        view_250: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_247, _shape_param_371);  view_247 = _shape_param_371 = None
        permute_219: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        slice_287: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_219, 1, -256, 9223372036854775807);  permute_219 = None
        slice_288: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_287, 3, -257, 9223372036854775807);  slice_287 = None
        where_38: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_56, full_54, slice_288);  convert_element_type_56 = full_54 = slice_288 = None
        copy_59: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_286, where_38);  slice_286 = where_38 = None
        slice_scatter_108: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_284, copy_59, 3, -257, 9223372036854775807);  slice_284 = copy_59 = None
        slice_scatter_109: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_217, slice_scatter_108, 1, -256, 9223372036854775807);  permute_217 = slice_scatter_108 = None
        permute_220: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_109, [0, 2, 1, 3]);  slice_scatter_109 = None
        permute_221: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_220, [0, 2, 1, 3]);  permute_220 = None
        add_50: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_208, permute_221);  permute_208 = permute_221 = None
        permute_222: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_50, [0, 2, 1, 3]);  add_50 = None
        permute_223: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_222, [0, 2, 1, 3]);  permute_222 = None
        convert_element_type_57: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_223, torch.float32);  permute_223 = None
        clone_44: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_57, memory_format = torch.contiguous_format);  convert_element_type_57 = None
        amax_4: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_44, [-1], True)
        sub_22: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_44, amax_4);  clone_44 = amax_4 = None
        exp_4: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_5: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_9: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        where_39: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_73, full_44, div_9);  unsqueeze_73 = full_44 = div_9 = None
        convert_element_type_58: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_39, torch.bfloat16);  where_39 = None
        permute_224: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_58, [0, 2, 1, 3]);  convert_element_type_58 = None
        clone_45: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
        view_251: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_45, _shape_param_372);  clone_45 = _shape_param_372 = None
        constant_pad_nd_18: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_251, _shape_param_373, 0.0);  view_251 = _shape_param_373 = None
        view_252: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_18, _shape_param_374);  constant_pad_nd_18 = _shape_param_374 = None
        slice_289: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_252, 2, 0, -256);  view_252 = None
        view_253: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_289, _shape_param_375);  slice_289 = _shape_param_375 = None
        slice_290: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_253, 3, 0, -1);  view_253 = None
        unsqueeze_88: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_290, 4);  slice_290 = None
        view_254: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_88, _shape_param_376);  unsqueeze_88 = _shape_param_376 = None
        clone_46: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_255: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_46, _shape_param_377);  clone_46 = _shape_param_377 = None
        permute_225: "bf16[768, 768]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        mm_18: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_255, permute_225);  view_255 = permute_225 = None
        view_256: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_18, _shape_param_378);  mm_18 = _shape_param_378 = None
        add_51: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_256, arg72_1);  view_256 = arg72_1 = None
        view_257: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_51, _shape_param_379);  add_51 = _shape_param_379 = None
        permute_226: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_257, [1, 0, 2, 3]);  view_257 = None
        permute_227: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_226, [0, 2, 1, 3]);  permute_226 = None
        view_258: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_227, _shape_param_380);  permute_227 = _shape_param_380 = None
        constant_pad_nd_19: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_258, _shape_param_381, -1.0);  view_258 = _shape_param_381 = None
        as_strided_24: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_19, _shape_param_382, _shape_param_383);  constant_pad_nd_19 = _shape_param_382 = _shape_param_383 = None
        unsqueeze_89: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_24, 4);  as_strided_24 = None
        clone_47: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_89, memory_format = torch.contiguous_format);  unsqueeze_89 = None
        view_259: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_47, _shape_param_384);  clone_47 = _shape_param_384 = None
        bmm_9: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_254, view_259);  view_254 = view_259 = None
        view_260: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_9, _shape_param_385);  bmm_9 = _shape_param_385 = None
        permute_228: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_260, [0, 1, 2, 4, 3]);  view_260 = None
        view_261: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_228, _shape_param_386);  permute_228 = _shape_param_386 = None
        view_262: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_261, _shape_param_387);  view_261 = _shape_param_387 = None
        permute_229: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
        permute_230: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_229, [1, 0, 2, 3]);  permute_229 = None
        clone_48: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
        view_263: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_48, _shape_param_388);  clone_48 = _shape_param_388 = None
        permute_231: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_263, [1, 0, 2]);  view_263 = None
        clone_49: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
        view_264: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_49, _shape_param_389);  clone_49 = _shape_param_389 = None
        permute_232: "bf16[768, 768]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        mm_19: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_264, permute_232);  view_264 = permute_232 = None
        view_265: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_19, _shape_param_390);  mm_19 = _shape_param_390 = None
        add_52: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_265, arg74_1);  view_265 = arg74_1 = None
        add_53: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_52, convert_element_type_51);  add_52 = convert_element_type_51 = None
        convert_element_type_59: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_53, torch.float32);  add_53 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_59, [2], correction = 0, keepdim = True)
        getitem_16: "f32[8, 1024, 1]" = var_mean_8[0]
        getitem_17: "f32[8, 1024, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_23: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_59, getitem_17);  convert_element_type_59 = getitem_17 = None
        add_54: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        mul_33: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_8);  sub_23 = rsqrt_8 = None
        mul_34: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_33, arg75_1);  mul_33 = arg75_1 = None
        add_55: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_34, arg76_1);  mul_34 = arg76_1 = None
        convert_element_type_60: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None
        view_266: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_60, _shape_param_391);  _shape_param_391 = None
        permute_233: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_8: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(arg78_1, view_266, permute_233);  arg78_1 = view_266 = permute_233 = None
        view_267: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_8, _shape_param_392);  addmm_8 = _shape_param_392 = None
        convert_element_type_61: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_267, torch.float32);  view_267 = None
        mul_35: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_61, 0.5)
        mul_36: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_61, 0.7071067811865476);  convert_element_type_61 = None
        erf_4: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
        add_56: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_37: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_35, add_56);  mul_35 = add_56 = None
        convert_element_type_62: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None
        view_268: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_62, _shape_param_393);  convert_element_type_62 = _shape_param_393 = None
        permute_234: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_9: "bf16[8192, 768]" = torch.ops.aten.addmm.default(arg80_1, view_268, permute_234);  arg80_1 = view_268 = permute_234 = None
        view_269: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_9, _shape_param_394);  addmm_9 = _shape_param_394 = None
        add_57: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_269, convert_element_type_60);  view_269 = convert_element_type_60 = None
        convert_element_type_63: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_57, torch.float32);  add_57 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_63, [2], correction = 0, keepdim = True)
        getitem_18: "f32[8, 1024, 1]" = var_mean_9[0]
        getitem_19: "f32[8, 1024, 1]" = var_mean_9[1];  var_mean_9 = None
        unsqueeze_90: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 2)
        unsqueeze_91: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, 3);  unsqueeze_90 = None
        full_55: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_56: "bf16[96, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_395, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_395 = None
        slice_291: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_56, 1, 0, -1)
        slice_292: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_56, 1, 0, -1)
        slice_293: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_292, 3, 256, 9223372036854775807);  slice_292 = None
        sub_24: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_63, getitem_19);  convert_element_type_63 = getitem_19 = None
        add_58: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_38: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_9);  sub_24 = rsqrt_9 = None
        mul_39: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_38, arg81_1);  mul_38 = arg81_1 = None
        add_59: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_39, arg82_1);  mul_39 = arg82_1 = None
        convert_element_type_64: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None
        permute_235: "bf16[1024, 8, 768]" = torch.ops.aten.permute.default(convert_element_type_64, [1, 0, 2])
        clone_50: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format)
        view_270: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_50, _shape_param_396);  clone_50 = _shape_param_396 = None
        permute_236: "bf16[768, 768]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_20: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_270, permute_236);  view_270 = permute_236 = None
        view_271: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_20, _shape_param_397);  mm_20 = _shape_param_397 = None
        add_60: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_271, arg84_1);  view_271 = arg84_1 = None
        div_10: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_60, 8.0);  add_60 = None
        view_272: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_10, _shape_param_398);  div_10 = _shape_param_398 = None
        permute_237: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_272, [1, 0, 2, 3]);  view_272 = None
        permute_238: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_237, [0, 2, 1, 3]);  permute_237 = None
        view_273: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_238, _shape_param_399);  permute_238 = _shape_param_399 = None
        view_274: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_273, _shape_param_400);  view_273 = _shape_param_400 = None
        as_strided_25: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_274, _shape_param_401, _shape_param_402);  view_274 = _shape_param_401 = _shape_param_402 = None
        unsqueeze_92: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_25, 4);  as_strided_25 = None
        clone_51: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_92, memory_format = torch.contiguous_format);  unsqueeze_92 = None
        view_275: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_51, _shape_param_403);  clone_51 = _shape_param_403 = None
        clone_52: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format)
        view_276: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_52, _shape_param_404);  clone_52 = _shape_param_404 = None
        permute_239: "bf16[768, 768]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_21: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_276, permute_239);  view_276 = permute_239 = None
        view_277: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_21, _shape_param_405);  mm_21 = _shape_param_405 = None
        add_61: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_277, arg86_1);  view_277 = arg86_1 = None
        view_278: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_61, _shape_param_406);  add_61 = _shape_param_406 = None
        permute_240: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_278, [1, 0, 2, 3]);  view_278 = None
        permute_241: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_240, [0, 2, 1, 3]);  permute_240 = None
        view_279: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_241, _shape_param_407);  permute_241 = _shape_param_407 = None
        view_280: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_279, _shape_param_408);  view_279 = _shape_param_408 = None
        as_strided_26: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_280, _shape_param_409, _shape_param_410);  view_280 = _shape_param_409 = _shape_param_410 = None
        unsqueeze_93: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_26, 4);  as_strided_26 = None
        permute_242: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_93, [0, 1, 4, 2, 3]);  unsqueeze_93 = None
        permute_243: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_242, [0, 1, 4, 3, 2]);  permute_242 = None
        clone_53: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
        view_281: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_53, _shape_param_411);  clone_53 = _shape_param_411 = None
        bmm_10: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_275, view_281);  view_275 = view_281 = None
        view_282: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_10, _shape_param_412);  bmm_10 = _shape_param_412 = None
        permute_244: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_282, [0, 1, 2, 4, 3]);  view_282 = None
        view_283: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_244, _shape_param_413);  permute_244 = _shape_param_413 = None
        constant_pad_nd_20: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_283, [0, 0, 0, 1], 0.0);  view_283 = None
        view_284: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_20, _shape_param_414);  constant_pad_nd_20 = _shape_param_414 = None
        slice_294: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_284, 2, 0, 256)
        slice_295: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_294, 3, 0, 257);  slice_294 = None
        copy_60: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_293, slice_295);  slice_293 = slice_295 = None
        slice_scatter_110: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_291, copy_60, 3, 256, 9223372036854775807);  slice_291 = copy_60 = None
        slice_scatter_111: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_56, slice_scatter_110, 1, 0, -1);  full_56 = slice_scatter_110 = None
        select_60: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_111, 1, -1)
        select_61: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_111, 1, -1)
        slice_296: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_61, 2, 256, 9223372036854775807);  select_61 = None
        select_62: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_284, 1, -1)
        slice_297: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_62, 1, 256, 9223372036854775807);  select_62 = None
        slice_298: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_297, 2, 0, 257);  slice_297 = None
        copy_61: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_296, slice_298);  slice_296 = slice_298 = None
        slice_scatter_112: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_60, copy_61, 2, 256, 9223372036854775807);  select_60 = copy_61 = None
        select_scatter_20: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_111, slice_scatter_112, 1, -1);  slice_scatter_111 = slice_scatter_112 = None
        slice_299: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_20, 1, 1, 9223372036854775807)
        slice_300: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_20, 1, 1, 9223372036854775807)
        slice_301: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_300, 3, 0, 256);  slice_300 = None
        slice_302: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_284, 2, -257, -1)
        slice_303: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_302, 3, 257, 9223372036854775807);  slice_302 = None
        copy_62: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_301, slice_303);  slice_301 = slice_303 = None
        slice_scatter_113: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_299, copy_62, 3, 0, 256);  slice_299 = copy_62 = None
        slice_scatter_114: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_20, slice_scatter_113, 1, 1, 9223372036854775807);  select_scatter_20 = slice_scatter_113 = None
        select_63: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_114, 1, 0)
        slice_304: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_63, 1, 1, 256)
        select_64: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_114, 1, 0)
        slice_305: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_64, 1, 1, 256);  select_64 = None
        slice_306: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_305, 2, 1, 256);  slice_305 = None
        select_65: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_284, 1, 0);  view_284 = None
        slice_307: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_65, 1, 0, 255);  select_65 = None
        slice_308: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_307, 2, -255, 9223372036854775807);  slice_307 = None
        copy_63: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_306, slice_308);  slice_306 = slice_308 = None
        slice_scatter_115: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_304, copy_63, 2, 1, 256);  slice_304 = copy_63 = None
        slice_scatter_116: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_63, slice_scatter_115, 1, 1, 256);  select_63 = slice_scatter_115 = None
        select_scatter_21: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_114, slice_scatter_116, 1, 0);  slice_scatter_114 = slice_scatter_116 = None
        view_285: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_21, _shape_param_415);  _shape_param_415 = None
        permute_245: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
        slice_309: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_245, 1, 0, 256)
        view_286: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_21, _shape_param_416);  _shape_param_416 = None
        permute_246: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
        slice_310: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_246, 1, 0, 256);  permute_246 = None
        slice_311: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_310, 3, 0, 257);  slice_310 = None
        iota_20: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_94: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_20, -2);  iota_20 = None
        iota_21: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_95: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_21, -1);  iota_21 = None
        sub_25: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_94, unsqueeze_95);  unsqueeze_94 = unsqueeze_95 = None
        le_10: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_25, 0);  sub_25 = None
        full_57: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_417, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_417 = None
        scalar_tensor_10: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_40: "bf16[256, 257]" = torch.ops.aten.where.self(le_10, full_57, scalar_tensor_10);  le_10 = full_57 = scalar_tensor_10 = None
        rev_20: "bf16[256, 257]" = torch.ops.prims.rev.default(where_40, [0]);  where_40 = None
        unsqueeze_96: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_20, 0);  rev_20 = None
        unsqueeze_97: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, 2);  unsqueeze_96 = None
        expand_20: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_97, _shape_param_418);  _shape_param_418 = None
        convert_element_type_65: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_20, torch.bool);  expand_20 = None
        full_58: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_419, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_419 = None
        permute_247: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_58, [0, 2, 1, 3]);  full_58 = None
        view_287: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_21, _shape_param_420);  select_scatter_21 = _shape_param_420 = None
        permute_248: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_287, [0, 2, 1, 3]);  view_287 = None
        slice_312: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_248, 1, 0, 256);  permute_248 = None
        slice_313: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_312, 3, 0, 257);  slice_312 = None
        where_41: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_65, permute_247, slice_313);  convert_element_type_65 = permute_247 = slice_313 = None
        copy_64: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_311, where_41);  slice_311 = where_41 = None
        slice_scatter_117: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_309, copy_64, 3, 0, 257);  slice_309 = copy_64 = None
        slice_scatter_118: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_245, slice_scatter_117, 1, 0, 256);  permute_245 = slice_scatter_117 = None
        permute_249: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_118, [0, 2, 1, 3]);  slice_scatter_118 = None
        view_288: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_249, _shape_param_421);  permute_249 = _shape_param_421 = None
        view_289: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_288, _shape_param_422);  _shape_param_422 = None
        permute_250: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None
        slice_314: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_250, 1, -256, 9223372036854775807)
        view_290: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_288, _shape_param_423);  _shape_param_423 = None
        permute_251: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
        slice_315: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_251, 1, -256, 9223372036854775807);  permute_251 = None
        slice_316: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_315, 3, -257, 9223372036854775807);  slice_315 = None
        rev_21: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_97, [1, 3]);  unsqueeze_97 = None
        expand_21: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_21, _shape_param_424);  rev_21 = _shape_param_424 = None
        convert_element_type_66: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_21, torch.bool);  expand_21 = None
        full_59: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_425, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_425 = None
        permute_252: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_59, [0, 2, 1, 3]);  full_59 = None
        view_291: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_288, _shape_param_426);  view_288 = _shape_param_426 = None
        permute_253: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        slice_317: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_253, 1, -256, 9223372036854775807);  permute_253 = None
        slice_318: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_317, 3, -257, 9223372036854775807);  slice_317 = None
        where_42: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_66, permute_252, slice_318);  convert_element_type_66 = permute_252 = slice_318 = None
        copy_65: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_316, where_42);  slice_316 = where_42 = None
        slice_scatter_119: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_314, copy_65, 3, -257, 9223372036854775807);  slice_314 = copy_65 = None
        slice_scatter_120: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_250, slice_scatter_119, 1, -256, 9223372036854775807);  permute_250 = slice_scatter_119 = None
        permute_254: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_120, [0, 2, 1, 3]);  slice_scatter_120 = None
        permute_255: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_254, [0, 2, 1, 3]);  permute_254 = None
        full_60: "bf16[8, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_427, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_427 = None
        slice_319: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_60, 1, 0, -1)
        slice_320: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_60, 1, 0, -1)
        slice_321: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_320, 3, 256, 9223372036854775807);  slice_320 = None
        full_61: "bf16[8, 1024, 1, 1]" = torch.ops.aten.full.default(_shape_param_428, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_428 = None
        permute_256: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(full_61, [0, 2, 1, 3]);  full_61 = None
        view_292: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_256, _shape_param_429);  permute_256 = _shape_param_429 = None
        view_293: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_292, _shape_param_430);  view_292 = _shape_param_430 = None
        as_strided_27: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_293, _shape_param_431, _shape_param_432);  view_293 = _shape_param_431 = _shape_param_432 = None
        unsqueeze_98: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_27, 4);  as_strided_27 = None
        permute_257: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_98, [0, 1, 2, 4, 3]);  unsqueeze_98 = None
        full_62: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ne_5: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg6_1, 0)
        unsqueeze_99: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_5, 2);  ne_5 = None
        unsqueeze_100: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 3);  unsqueeze_99 = None
        convert_element_type_67: "bf16[8, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_100, torch.bfloat16)
        where_43: "bf16[8, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_100, full_62, convert_element_type_67);  unsqueeze_100 = full_62 = convert_element_type_67 = None
        permute_258: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(where_43, [0, 2, 1, 3]);  where_43 = None
        view_294: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_258, _shape_param_433);  permute_258 = _shape_param_433 = None
        view_295: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_294, _shape_param_434);  view_294 = _shape_param_434 = None
        as_strided_28: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_295, _shape_param_435, _shape_param_436);  view_295 = _shape_param_435 = _shape_param_436 = None
        unsqueeze_101: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_28, 4);  as_strided_28 = None
        permute_259: "bf16[8, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_101, [0, 1, 4, 2, 3]);  unsqueeze_101 = None
        mul_40: "bf16[8, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_257, permute_259);  permute_257 = permute_259 = None
        view_296: "bf16[8, 3, 512, 512]" = torch.ops.aten.view.default(mul_40, _shape_param_437);  mul_40 = _shape_param_437 = None
        constant_pad_nd_21: "bf16[8, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_296, [0, 0, 0, 1], 0.0);  view_296 = None
        view_297: "bf16[8, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_21, _shape_param_438);  constant_pad_nd_21 = _shape_param_438 = None
        slice_322: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_297, 2, 0, 256)
        slice_323: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_322, 3, 0, 257);  slice_322 = None
        copy_66: "bf16[8, 3, 256, 257]" = torch.ops.aten.copy.default(slice_321, slice_323);  slice_321 = slice_323 = None
        slice_scatter_121: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_319, copy_66, 3, 256, 9223372036854775807);  slice_319 = copy_66 = None
        slice_scatter_122: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_60, slice_scatter_121, 1, 0, -1);  full_60 = slice_scatter_121 = None
        select_66: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_122, 1, -1)
        select_67: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_122, 1, -1)
        slice_324: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(select_67, 2, 256, 9223372036854775807);  select_67 = None
        select_68: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_297, 1, -1)
        slice_325: "bf16[8, 256, 513]" = torch.ops.aten.slice.Tensor(select_68, 1, 256, 9223372036854775807);  select_68 = None
        slice_326: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(slice_325, 2, 0, 257);  slice_325 = None
        copy_67: "bf16[8, 256, 257]" = torch.ops.aten.copy.default(slice_324, slice_326);  slice_324 = slice_326 = None
        slice_scatter_123: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_66, copy_67, 2, 256, 9223372036854775807);  select_66 = copy_67 = None
        select_scatter_22: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_122, slice_scatter_123, 1, -1);  slice_scatter_122 = slice_scatter_123 = None
        slice_327: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_22, 1, 1, 9223372036854775807)
        slice_328: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_22, 1, 1, 9223372036854775807)
        slice_329: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_328, 3, 0, 256);  slice_328 = None
        slice_330: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_297, 2, -257, -1)
        slice_331: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_330, 3, 257, 9223372036854775807);  slice_330 = None
        copy_68: "bf16[8, 3, 256, 256]" = torch.ops.aten.copy.default(slice_329, slice_331);  slice_329 = slice_331 = None
        slice_scatter_124: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_327, copy_68, 3, 0, 256);  slice_327 = copy_68 = None
        slice_scatter_125: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_22, slice_scatter_124, 1, 1, 9223372036854775807);  select_scatter_22 = slice_scatter_124 = None
        select_69: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_125, 1, 0)
        slice_332: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_69, 1, 1, 256)
        select_70: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_125, 1, 0)
        slice_333: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_70, 1, 1, 256);  select_70 = None
        slice_334: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_333, 2, 1, 256);  slice_333 = None
        select_71: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_297, 1, 0);  view_297 = None
        slice_335: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_71, 1, 0, 255);  select_71 = None
        slice_336: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_335, 2, -255, 9223372036854775807);  slice_335 = None
        copy_69: "bf16[8, 255, 255]" = torch.ops.aten.copy.default(slice_334, slice_336);  slice_334 = slice_336 = None
        slice_scatter_126: "bf16[8, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_332, copy_69, 2, 1, 256);  slice_332 = copy_69 = None
        slice_scatter_127: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_69, slice_scatter_126, 1, 1, 256);  select_69 = slice_scatter_126 = None
        select_scatter_23: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_125, slice_scatter_127, 1, 0);  slice_scatter_125 = slice_scatter_127 = None
        view_298: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_23, _shape_param_439);  _shape_param_439 = None
        permute_260: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_298, [0, 2, 1, 3]);  view_298 = None
        slice_337: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_260, 1, 0, 256)
        view_299: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_23, _shape_param_440);  _shape_param_440 = None
        permute_261: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_299, [0, 2, 1, 3]);  view_299 = None
        slice_338: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_261, 1, 0, 256);  permute_261 = None
        slice_339: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_338, 3, 0, 257);  slice_338 = None
        iota_22: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_102: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_22, -2);  iota_22 = None
        iota_23: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_103: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_23, -1);  iota_23 = None
        sub_26: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_102, unsqueeze_103);  unsqueeze_102 = unsqueeze_103 = None
        le_11: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_26, 0);  sub_26 = None
        full_63: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_441, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_441 = None
        scalar_tensor_11: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_44: "bf16[256, 257]" = torch.ops.aten.where.self(le_11, full_63, scalar_tensor_11);  le_11 = full_63 = scalar_tensor_11 = None
        rev_22: "bf16[256, 257]" = torch.ops.prims.rev.default(where_44, [0]);  where_44 = None
        unsqueeze_104: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_22, 0);  rev_22 = None
        unsqueeze_105: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, 2);  unsqueeze_104 = None
        expand_22: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_105, _shape_param_442);  _shape_param_442 = None
        convert_element_type_68: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_22, torch.bool);  expand_22 = None
        full_64: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_443, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_443 = None
        view_300: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_23, _shape_param_444);  select_scatter_23 = _shape_param_444 = None
        permute_262: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
        slice_340: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_262, 1, 0, 256);  permute_262 = None
        slice_341: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_340, 3, 0, 257);  slice_340 = None
        where_45: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_68, full_64, slice_341);  convert_element_type_68 = full_64 = slice_341 = None
        copy_70: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_339, where_45);  slice_339 = where_45 = None
        slice_scatter_128: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_337, copy_70, 3, 0, 257);  slice_337 = copy_70 = None
        slice_scatter_129: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_260, slice_scatter_128, 1, 0, 256);  permute_260 = slice_scatter_128 = None
        permute_263: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_129, [0, 2, 1, 3]);  slice_scatter_129 = None
        view_301: "bf16[8, 4, 256, 513]" = torch.ops.aten.view.default(permute_263, _shape_param_445);  permute_263 = _shape_param_445 = None
        view_302: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_301, _shape_param_446);  _shape_param_446 = None
        permute_264: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
        slice_342: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_264, 1, -256, 9223372036854775807)
        view_303: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_301, _shape_param_447);  _shape_param_447 = None
        permute_265: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
        slice_343: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_265, 1, -256, 9223372036854775807);  permute_265 = None
        slice_344: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_343, 3, -257, 9223372036854775807);  slice_343 = None
        rev_23: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_105, [1, 3]);  unsqueeze_105 = None
        expand_23: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(rev_23, _shape_param_448);  rev_23 = _shape_param_448 = None
        convert_element_type_69: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_23, torch.bool);  expand_23 = None
        full_65: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_449, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_449 = None
        view_304: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_301, _shape_param_450);  view_301 = _shape_param_450 = None
        permute_266: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None
        slice_345: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_266, 1, -256, 9223372036854775807);  permute_266 = None
        slice_346: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_345, 3, -257, 9223372036854775807);  slice_345 = None
        where_46: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_69, full_65, slice_346);  convert_element_type_69 = full_65 = slice_346 = None
        copy_71: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_344, where_46);  slice_344 = where_46 = None
        slice_scatter_130: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_342, copy_71, 3, -257, 9223372036854775807);  slice_342 = copy_71 = None
        slice_scatter_131: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_264, slice_scatter_130, 1, -256, 9223372036854775807);  permute_264 = slice_scatter_130 = None
        permute_267: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_131, [0, 2, 1, 3]);  slice_scatter_131 = None
        permute_268: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_267, [0, 2, 1, 3]);  permute_267 = None
        add_62: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_255, permute_268);  permute_255 = permute_268 = None
        permute_269: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_62, [0, 2, 1, 3]);  add_62 = None
        permute_270: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_269, [0, 2, 1, 3]);  permute_269 = None
        convert_element_type_70: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_270, torch.float32);  permute_270 = None
        clone_54: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_70, memory_format = torch.contiguous_format);  convert_element_type_70 = None
        amax_5: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_54, [-1], True)
        sub_27: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_54, amax_5);  clone_54 = amax_5 = None
        exp_5: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_6: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_11: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        where_47: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_91, full_55, div_11);  unsqueeze_91 = full_55 = div_11 = None
        convert_element_type_71: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_47, torch.bfloat16);  where_47 = None
        permute_271: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_71, [0, 2, 1, 3]);  convert_element_type_71 = None
        clone_55: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_271, memory_format = torch.contiguous_format);  permute_271 = None
        view_305: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_55, _shape_param_451);  clone_55 = _shape_param_451 = None
        constant_pad_nd_22: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_305, _shape_param_452, 0.0);  view_305 = _shape_param_452 = None
        view_306: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_22, _shape_param_453);  constant_pad_nd_22 = _shape_param_453 = None
        slice_347: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_306, 2, 0, -256);  view_306 = None
        view_307: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_347, _shape_param_454);  slice_347 = _shape_param_454 = None
        slice_348: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_307, 3, 0, -1);  view_307 = None
        unsqueeze_106: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_348, 4);  slice_348 = None
        view_308: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_106, _shape_param_455);  unsqueeze_106 = _shape_param_455 = None
        clone_56: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
        view_309: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_56, _shape_param_456);  clone_56 = _shape_param_456 = None
        permute_272: "bf16[768, 768]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        mm_22: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_309, permute_272);  view_309 = permute_272 = None
        view_310: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_22, _shape_param_457);  mm_22 = _shape_param_457 = None
        add_63: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_310, arg88_1);  view_310 = arg88_1 = None
        view_311: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_63, _shape_param_458);  add_63 = _shape_param_458 = None
        permute_273: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_311, [1, 0, 2, 3]);  view_311 = None
        permute_274: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_273, [0, 2, 1, 3]);  permute_273 = None
        view_312: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_274, _shape_param_459);  permute_274 = _shape_param_459 = None
        constant_pad_nd_23: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_312, _shape_param_460, -1.0);  view_312 = _shape_param_460 = None
        as_strided_29: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_23, _shape_param_461, _shape_param_462);  constant_pad_nd_23 = _shape_param_461 = _shape_param_462 = None
        unsqueeze_107: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_29, 4);  as_strided_29 = None
        clone_57: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_107, memory_format = torch.contiguous_format);  unsqueeze_107 = None
        view_313: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_57, _shape_param_463);  clone_57 = _shape_param_463 = None
        bmm_11: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_308, view_313);  view_308 = view_313 = None
        view_314: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_464);  bmm_11 = _shape_param_464 = None
        permute_275: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_314, [0, 1, 2, 4, 3]);  view_314 = None
        view_315: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_275, _shape_param_465);  permute_275 = _shape_param_465 = None
        view_316: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_315, _shape_param_466);  view_315 = _shape_param_466 = None
        permute_276: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
        permute_277: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_276, [1, 0, 2, 3]);  permute_276 = None
        clone_58: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_277, memory_format = torch.contiguous_format);  permute_277 = None
        view_317: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_58, _shape_param_467);  clone_58 = _shape_param_467 = None
        permute_278: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_317, [1, 0, 2]);  view_317 = None
        clone_59: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_278, memory_format = torch.contiguous_format);  permute_278 = None
        view_318: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_59, _shape_param_468);  clone_59 = _shape_param_468 = None
        permute_279: "bf16[768, 768]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        mm_23: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_318, permute_279);  view_318 = permute_279 = None
        view_319: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_23, _shape_param_469);  mm_23 = _shape_param_469 = None
        add_64: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_319, arg90_1);  view_319 = arg90_1 = None
        add_65: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_64, convert_element_type_64);  add_64 = convert_element_type_64 = None
        convert_element_type_72: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_65, torch.float32);  add_65 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_72, [2], correction = 0, keepdim = True)
        getitem_20: "f32[8, 1024, 1]" = var_mean_10[0]
        getitem_21: "f32[8, 1024, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_28: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_72, getitem_21);  convert_element_type_72 = getitem_21 = None
        add_66: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_41: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_10);  sub_28 = rsqrt_10 = None
        mul_42: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_41, arg91_1);  mul_41 = arg91_1 = None
        add_67: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_42, arg92_1);  mul_42 = arg92_1 = None
        convert_element_type_73: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_67, torch.bfloat16);  add_67 = None
        view_320: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_73, _shape_param_470);  _shape_param_470 = None
        permute_280: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_10: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(arg94_1, view_320, permute_280);  arg94_1 = view_320 = permute_280 = None
        view_321: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_10, _shape_param_471);  addmm_10 = _shape_param_471 = None
        convert_element_type_74: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_321, torch.float32);  view_321 = None
        mul_43: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_74, 0.5)
        mul_44: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_74, 0.7071067811865476);  convert_element_type_74 = None
        erf_5: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_44);  mul_44 = None
        add_68: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_45: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_43, add_68);  mul_43 = add_68 = None
        convert_element_type_75: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        view_322: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_75, _shape_param_472);  convert_element_type_75 = _shape_param_472 = None
        permute_281: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_11: "bf16[8192, 768]" = torch.ops.aten.addmm.default(arg96_1, view_322, permute_281);  arg96_1 = view_322 = permute_281 = None
        view_323: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_11, _shape_param_473);  addmm_11 = _shape_param_473 = None
        add_69: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_323, convert_element_type_73);  view_323 = convert_element_type_73 = None
        convert_element_type_76: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_69, torch.float32);  add_69 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_76, [2], correction = 0, keepdim = True)
        getitem_22: "f32[8, 1024, 1]" = var_mean_11[0]
        getitem_23: "f32[8, 1024, 1]" = var_mean_11[1];  var_mean_11 = None
        unsqueeze_108: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 2)
        unsqueeze_109: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 3);  unsqueeze_108 = None
        full_66: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_67: "bf16[96, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_474, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_474 = None
        slice_349: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_67, 1, 0, -1)
        slice_350: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_67, 1, 0, -1)
        slice_351: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_350, 3, 256, 9223372036854775807);  slice_350 = None
        sub_29: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_76, getitem_23);  convert_element_type_76 = getitem_23 = None
        add_70: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_46: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_11);  sub_29 = rsqrt_11 = None
        mul_47: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_46, arg97_1);  mul_46 = arg97_1 = None
        add_71: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_47, arg98_1);  mul_47 = arg98_1 = None
        convert_element_type_77: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16);  add_71 = None
        permute_282: "bf16[1024, 8, 768]" = torch.ops.aten.permute.default(convert_element_type_77, [1, 0, 2])
        clone_60: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_282, memory_format = torch.contiguous_format)
        view_324: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_60, _shape_param_475);  clone_60 = _shape_param_475 = None
        permute_283: "bf16[768, 768]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_24: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_324, permute_283);  view_324 = permute_283 = None
        view_325: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_24, _shape_param_476);  mm_24 = _shape_param_476 = None
        add_72: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_325, arg100_1);  view_325 = arg100_1 = None
        div_12: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_72, 8.0);  add_72 = None
        view_326: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_12, _shape_param_477);  div_12 = _shape_param_477 = None
        permute_284: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_326, [1, 0, 2, 3]);  view_326 = None
        permute_285: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_284, [0, 2, 1, 3]);  permute_284 = None
        view_327: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_285, _shape_param_478);  permute_285 = _shape_param_478 = None
        view_328: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_327, _shape_param_479);  view_327 = _shape_param_479 = None
        as_strided_30: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_328, _shape_param_480, _shape_param_481);  view_328 = _shape_param_480 = _shape_param_481 = None
        unsqueeze_110: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_30, 4);  as_strided_30 = None
        clone_61: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_110, memory_format = torch.contiguous_format);  unsqueeze_110 = None
        view_329: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_61, _shape_param_482);  clone_61 = _shape_param_482 = None
        clone_62: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_282, memory_format = torch.contiguous_format)
        view_330: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_62, _shape_param_483);  clone_62 = _shape_param_483 = None
        permute_286: "bf16[768, 768]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_25: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_330, permute_286);  view_330 = permute_286 = None
        view_331: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_25, _shape_param_484);  mm_25 = _shape_param_484 = None
        add_73: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_331, arg102_1);  view_331 = arg102_1 = None
        view_332: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_73, _shape_param_485);  add_73 = _shape_param_485 = None
        permute_287: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_332, [1, 0, 2, 3]);  view_332 = None
        permute_288: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_287, [0, 2, 1, 3]);  permute_287 = None
        view_333: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_288, _shape_param_486);  permute_288 = _shape_param_486 = None
        view_334: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_333, _shape_param_487);  view_333 = _shape_param_487 = None
        as_strided_31: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_334, _shape_param_488, _shape_param_489);  view_334 = _shape_param_488 = _shape_param_489 = None
        unsqueeze_111: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_31, 4);  as_strided_31 = None
        permute_289: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_111, [0, 1, 4, 2, 3]);  unsqueeze_111 = None
        permute_290: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_289, [0, 1, 4, 3, 2]);  permute_289 = None
        clone_63: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_290, memory_format = torch.contiguous_format);  permute_290 = None
        view_335: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_63, _shape_param_490);  clone_63 = _shape_param_490 = None
        bmm_12: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_329, view_335);  view_329 = view_335 = None
        view_336: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_12, _shape_param_491);  bmm_12 = _shape_param_491 = None
        permute_291: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_336, [0, 1, 2, 4, 3]);  view_336 = None
        view_337: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_291, _shape_param_492);  permute_291 = _shape_param_492 = None
        constant_pad_nd_24: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_337, [0, 0, 0, 1], 0.0);  view_337 = None
        view_338: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_24, _shape_param_493);  constant_pad_nd_24 = _shape_param_493 = None
        slice_352: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_338, 2, 0, 256)
        slice_353: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_352, 3, 0, 257);  slice_352 = None
        copy_72: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_351, slice_353);  slice_351 = slice_353 = None
        slice_scatter_132: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_349, copy_72, 3, 256, 9223372036854775807);  slice_349 = copy_72 = None
        slice_scatter_133: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_67, slice_scatter_132, 1, 0, -1);  full_67 = slice_scatter_132 = None
        select_72: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_133, 1, -1)
        select_73: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_133, 1, -1)
        slice_354: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_73, 2, 256, 9223372036854775807);  select_73 = None
        select_74: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_338, 1, -1)
        slice_355: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_74, 1, 256, 9223372036854775807);  select_74 = None
        slice_356: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_355, 2, 0, 257);  slice_355 = None
        copy_73: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_354, slice_356);  slice_354 = slice_356 = None
        slice_scatter_134: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_72, copy_73, 2, 256, 9223372036854775807);  select_72 = copy_73 = None
        select_scatter_24: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_133, slice_scatter_134, 1, -1);  slice_scatter_133 = slice_scatter_134 = None
        slice_357: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_24, 1, 1, 9223372036854775807)
        slice_358: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_24, 1, 1, 9223372036854775807)
        slice_359: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_358, 3, 0, 256);  slice_358 = None
        slice_360: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_338, 2, -257, -1)
        slice_361: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_360, 3, 257, 9223372036854775807);  slice_360 = None
        copy_74: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_359, slice_361);  slice_359 = slice_361 = None
        slice_scatter_135: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_357, copy_74, 3, 0, 256);  slice_357 = copy_74 = None
        slice_scatter_136: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_24, slice_scatter_135, 1, 1, 9223372036854775807);  select_scatter_24 = slice_scatter_135 = None
        select_75: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_136, 1, 0)
        slice_362: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_75, 1, 1, 256)
        select_76: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_136, 1, 0)
        slice_363: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_76, 1, 1, 256);  select_76 = None
        slice_364: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_363, 2, 1, 256);  slice_363 = None
        select_77: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_338, 1, 0);  view_338 = None
        slice_365: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_77, 1, 0, 255);  select_77 = None
        slice_366: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_365, 2, -255, 9223372036854775807);  slice_365 = None
        copy_75: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_364, slice_366);  slice_364 = slice_366 = None
        slice_scatter_137: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_362, copy_75, 2, 1, 256);  slice_362 = copy_75 = None
        slice_scatter_138: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_75, slice_scatter_137, 1, 1, 256);  select_75 = slice_scatter_137 = None
        select_scatter_25: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_136, slice_scatter_138, 1, 0);  slice_scatter_136 = slice_scatter_138 = None
        view_339: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_25, _shape_param_494);  _shape_param_494 = None
        permute_292: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
        slice_367: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_292, 1, 0, 256)
        view_340: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_25, _shape_param_495);  _shape_param_495 = None
        permute_293: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
        slice_368: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_293, 1, 0, 256);  permute_293 = None
        slice_369: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_368, 3, 0, 257);  slice_368 = None
        iota_24: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_112: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_24, -2);  iota_24 = None
        iota_25: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_113: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_25, -1);  iota_25 = None
        sub_30: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_112, unsqueeze_113);  unsqueeze_112 = unsqueeze_113 = None
        le_12: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_30, 0);  sub_30 = None
        full_68: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_496, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_496 = None
        scalar_tensor_12: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_48: "bf16[256, 257]" = torch.ops.aten.where.self(le_12, full_68, scalar_tensor_12);  le_12 = full_68 = scalar_tensor_12 = None
        rev_24: "bf16[256, 257]" = torch.ops.prims.rev.default(where_48, [0]);  where_48 = None
        unsqueeze_114: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_24, 0);  rev_24 = None
        unsqueeze_115: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 2);  unsqueeze_114 = None
        expand_24: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_115, _shape_param_497);  _shape_param_497 = None
        convert_element_type_78: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_24, torch.bool);  expand_24 = None
        full_69: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_498, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_498 = None
        permute_294: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_69, [0, 2, 1, 3]);  full_69 = None
        view_341: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_25, _shape_param_499);  select_scatter_25 = _shape_param_499 = None
        permute_295: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None
        slice_370: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_295, 1, 0, 256);  permute_295 = None
        slice_371: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_370, 3, 0, 257);  slice_370 = None
        where_49: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_78, permute_294, slice_371);  convert_element_type_78 = permute_294 = slice_371 = None
        copy_76: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_369, where_49);  slice_369 = where_49 = None
        slice_scatter_139: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_367, copy_76, 3, 0, 257);  slice_367 = copy_76 = None
        slice_scatter_140: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_292, slice_scatter_139, 1, 0, 256);  permute_292 = slice_scatter_139 = None
        permute_296: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_140, [0, 2, 1, 3]);  slice_scatter_140 = None
        view_342: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_296, _shape_param_500);  permute_296 = _shape_param_500 = None
        view_343: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_342, _shape_param_501);  _shape_param_501 = None
        permute_297: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None
        slice_372: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_297, 1, -256, 9223372036854775807)
        view_344: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_342, _shape_param_502);  _shape_param_502 = None
        permute_298: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
        slice_373: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_298, 1, -256, 9223372036854775807);  permute_298 = None
        slice_374: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_373, 3, -257, 9223372036854775807);  slice_373 = None
        rev_25: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_115, [1, 3]);  unsqueeze_115 = None
        expand_25: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_25, _shape_param_503);  rev_25 = _shape_param_503 = None
        convert_element_type_79: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_25, torch.bool);  expand_25 = None
        full_70: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_504, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_504 = None
        permute_299: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_70, [0, 2, 1, 3]);  full_70 = None
        view_345: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_342, _shape_param_505);  view_342 = _shape_param_505 = None
        permute_300: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
        slice_375: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_300, 1, -256, 9223372036854775807);  permute_300 = None
        slice_376: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_375, 3, -257, 9223372036854775807);  slice_375 = None
        where_50: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_79, permute_299, slice_376);  convert_element_type_79 = permute_299 = slice_376 = None
        copy_77: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_374, where_50);  slice_374 = where_50 = None
        slice_scatter_141: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_372, copy_77, 3, -257, 9223372036854775807);  slice_372 = copy_77 = None
        slice_scatter_142: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_297, slice_scatter_141, 1, -256, 9223372036854775807);  permute_297 = slice_scatter_141 = None
        permute_301: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_142, [0, 2, 1, 3]);  slice_scatter_142 = None
        permute_302: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_301, [0, 2, 1, 3]);  permute_301 = None
        full_71: "bf16[8, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_506, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_506 = None
        slice_377: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_71, 1, 0, -1)
        slice_378: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_71, 1, 0, -1)
        slice_379: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_378, 3, 256, 9223372036854775807);  slice_378 = None
        full_72: "bf16[8, 1024, 1, 1]" = torch.ops.aten.full.default(_shape_param_507, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_507 = None
        permute_303: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(full_72, [0, 2, 1, 3]);  full_72 = None
        view_346: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_303, _shape_param_508);  permute_303 = _shape_param_508 = None
        view_347: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_346, _shape_param_509);  view_346 = _shape_param_509 = None
        as_strided_32: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_347, _shape_param_510, _shape_param_511);  view_347 = _shape_param_510 = _shape_param_511 = None
        unsqueeze_116: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_32, 4);  as_strided_32 = None
        permute_304: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_116, [0, 1, 2, 4, 3]);  unsqueeze_116 = None
        full_73: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ne_6: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg6_1, 0)
        unsqueeze_117: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_6, 2);  ne_6 = None
        unsqueeze_118: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 3);  unsqueeze_117 = None
        convert_element_type_80: "bf16[8, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_118, torch.bfloat16)
        where_51: "bf16[8, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_118, full_73, convert_element_type_80);  unsqueeze_118 = full_73 = convert_element_type_80 = None
        permute_305: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(where_51, [0, 2, 1, 3]);  where_51 = None
        view_348: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_305, _shape_param_512);  permute_305 = _shape_param_512 = None
        view_349: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_348, _shape_param_513);  view_348 = _shape_param_513 = None
        as_strided_33: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_349, _shape_param_514, _shape_param_515);  view_349 = _shape_param_514 = _shape_param_515 = None
        unsqueeze_119: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_33, 4);  as_strided_33 = None
        permute_306: "bf16[8, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_119, [0, 1, 4, 2, 3]);  unsqueeze_119 = None
        mul_48: "bf16[8, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_304, permute_306);  permute_304 = permute_306 = None
        view_350: "bf16[8, 3, 512, 512]" = torch.ops.aten.view.default(mul_48, _shape_param_516);  mul_48 = _shape_param_516 = None
        constant_pad_nd_25: "bf16[8, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_350, [0, 0, 0, 1], 0.0);  view_350 = None
        view_351: "bf16[8, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_25, _shape_param_517);  constant_pad_nd_25 = _shape_param_517 = None
        slice_380: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_351, 2, 0, 256)
        slice_381: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_380, 3, 0, 257);  slice_380 = None
        copy_78: "bf16[8, 3, 256, 257]" = torch.ops.aten.copy.default(slice_379, slice_381);  slice_379 = slice_381 = None
        slice_scatter_143: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_377, copy_78, 3, 256, 9223372036854775807);  slice_377 = copy_78 = None
        slice_scatter_144: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_71, slice_scatter_143, 1, 0, -1);  full_71 = slice_scatter_143 = None
        select_78: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_144, 1, -1)
        select_79: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_144, 1, -1)
        slice_382: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(select_79, 2, 256, 9223372036854775807);  select_79 = None
        select_80: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_351, 1, -1)
        slice_383: "bf16[8, 256, 513]" = torch.ops.aten.slice.Tensor(select_80, 1, 256, 9223372036854775807);  select_80 = None
        slice_384: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(slice_383, 2, 0, 257);  slice_383 = None
        copy_79: "bf16[8, 256, 257]" = torch.ops.aten.copy.default(slice_382, slice_384);  slice_382 = slice_384 = None
        slice_scatter_145: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_78, copy_79, 2, 256, 9223372036854775807);  select_78 = copy_79 = None
        select_scatter_26: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_144, slice_scatter_145, 1, -1);  slice_scatter_144 = slice_scatter_145 = None
        slice_385: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_26, 1, 1, 9223372036854775807)
        slice_386: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_26, 1, 1, 9223372036854775807)
        slice_387: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_386, 3, 0, 256);  slice_386 = None
        slice_388: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_351, 2, -257, -1)
        slice_389: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_388, 3, 257, 9223372036854775807);  slice_388 = None
        copy_80: "bf16[8, 3, 256, 256]" = torch.ops.aten.copy.default(slice_387, slice_389);  slice_387 = slice_389 = None
        slice_scatter_146: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_385, copy_80, 3, 0, 256);  slice_385 = copy_80 = None
        slice_scatter_147: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_26, slice_scatter_146, 1, 1, 9223372036854775807);  select_scatter_26 = slice_scatter_146 = None
        select_81: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_147, 1, 0)
        slice_390: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_81, 1, 1, 256)
        select_82: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_147, 1, 0)
        slice_391: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_82, 1, 1, 256);  select_82 = None
        slice_392: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_391, 2, 1, 256);  slice_391 = None
        select_83: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_351, 1, 0);  view_351 = None
        slice_393: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_83, 1, 0, 255);  select_83 = None
        slice_394: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_393, 2, -255, 9223372036854775807);  slice_393 = None
        copy_81: "bf16[8, 255, 255]" = torch.ops.aten.copy.default(slice_392, slice_394);  slice_392 = slice_394 = None
        slice_scatter_148: "bf16[8, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_390, copy_81, 2, 1, 256);  slice_390 = copy_81 = None
        slice_scatter_149: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_81, slice_scatter_148, 1, 1, 256);  select_81 = slice_scatter_148 = None
        select_scatter_27: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_147, slice_scatter_149, 1, 0);  slice_scatter_147 = slice_scatter_149 = None
        view_352: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_27, _shape_param_518);  _shape_param_518 = None
        permute_307: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None
        slice_395: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_307, 1, 0, 256)
        view_353: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_27, _shape_param_519);  _shape_param_519 = None
        permute_308: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None
        slice_396: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_308, 1, 0, 256);  permute_308 = None
        slice_397: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_396, 3, 0, 257);  slice_396 = None
        iota_26: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_120: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_26, -2);  iota_26 = None
        iota_27: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_121: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_27, -1);  iota_27 = None
        sub_31: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_120, unsqueeze_121);  unsqueeze_120 = unsqueeze_121 = None
        le_13: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_31, 0);  sub_31 = None
        full_74: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_520, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_520 = None
        scalar_tensor_13: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_52: "bf16[256, 257]" = torch.ops.aten.where.self(le_13, full_74, scalar_tensor_13);  le_13 = full_74 = scalar_tensor_13 = None
        rev_26: "bf16[256, 257]" = torch.ops.prims.rev.default(where_52, [0]);  where_52 = None
        unsqueeze_122: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_26, 0);  rev_26 = None
        unsqueeze_123: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, 2);  unsqueeze_122 = None
        expand_26: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_123, _shape_param_521);  _shape_param_521 = None
        convert_element_type_81: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_26, torch.bool);  expand_26 = None
        full_75: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_522, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_522 = None
        view_354: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_27, _shape_param_523);  select_scatter_27 = _shape_param_523 = None
        permute_309: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
        slice_398: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_309, 1, 0, 256);  permute_309 = None
        slice_399: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_398, 3, 0, 257);  slice_398 = None
        where_53: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_81, full_75, slice_399);  convert_element_type_81 = full_75 = slice_399 = None
        copy_82: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_397, where_53);  slice_397 = where_53 = None
        slice_scatter_150: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_395, copy_82, 3, 0, 257);  slice_395 = copy_82 = None
        slice_scatter_151: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_307, slice_scatter_150, 1, 0, 256);  permute_307 = slice_scatter_150 = None
        permute_310: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_151, [0, 2, 1, 3]);  slice_scatter_151 = None
        view_355: "bf16[8, 4, 256, 513]" = torch.ops.aten.view.default(permute_310, _shape_param_524);  permute_310 = _shape_param_524 = None
        view_356: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_355, _shape_param_525);  _shape_param_525 = None
        permute_311: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None
        slice_400: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_311, 1, -256, 9223372036854775807)
        view_357: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_355, _shape_param_526);  _shape_param_526 = None
        permute_312: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None
        slice_401: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_312, 1, -256, 9223372036854775807);  permute_312 = None
        slice_402: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_401, 3, -257, 9223372036854775807);  slice_401 = None
        rev_27: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_123, [1, 3]);  unsqueeze_123 = None
        expand_27: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(rev_27, _shape_param_527);  rev_27 = _shape_param_527 = None
        convert_element_type_82: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_27, torch.bool);  expand_27 = None
        full_76: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_528, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_528 = None
        view_358: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_355, _shape_param_529);  view_355 = _shape_param_529 = None
        permute_313: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
        slice_403: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_313, 1, -256, 9223372036854775807);  permute_313 = None
        slice_404: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_403, 3, -257, 9223372036854775807);  slice_403 = None
        where_54: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_82, full_76, slice_404);  convert_element_type_82 = full_76 = slice_404 = None
        copy_83: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_402, where_54);  slice_402 = where_54 = None
        slice_scatter_152: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_400, copy_83, 3, -257, 9223372036854775807);  slice_400 = copy_83 = None
        slice_scatter_153: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_311, slice_scatter_152, 1, -256, 9223372036854775807);  permute_311 = slice_scatter_152 = None
        permute_314: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_153, [0, 2, 1, 3]);  slice_scatter_153 = None
        permute_315: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_314, [0, 2, 1, 3]);  permute_314 = None
        add_74: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_302, permute_315);  permute_302 = permute_315 = None
        permute_316: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_74, [0, 2, 1, 3]);  add_74 = None
        permute_317: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_316, [0, 2, 1, 3]);  permute_316 = None
        convert_element_type_83: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_317, torch.float32);  permute_317 = None
        clone_64: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_83, memory_format = torch.contiguous_format);  convert_element_type_83 = None
        amax_6: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_64, [-1], True)
        sub_32: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_64, amax_6);  clone_64 = amax_6 = None
        exp_6: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
        sum_7: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_13: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        where_55: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_109, full_66, div_13);  unsqueeze_109 = full_66 = div_13 = None
        convert_element_type_84: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_55, torch.bfloat16);  where_55 = None
        permute_318: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_84, [0, 2, 1, 3]);  convert_element_type_84 = None
        clone_65: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_318, memory_format = torch.contiguous_format);  permute_318 = None
        view_359: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_65, _shape_param_530);  clone_65 = _shape_param_530 = None
        constant_pad_nd_26: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_359, _shape_param_531, 0.0);  view_359 = _shape_param_531 = None
        view_360: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_26, _shape_param_532);  constant_pad_nd_26 = _shape_param_532 = None
        slice_405: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_360, 2, 0, -256);  view_360 = None
        view_361: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_405, _shape_param_533);  slice_405 = _shape_param_533 = None
        slice_406: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_361, 3, 0, -1);  view_361 = None
        unsqueeze_124: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_406, 4);  slice_406 = None
        view_362: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_124, _shape_param_534);  unsqueeze_124 = _shape_param_534 = None
        clone_66: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_282, memory_format = torch.contiguous_format);  permute_282 = None
        view_363: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_66, _shape_param_535);  clone_66 = _shape_param_535 = None
        permute_319: "bf16[768, 768]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        mm_26: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_363, permute_319);  view_363 = permute_319 = None
        view_364: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_26, _shape_param_536);  mm_26 = _shape_param_536 = None
        add_75: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_364, arg104_1);  view_364 = arg104_1 = None
        view_365: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_75, _shape_param_537);  add_75 = _shape_param_537 = None
        permute_320: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_365, [1, 0, 2, 3]);  view_365 = None
        permute_321: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_320, [0, 2, 1, 3]);  permute_320 = None
        view_366: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_321, _shape_param_538);  permute_321 = _shape_param_538 = None
        constant_pad_nd_27: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_366, _shape_param_539, -1.0);  view_366 = _shape_param_539 = None
        as_strided_34: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_27, _shape_param_540, _shape_param_541);  constant_pad_nd_27 = _shape_param_540 = _shape_param_541 = None
        unsqueeze_125: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_34, 4);  as_strided_34 = None
        clone_67: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_125, memory_format = torch.contiguous_format);  unsqueeze_125 = None
        view_367: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_67, _shape_param_542);  clone_67 = _shape_param_542 = None
        bmm_13: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_362, view_367);  view_362 = view_367 = None
        view_368: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_13, _shape_param_543);  bmm_13 = _shape_param_543 = None
        permute_322: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_368, [0, 1, 2, 4, 3]);  view_368 = None
        view_369: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_322, _shape_param_544);  permute_322 = _shape_param_544 = None
        view_370: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_369, _shape_param_545);  view_369 = _shape_param_545 = None
        permute_323: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
        permute_324: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_323, [1, 0, 2, 3]);  permute_323 = None
        clone_68: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_324, memory_format = torch.contiguous_format);  permute_324 = None
        view_371: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_68, _shape_param_546);  clone_68 = _shape_param_546 = None
        permute_325: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_371, [1, 0, 2]);  view_371 = None
        clone_69: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
        view_372: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_69, _shape_param_547);  clone_69 = _shape_param_547 = None
        permute_326: "bf16[768, 768]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        mm_27: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_372, permute_326);  view_372 = permute_326 = None
        view_373: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_27, _shape_param_548);  mm_27 = _shape_param_548 = None
        add_76: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_373, arg106_1);  view_373 = arg106_1 = None
        add_77: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_76, convert_element_type_77);  add_76 = convert_element_type_77 = None
        convert_element_type_85: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_77, torch.float32);  add_77 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_85, [2], correction = 0, keepdim = True)
        getitem_24: "f32[8, 1024, 1]" = var_mean_12[0]
        getitem_25: "f32[8, 1024, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_33: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_85, getitem_25);  convert_element_type_85 = getitem_25 = None
        add_78: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_49: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_12);  sub_33 = rsqrt_12 = None
        mul_50: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_49, arg107_1);  mul_49 = arg107_1 = None
        add_79: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_50, arg108_1);  mul_50 = arg108_1 = None
        convert_element_type_86: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None
        view_374: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_86, _shape_param_549);  _shape_param_549 = None
        permute_327: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_12: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(arg110_1, view_374, permute_327);  arg110_1 = view_374 = permute_327 = None
        view_375: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_12, _shape_param_550);  addmm_12 = _shape_param_550 = None
        convert_element_type_87: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_375, torch.float32);  view_375 = None
        mul_51: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_87, 0.5)
        mul_52: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_87, 0.7071067811865476);  convert_element_type_87 = None
        erf_6: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_80: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_53: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_51, add_80);  mul_51 = add_80 = None
        convert_element_type_88: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16);  mul_53 = None
        view_376: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_88, _shape_param_551);  convert_element_type_88 = _shape_param_551 = None
        permute_328: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_13: "bf16[8192, 768]" = torch.ops.aten.addmm.default(arg112_1, view_376, permute_328);  arg112_1 = view_376 = permute_328 = None
        view_377: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_13, _shape_param_552);  addmm_13 = _shape_param_552 = None
        add_81: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_377, convert_element_type_86);  view_377 = convert_element_type_86 = None
        convert_element_type_89: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_81, torch.float32);  add_81 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_89, [2], correction = 0, keepdim = True)
        getitem_26: "f32[8, 1024, 1]" = var_mean_13[0]
        getitem_27: "f32[8, 1024, 1]" = var_mean_13[1];  var_mean_13 = None
        unsqueeze_126: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 2)
        unsqueeze_127: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 3);  unsqueeze_126 = None
        full_77: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_78: "bf16[96, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_553, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_553 = None
        slice_407: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_78, 1, 0, -1)
        slice_408: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_78, 1, 0, -1)
        slice_409: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_408, 3, 256, 9223372036854775807);  slice_408 = None
        sub_34: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_89, getitem_27);  convert_element_type_89 = getitem_27 = None
        add_82: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_54: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_13);  sub_34 = rsqrt_13 = None
        mul_55: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_54, arg113_1);  mul_54 = arg113_1 = None
        add_83: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_55, arg114_1);  mul_55 = arg114_1 = None
        convert_element_type_90: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None
        permute_329: "bf16[1024, 8, 768]" = torch.ops.aten.permute.default(convert_element_type_90, [1, 0, 2])
        clone_70: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format)
        view_378: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_70, _shape_param_554);  clone_70 = _shape_param_554 = None
        permute_330: "bf16[768, 768]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        mm_28: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_378, permute_330);  view_378 = permute_330 = None
        view_379: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_28, _shape_param_555);  mm_28 = _shape_param_555 = None
        add_84: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_379, arg116_1);  view_379 = arg116_1 = None
        div_14: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_84, 8.0);  add_84 = None
        view_380: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_14, _shape_param_556);  div_14 = _shape_param_556 = None
        permute_331: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_380, [1, 0, 2, 3]);  view_380 = None
        permute_332: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_331, [0, 2, 1, 3]);  permute_331 = None
        view_381: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_332, _shape_param_557);  permute_332 = _shape_param_557 = None
        view_382: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_381, _shape_param_558);  view_381 = _shape_param_558 = None
        as_strided_35: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_382, _shape_param_559, _shape_param_560);  view_382 = _shape_param_559 = _shape_param_560 = None
        unsqueeze_128: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_35, 4);  as_strided_35 = None
        clone_71: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_128, memory_format = torch.contiguous_format);  unsqueeze_128 = None
        view_383: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_71, _shape_param_561);  clone_71 = _shape_param_561 = None
        clone_72: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format)
        view_384: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_72, _shape_param_562);  clone_72 = _shape_param_562 = None
        permute_333: "bf16[768, 768]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_29: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_384, permute_333);  view_384 = permute_333 = None
        view_385: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_29, _shape_param_563);  mm_29 = _shape_param_563 = None
        add_85: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_385, arg118_1);  view_385 = arg118_1 = None
        view_386: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_85, _shape_param_564);  add_85 = _shape_param_564 = None
        permute_334: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_386, [1, 0, 2, 3]);  view_386 = None
        permute_335: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_334, [0, 2, 1, 3]);  permute_334 = None
        view_387: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_335, _shape_param_565);  permute_335 = _shape_param_565 = None
        view_388: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_387, _shape_param_566);  view_387 = _shape_param_566 = None
        as_strided_36: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_388, _shape_param_567, _shape_param_568);  view_388 = _shape_param_567 = _shape_param_568 = None
        unsqueeze_129: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_36, 4);  as_strided_36 = None
        permute_336: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_129, [0, 1, 4, 2, 3]);  unsqueeze_129 = None
        permute_337: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_336, [0, 1, 4, 3, 2]);  permute_336 = None
        clone_73: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
        view_389: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_73, _shape_param_569);  clone_73 = _shape_param_569 = None
        bmm_14: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_383, view_389);  view_383 = view_389 = None
        view_390: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_14, _shape_param_570);  bmm_14 = _shape_param_570 = None
        permute_338: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_390, [0, 1, 2, 4, 3]);  view_390 = None
        view_391: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_338, _shape_param_571);  permute_338 = _shape_param_571 = None
        constant_pad_nd_28: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_391, [0, 0, 0, 1], 0.0);  view_391 = None
        view_392: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_28, _shape_param_572);  constant_pad_nd_28 = _shape_param_572 = None
        slice_410: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_392, 2, 0, 256)
        slice_411: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_410, 3, 0, 257);  slice_410 = None
        copy_84: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_409, slice_411);  slice_409 = slice_411 = None
        slice_scatter_154: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_407, copy_84, 3, 256, 9223372036854775807);  slice_407 = copy_84 = None
        slice_scatter_155: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_78, slice_scatter_154, 1, 0, -1);  full_78 = slice_scatter_154 = None
        select_84: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_155, 1, -1)
        select_85: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_155, 1, -1)
        slice_412: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_85, 2, 256, 9223372036854775807);  select_85 = None
        select_86: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_392, 1, -1)
        slice_413: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_86, 1, 256, 9223372036854775807);  select_86 = None
        slice_414: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_413, 2, 0, 257);  slice_413 = None
        copy_85: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_412, slice_414);  slice_412 = slice_414 = None
        slice_scatter_156: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_84, copy_85, 2, 256, 9223372036854775807);  select_84 = copy_85 = None
        select_scatter_28: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_155, slice_scatter_156, 1, -1);  slice_scatter_155 = slice_scatter_156 = None
        slice_415: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_28, 1, 1, 9223372036854775807)
        slice_416: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_28, 1, 1, 9223372036854775807)
        slice_417: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_416, 3, 0, 256);  slice_416 = None
        slice_418: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_392, 2, -257, -1)
        slice_419: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_418, 3, 257, 9223372036854775807);  slice_418 = None
        copy_86: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_417, slice_419);  slice_417 = slice_419 = None
        slice_scatter_157: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_415, copy_86, 3, 0, 256);  slice_415 = copy_86 = None
        slice_scatter_158: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_28, slice_scatter_157, 1, 1, 9223372036854775807);  select_scatter_28 = slice_scatter_157 = None
        select_87: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_158, 1, 0)
        slice_420: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_87, 1, 1, 256)
        select_88: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_158, 1, 0)
        slice_421: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_88, 1, 1, 256);  select_88 = None
        slice_422: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_421, 2, 1, 256);  slice_421 = None
        select_89: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_392, 1, 0);  view_392 = None
        slice_423: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_89, 1, 0, 255);  select_89 = None
        slice_424: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_423, 2, -255, 9223372036854775807);  slice_423 = None
        copy_87: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_422, slice_424);  slice_422 = slice_424 = None
        slice_scatter_159: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_420, copy_87, 2, 1, 256);  slice_420 = copy_87 = None
        slice_scatter_160: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_87, slice_scatter_159, 1, 1, 256);  select_87 = slice_scatter_159 = None
        select_scatter_29: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_158, slice_scatter_160, 1, 0);  slice_scatter_158 = slice_scatter_160 = None
        view_393: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_29, _shape_param_573);  _shape_param_573 = None
        permute_339: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
        slice_425: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_339, 1, 0, 256)
        view_394: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_29, _shape_param_574);  _shape_param_574 = None
        permute_340: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_394, [0, 2, 1, 3]);  view_394 = None
        slice_426: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_340, 1, 0, 256);  permute_340 = None
        slice_427: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_426, 3, 0, 257);  slice_426 = None
        iota_28: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_130: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_28, -2);  iota_28 = None
        iota_29: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_131: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_29, -1);  iota_29 = None
        sub_35: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_130, unsqueeze_131);  unsqueeze_130 = unsqueeze_131 = None
        le_14: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_35, 0);  sub_35 = None
        full_79: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_575, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_575 = None
        scalar_tensor_14: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_56: "bf16[256, 257]" = torch.ops.aten.where.self(le_14, full_79, scalar_tensor_14);  le_14 = full_79 = scalar_tensor_14 = None
        rev_28: "bf16[256, 257]" = torch.ops.prims.rev.default(where_56, [0]);  where_56 = None
        unsqueeze_132: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_28, 0);  rev_28 = None
        unsqueeze_133: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 2);  unsqueeze_132 = None
        expand_28: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_133, _shape_param_576);  _shape_param_576 = None
        convert_element_type_91: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_28, torch.bool);  expand_28 = None
        full_80: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_577, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_577 = None
        permute_341: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_80, [0, 2, 1, 3]);  full_80 = None
        view_395: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_29, _shape_param_578);  select_scatter_29 = _shape_param_578 = None
        permute_342: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_395, [0, 2, 1, 3]);  view_395 = None
        slice_428: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_342, 1, 0, 256);  permute_342 = None
        slice_429: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_428, 3, 0, 257);  slice_428 = None
        where_57: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_91, permute_341, slice_429);  convert_element_type_91 = permute_341 = slice_429 = None
        copy_88: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_427, where_57);  slice_427 = where_57 = None
        slice_scatter_161: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_425, copy_88, 3, 0, 257);  slice_425 = copy_88 = None
        slice_scatter_162: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_339, slice_scatter_161, 1, 0, 256);  permute_339 = slice_scatter_161 = None
        permute_343: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_162, [0, 2, 1, 3]);  slice_scatter_162 = None
        view_396: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_343, _shape_param_579);  permute_343 = _shape_param_579 = None
        view_397: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_396, _shape_param_580);  _shape_param_580 = None
        permute_344: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_397, [0, 2, 1, 3]);  view_397 = None
        slice_430: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_344, 1, -256, 9223372036854775807)
        view_398: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_396, _shape_param_581);  _shape_param_581 = None
        permute_345: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
        slice_431: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_345, 1, -256, 9223372036854775807);  permute_345 = None
        slice_432: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_431, 3, -257, 9223372036854775807);  slice_431 = None
        rev_29: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_133, [1, 3]);  unsqueeze_133 = None
        expand_29: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_29, _shape_param_582);  rev_29 = _shape_param_582 = None
        convert_element_type_92: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_29, torch.bool);  expand_29 = None
        full_81: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_583, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_583 = None
        permute_346: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_81, [0, 2, 1, 3]);  full_81 = None
        view_399: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_396, _shape_param_584);  view_396 = _shape_param_584 = None
        permute_347: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
        slice_433: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_347, 1, -256, 9223372036854775807);  permute_347 = None
        slice_434: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_433, 3, -257, 9223372036854775807);  slice_433 = None
        where_58: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_92, permute_346, slice_434);  convert_element_type_92 = permute_346 = slice_434 = None
        copy_89: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_432, where_58);  slice_432 = where_58 = None
        slice_scatter_163: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_430, copy_89, 3, -257, 9223372036854775807);  slice_430 = copy_89 = None
        slice_scatter_164: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_344, slice_scatter_163, 1, -256, 9223372036854775807);  permute_344 = slice_scatter_163 = None
        permute_348: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_164, [0, 2, 1, 3]);  slice_scatter_164 = None
        permute_349: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_348, [0, 2, 1, 3]);  permute_348 = None
        full_82: "bf16[8, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_585, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_585 = None
        slice_435: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_82, 1, 0, -1)
        slice_436: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_82, 1, 0, -1)
        slice_437: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_436, 3, 256, 9223372036854775807);  slice_436 = None
        full_83: "bf16[8, 1024, 1, 1]" = torch.ops.aten.full.default(_shape_param_586, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_586 = None
        permute_350: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(full_83, [0, 2, 1, 3]);  full_83 = None
        view_400: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_350, _shape_param_587);  permute_350 = _shape_param_587 = None
        view_401: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_400, _shape_param_588);  view_400 = _shape_param_588 = None
        as_strided_37: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_401, _shape_param_589, _shape_param_590);  view_401 = _shape_param_589 = _shape_param_590 = None
        unsqueeze_134: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_37, 4);  as_strided_37 = None
        permute_351: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_134, [0, 1, 2, 4, 3]);  unsqueeze_134 = None
        full_84: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ne_7: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg6_1, 0)
        unsqueeze_135: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_7, 2);  ne_7 = None
        unsqueeze_136: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 3);  unsqueeze_135 = None
        convert_element_type_93: "bf16[8, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_136, torch.bfloat16)
        where_59: "bf16[8, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_136, full_84, convert_element_type_93);  unsqueeze_136 = full_84 = convert_element_type_93 = None
        permute_352: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(where_59, [0, 2, 1, 3]);  where_59 = None
        view_402: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_352, _shape_param_591);  permute_352 = _shape_param_591 = None
        view_403: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_402, _shape_param_592);  view_402 = _shape_param_592 = None
        as_strided_38: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_403, _shape_param_593, _shape_param_594);  view_403 = _shape_param_593 = _shape_param_594 = None
        unsqueeze_137: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_38, 4);  as_strided_38 = None
        permute_353: "bf16[8, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_137, [0, 1, 4, 2, 3]);  unsqueeze_137 = None
        mul_56: "bf16[8, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_351, permute_353);  permute_351 = permute_353 = None
        view_404: "bf16[8, 3, 512, 512]" = torch.ops.aten.view.default(mul_56, _shape_param_595);  mul_56 = _shape_param_595 = None
        constant_pad_nd_29: "bf16[8, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_404, [0, 0, 0, 1], 0.0);  view_404 = None
        view_405: "bf16[8, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_29, _shape_param_596);  constant_pad_nd_29 = _shape_param_596 = None
        slice_438: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_405, 2, 0, 256)
        slice_439: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_438, 3, 0, 257);  slice_438 = None
        copy_90: "bf16[8, 3, 256, 257]" = torch.ops.aten.copy.default(slice_437, slice_439);  slice_437 = slice_439 = None
        slice_scatter_165: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_435, copy_90, 3, 256, 9223372036854775807);  slice_435 = copy_90 = None
        slice_scatter_166: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_82, slice_scatter_165, 1, 0, -1);  full_82 = slice_scatter_165 = None
        select_90: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_166, 1, -1)
        select_91: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_166, 1, -1)
        slice_440: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(select_91, 2, 256, 9223372036854775807);  select_91 = None
        select_92: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_405, 1, -1)
        slice_441: "bf16[8, 256, 513]" = torch.ops.aten.slice.Tensor(select_92, 1, 256, 9223372036854775807);  select_92 = None
        slice_442: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(slice_441, 2, 0, 257);  slice_441 = None
        copy_91: "bf16[8, 256, 257]" = torch.ops.aten.copy.default(slice_440, slice_442);  slice_440 = slice_442 = None
        slice_scatter_167: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_90, copy_91, 2, 256, 9223372036854775807);  select_90 = copy_91 = None
        select_scatter_30: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_166, slice_scatter_167, 1, -1);  slice_scatter_166 = slice_scatter_167 = None
        slice_443: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_30, 1, 1, 9223372036854775807)
        slice_444: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_30, 1, 1, 9223372036854775807)
        slice_445: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_444, 3, 0, 256);  slice_444 = None
        slice_446: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_405, 2, -257, -1)
        slice_447: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_446, 3, 257, 9223372036854775807);  slice_446 = None
        copy_92: "bf16[8, 3, 256, 256]" = torch.ops.aten.copy.default(slice_445, slice_447);  slice_445 = slice_447 = None
        slice_scatter_168: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_443, copy_92, 3, 0, 256);  slice_443 = copy_92 = None
        slice_scatter_169: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_30, slice_scatter_168, 1, 1, 9223372036854775807);  select_scatter_30 = slice_scatter_168 = None
        select_93: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_169, 1, 0)
        slice_448: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_93, 1, 1, 256)
        select_94: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_169, 1, 0)
        slice_449: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_94, 1, 1, 256);  select_94 = None
        slice_450: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_449, 2, 1, 256);  slice_449 = None
        select_95: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_405, 1, 0);  view_405 = None
        slice_451: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_95, 1, 0, 255);  select_95 = None
        slice_452: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_451, 2, -255, 9223372036854775807);  slice_451 = None
        copy_93: "bf16[8, 255, 255]" = torch.ops.aten.copy.default(slice_450, slice_452);  slice_450 = slice_452 = None
        slice_scatter_170: "bf16[8, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_448, copy_93, 2, 1, 256);  slice_448 = copy_93 = None
        slice_scatter_171: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_93, slice_scatter_170, 1, 1, 256);  select_93 = slice_scatter_170 = None
        select_scatter_31: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_169, slice_scatter_171, 1, 0);  slice_scatter_169 = slice_scatter_171 = None
        view_406: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_31, _shape_param_597);  _shape_param_597 = None
        permute_354: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
        slice_453: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_354, 1, 0, 256)
        view_407: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_31, _shape_param_598);  _shape_param_598 = None
        permute_355: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_407, [0, 2, 1, 3]);  view_407 = None
        slice_454: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_355, 1, 0, 256);  permute_355 = None
        slice_455: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_454, 3, 0, 257);  slice_454 = None
        iota_30: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_138: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_30, -2);  iota_30 = None
        iota_31: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_139: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_31, -1);  iota_31 = None
        sub_36: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_138, unsqueeze_139);  unsqueeze_138 = unsqueeze_139 = None
        le_15: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_36, 0);  sub_36 = None
        full_85: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_599, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_599 = None
        scalar_tensor_15: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_60: "bf16[256, 257]" = torch.ops.aten.where.self(le_15, full_85, scalar_tensor_15);  le_15 = full_85 = scalar_tensor_15 = None
        rev_30: "bf16[256, 257]" = torch.ops.prims.rev.default(where_60, [0]);  where_60 = None
        unsqueeze_140: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_30, 0);  rev_30 = None
        unsqueeze_141: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 2);  unsqueeze_140 = None
        expand_30: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_141, _shape_param_600);  _shape_param_600 = None
        convert_element_type_94: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_30, torch.bool);  expand_30 = None
        full_86: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_601, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_601 = None
        view_408: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_31, _shape_param_602);  select_scatter_31 = _shape_param_602 = None
        permute_356: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_408, [0, 2, 1, 3]);  view_408 = None
        slice_456: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_356, 1, 0, 256);  permute_356 = None
        slice_457: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_456, 3, 0, 257);  slice_456 = None
        where_61: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_94, full_86, slice_457);  convert_element_type_94 = full_86 = slice_457 = None
        copy_94: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_455, where_61);  slice_455 = where_61 = None
        slice_scatter_172: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_453, copy_94, 3, 0, 257);  slice_453 = copy_94 = None
        slice_scatter_173: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_354, slice_scatter_172, 1, 0, 256);  permute_354 = slice_scatter_172 = None
        permute_357: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_173, [0, 2, 1, 3]);  slice_scatter_173 = None
        view_409: "bf16[8, 4, 256, 513]" = torch.ops.aten.view.default(permute_357, _shape_param_603);  permute_357 = _shape_param_603 = None
        view_410: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_409, _shape_param_604);  _shape_param_604 = None
        permute_358: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
        slice_458: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_358, 1, -256, 9223372036854775807)
        view_411: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_409, _shape_param_605);  _shape_param_605 = None
        permute_359: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_411, [0, 2, 1, 3]);  view_411 = None
        slice_459: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_359, 1, -256, 9223372036854775807);  permute_359 = None
        slice_460: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_459, 3, -257, 9223372036854775807);  slice_459 = None
        rev_31: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_141, [1, 3]);  unsqueeze_141 = None
        expand_31: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(rev_31, _shape_param_606);  rev_31 = _shape_param_606 = None
        convert_element_type_95: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_31, torch.bool);  expand_31 = None
        full_87: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_607, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_607 = None
        view_412: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_409, _shape_param_608);  view_409 = _shape_param_608 = None
        permute_360: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
        slice_461: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_360, 1, -256, 9223372036854775807);  permute_360 = None
        slice_462: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_461, 3, -257, 9223372036854775807);  slice_461 = None
        where_62: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_95, full_87, slice_462);  convert_element_type_95 = full_87 = slice_462 = None
        copy_95: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_460, where_62);  slice_460 = where_62 = None
        slice_scatter_174: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_458, copy_95, 3, -257, 9223372036854775807);  slice_458 = copy_95 = None
        slice_scatter_175: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_358, slice_scatter_174, 1, -256, 9223372036854775807);  permute_358 = slice_scatter_174 = None
        permute_361: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_175, [0, 2, 1, 3]);  slice_scatter_175 = None
        permute_362: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_361, [0, 2, 1, 3]);  permute_361 = None
        add_86: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_349, permute_362);  permute_349 = permute_362 = None
        permute_363: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_86, [0, 2, 1, 3]);  add_86 = None
        permute_364: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_363, [0, 2, 1, 3]);  permute_363 = None
        convert_element_type_96: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_364, torch.float32);  permute_364 = None
        clone_74: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_96, memory_format = torch.contiguous_format);  convert_element_type_96 = None
        amax_7: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_74, [-1], True)
        sub_37: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_74, amax_7);  clone_74 = amax_7 = None
        exp_7: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_8: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_15: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        where_63: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_127, full_77, div_15);  unsqueeze_127 = full_77 = div_15 = None
        convert_element_type_97: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_63, torch.bfloat16);  where_63 = None
        permute_365: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_97, [0, 2, 1, 3]);  convert_element_type_97 = None
        clone_75: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_365, memory_format = torch.contiguous_format);  permute_365 = None
        view_413: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_75, _shape_param_609);  clone_75 = _shape_param_609 = None
        constant_pad_nd_30: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_413, _shape_param_610, 0.0);  view_413 = _shape_param_610 = None
        view_414: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_30, _shape_param_611);  constant_pad_nd_30 = _shape_param_611 = None
        slice_463: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_414, 2, 0, -256);  view_414 = None
        view_415: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_463, _shape_param_612);  slice_463 = _shape_param_612 = None
        slice_464: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_415, 3, 0, -1);  view_415 = None
        unsqueeze_142: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_464, 4);  slice_464 = None
        view_416: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_142, _shape_param_613);  unsqueeze_142 = _shape_param_613 = None
        clone_76: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format);  permute_329 = None
        view_417: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_76, _shape_param_614);  clone_76 = _shape_param_614 = None
        permute_366: "bf16[768, 768]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        mm_30: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_417, permute_366);  view_417 = permute_366 = None
        view_418: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_30, _shape_param_615);  mm_30 = _shape_param_615 = None
        add_87: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_418, arg120_1);  view_418 = arg120_1 = None
        view_419: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_87, _shape_param_616);  add_87 = _shape_param_616 = None
        permute_367: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_419, [1, 0, 2, 3]);  view_419 = None
        permute_368: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_367, [0, 2, 1, 3]);  permute_367 = None
        view_420: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_368, _shape_param_617);  permute_368 = _shape_param_617 = None
        constant_pad_nd_31: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_420, _shape_param_618, -1.0);  view_420 = _shape_param_618 = None
        as_strided_39: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_31, _shape_param_619, _shape_param_620);  constant_pad_nd_31 = _shape_param_619 = _shape_param_620 = None
        unsqueeze_143: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_39, 4);  as_strided_39 = None
        clone_77: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_143, memory_format = torch.contiguous_format);  unsqueeze_143 = None
        view_421: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_77, _shape_param_621);  clone_77 = _shape_param_621 = None
        bmm_15: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_416, view_421);  view_416 = view_421 = None
        view_422: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_622);  bmm_15 = _shape_param_622 = None
        permute_369: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_422, [0, 1, 2, 4, 3]);  view_422 = None
        view_423: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_369, _shape_param_623);  permute_369 = _shape_param_623 = None
        view_424: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_423, _shape_param_624);  view_423 = _shape_param_624 = None
        permute_370: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
        permute_371: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_370, [1, 0, 2, 3]);  permute_370 = None
        clone_78: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_371, memory_format = torch.contiguous_format);  permute_371 = None
        view_425: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_78, _shape_param_625);  clone_78 = _shape_param_625 = None
        permute_372: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_425, [1, 0, 2]);  view_425 = None
        clone_79: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_372, memory_format = torch.contiguous_format);  permute_372 = None
        view_426: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_79, _shape_param_626);  clone_79 = _shape_param_626 = None
        permute_373: "bf16[768, 768]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        mm_31: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_426, permute_373);  view_426 = permute_373 = None
        view_427: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_31, _shape_param_627);  mm_31 = _shape_param_627 = None
        add_88: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_427, arg122_1);  view_427 = arg122_1 = None
        add_89: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_88, convert_element_type_90);  add_88 = convert_element_type_90 = None
        convert_element_type_98: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_89, torch.float32);  add_89 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_98, [2], correction = 0, keepdim = True)
        getitem_28: "f32[8, 1024, 1]" = var_mean_14[0]
        getitem_29: "f32[8, 1024, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_38: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_98, getitem_29);  convert_element_type_98 = getitem_29 = None
        add_90: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_57: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_14);  sub_38 = rsqrt_14 = None
        mul_58: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_57, arg123_1);  mul_57 = arg123_1 = None
        add_91: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_58, arg124_1);  mul_58 = arg124_1 = None
        convert_element_type_99: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16);  add_91 = None
        view_428: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_99, _shape_param_628);  _shape_param_628 = None
        permute_374: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_14: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(arg126_1, view_428, permute_374);  arg126_1 = view_428 = permute_374 = None
        view_429: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_14, _shape_param_629);  addmm_14 = _shape_param_629 = None
        convert_element_type_100: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_429, torch.float32);  view_429 = None
        mul_59: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_100, 0.5)
        mul_60: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_100, 0.7071067811865476);  convert_element_type_100 = None
        erf_7: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
        add_92: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_61: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_59, add_92);  mul_59 = add_92 = None
        convert_element_type_101: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None
        view_430: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_101, _shape_param_630);  convert_element_type_101 = _shape_param_630 = None
        permute_375: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_15: "bf16[8192, 768]" = torch.ops.aten.addmm.default(arg128_1, view_430, permute_375);  arg128_1 = view_430 = permute_375 = None
        view_431: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_15, _shape_param_631);  addmm_15 = _shape_param_631 = None
        add_93: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_431, convert_element_type_99);  view_431 = convert_element_type_99 = None
        convert_element_type_102: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_93, torch.float32);  add_93 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_102, [2], correction = 0, keepdim = True)
        getitem_30: "f32[8, 1024, 1]" = var_mean_15[0]
        getitem_31: "f32[8, 1024, 1]" = var_mean_15[1];  var_mean_15 = None
        unsqueeze_144: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 2)
        unsqueeze_145: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 3);  unsqueeze_144 = None
        full_88: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_89: "bf16[96, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_632, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_632 = None
        slice_465: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_89, 1, 0, -1)
        slice_466: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_89, 1, 0, -1)
        slice_467: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_466, 3, 256, 9223372036854775807);  slice_466 = None
        sub_39: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_102, getitem_31);  convert_element_type_102 = getitem_31 = None
        add_94: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_62: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_15);  sub_39 = rsqrt_15 = None
        mul_63: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_62, arg129_1);  mul_62 = arg129_1 = None
        add_95: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_63, arg130_1);  mul_63 = arg130_1 = None
        convert_element_type_103: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_95, torch.bfloat16);  add_95 = None
        permute_376: "bf16[1024, 8, 768]" = torch.ops.aten.permute.default(convert_element_type_103, [1, 0, 2])
        clone_80: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_376, memory_format = torch.contiguous_format)
        view_432: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_80, _shape_param_633);  clone_80 = _shape_param_633 = None
        permute_377: "bf16[768, 768]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        mm_32: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_432, permute_377);  view_432 = permute_377 = None
        view_433: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_32, _shape_param_634);  mm_32 = _shape_param_634 = None
        add_96: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_433, arg132_1);  view_433 = arg132_1 = None
        div_16: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_96, 8.0);  add_96 = None
        view_434: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_16, _shape_param_635);  div_16 = _shape_param_635 = None
        permute_378: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_434, [1, 0, 2, 3]);  view_434 = None
        permute_379: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_378, [0, 2, 1, 3]);  permute_378 = None
        view_435: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_379, _shape_param_636);  permute_379 = _shape_param_636 = None
        view_436: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_435, _shape_param_637);  view_435 = _shape_param_637 = None
        as_strided_40: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_436, _shape_param_638, _shape_param_639);  view_436 = _shape_param_638 = _shape_param_639 = None
        unsqueeze_146: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_40, 4);  as_strided_40 = None
        clone_81: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_146, memory_format = torch.contiguous_format);  unsqueeze_146 = None
        view_437: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_81, _shape_param_640);  clone_81 = _shape_param_640 = None
        clone_82: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_376, memory_format = torch.contiguous_format)
        view_438: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_82, _shape_param_641);  clone_82 = _shape_param_641 = None
        permute_380: "bf16[768, 768]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        mm_33: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_438, permute_380);  view_438 = permute_380 = None
        view_439: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_33, _shape_param_642);  mm_33 = _shape_param_642 = None
        add_97: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_439, arg134_1);  view_439 = arg134_1 = None
        view_440: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_97, _shape_param_643);  add_97 = _shape_param_643 = None
        permute_381: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_440, [1, 0, 2, 3]);  view_440 = None
        permute_382: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_381, [0, 2, 1, 3]);  permute_381 = None
        view_441: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_382, _shape_param_644);  permute_382 = _shape_param_644 = None
        view_442: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_441, _shape_param_645);  view_441 = _shape_param_645 = None
        as_strided_41: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_442, _shape_param_646, _shape_param_647);  view_442 = _shape_param_646 = _shape_param_647 = None
        unsqueeze_147: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_41, 4);  as_strided_41 = None
        permute_383: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_147, [0, 1, 4, 2, 3]);  unsqueeze_147 = None
        permute_384: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_383, [0, 1, 4, 3, 2]);  permute_383 = None
        clone_83: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_384, memory_format = torch.contiguous_format);  permute_384 = None
        view_443: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_83, _shape_param_648);  clone_83 = _shape_param_648 = None
        bmm_16: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_437, view_443);  view_437 = view_443 = None
        view_444: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_16, _shape_param_649);  bmm_16 = _shape_param_649 = None
        permute_385: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_444, [0, 1, 2, 4, 3]);  view_444 = None
        view_445: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_385, _shape_param_650);  permute_385 = _shape_param_650 = None
        constant_pad_nd_32: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_445, [0, 0, 0, 1], 0.0);  view_445 = None
        view_446: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_32, _shape_param_651);  constant_pad_nd_32 = _shape_param_651 = None
        slice_468: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_446, 2, 0, 256)
        slice_469: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_468, 3, 0, 257);  slice_468 = None
        copy_96: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_467, slice_469);  slice_467 = slice_469 = None
        slice_scatter_176: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_465, copy_96, 3, 256, 9223372036854775807);  slice_465 = copy_96 = None
        slice_scatter_177: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_89, slice_scatter_176, 1, 0, -1);  full_89 = slice_scatter_176 = None
        select_96: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_177, 1, -1)
        select_97: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_177, 1, -1)
        slice_470: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_97, 2, 256, 9223372036854775807);  select_97 = None
        select_98: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_446, 1, -1)
        slice_471: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_98, 1, 256, 9223372036854775807);  select_98 = None
        slice_472: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_471, 2, 0, 257);  slice_471 = None
        copy_97: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_470, slice_472);  slice_470 = slice_472 = None
        slice_scatter_178: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_96, copy_97, 2, 256, 9223372036854775807);  select_96 = copy_97 = None
        select_scatter_32: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_177, slice_scatter_178, 1, -1);  slice_scatter_177 = slice_scatter_178 = None
        slice_473: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_32, 1, 1, 9223372036854775807)
        slice_474: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_32, 1, 1, 9223372036854775807)
        slice_475: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_474, 3, 0, 256);  slice_474 = None
        slice_476: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_446, 2, -257, -1)
        slice_477: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_476, 3, 257, 9223372036854775807);  slice_476 = None
        copy_98: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_475, slice_477);  slice_475 = slice_477 = None
        slice_scatter_179: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_473, copy_98, 3, 0, 256);  slice_473 = copy_98 = None
        slice_scatter_180: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_32, slice_scatter_179, 1, 1, 9223372036854775807);  select_scatter_32 = slice_scatter_179 = None
        select_99: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_180, 1, 0)
        slice_478: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_99, 1, 1, 256)
        select_100: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_180, 1, 0)
        slice_479: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_100, 1, 1, 256);  select_100 = None
        slice_480: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_479, 2, 1, 256);  slice_479 = None
        select_101: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_446, 1, 0);  view_446 = None
        slice_481: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_101, 1, 0, 255);  select_101 = None
        slice_482: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_481, 2, -255, 9223372036854775807);  slice_481 = None
        copy_99: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_480, slice_482);  slice_480 = slice_482 = None
        slice_scatter_181: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_478, copy_99, 2, 1, 256);  slice_478 = copy_99 = None
        slice_scatter_182: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_99, slice_scatter_181, 1, 1, 256);  select_99 = slice_scatter_181 = None
        select_scatter_33: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_180, slice_scatter_182, 1, 0);  slice_scatter_180 = slice_scatter_182 = None
        view_447: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_33, _shape_param_652);  _shape_param_652 = None
        permute_386: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None
        slice_483: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_386, 1, 0, 256)
        view_448: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_33, _shape_param_653);  _shape_param_653 = None
        permute_387: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
        slice_484: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_387, 1, 0, 256);  permute_387 = None
        slice_485: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_484, 3, 0, 257);  slice_484 = None
        iota_32: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_148: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_32, -2);  iota_32 = None
        iota_33: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_149: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_33, -1);  iota_33 = None
        sub_40: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_148, unsqueeze_149);  unsqueeze_148 = unsqueeze_149 = None
        le_16: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_40, 0);  sub_40 = None
        full_90: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_654, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_654 = None
        scalar_tensor_16: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_64: "bf16[256, 257]" = torch.ops.aten.where.self(le_16, full_90, scalar_tensor_16);  le_16 = full_90 = scalar_tensor_16 = None
        rev_32: "bf16[256, 257]" = torch.ops.prims.rev.default(where_64, [0]);  where_64 = None
        unsqueeze_150: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_32, 0);  rev_32 = None
        unsqueeze_151: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 2);  unsqueeze_150 = None
        expand_32: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_151, _shape_param_655);  _shape_param_655 = None
        convert_element_type_104: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_32, torch.bool);  expand_32 = None
        full_91: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_656, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_656 = None
        permute_388: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_91, [0, 2, 1, 3]);  full_91 = None
        view_449: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_33, _shape_param_657);  select_scatter_33 = _shape_param_657 = None
        permute_389: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_449, [0, 2, 1, 3]);  view_449 = None
        slice_486: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_389, 1, 0, 256);  permute_389 = None
        slice_487: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_486, 3, 0, 257);  slice_486 = None
        where_65: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_104, permute_388, slice_487);  convert_element_type_104 = permute_388 = slice_487 = None
        copy_100: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_485, where_65);  slice_485 = where_65 = None
        slice_scatter_183: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_483, copy_100, 3, 0, 257);  slice_483 = copy_100 = None
        slice_scatter_184: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_386, slice_scatter_183, 1, 0, 256);  permute_386 = slice_scatter_183 = None
        permute_390: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_184, [0, 2, 1, 3]);  slice_scatter_184 = None
        view_450: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_390, _shape_param_658);  permute_390 = _shape_param_658 = None
        view_451: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_450, _shape_param_659);  _shape_param_659 = None
        permute_391: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_451, [0, 2, 1, 3]);  view_451 = None
        slice_488: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_391, 1, -256, 9223372036854775807)
        view_452: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_450, _shape_param_660);  _shape_param_660 = None
        permute_392: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None
        slice_489: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_392, 1, -256, 9223372036854775807);  permute_392 = None
        slice_490: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_489, 3, -257, 9223372036854775807);  slice_489 = None
        rev_33: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_151, [1, 3]);  unsqueeze_151 = None
        expand_33: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_33, _shape_param_661);  rev_33 = _shape_param_661 = None
        convert_element_type_105: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_33, torch.bool);  expand_33 = None
        full_92: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_662, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_662 = None
        permute_393: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_92, [0, 2, 1, 3]);  full_92 = None
        view_453: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_450, _shape_param_663);  view_450 = _shape_param_663 = None
        permute_394: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
        slice_491: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_394, 1, -256, 9223372036854775807);  permute_394 = None
        slice_492: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_491, 3, -257, 9223372036854775807);  slice_491 = None
        where_66: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_105, permute_393, slice_492);  convert_element_type_105 = permute_393 = slice_492 = None
        copy_101: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_490, where_66);  slice_490 = where_66 = None
        slice_scatter_185: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_488, copy_101, 3, -257, 9223372036854775807);  slice_488 = copy_101 = None
        slice_scatter_186: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_391, slice_scatter_185, 1, -256, 9223372036854775807);  permute_391 = slice_scatter_185 = None
        permute_395: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_186, [0, 2, 1, 3]);  slice_scatter_186 = None
        permute_396: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_395, [0, 2, 1, 3]);  permute_395 = None
        full_93: "bf16[8, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_664, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_664 = None
        slice_493: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_93, 1, 0, -1)
        slice_494: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_93, 1, 0, -1)
        slice_495: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_494, 3, 256, 9223372036854775807);  slice_494 = None
        full_94: "bf16[8, 1024, 1, 1]" = torch.ops.aten.full.default(_shape_param_665, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_665 = None
        permute_397: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(full_94, [0, 2, 1, 3]);  full_94 = None
        view_454: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_397, _shape_param_666);  permute_397 = _shape_param_666 = None
        view_455: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_454, _shape_param_667);  view_454 = _shape_param_667 = None
        as_strided_42: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_455, _shape_param_668, _shape_param_669);  view_455 = _shape_param_668 = _shape_param_669 = None
        unsqueeze_152: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_42, 4);  as_strided_42 = None
        permute_398: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_152, [0, 1, 2, 4, 3]);  unsqueeze_152 = None
        full_95: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ne_8: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg6_1, 0)
        unsqueeze_153: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_8, 2);  ne_8 = None
        unsqueeze_154: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 3);  unsqueeze_153 = None
        convert_element_type_106: "bf16[8, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_154, torch.bfloat16)
        where_67: "bf16[8, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_154, full_95, convert_element_type_106);  unsqueeze_154 = full_95 = convert_element_type_106 = None
        permute_399: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(where_67, [0, 2, 1, 3]);  where_67 = None
        view_456: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_399, _shape_param_670);  permute_399 = _shape_param_670 = None
        view_457: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_456, _shape_param_671);  view_456 = _shape_param_671 = None
        as_strided_43: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_457, _shape_param_672, _shape_param_673);  view_457 = _shape_param_672 = _shape_param_673 = None
        unsqueeze_155: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_43, 4);  as_strided_43 = None
        permute_400: "bf16[8, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_155, [0, 1, 4, 2, 3]);  unsqueeze_155 = None
        mul_64: "bf16[8, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_398, permute_400);  permute_398 = permute_400 = None
        view_458: "bf16[8, 3, 512, 512]" = torch.ops.aten.view.default(mul_64, _shape_param_674);  mul_64 = _shape_param_674 = None
        constant_pad_nd_33: "bf16[8, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_458, [0, 0, 0, 1], 0.0);  view_458 = None
        view_459: "bf16[8, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_33, _shape_param_675);  constant_pad_nd_33 = _shape_param_675 = None
        slice_496: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_459, 2, 0, 256)
        slice_497: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_496, 3, 0, 257);  slice_496 = None
        copy_102: "bf16[8, 3, 256, 257]" = torch.ops.aten.copy.default(slice_495, slice_497);  slice_495 = slice_497 = None
        slice_scatter_187: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_493, copy_102, 3, 256, 9223372036854775807);  slice_493 = copy_102 = None
        slice_scatter_188: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_93, slice_scatter_187, 1, 0, -1);  full_93 = slice_scatter_187 = None
        select_102: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_188, 1, -1)
        select_103: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_188, 1, -1)
        slice_498: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(select_103, 2, 256, 9223372036854775807);  select_103 = None
        select_104: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_459, 1, -1)
        slice_499: "bf16[8, 256, 513]" = torch.ops.aten.slice.Tensor(select_104, 1, 256, 9223372036854775807);  select_104 = None
        slice_500: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(slice_499, 2, 0, 257);  slice_499 = None
        copy_103: "bf16[8, 256, 257]" = torch.ops.aten.copy.default(slice_498, slice_500);  slice_498 = slice_500 = None
        slice_scatter_189: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_102, copy_103, 2, 256, 9223372036854775807);  select_102 = copy_103 = None
        select_scatter_34: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_188, slice_scatter_189, 1, -1);  slice_scatter_188 = slice_scatter_189 = None
        slice_501: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_34, 1, 1, 9223372036854775807)
        slice_502: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_34, 1, 1, 9223372036854775807)
        slice_503: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_502, 3, 0, 256);  slice_502 = None
        slice_504: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_459, 2, -257, -1)
        slice_505: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_504, 3, 257, 9223372036854775807);  slice_504 = None
        copy_104: "bf16[8, 3, 256, 256]" = torch.ops.aten.copy.default(slice_503, slice_505);  slice_503 = slice_505 = None
        slice_scatter_190: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_501, copy_104, 3, 0, 256);  slice_501 = copy_104 = None
        slice_scatter_191: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_34, slice_scatter_190, 1, 1, 9223372036854775807);  select_scatter_34 = slice_scatter_190 = None
        select_105: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_191, 1, 0)
        slice_506: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_105, 1, 1, 256)
        select_106: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_191, 1, 0)
        slice_507: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_106, 1, 1, 256);  select_106 = None
        slice_508: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_507, 2, 1, 256);  slice_507 = None
        select_107: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_459, 1, 0);  view_459 = None
        slice_509: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_107, 1, 0, 255);  select_107 = None
        slice_510: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_509, 2, -255, 9223372036854775807);  slice_509 = None
        copy_105: "bf16[8, 255, 255]" = torch.ops.aten.copy.default(slice_508, slice_510);  slice_508 = slice_510 = None
        slice_scatter_192: "bf16[8, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_506, copy_105, 2, 1, 256);  slice_506 = copy_105 = None
        slice_scatter_193: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_105, slice_scatter_192, 1, 1, 256);  select_105 = slice_scatter_192 = None
        select_scatter_35: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_191, slice_scatter_193, 1, 0);  slice_scatter_191 = slice_scatter_193 = None
        view_460: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_35, _shape_param_676);  _shape_param_676 = None
        permute_401: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_460, [0, 2, 1, 3]);  view_460 = None
        slice_511: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_401, 1, 0, 256)
        view_461: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_35, _shape_param_677);  _shape_param_677 = None
        permute_402: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_461, [0, 2, 1, 3]);  view_461 = None
        slice_512: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_402, 1, 0, 256);  permute_402 = None
        slice_513: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_512, 3, 0, 257);  slice_512 = None
        iota_34: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_156: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_34, -2);  iota_34 = None
        iota_35: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_157: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_35, -1);  iota_35 = None
        sub_41: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_156, unsqueeze_157);  unsqueeze_156 = unsqueeze_157 = None
        le_17: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_41, 0);  sub_41 = None
        full_96: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_678, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_678 = None
        scalar_tensor_17: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_68: "bf16[256, 257]" = torch.ops.aten.where.self(le_17, full_96, scalar_tensor_17);  le_17 = full_96 = scalar_tensor_17 = None
        rev_34: "bf16[256, 257]" = torch.ops.prims.rev.default(where_68, [0]);  where_68 = None
        unsqueeze_158: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_34, 0);  rev_34 = None
        unsqueeze_159: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 2);  unsqueeze_158 = None
        expand_34: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_159, _shape_param_679);  _shape_param_679 = None
        convert_element_type_107: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_34, torch.bool);  expand_34 = None
        full_97: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_680, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_680 = None
        view_462: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_35, _shape_param_681);  select_scatter_35 = _shape_param_681 = None
        permute_403: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None
        slice_514: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_403, 1, 0, 256);  permute_403 = None
        slice_515: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_514, 3, 0, 257);  slice_514 = None
        where_69: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_107, full_97, slice_515);  convert_element_type_107 = full_97 = slice_515 = None
        copy_106: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_513, where_69);  slice_513 = where_69 = None
        slice_scatter_194: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_511, copy_106, 3, 0, 257);  slice_511 = copy_106 = None
        slice_scatter_195: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_401, slice_scatter_194, 1, 0, 256);  permute_401 = slice_scatter_194 = None
        permute_404: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_195, [0, 2, 1, 3]);  slice_scatter_195 = None
        view_463: "bf16[8, 4, 256, 513]" = torch.ops.aten.view.default(permute_404, _shape_param_682);  permute_404 = _shape_param_682 = None
        view_464: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_463, _shape_param_683);  _shape_param_683 = None
        permute_405: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_464, [0, 2, 1, 3]);  view_464 = None
        slice_516: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_405, 1, -256, 9223372036854775807)
        view_465: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_463, _shape_param_684);  _shape_param_684 = None
        permute_406: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_465, [0, 2, 1, 3]);  view_465 = None
        slice_517: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_406, 1, -256, 9223372036854775807);  permute_406 = None
        slice_518: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_517, 3, -257, 9223372036854775807);  slice_517 = None
        rev_35: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_159, [1, 3]);  unsqueeze_159 = None
        expand_35: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(rev_35, _shape_param_685);  rev_35 = _shape_param_685 = None
        convert_element_type_108: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_35, torch.bool);  expand_35 = None
        full_98: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_686, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_686 = None
        view_466: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_463, _shape_param_687);  view_463 = _shape_param_687 = None
        permute_407: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None
        slice_519: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_407, 1, -256, 9223372036854775807);  permute_407 = None
        slice_520: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_519, 3, -257, 9223372036854775807);  slice_519 = None
        where_70: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_108, full_98, slice_520);  convert_element_type_108 = full_98 = slice_520 = None
        copy_107: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_518, where_70);  slice_518 = where_70 = None
        slice_scatter_196: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_516, copy_107, 3, -257, 9223372036854775807);  slice_516 = copy_107 = None
        slice_scatter_197: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_405, slice_scatter_196, 1, -256, 9223372036854775807);  permute_405 = slice_scatter_196 = None
        permute_408: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_197, [0, 2, 1, 3]);  slice_scatter_197 = None
        permute_409: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_408, [0, 2, 1, 3]);  permute_408 = None
        add_98: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_396, permute_409);  permute_396 = permute_409 = None
        permute_410: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_98, [0, 2, 1, 3]);  add_98 = None
        permute_411: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_410, [0, 2, 1, 3]);  permute_410 = None
        convert_element_type_109: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_411, torch.float32);  permute_411 = None
        clone_84: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_109, memory_format = torch.contiguous_format);  convert_element_type_109 = None
        amax_8: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_84, [-1], True)
        sub_42: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_84, amax_8);  clone_84 = amax_8 = None
        exp_8: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
        sum_9: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_17: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        where_71: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_145, full_88, div_17);  unsqueeze_145 = full_88 = div_17 = None
        convert_element_type_110: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_71, torch.bfloat16);  where_71 = None
        permute_412: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_110, [0, 2, 1, 3]);  convert_element_type_110 = None
        clone_85: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_412, memory_format = torch.contiguous_format);  permute_412 = None
        view_467: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_85, _shape_param_688);  clone_85 = _shape_param_688 = None
        constant_pad_nd_34: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_467, _shape_param_689, 0.0);  view_467 = _shape_param_689 = None
        view_468: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_34, _shape_param_690);  constant_pad_nd_34 = _shape_param_690 = None
        slice_521: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_468, 2, 0, -256);  view_468 = None
        view_469: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_521, _shape_param_691);  slice_521 = _shape_param_691 = None
        slice_522: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_469, 3, 0, -1);  view_469 = None
        unsqueeze_160: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_522, 4);  slice_522 = None
        view_470: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_160, _shape_param_692);  unsqueeze_160 = _shape_param_692 = None
        clone_86: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_376, memory_format = torch.contiguous_format);  permute_376 = None
        view_471: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_86, _shape_param_693);  clone_86 = _shape_param_693 = None
        permute_413: "bf16[768, 768]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        mm_34: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_471, permute_413);  view_471 = permute_413 = None
        view_472: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_34, _shape_param_694);  mm_34 = _shape_param_694 = None
        add_99: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_472, arg136_1);  view_472 = arg136_1 = None
        view_473: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_99, _shape_param_695);  add_99 = _shape_param_695 = None
        permute_414: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_473, [1, 0, 2, 3]);  view_473 = None
        permute_415: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_414, [0, 2, 1, 3]);  permute_414 = None
        view_474: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_415, _shape_param_696);  permute_415 = _shape_param_696 = None
        constant_pad_nd_35: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_474, _shape_param_697, -1.0);  view_474 = _shape_param_697 = None
        as_strided_44: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_35, _shape_param_698, _shape_param_699);  constant_pad_nd_35 = _shape_param_698 = _shape_param_699 = None
        unsqueeze_161: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_44, 4);  as_strided_44 = None
        clone_87: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_161, memory_format = torch.contiguous_format);  unsqueeze_161 = None
        view_475: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_87, _shape_param_700);  clone_87 = _shape_param_700 = None
        bmm_17: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_470, view_475);  view_470 = view_475 = None
        view_476: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_17, _shape_param_701);  bmm_17 = _shape_param_701 = None
        permute_416: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_476, [0, 1, 2, 4, 3]);  view_476 = None
        view_477: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_416, _shape_param_702);  permute_416 = _shape_param_702 = None
        view_478: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_477, _shape_param_703);  view_477 = _shape_param_703 = None
        permute_417: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_478, [0, 2, 1, 3]);  view_478 = None
        permute_418: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_417, [1, 0, 2, 3]);  permute_417 = None
        clone_88: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
        view_479: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_88, _shape_param_704);  clone_88 = _shape_param_704 = None
        permute_419: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_479, [1, 0, 2]);  view_479 = None
        clone_89: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
        view_480: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_89, _shape_param_705);  clone_89 = _shape_param_705 = None
        permute_420: "bf16[768, 768]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        mm_35: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_480, permute_420);  view_480 = permute_420 = None
        view_481: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_35, _shape_param_706);  mm_35 = _shape_param_706 = None
        add_100: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_481, arg138_1);  view_481 = arg138_1 = None
        add_101: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_100, convert_element_type_103);  add_100 = convert_element_type_103 = None
        convert_element_type_111: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_101, torch.float32);  add_101 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_111, [2], correction = 0, keepdim = True)
        getitem_32: "f32[8, 1024, 1]" = var_mean_16[0]
        getitem_33: "f32[8, 1024, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_43: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_111, getitem_33);  convert_element_type_111 = getitem_33 = None
        add_102: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_65: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_16);  sub_43 = rsqrt_16 = None
        mul_66: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_65, arg139_1);  mul_65 = arg139_1 = None
        add_103: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_66, arg140_1);  mul_66 = arg140_1 = None
        convert_element_type_112: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16);  add_103 = None
        view_482: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_112, _shape_param_707);  _shape_param_707 = None
        permute_421: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_16: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(arg142_1, view_482, permute_421);  arg142_1 = view_482 = permute_421 = None
        view_483: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_16, _shape_param_708);  addmm_16 = _shape_param_708 = None
        convert_element_type_113: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_483, torch.float32);  view_483 = None
        mul_67: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_113, 0.5)
        mul_68: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_113, 0.7071067811865476);  convert_element_type_113 = None
        erf_8: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_104: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_69: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_104);  mul_67 = add_104 = None
        convert_element_type_114: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        view_484: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_114, _shape_param_709);  convert_element_type_114 = _shape_param_709 = None
        permute_422: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_17: "bf16[8192, 768]" = torch.ops.aten.addmm.default(arg144_1, view_484, permute_422);  arg144_1 = view_484 = permute_422 = None
        view_485: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_17, _shape_param_710);  addmm_17 = _shape_param_710 = None
        add_105: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_485, convert_element_type_112);  view_485 = convert_element_type_112 = None
        convert_element_type_115: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_105, torch.float32);  add_105 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_115, [2], correction = 0, keepdim = True)
        getitem_34: "f32[8, 1024, 1]" = var_mean_17[0]
        getitem_35: "f32[8, 1024, 1]" = var_mean_17[1];  var_mean_17 = None
        unsqueeze_162: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 2)
        unsqueeze_163: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 3);  unsqueeze_162 = None
        full_99: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_100: "bf16[96, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_711, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_711 = None
        slice_523: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_100, 1, 0, -1)
        slice_524: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_100, 1, 0, -1)
        slice_525: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_524, 3, 256, 9223372036854775807);  slice_524 = None
        sub_44: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_115, getitem_35);  convert_element_type_115 = getitem_35 = None
        add_106: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        mul_70: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_17);  sub_44 = rsqrt_17 = None
        mul_71: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_70, arg145_1);  mul_70 = arg145_1 = None
        add_107: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_71, arg146_1);  mul_71 = arg146_1 = None
        convert_element_type_116: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_107, torch.bfloat16);  add_107 = None
        permute_423: "bf16[1024, 8, 768]" = torch.ops.aten.permute.default(convert_element_type_116, [1, 0, 2])
        clone_90: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_423, memory_format = torch.contiguous_format)
        view_486: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_90, _shape_param_712);  clone_90 = _shape_param_712 = None
        permute_424: "bf16[768, 768]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        mm_36: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_486, permute_424);  view_486 = permute_424 = None
        view_487: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_36, _shape_param_713);  mm_36 = _shape_param_713 = None
        add_108: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_487, arg148_1);  view_487 = arg148_1 = None
        div_18: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_108, 8.0);  add_108 = None
        view_488: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_18, _shape_param_714);  div_18 = _shape_param_714 = None
        permute_425: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_488, [1, 0, 2, 3]);  view_488 = None
        permute_426: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_425, [0, 2, 1, 3]);  permute_425 = None
        view_489: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_426, _shape_param_715);  permute_426 = _shape_param_715 = None
        view_490: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_489, _shape_param_716);  view_489 = _shape_param_716 = None
        as_strided_45: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_490, _shape_param_717, _shape_param_718);  view_490 = _shape_param_717 = _shape_param_718 = None
        unsqueeze_164: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_45, 4);  as_strided_45 = None
        clone_91: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_164, memory_format = torch.contiguous_format);  unsqueeze_164 = None
        view_491: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_91, _shape_param_719);  clone_91 = _shape_param_719 = None
        clone_92: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_423, memory_format = torch.contiguous_format)
        view_492: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_92, _shape_param_720);  clone_92 = _shape_param_720 = None
        permute_427: "bf16[768, 768]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        mm_37: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_492, permute_427);  view_492 = permute_427 = None
        view_493: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_37, _shape_param_721);  mm_37 = _shape_param_721 = None
        add_109: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_493, arg150_1);  view_493 = arg150_1 = None
        view_494: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_109, _shape_param_722);  add_109 = _shape_param_722 = None
        permute_428: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_494, [1, 0, 2, 3]);  view_494 = None
        permute_429: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_428, [0, 2, 1, 3]);  permute_428 = None
        view_495: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_429, _shape_param_723);  permute_429 = _shape_param_723 = None
        view_496: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_495, _shape_param_724);  view_495 = _shape_param_724 = None
        as_strided_46: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_496, _shape_param_725, _shape_param_726);  view_496 = _shape_param_725 = _shape_param_726 = None
        unsqueeze_165: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_46, 4);  as_strided_46 = None
        permute_430: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_165, [0, 1, 4, 2, 3]);  unsqueeze_165 = None
        permute_431: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_430, [0, 1, 4, 3, 2]);  permute_430 = None
        clone_93: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_431, memory_format = torch.contiguous_format);  permute_431 = None
        view_497: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_93, _shape_param_727);  clone_93 = _shape_param_727 = None
        bmm_18: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_491, view_497);  view_491 = view_497 = None
        view_498: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_18, _shape_param_728);  bmm_18 = _shape_param_728 = None
        permute_432: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_498, [0, 1, 2, 4, 3]);  view_498 = None
        view_499: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_432, _shape_param_729);  permute_432 = _shape_param_729 = None
        constant_pad_nd_36: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_499, [0, 0, 0, 1], 0.0);  view_499 = None
        view_500: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_36, _shape_param_730);  constant_pad_nd_36 = _shape_param_730 = None
        slice_526: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_500, 2, 0, 256)
        slice_527: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_526, 3, 0, 257);  slice_526 = None
        copy_108: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_525, slice_527);  slice_525 = slice_527 = None
        slice_scatter_198: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_523, copy_108, 3, 256, 9223372036854775807);  slice_523 = copy_108 = None
        slice_scatter_199: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_100, slice_scatter_198, 1, 0, -1);  full_100 = slice_scatter_198 = None
        select_108: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_199, 1, -1)
        select_109: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_199, 1, -1)
        slice_528: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_109, 2, 256, 9223372036854775807);  select_109 = None
        select_110: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_500, 1, -1)
        slice_529: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_110, 1, 256, 9223372036854775807);  select_110 = None
        slice_530: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_529, 2, 0, 257);  slice_529 = None
        copy_109: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_528, slice_530);  slice_528 = slice_530 = None
        slice_scatter_200: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_108, copy_109, 2, 256, 9223372036854775807);  select_108 = copy_109 = None
        select_scatter_36: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_199, slice_scatter_200, 1, -1);  slice_scatter_199 = slice_scatter_200 = None
        slice_531: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_36, 1, 1, 9223372036854775807)
        slice_532: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_36, 1, 1, 9223372036854775807)
        slice_533: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_532, 3, 0, 256);  slice_532 = None
        slice_534: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_500, 2, -257, -1)
        slice_535: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_534, 3, 257, 9223372036854775807);  slice_534 = None
        copy_110: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_533, slice_535);  slice_533 = slice_535 = None
        slice_scatter_201: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_531, copy_110, 3, 0, 256);  slice_531 = copy_110 = None
        slice_scatter_202: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_36, slice_scatter_201, 1, 1, 9223372036854775807);  select_scatter_36 = slice_scatter_201 = None
        select_111: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_202, 1, 0)
        slice_536: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_111, 1, 1, 256)
        select_112: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_202, 1, 0)
        slice_537: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_112, 1, 1, 256);  select_112 = None
        slice_538: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_537, 2, 1, 256);  slice_537 = None
        select_113: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_500, 1, 0);  view_500 = None
        slice_539: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_113, 1, 0, 255);  select_113 = None
        slice_540: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_539, 2, -255, 9223372036854775807);  slice_539 = None
        copy_111: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_538, slice_540);  slice_538 = slice_540 = None
        slice_scatter_203: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_536, copy_111, 2, 1, 256);  slice_536 = copy_111 = None
        slice_scatter_204: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_111, slice_scatter_203, 1, 1, 256);  select_111 = slice_scatter_203 = None
        select_scatter_37: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_202, slice_scatter_204, 1, 0);  slice_scatter_202 = slice_scatter_204 = None
        view_501: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_37, _shape_param_731);  _shape_param_731 = None
        permute_433: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_501, [0, 2, 1, 3]);  view_501 = None
        slice_541: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_433, 1, 0, 256)
        view_502: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_37, _shape_param_732);  _shape_param_732 = None
        permute_434: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_502, [0, 2, 1, 3]);  view_502 = None
        slice_542: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_434, 1, 0, 256);  permute_434 = None
        slice_543: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_542, 3, 0, 257);  slice_542 = None
        iota_36: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_166: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_36, -2);  iota_36 = None
        iota_37: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_167: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_37, -1);  iota_37 = None
        sub_45: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_166, unsqueeze_167);  unsqueeze_166 = unsqueeze_167 = None
        le_18: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_45, 0);  sub_45 = None
        full_101: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_733, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_733 = None
        scalar_tensor_18: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_72: "bf16[256, 257]" = torch.ops.aten.where.self(le_18, full_101, scalar_tensor_18);  le_18 = full_101 = scalar_tensor_18 = None
        rev_36: "bf16[256, 257]" = torch.ops.prims.rev.default(where_72, [0]);  where_72 = None
        unsqueeze_168: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_36, 0);  rev_36 = None
        unsqueeze_169: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 2);  unsqueeze_168 = None
        expand_36: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_169, _shape_param_734);  _shape_param_734 = None
        convert_element_type_117: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_36, torch.bool);  expand_36 = None
        full_102: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_735, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_735 = None
        permute_435: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_102, [0, 2, 1, 3]);  full_102 = None
        view_503: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_37, _shape_param_736);  select_scatter_37 = _shape_param_736 = None
        permute_436: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_503, [0, 2, 1, 3]);  view_503 = None
        slice_544: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_436, 1, 0, 256);  permute_436 = None
        slice_545: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_544, 3, 0, 257);  slice_544 = None
        where_73: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_117, permute_435, slice_545);  convert_element_type_117 = permute_435 = slice_545 = None
        copy_112: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_543, where_73);  slice_543 = where_73 = None
        slice_scatter_205: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_541, copy_112, 3, 0, 257);  slice_541 = copy_112 = None
        slice_scatter_206: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_433, slice_scatter_205, 1, 0, 256);  permute_433 = slice_scatter_205 = None
        permute_437: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_206, [0, 2, 1, 3]);  slice_scatter_206 = None
        view_504: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_437, _shape_param_737);  permute_437 = _shape_param_737 = None
        view_505: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_504, _shape_param_738);  _shape_param_738 = None
        permute_438: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_505, [0, 2, 1, 3]);  view_505 = None
        slice_546: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_438, 1, -256, 9223372036854775807)
        view_506: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_504, _shape_param_739);  _shape_param_739 = None
        permute_439: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_506, [0, 2, 1, 3]);  view_506 = None
        slice_547: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_439, 1, -256, 9223372036854775807);  permute_439 = None
        slice_548: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_547, 3, -257, 9223372036854775807);  slice_547 = None
        rev_37: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_169, [1, 3]);  unsqueeze_169 = None
        expand_37: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_37, _shape_param_740);  rev_37 = _shape_param_740 = None
        convert_element_type_118: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_37, torch.bool);  expand_37 = None
        full_103: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_741, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_741 = None
        permute_440: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_103, [0, 2, 1, 3]);  full_103 = None
        view_507: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_504, _shape_param_742);  view_504 = _shape_param_742 = None
        permute_441: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_507, [0, 2, 1, 3]);  view_507 = None
        slice_549: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_441, 1, -256, 9223372036854775807);  permute_441 = None
        slice_550: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_549, 3, -257, 9223372036854775807);  slice_549 = None
        where_74: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_118, permute_440, slice_550);  convert_element_type_118 = permute_440 = slice_550 = None
        copy_113: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_548, where_74);  slice_548 = where_74 = None
        slice_scatter_207: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_546, copy_113, 3, -257, 9223372036854775807);  slice_546 = copy_113 = None
        slice_scatter_208: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_438, slice_scatter_207, 1, -256, 9223372036854775807);  permute_438 = slice_scatter_207 = None
        permute_442: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_208, [0, 2, 1, 3]);  slice_scatter_208 = None
        permute_443: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_442, [0, 2, 1, 3]);  permute_442 = None
        full_104: "bf16[8, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_743, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_743 = None
        slice_551: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_104, 1, 0, -1)
        slice_552: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_104, 1, 0, -1)
        slice_553: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_552, 3, 256, 9223372036854775807);  slice_552 = None
        full_105: "bf16[8, 1024, 1, 1]" = torch.ops.aten.full.default(_shape_param_744, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_744 = None
        permute_444: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(full_105, [0, 2, 1, 3]);  full_105 = None
        view_508: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_444, _shape_param_745);  permute_444 = _shape_param_745 = None
        view_509: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_508, _shape_param_746);  view_508 = _shape_param_746 = None
        as_strided_47: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_509, _shape_param_747, _shape_param_748);  view_509 = _shape_param_747 = _shape_param_748 = None
        unsqueeze_170: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_47, 4);  as_strided_47 = None
        permute_445: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_170, [0, 1, 2, 4, 3]);  unsqueeze_170 = None
        full_106: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ne_9: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg6_1, 0)
        unsqueeze_171: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_9, 2);  ne_9 = None
        unsqueeze_172: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 3);  unsqueeze_171 = None
        convert_element_type_119: "bf16[8, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_172, torch.bfloat16)
        where_75: "bf16[8, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_172, full_106, convert_element_type_119);  unsqueeze_172 = full_106 = convert_element_type_119 = None
        permute_446: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(where_75, [0, 2, 1, 3]);  where_75 = None
        view_510: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_446, _shape_param_749);  permute_446 = _shape_param_749 = None
        view_511: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_510, _shape_param_750);  view_510 = _shape_param_750 = None
        as_strided_48: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_511, _shape_param_751, _shape_param_752);  view_511 = _shape_param_751 = _shape_param_752 = None
        unsqueeze_173: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_48, 4);  as_strided_48 = None
        permute_447: "bf16[8, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_173, [0, 1, 4, 2, 3]);  unsqueeze_173 = None
        mul_72: "bf16[8, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_445, permute_447);  permute_445 = permute_447 = None
        view_512: "bf16[8, 3, 512, 512]" = torch.ops.aten.view.default(mul_72, _shape_param_753);  mul_72 = _shape_param_753 = None
        constant_pad_nd_37: "bf16[8, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_512, [0, 0, 0, 1], 0.0);  view_512 = None
        view_513: "bf16[8, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_37, _shape_param_754);  constant_pad_nd_37 = _shape_param_754 = None
        slice_554: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_513, 2, 0, 256)
        slice_555: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_554, 3, 0, 257);  slice_554 = None
        copy_114: "bf16[8, 3, 256, 257]" = torch.ops.aten.copy.default(slice_553, slice_555);  slice_553 = slice_555 = None
        slice_scatter_209: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_551, copy_114, 3, 256, 9223372036854775807);  slice_551 = copy_114 = None
        slice_scatter_210: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_104, slice_scatter_209, 1, 0, -1);  full_104 = slice_scatter_209 = None
        select_114: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_210, 1, -1)
        select_115: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_210, 1, -1)
        slice_556: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(select_115, 2, 256, 9223372036854775807);  select_115 = None
        select_116: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_513, 1, -1)
        slice_557: "bf16[8, 256, 513]" = torch.ops.aten.slice.Tensor(select_116, 1, 256, 9223372036854775807);  select_116 = None
        slice_558: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(slice_557, 2, 0, 257);  slice_557 = None
        copy_115: "bf16[8, 256, 257]" = torch.ops.aten.copy.default(slice_556, slice_558);  slice_556 = slice_558 = None
        slice_scatter_211: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_114, copy_115, 2, 256, 9223372036854775807);  select_114 = copy_115 = None
        select_scatter_38: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_210, slice_scatter_211, 1, -1);  slice_scatter_210 = slice_scatter_211 = None
        slice_559: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_38, 1, 1, 9223372036854775807)
        slice_560: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_38, 1, 1, 9223372036854775807)
        slice_561: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_560, 3, 0, 256);  slice_560 = None
        slice_562: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_513, 2, -257, -1)
        slice_563: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_562, 3, 257, 9223372036854775807);  slice_562 = None
        copy_116: "bf16[8, 3, 256, 256]" = torch.ops.aten.copy.default(slice_561, slice_563);  slice_561 = slice_563 = None
        slice_scatter_212: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_559, copy_116, 3, 0, 256);  slice_559 = copy_116 = None
        slice_scatter_213: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_38, slice_scatter_212, 1, 1, 9223372036854775807);  select_scatter_38 = slice_scatter_212 = None
        select_117: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_213, 1, 0)
        slice_564: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_117, 1, 1, 256)
        select_118: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_213, 1, 0)
        slice_565: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_118, 1, 1, 256);  select_118 = None
        slice_566: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_565, 2, 1, 256);  slice_565 = None
        select_119: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_513, 1, 0);  view_513 = None
        slice_567: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_119, 1, 0, 255);  select_119 = None
        slice_568: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_567, 2, -255, 9223372036854775807);  slice_567 = None
        copy_117: "bf16[8, 255, 255]" = torch.ops.aten.copy.default(slice_566, slice_568);  slice_566 = slice_568 = None
        slice_scatter_214: "bf16[8, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_564, copy_117, 2, 1, 256);  slice_564 = copy_117 = None
        slice_scatter_215: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_117, slice_scatter_214, 1, 1, 256);  select_117 = slice_scatter_214 = None
        select_scatter_39: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_213, slice_scatter_215, 1, 0);  slice_scatter_213 = slice_scatter_215 = None
        view_514: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_39, _shape_param_755);  _shape_param_755 = None
        permute_448: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
        slice_569: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_448, 1, 0, 256)
        view_515: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_39, _shape_param_756);  _shape_param_756 = None
        permute_449: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_515, [0, 2, 1, 3]);  view_515 = None
        slice_570: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_449, 1, 0, 256);  permute_449 = None
        slice_571: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_570, 3, 0, 257);  slice_570 = None
        iota_38: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_174: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_38, -2);  iota_38 = None
        iota_39: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_175: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_39, -1);  iota_39 = None
        sub_46: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_174, unsqueeze_175);  unsqueeze_174 = unsqueeze_175 = None
        le_19: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_46, 0);  sub_46 = None
        full_107: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_757, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_757 = None
        scalar_tensor_19: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_76: "bf16[256, 257]" = torch.ops.aten.where.self(le_19, full_107, scalar_tensor_19);  le_19 = full_107 = scalar_tensor_19 = None
        rev_38: "bf16[256, 257]" = torch.ops.prims.rev.default(where_76, [0]);  where_76 = None
        unsqueeze_176: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_38, 0);  rev_38 = None
        unsqueeze_177: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 2);  unsqueeze_176 = None
        expand_38: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_177, _shape_param_758);  _shape_param_758 = None
        convert_element_type_120: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_38, torch.bool);  expand_38 = None
        full_108: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_759, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_759 = None
        view_516: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_39, _shape_param_760);  select_scatter_39 = _shape_param_760 = None
        permute_450: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None
        slice_572: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_450, 1, 0, 256);  permute_450 = None
        slice_573: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_572, 3, 0, 257);  slice_572 = None
        where_77: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_120, full_108, slice_573);  convert_element_type_120 = full_108 = slice_573 = None
        copy_118: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_571, where_77);  slice_571 = where_77 = None
        slice_scatter_216: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_569, copy_118, 3, 0, 257);  slice_569 = copy_118 = None
        slice_scatter_217: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_448, slice_scatter_216, 1, 0, 256);  permute_448 = slice_scatter_216 = None
        permute_451: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_217, [0, 2, 1, 3]);  slice_scatter_217 = None
        view_517: "bf16[8, 4, 256, 513]" = torch.ops.aten.view.default(permute_451, _shape_param_761);  permute_451 = _shape_param_761 = None
        view_518: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_517, _shape_param_762);  _shape_param_762 = None
        permute_452: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_518, [0, 2, 1, 3]);  view_518 = None
        slice_574: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_452, 1, -256, 9223372036854775807)
        view_519: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_517, _shape_param_763);  _shape_param_763 = None
        permute_453: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_519, [0, 2, 1, 3]);  view_519 = None
        slice_575: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_453, 1, -256, 9223372036854775807);  permute_453 = None
        slice_576: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_575, 3, -257, 9223372036854775807);  slice_575 = None
        rev_39: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_177, [1, 3]);  unsqueeze_177 = None
        expand_39: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(rev_39, _shape_param_764);  rev_39 = _shape_param_764 = None
        convert_element_type_121: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_39, torch.bool);  expand_39 = None
        full_109: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_765, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_765 = None
        view_520: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_517, _shape_param_766);  view_517 = _shape_param_766 = None
        permute_454: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_520, [0, 2, 1, 3]);  view_520 = None
        slice_577: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_454, 1, -256, 9223372036854775807);  permute_454 = None
        slice_578: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_577, 3, -257, 9223372036854775807);  slice_577 = None
        where_78: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_121, full_109, slice_578);  convert_element_type_121 = full_109 = slice_578 = None
        copy_119: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_576, where_78);  slice_576 = where_78 = None
        slice_scatter_218: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_574, copy_119, 3, -257, 9223372036854775807);  slice_574 = copy_119 = None
        slice_scatter_219: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_452, slice_scatter_218, 1, -256, 9223372036854775807);  permute_452 = slice_scatter_218 = None
        permute_455: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_219, [0, 2, 1, 3]);  slice_scatter_219 = None
        permute_456: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_455, [0, 2, 1, 3]);  permute_455 = None
        add_110: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_443, permute_456);  permute_443 = permute_456 = None
        permute_457: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_110, [0, 2, 1, 3]);  add_110 = None
        permute_458: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_457, [0, 2, 1, 3]);  permute_457 = None
        convert_element_type_122: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_458, torch.float32);  permute_458 = None
        clone_94: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_122, memory_format = torch.contiguous_format);  convert_element_type_122 = None
        amax_9: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_94, [-1], True)
        sub_47: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_94, amax_9);  clone_94 = amax_9 = None
        exp_9: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_47);  sub_47 = None
        sum_10: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_19: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        where_79: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_163, full_99, div_19);  unsqueeze_163 = full_99 = div_19 = None
        convert_element_type_123: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_79, torch.bfloat16);  where_79 = None
        permute_459: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_123, [0, 2, 1, 3]);  convert_element_type_123 = None
        clone_95: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
        view_521: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_95, _shape_param_767);  clone_95 = _shape_param_767 = None
        constant_pad_nd_38: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_521, _shape_param_768, 0.0);  view_521 = _shape_param_768 = None
        view_522: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_38, _shape_param_769);  constant_pad_nd_38 = _shape_param_769 = None
        slice_579: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_522, 2, 0, -256);  view_522 = None
        view_523: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_579, _shape_param_770);  slice_579 = _shape_param_770 = None
        slice_580: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_523, 3, 0, -1);  view_523 = None
        unsqueeze_178: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_580, 4);  slice_580 = None
        view_524: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_178, _shape_param_771);  unsqueeze_178 = _shape_param_771 = None
        clone_96: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_423, memory_format = torch.contiguous_format);  permute_423 = None
        view_525: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_96, _shape_param_772);  clone_96 = _shape_param_772 = None
        permute_460: "bf16[768, 768]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        mm_38: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_525, permute_460);  view_525 = permute_460 = None
        view_526: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_38, _shape_param_773);  mm_38 = _shape_param_773 = None
        add_111: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_526, arg152_1);  view_526 = arg152_1 = None
        view_527: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_111, _shape_param_774);  add_111 = _shape_param_774 = None
        permute_461: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_527, [1, 0, 2, 3]);  view_527 = None
        permute_462: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_461, [0, 2, 1, 3]);  permute_461 = None
        view_528: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_462, _shape_param_775);  permute_462 = _shape_param_775 = None
        constant_pad_nd_39: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_528, _shape_param_776, -1.0);  view_528 = _shape_param_776 = None
        as_strided_49: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_39, _shape_param_777, _shape_param_778);  constant_pad_nd_39 = _shape_param_777 = _shape_param_778 = None
        unsqueeze_179: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_49, 4);  as_strided_49 = None
        clone_97: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_179, memory_format = torch.contiguous_format);  unsqueeze_179 = None
        view_529: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_97, _shape_param_779);  clone_97 = _shape_param_779 = None
        bmm_19: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_524, view_529);  view_524 = view_529 = None
        view_530: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_780);  bmm_19 = _shape_param_780 = None
        permute_463: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_530, [0, 1, 2, 4, 3]);  view_530 = None
        view_531: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_463, _shape_param_781);  permute_463 = _shape_param_781 = None
        view_532: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_531, _shape_param_782);  view_531 = _shape_param_782 = None
        permute_464: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
        permute_465: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_464, [1, 0, 2, 3]);  permute_464 = None
        clone_98: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_465, memory_format = torch.contiguous_format);  permute_465 = None
        view_533: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_98, _shape_param_783);  clone_98 = _shape_param_783 = None
        permute_466: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_533, [1, 0, 2]);  view_533 = None
        clone_99: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_466, memory_format = torch.contiguous_format);  permute_466 = None
        view_534: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_99, _shape_param_784);  clone_99 = _shape_param_784 = None
        permute_467: "bf16[768, 768]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        mm_39: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_534, permute_467);  view_534 = permute_467 = None
        view_535: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_39, _shape_param_785);  mm_39 = _shape_param_785 = None
        add_112: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_535, arg154_1);  view_535 = arg154_1 = None
        add_113: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_112, convert_element_type_116);  add_112 = convert_element_type_116 = None
        convert_element_type_124: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_113, torch.float32);  add_113 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_124, [2], correction = 0, keepdim = True)
        getitem_36: "f32[8, 1024, 1]" = var_mean_18[0]
        getitem_37: "f32[8, 1024, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_48: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_124, getitem_37);  convert_element_type_124 = getitem_37 = None
        add_114: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        mul_73: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_18);  sub_48 = rsqrt_18 = None
        mul_74: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_73, arg155_1);  mul_73 = arg155_1 = None
        add_115: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_74, arg156_1);  mul_74 = arg156_1 = None
        convert_element_type_125: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_115, torch.bfloat16);  add_115 = None
        view_536: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_125, _shape_param_786);  _shape_param_786 = None
        permute_468: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_18: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(arg158_1, view_536, permute_468);  arg158_1 = view_536 = permute_468 = None
        view_537: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_18, _shape_param_787);  addmm_18 = _shape_param_787 = None
        convert_element_type_126: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_537, torch.float32);  view_537 = None
        mul_75: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_126, 0.5)
        mul_76: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_126, 0.7071067811865476);  convert_element_type_126 = None
        erf_9: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_116: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_77: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_75, add_116);  mul_75 = add_116 = None
        convert_element_type_127: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None
        view_538: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_127, _shape_param_788);  convert_element_type_127 = _shape_param_788 = None
        permute_469: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_19: "bf16[8192, 768]" = torch.ops.aten.addmm.default(arg160_1, view_538, permute_469);  arg160_1 = view_538 = permute_469 = None
        view_539: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_19, _shape_param_789);  addmm_19 = _shape_param_789 = None
        add_117: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_539, convert_element_type_125);  view_539 = convert_element_type_125 = None
        convert_element_type_128: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_117, torch.float32);  add_117 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_128, [2], correction = 0, keepdim = True)
        getitem_38: "f32[8, 1024, 1]" = var_mean_19[0]
        getitem_39: "f32[8, 1024, 1]" = var_mean_19[1];  var_mean_19 = None
        unsqueeze_180: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 2)
        unsqueeze_181: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 3);  unsqueeze_180 = None
        full_110: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_111: "bf16[96, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_790, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_790 = None
        slice_581: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_111, 1, 0, -1)
        slice_582: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_111, 1, 0, -1)
        slice_583: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_582, 3, 256, 9223372036854775807);  slice_582 = None
        sub_49: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_128, getitem_39);  convert_element_type_128 = getitem_39 = None
        add_118: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_78: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_19);  sub_49 = rsqrt_19 = None
        mul_79: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_78, arg161_1);  mul_78 = arg161_1 = None
        add_119: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_79, arg162_1);  mul_79 = arg162_1 = None
        convert_element_type_129: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None
        permute_470: "bf16[1024, 8, 768]" = torch.ops.aten.permute.default(convert_element_type_129, [1, 0, 2])
        clone_100: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_470, memory_format = torch.contiguous_format)
        view_540: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_100, _shape_param_791);  clone_100 = _shape_param_791 = None
        permute_471: "bf16[768, 768]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        mm_40: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_540, permute_471);  view_540 = permute_471 = None
        view_541: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_40, _shape_param_792);  mm_40 = _shape_param_792 = None
        add_120: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_541, arg164_1);  view_541 = arg164_1 = None
        div_20: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_120, 8.0);  add_120 = None
        view_542: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_20, _shape_param_793);  div_20 = _shape_param_793 = None
        permute_472: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_542, [1, 0, 2, 3]);  view_542 = None
        permute_473: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_472, [0, 2, 1, 3]);  permute_472 = None
        view_543: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_473, _shape_param_794);  permute_473 = _shape_param_794 = None
        view_544: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_543, _shape_param_795);  view_543 = _shape_param_795 = None
        as_strided_50: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_544, _shape_param_796, _shape_param_797);  view_544 = _shape_param_796 = _shape_param_797 = None
        unsqueeze_182: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_50, 4);  as_strided_50 = None
        clone_101: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_182, memory_format = torch.contiguous_format);  unsqueeze_182 = None
        view_545: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_101, _shape_param_798);  clone_101 = _shape_param_798 = None
        clone_102: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_470, memory_format = torch.contiguous_format)
        view_546: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_102, _shape_param_799);  clone_102 = _shape_param_799 = None
        permute_474: "bf16[768, 768]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        mm_41: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_546, permute_474);  view_546 = permute_474 = None
        view_547: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_41, _shape_param_800);  mm_41 = _shape_param_800 = None
        add_121: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_547, arg166_1);  view_547 = arg166_1 = None
        view_548: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_121, _shape_param_801);  add_121 = _shape_param_801 = None
        permute_475: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_548, [1, 0, 2, 3]);  view_548 = None
        permute_476: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_475, [0, 2, 1, 3]);  permute_475 = None
        view_549: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_476, _shape_param_802);  permute_476 = _shape_param_802 = None
        view_550: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_549, _shape_param_803);  view_549 = _shape_param_803 = None
        as_strided_51: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_550, _shape_param_804, _shape_param_805);  view_550 = _shape_param_804 = _shape_param_805 = None
        unsqueeze_183: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_51, 4);  as_strided_51 = None
        permute_477: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_183, [0, 1, 4, 2, 3]);  unsqueeze_183 = None
        permute_478: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_477, [0, 1, 4, 3, 2]);  permute_477 = None
        clone_103: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_478, memory_format = torch.contiguous_format);  permute_478 = None
        view_551: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_103, _shape_param_806);  clone_103 = _shape_param_806 = None
        bmm_20: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_545, view_551);  view_545 = view_551 = None
        view_552: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_20, _shape_param_807);  bmm_20 = _shape_param_807 = None
        permute_479: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_552, [0, 1, 2, 4, 3]);  view_552 = None
        view_553: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_479, _shape_param_808);  permute_479 = _shape_param_808 = None
        constant_pad_nd_40: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_553, [0, 0, 0, 1], 0.0);  view_553 = None
        view_554: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_40, _shape_param_809);  constant_pad_nd_40 = _shape_param_809 = None
        slice_584: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_554, 2, 0, 256)
        slice_585: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_584, 3, 0, 257);  slice_584 = None
        copy_120: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_583, slice_585);  slice_583 = slice_585 = None
        slice_scatter_220: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_581, copy_120, 3, 256, 9223372036854775807);  slice_581 = copy_120 = None
        slice_scatter_221: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_111, slice_scatter_220, 1, 0, -1);  full_111 = slice_scatter_220 = None
        select_120: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_221, 1, -1)
        select_121: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_221, 1, -1)
        slice_586: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_121, 2, 256, 9223372036854775807);  select_121 = None
        select_122: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_554, 1, -1)
        slice_587: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_122, 1, 256, 9223372036854775807);  select_122 = None
        slice_588: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_587, 2, 0, 257);  slice_587 = None
        copy_121: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_586, slice_588);  slice_586 = slice_588 = None
        slice_scatter_222: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_120, copy_121, 2, 256, 9223372036854775807);  select_120 = copy_121 = None
        select_scatter_40: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_221, slice_scatter_222, 1, -1);  slice_scatter_221 = slice_scatter_222 = None
        slice_589: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_40, 1, 1, 9223372036854775807)
        slice_590: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_40, 1, 1, 9223372036854775807)
        slice_591: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_590, 3, 0, 256);  slice_590 = None
        slice_592: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_554, 2, -257, -1)
        slice_593: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_592, 3, 257, 9223372036854775807);  slice_592 = None
        copy_122: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_591, slice_593);  slice_591 = slice_593 = None
        slice_scatter_223: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_589, copy_122, 3, 0, 256);  slice_589 = copy_122 = None
        slice_scatter_224: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_40, slice_scatter_223, 1, 1, 9223372036854775807);  select_scatter_40 = slice_scatter_223 = None
        select_123: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_224, 1, 0)
        slice_594: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_123, 1, 1, 256)
        select_124: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_224, 1, 0)
        slice_595: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_124, 1, 1, 256);  select_124 = None
        slice_596: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_595, 2, 1, 256);  slice_595 = None
        select_125: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_554, 1, 0);  view_554 = None
        slice_597: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_125, 1, 0, 255);  select_125 = None
        slice_598: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_597, 2, -255, 9223372036854775807);  slice_597 = None
        copy_123: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_596, slice_598);  slice_596 = slice_598 = None
        slice_scatter_225: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_594, copy_123, 2, 1, 256);  slice_594 = copy_123 = None
        slice_scatter_226: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_123, slice_scatter_225, 1, 1, 256);  select_123 = slice_scatter_225 = None
        select_scatter_41: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_224, slice_scatter_226, 1, 0);  slice_scatter_224 = slice_scatter_226 = None
        view_555: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_41, _shape_param_810);  _shape_param_810 = None
        permute_480: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_555, [0, 2, 1, 3]);  view_555 = None
        slice_599: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_480, 1, 0, 256)
        view_556: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_41, _shape_param_811);  _shape_param_811 = None
        permute_481: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None
        slice_600: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_481, 1, 0, 256);  permute_481 = None
        slice_601: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_600, 3, 0, 257);  slice_600 = None
        iota_40: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_184: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_40, -2);  iota_40 = None
        iota_41: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_185: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_41, -1);  iota_41 = None
        sub_50: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_184, unsqueeze_185);  unsqueeze_184 = unsqueeze_185 = None
        le_20: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_50, 0);  sub_50 = None
        full_112: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_812, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_812 = None
        scalar_tensor_20: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_80: "bf16[256, 257]" = torch.ops.aten.where.self(le_20, full_112, scalar_tensor_20);  le_20 = full_112 = scalar_tensor_20 = None
        rev_40: "bf16[256, 257]" = torch.ops.prims.rev.default(where_80, [0]);  where_80 = None
        unsqueeze_186: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_40, 0);  rev_40 = None
        unsqueeze_187: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 2);  unsqueeze_186 = None
        expand_40: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_187, _shape_param_813);  _shape_param_813 = None
        convert_element_type_130: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_40, torch.bool);  expand_40 = None
        full_113: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_814, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_814 = None
        permute_482: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_113, [0, 2, 1, 3]);  full_113 = None
        view_557: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_41, _shape_param_815);  select_scatter_41 = _shape_param_815 = None
        permute_483: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_557, [0, 2, 1, 3]);  view_557 = None
        slice_602: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_483, 1, 0, 256);  permute_483 = None
        slice_603: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_602, 3, 0, 257);  slice_602 = None
        where_81: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_130, permute_482, slice_603);  convert_element_type_130 = permute_482 = slice_603 = None
        copy_124: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_601, where_81);  slice_601 = where_81 = None
        slice_scatter_227: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_599, copy_124, 3, 0, 257);  slice_599 = copy_124 = None
        slice_scatter_228: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_480, slice_scatter_227, 1, 0, 256);  permute_480 = slice_scatter_227 = None
        permute_484: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_228, [0, 2, 1, 3]);  slice_scatter_228 = None
        view_558: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_484, _shape_param_816);  permute_484 = _shape_param_816 = None
        view_559: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_558, _shape_param_817);  _shape_param_817 = None
        permute_485: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_559, [0, 2, 1, 3]);  view_559 = None
        slice_604: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_485, 1, -256, 9223372036854775807)
        view_560: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_558, _shape_param_818);  _shape_param_818 = None
        permute_486: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None
        slice_605: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_486, 1, -256, 9223372036854775807);  permute_486 = None
        slice_606: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_605, 3, -257, 9223372036854775807);  slice_605 = None
        rev_41: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_187, [1, 3]);  unsqueeze_187 = None
        expand_41: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_41, _shape_param_819);  rev_41 = _shape_param_819 = None
        convert_element_type_131: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_41, torch.bool);  expand_41 = None
        full_114: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_820, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_820 = None
        permute_487: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_114, [0, 2, 1, 3]);  full_114 = None
        view_561: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_558, _shape_param_821);  view_558 = _shape_param_821 = None
        permute_488: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_561, [0, 2, 1, 3]);  view_561 = None
        slice_607: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_488, 1, -256, 9223372036854775807);  permute_488 = None
        slice_608: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_607, 3, -257, 9223372036854775807);  slice_607 = None
        where_82: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_131, permute_487, slice_608);  convert_element_type_131 = permute_487 = slice_608 = None
        copy_125: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_606, where_82);  slice_606 = where_82 = None
        slice_scatter_229: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_604, copy_125, 3, -257, 9223372036854775807);  slice_604 = copy_125 = None
        slice_scatter_230: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_485, slice_scatter_229, 1, -256, 9223372036854775807);  permute_485 = slice_scatter_229 = None
        permute_489: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_230, [0, 2, 1, 3]);  slice_scatter_230 = None
        permute_490: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_489, [0, 2, 1, 3]);  permute_489 = None
        full_115: "bf16[8, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_822, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_822 = None
        slice_609: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_115, 1, 0, -1)
        slice_610: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_115, 1, 0, -1)
        slice_611: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_610, 3, 256, 9223372036854775807);  slice_610 = None
        full_116: "bf16[8, 1024, 1, 1]" = torch.ops.aten.full.default(_shape_param_823, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_823 = None
        permute_491: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(full_116, [0, 2, 1, 3]);  full_116 = None
        view_562: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_491, _shape_param_824);  permute_491 = _shape_param_824 = None
        view_563: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_562, _shape_param_825);  view_562 = _shape_param_825 = None
        as_strided_52: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_563, _shape_param_826, _shape_param_827);  view_563 = _shape_param_826 = _shape_param_827 = None
        unsqueeze_188: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_52, 4);  as_strided_52 = None
        permute_492: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_188, [0, 1, 2, 4, 3]);  unsqueeze_188 = None
        full_117: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ne_10: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg6_1, 0)
        unsqueeze_189: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_10, 2);  ne_10 = None
        unsqueeze_190: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 3);  unsqueeze_189 = None
        convert_element_type_132: "bf16[8, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_190, torch.bfloat16)
        where_83: "bf16[8, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_190, full_117, convert_element_type_132);  unsqueeze_190 = full_117 = convert_element_type_132 = None
        permute_493: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(where_83, [0, 2, 1, 3]);  where_83 = None
        view_564: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_493, _shape_param_828);  permute_493 = _shape_param_828 = None
        view_565: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_564, _shape_param_829);  view_564 = _shape_param_829 = None
        as_strided_53: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_565, _shape_param_830, _shape_param_831);  view_565 = _shape_param_830 = _shape_param_831 = None
        unsqueeze_191: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_53, 4);  as_strided_53 = None
        permute_494: "bf16[8, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_191, [0, 1, 4, 2, 3]);  unsqueeze_191 = None
        mul_80: "bf16[8, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_492, permute_494);  permute_492 = permute_494 = None
        view_566: "bf16[8, 3, 512, 512]" = torch.ops.aten.view.default(mul_80, _shape_param_832);  mul_80 = _shape_param_832 = None
        constant_pad_nd_41: "bf16[8, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_566, [0, 0, 0, 1], 0.0);  view_566 = None
        view_567: "bf16[8, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_41, _shape_param_833);  constant_pad_nd_41 = _shape_param_833 = None
        slice_612: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_567, 2, 0, 256)
        slice_613: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_612, 3, 0, 257);  slice_612 = None
        copy_126: "bf16[8, 3, 256, 257]" = torch.ops.aten.copy.default(slice_611, slice_613);  slice_611 = slice_613 = None
        slice_scatter_231: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_609, copy_126, 3, 256, 9223372036854775807);  slice_609 = copy_126 = None
        slice_scatter_232: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_115, slice_scatter_231, 1, 0, -1);  full_115 = slice_scatter_231 = None
        select_126: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_232, 1, -1)
        select_127: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_232, 1, -1)
        slice_614: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(select_127, 2, 256, 9223372036854775807);  select_127 = None
        select_128: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_567, 1, -1)
        slice_615: "bf16[8, 256, 513]" = torch.ops.aten.slice.Tensor(select_128, 1, 256, 9223372036854775807);  select_128 = None
        slice_616: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(slice_615, 2, 0, 257);  slice_615 = None
        copy_127: "bf16[8, 256, 257]" = torch.ops.aten.copy.default(slice_614, slice_616);  slice_614 = slice_616 = None
        slice_scatter_233: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_126, copy_127, 2, 256, 9223372036854775807);  select_126 = copy_127 = None
        select_scatter_42: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_232, slice_scatter_233, 1, -1);  slice_scatter_232 = slice_scatter_233 = None
        slice_617: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_42, 1, 1, 9223372036854775807)
        slice_618: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_42, 1, 1, 9223372036854775807)
        slice_619: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_618, 3, 0, 256);  slice_618 = None
        slice_620: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_567, 2, -257, -1)
        slice_621: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_620, 3, 257, 9223372036854775807);  slice_620 = None
        copy_128: "bf16[8, 3, 256, 256]" = torch.ops.aten.copy.default(slice_619, slice_621);  slice_619 = slice_621 = None
        slice_scatter_234: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_617, copy_128, 3, 0, 256);  slice_617 = copy_128 = None
        slice_scatter_235: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_42, slice_scatter_234, 1, 1, 9223372036854775807);  select_scatter_42 = slice_scatter_234 = None
        select_129: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_235, 1, 0)
        slice_622: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_129, 1, 1, 256)
        select_130: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_235, 1, 0)
        slice_623: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_130, 1, 1, 256);  select_130 = None
        slice_624: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_623, 2, 1, 256);  slice_623 = None
        select_131: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_567, 1, 0);  view_567 = None
        slice_625: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_131, 1, 0, 255);  select_131 = None
        slice_626: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_625, 2, -255, 9223372036854775807);  slice_625 = None
        copy_129: "bf16[8, 255, 255]" = torch.ops.aten.copy.default(slice_624, slice_626);  slice_624 = slice_626 = None
        slice_scatter_236: "bf16[8, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_622, copy_129, 2, 1, 256);  slice_622 = copy_129 = None
        slice_scatter_237: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_129, slice_scatter_236, 1, 1, 256);  select_129 = slice_scatter_236 = None
        select_scatter_43: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_235, slice_scatter_237, 1, 0);  slice_scatter_235 = slice_scatter_237 = None
        view_568: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_43, _shape_param_834);  _shape_param_834 = None
        permute_495: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
        slice_627: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_495, 1, 0, 256)
        view_569: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_43, _shape_param_835);  _shape_param_835 = None
        permute_496: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_569, [0, 2, 1, 3]);  view_569 = None
        slice_628: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_496, 1, 0, 256);  permute_496 = None
        slice_629: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_628, 3, 0, 257);  slice_628 = None
        iota_42: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_192: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_42, -2);  iota_42 = None
        iota_43: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_193: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_43, -1);  iota_43 = None
        sub_51: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_192, unsqueeze_193);  unsqueeze_192 = unsqueeze_193 = None
        le_21: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_51, 0);  sub_51 = None
        full_118: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_836, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_836 = None
        scalar_tensor_21: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_84: "bf16[256, 257]" = torch.ops.aten.where.self(le_21, full_118, scalar_tensor_21);  le_21 = full_118 = scalar_tensor_21 = None
        rev_42: "bf16[256, 257]" = torch.ops.prims.rev.default(where_84, [0]);  where_84 = None
        unsqueeze_194: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_42, 0);  rev_42 = None
        unsqueeze_195: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 2);  unsqueeze_194 = None
        expand_42: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_195, _shape_param_837);  _shape_param_837 = None
        convert_element_type_133: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_42, torch.bool);  expand_42 = None
        full_119: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_838, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_838 = None
        view_570: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_43, _shape_param_839);  select_scatter_43 = _shape_param_839 = None
        permute_497: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
        slice_630: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_497, 1, 0, 256);  permute_497 = None
        slice_631: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_630, 3, 0, 257);  slice_630 = None
        where_85: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_133, full_119, slice_631);  convert_element_type_133 = full_119 = slice_631 = None
        copy_130: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_629, where_85);  slice_629 = where_85 = None
        slice_scatter_238: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_627, copy_130, 3, 0, 257);  slice_627 = copy_130 = None
        slice_scatter_239: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_495, slice_scatter_238, 1, 0, 256);  permute_495 = slice_scatter_238 = None
        permute_498: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_239, [0, 2, 1, 3]);  slice_scatter_239 = None
        view_571: "bf16[8, 4, 256, 513]" = torch.ops.aten.view.default(permute_498, _shape_param_840);  permute_498 = _shape_param_840 = None
        view_572: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_571, _shape_param_841);  _shape_param_841 = None
        permute_499: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
        slice_632: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_499, 1, -256, 9223372036854775807)
        view_573: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_571, _shape_param_842);  _shape_param_842 = None
        permute_500: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None
        slice_633: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_500, 1, -256, 9223372036854775807);  permute_500 = None
        slice_634: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_633, 3, -257, 9223372036854775807);  slice_633 = None
        rev_43: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_195, [1, 3]);  unsqueeze_195 = None
        expand_43: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(rev_43, _shape_param_843);  rev_43 = _shape_param_843 = None
        convert_element_type_134: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_43, torch.bool);  expand_43 = None
        full_120: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_844, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_844 = None
        view_574: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_571, _shape_param_845);  view_571 = _shape_param_845 = None
        permute_501: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_574, [0, 2, 1, 3]);  view_574 = None
        slice_635: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_501, 1, -256, 9223372036854775807);  permute_501 = None
        slice_636: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_635, 3, -257, 9223372036854775807);  slice_635 = None
        where_86: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_134, full_120, slice_636);  convert_element_type_134 = full_120 = slice_636 = None
        copy_131: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_634, where_86);  slice_634 = where_86 = None
        slice_scatter_240: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_632, copy_131, 3, -257, 9223372036854775807);  slice_632 = copy_131 = None
        slice_scatter_241: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_499, slice_scatter_240, 1, -256, 9223372036854775807);  permute_499 = slice_scatter_240 = None
        permute_502: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_241, [0, 2, 1, 3]);  slice_scatter_241 = None
        permute_503: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_502, [0, 2, 1, 3]);  permute_502 = None
        add_122: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_490, permute_503);  permute_490 = permute_503 = None
        permute_504: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_122, [0, 2, 1, 3]);  add_122 = None
        permute_505: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_504, [0, 2, 1, 3]);  permute_504 = None
        convert_element_type_135: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_505, torch.float32);  permute_505 = None
        clone_104: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_135, memory_format = torch.contiguous_format);  convert_element_type_135 = None
        amax_10: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_104, [-1], True)
        sub_52: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_104, amax_10);  clone_104 = amax_10 = None
        exp_10: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_11: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_21: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        where_87: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_181, full_110, div_21);  unsqueeze_181 = full_110 = div_21 = None
        convert_element_type_136: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_87, torch.bfloat16);  where_87 = None
        permute_506: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_136, [0, 2, 1, 3]);  convert_element_type_136 = None
        clone_105: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_506, memory_format = torch.contiguous_format);  permute_506 = None
        view_575: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_105, _shape_param_846);  clone_105 = _shape_param_846 = None
        constant_pad_nd_42: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_575, _shape_param_847, 0.0);  view_575 = _shape_param_847 = None
        view_576: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_42, _shape_param_848);  constant_pad_nd_42 = _shape_param_848 = None
        slice_637: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_576, 2, 0, -256);  view_576 = None
        view_577: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_637, _shape_param_849);  slice_637 = _shape_param_849 = None
        slice_638: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_577, 3, 0, -1);  view_577 = None
        unsqueeze_196: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_638, 4);  slice_638 = None
        view_578: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_196, _shape_param_850);  unsqueeze_196 = _shape_param_850 = None
        clone_106: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_470, memory_format = torch.contiguous_format);  permute_470 = None
        view_579: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_106, _shape_param_851);  clone_106 = _shape_param_851 = None
        permute_507: "bf16[768, 768]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        mm_42: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_579, permute_507);  view_579 = permute_507 = None
        view_580: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_42, _shape_param_852);  mm_42 = _shape_param_852 = None
        add_123: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_580, arg168_1);  view_580 = arg168_1 = None
        view_581: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_123, _shape_param_853);  add_123 = _shape_param_853 = None
        permute_508: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_581, [1, 0, 2, 3]);  view_581 = None
        permute_509: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_508, [0, 2, 1, 3]);  permute_508 = None
        view_582: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_509, _shape_param_854);  permute_509 = _shape_param_854 = None
        constant_pad_nd_43: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_582, _shape_param_855, -1.0);  view_582 = _shape_param_855 = None
        as_strided_54: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_43, _shape_param_856, _shape_param_857);  constant_pad_nd_43 = _shape_param_856 = _shape_param_857 = None
        unsqueeze_197: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_54, 4);  as_strided_54 = None
        clone_107: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_197, memory_format = torch.contiguous_format);  unsqueeze_197 = None
        view_583: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_107, _shape_param_858);  clone_107 = _shape_param_858 = None
        bmm_21: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_578, view_583);  view_578 = view_583 = None
        view_584: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_21, _shape_param_859);  bmm_21 = _shape_param_859 = None
        permute_510: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_584, [0, 1, 2, 4, 3]);  view_584 = None
        view_585: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_510, _shape_param_860);  permute_510 = _shape_param_860 = None
        view_586: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_585, _shape_param_861);  view_585 = _shape_param_861 = None
        permute_511: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_586, [0, 2, 1, 3]);  view_586 = None
        permute_512: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_511, [1, 0, 2, 3]);  permute_511 = None
        clone_108: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_512, memory_format = torch.contiguous_format);  permute_512 = None
        view_587: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_108, _shape_param_862);  clone_108 = _shape_param_862 = None
        permute_513: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_587, [1, 0, 2]);  view_587 = None
        clone_109: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_513, memory_format = torch.contiguous_format);  permute_513 = None
        view_588: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_109, _shape_param_863);  clone_109 = _shape_param_863 = None
        permute_514: "bf16[768, 768]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        mm_43: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_588, permute_514);  view_588 = permute_514 = None
        view_589: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_43, _shape_param_864);  mm_43 = _shape_param_864 = None
        add_124: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_589, arg170_1);  view_589 = arg170_1 = None
        add_125: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_124, convert_element_type_129);  add_124 = convert_element_type_129 = None
        convert_element_type_137: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_125, torch.float32);  add_125 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_137, [2], correction = 0, keepdim = True)
        getitem_40: "f32[8, 1024, 1]" = var_mean_20[0]
        getitem_41: "f32[8, 1024, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_53: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_137, getitem_41);  convert_element_type_137 = getitem_41 = None
        add_126: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_81: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_20);  sub_53 = rsqrt_20 = None
        mul_82: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_81, arg171_1);  mul_81 = arg171_1 = None
        add_127: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_82, arg172_1);  mul_82 = arg172_1 = None
        convert_element_type_138: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_127, torch.bfloat16);  add_127 = None
        view_590: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_138, _shape_param_865);  _shape_param_865 = None
        permute_515: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_20: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(arg174_1, view_590, permute_515);  arg174_1 = view_590 = permute_515 = None
        view_591: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_20, _shape_param_866);  addmm_20 = _shape_param_866 = None
        convert_element_type_139: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_591, torch.float32);  view_591 = None
        mul_83: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.5)
        mul_84: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.7071067811865476);  convert_element_type_139 = None
        erf_10: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
        add_128: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_85: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_83, add_128);  mul_83 = add_128 = None
        convert_element_type_140: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_85, torch.bfloat16);  mul_85 = None
        view_592: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_140, _shape_param_867);  convert_element_type_140 = _shape_param_867 = None
        permute_516: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_21: "bf16[8192, 768]" = torch.ops.aten.addmm.default(arg176_1, view_592, permute_516);  arg176_1 = view_592 = permute_516 = None
        view_593: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_21, _shape_param_868);  addmm_21 = _shape_param_868 = None
        add_129: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_593, convert_element_type_138);  view_593 = convert_element_type_138 = None
        convert_element_type_141: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_129, torch.float32);  add_129 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_141, [2], correction = 0, keepdim = True)
        getitem_42: "f32[8, 1024, 1]" = var_mean_21[0]
        getitem_43: "f32[8, 1024, 1]" = var_mean_21[1];  var_mean_21 = None
        unsqueeze_198: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg0_1, 2);  arg0_1 = None
        unsqueeze_199: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None
        full_121: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_122: "bf16[96, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_869, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_869 = None
        slice_639: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_122, 1, 0, -1)
        slice_640: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_122, 1, 0, -1)
        slice_641: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_640, 3, 256, 9223372036854775807);  slice_640 = None
        sub_54: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_141, getitem_43);  convert_element_type_141 = getitem_43 = None
        add_130: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        mul_86: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_21);  sub_54 = rsqrt_21 = None
        mul_87: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_86, arg177_1);  mul_86 = arg177_1 = None
        add_131: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_87, arg178_1);  mul_87 = arg178_1 = None
        convert_element_type_142: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16);  add_131 = None
        permute_517: "bf16[1024, 8, 768]" = torch.ops.aten.permute.default(convert_element_type_142, [1, 0, 2])
        clone_110: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format)
        view_594: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_110, _shape_param_870);  clone_110 = _shape_param_870 = None
        permute_518: "bf16[768, 768]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        mm_44: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_594, permute_518);  view_594 = permute_518 = None
        view_595: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_44, _shape_param_871);  mm_44 = _shape_param_871 = None
        add_132: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_595, arg180_1);  view_595 = arg180_1 = None
        div_22: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add_132, 8.0);  add_132 = None
        view_596: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div_22, _shape_param_872);  div_22 = _shape_param_872 = None
        permute_519: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_596, [1, 0, 2, 3]);  view_596 = None
        permute_520: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_519, [0, 2, 1, 3]);  permute_519 = None
        view_597: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_520, _shape_param_873);  permute_520 = _shape_param_873 = None
        view_598: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_597, _shape_param_874);  view_597 = _shape_param_874 = None
        as_strided_55: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_598, _shape_param_875, _shape_param_876);  view_598 = _shape_param_875 = _shape_param_876 = None
        unsqueeze_200: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_55, 4);  as_strided_55 = None
        clone_111: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_200, memory_format = torch.contiguous_format);  unsqueeze_200 = None
        view_599: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_111, _shape_param_877);  clone_111 = _shape_param_877 = None
        clone_112: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format)
        view_600: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_112, _shape_param_878);  clone_112 = _shape_param_878 = None
        permute_521: "bf16[768, 768]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        mm_45: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_600, permute_521);  view_600 = permute_521 = None
        view_601: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_45, _shape_param_879);  mm_45 = _shape_param_879 = None
        add_133: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_601, arg182_1);  view_601 = arg182_1 = None
        view_602: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_133, _shape_param_880);  add_133 = _shape_param_880 = None
        permute_522: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_602, [1, 0, 2, 3]);  view_602 = None
        permute_523: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_522, [0, 2, 1, 3]);  permute_522 = None
        view_603: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_523, _shape_param_881);  permute_523 = _shape_param_881 = None
        view_604: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_603, _shape_param_882);  view_603 = _shape_param_882 = None
        as_strided_56: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_604, _shape_param_883, _shape_param_884);  view_604 = _shape_param_883 = _shape_param_884 = None
        unsqueeze_201: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_56, 4);  as_strided_56 = None
        permute_524: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_201, [0, 1, 4, 2, 3]);  unsqueeze_201 = None
        permute_525: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_524, [0, 1, 4, 3, 2]);  permute_524 = None
        clone_113: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_525, memory_format = torch.contiguous_format);  permute_525 = None
        view_605: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_113, _shape_param_885);  clone_113 = _shape_param_885 = None
        bmm_22: "bf16[288, 512, 512]" = torch.ops.aten.bmm.default(view_599, view_605);  view_599 = view_605 = None
        view_606: "bf16[96, 3, 512, 1, 512]" = torch.ops.aten.view.default(bmm_22, _shape_param_886);  bmm_22 = _shape_param_886 = None
        permute_526: "bf16[96, 3, 512, 512, 1]" = torch.ops.aten.permute.default(view_606, [0, 1, 2, 4, 3]);  view_606 = None
        view_607: "bf16[96, 3, 512, 512]" = torch.ops.aten.view.default(permute_526, _shape_param_887);  permute_526 = _shape_param_887 = None
        constant_pad_nd_44: "bf16[96, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_607, [0, 0, 0, 1], 0.0);  view_607 = None
        view_608: "bf16[96, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_44, _shape_param_888);  constant_pad_nd_44 = _shape_param_888 = None
        slice_642: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_608, 2, 0, 256)
        slice_643: "bf16[96, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_642, 3, 0, 257);  slice_642 = None
        copy_132: "bf16[96, 3, 256, 257]" = torch.ops.aten.copy.default(slice_641, slice_643);  slice_641 = slice_643 = None
        slice_scatter_242: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_639, copy_132, 3, 256, 9223372036854775807);  slice_639 = copy_132 = None
        slice_scatter_243: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_122, slice_scatter_242, 1, 0, -1);  full_122 = slice_scatter_242 = None
        select_132: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_243, 1, -1)
        select_133: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_243, 1, -1)
        slice_644: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(select_133, 2, 256, 9223372036854775807);  select_133 = None
        select_134: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_608, 1, -1)
        slice_645: "bf16[96, 256, 513]" = torch.ops.aten.slice.Tensor(select_134, 1, 256, 9223372036854775807);  select_134 = None
        slice_646: "bf16[96, 256, 257]" = torch.ops.aten.slice.Tensor(slice_645, 2, 0, 257);  slice_645 = None
        copy_133: "bf16[96, 256, 257]" = torch.ops.aten.copy.default(slice_644, slice_646);  slice_644 = slice_646 = None
        slice_scatter_244: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_132, copy_133, 2, 256, 9223372036854775807);  select_132 = copy_133 = None
        select_scatter_44: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_243, slice_scatter_244, 1, -1);  slice_scatter_243 = slice_scatter_244 = None
        slice_647: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_44, 1, 1, 9223372036854775807)
        slice_648: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_44, 1, 1, 9223372036854775807)
        slice_649: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_648, 3, 0, 256);  slice_648 = None
        slice_650: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_608, 2, -257, -1)
        slice_651: "bf16[96, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_650, 3, 257, 9223372036854775807);  slice_650 = None
        copy_134: "bf16[96, 3, 256, 256]" = torch.ops.aten.copy.default(slice_649, slice_651);  slice_649 = slice_651 = None
        slice_scatter_245: "bf16[96, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_647, copy_134, 3, 0, 256);  slice_647 = copy_134 = None
        slice_scatter_246: "bf16[96, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_44, slice_scatter_245, 1, 1, 9223372036854775807);  select_scatter_44 = slice_scatter_245 = None
        select_135: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_246, 1, 0)
        slice_652: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_135, 1, 1, 256)
        select_136: "bf16[96, 256, 513]" = torch.ops.aten.select.int(slice_scatter_246, 1, 0)
        slice_653: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_136, 1, 1, 256);  select_136 = None
        slice_654: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_653, 2, 1, 256);  slice_653 = None
        select_137: "bf16[96, 512, 513]" = torch.ops.aten.select.int(view_608, 1, 0);  view_608 = None
        slice_655: "bf16[96, 255, 513]" = torch.ops.aten.slice.Tensor(select_137, 1, 0, 255);  select_137 = None
        slice_656: "bf16[96, 255, 255]" = torch.ops.aten.slice.Tensor(slice_655, 2, -255, 9223372036854775807);  slice_655 = None
        copy_135: "bf16[96, 255, 255]" = torch.ops.aten.copy.default(slice_654, slice_656);  slice_654 = slice_656 = None
        slice_scatter_247: "bf16[96, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_652, copy_135, 2, 1, 256);  slice_652 = copy_135 = None
        slice_scatter_248: "bf16[96, 256, 513]" = torch.ops.aten.slice_scatter.default(select_135, slice_scatter_247, 1, 1, 256);  select_135 = slice_scatter_247 = None
        select_scatter_45: "bf16[96, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_246, slice_scatter_248, 1, 0);  slice_scatter_246 = slice_scatter_248 = None
        view_609: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_45, _shape_param_889);  _shape_param_889 = None
        permute_527: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_609, [0, 2, 1, 3]);  view_609 = None
        slice_657: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_527, 1, 0, 256)
        view_610: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_45, _shape_param_890);  _shape_param_890 = None
        permute_528: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_610, [0, 2, 1, 3]);  view_610 = None
        slice_658: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_528, 1, 0, 256);  permute_528 = None
        slice_659: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_658, 3, 0, 257);  slice_658 = None
        iota_44: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_202: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_44, -2);  iota_44 = None
        iota_45: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_203: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_45, -1);  iota_45 = None
        sub_55: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_202, unsqueeze_203);  unsqueeze_202 = unsqueeze_203 = None
        le_22: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_55, 0);  sub_55 = None
        full_123: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_891, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_891 = None
        scalar_tensor_22: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_88: "bf16[256, 257]" = torch.ops.aten.where.self(le_22, full_123, scalar_tensor_22);  le_22 = full_123 = scalar_tensor_22 = None
        rev_44: "bf16[256, 257]" = torch.ops.prims.rev.default(where_88, [0]);  where_88 = None
        unsqueeze_204: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_44, 0);  rev_44 = None
        unsqueeze_205: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 2);  unsqueeze_204 = None
        expand_44: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_205, _shape_param_892);  _shape_param_892 = None
        convert_element_type_143: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_44, torch.bool);  expand_44 = None
        full_124: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_893, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_893 = None
        permute_529: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_124, [0, 2, 1, 3]);  full_124 = None
        view_611: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(select_scatter_45, _shape_param_894);  select_scatter_45 = _shape_param_894 = None
        permute_530: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_611, [0, 2, 1, 3]);  view_611 = None
        slice_660: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_530, 1, 0, 256);  permute_530 = None
        slice_661: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_660, 3, 0, 257);  slice_660 = None
        where_89: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_143, permute_529, slice_661);  convert_element_type_143 = permute_529 = slice_661 = None
        copy_136: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_659, where_89);  slice_659 = where_89 = None
        slice_scatter_249: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_657, copy_136, 3, 0, 257);  slice_657 = copy_136 = None
        slice_scatter_250: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_527, slice_scatter_249, 1, 0, 256);  permute_527 = slice_scatter_249 = None
        permute_531: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_250, [0, 2, 1, 3]);  slice_scatter_250 = None
        view_612: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(permute_531, _shape_param_895);  permute_531 = _shape_param_895 = None
        view_613: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_612, _shape_param_896);  _shape_param_896 = None
        permute_532: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_613, [0, 2, 1, 3]);  view_613 = None
        slice_662: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_532, 1, -256, 9223372036854775807)
        view_614: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_612, _shape_param_897);  _shape_param_897 = None
        permute_533: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None
        slice_663: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_533, 1, -256, 9223372036854775807);  permute_533 = None
        slice_664: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_663, 3, -257, 9223372036854775807);  slice_663 = None
        rev_45: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_205, [1, 3]);  unsqueeze_205 = None
        expand_45: "bf16[8, 256, 12, 257]" = torch.ops.aten.expand.default(rev_45, _shape_param_898);  rev_45 = _shape_param_898 = None
        convert_element_type_144: "b8[8, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_45, torch.bool);  expand_45 = None
        full_125: "bf16[8, 12, 256, 257]" = torch.ops.aten.full.default(_shape_param_899, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_899 = None
        permute_534: "bf16[8, 256, 12, 257]" = torch.ops.aten.permute.default(full_125, [0, 2, 1, 3]);  full_125 = None
        view_615: "bf16[8, 12, 1024, 513]" = torch.ops.aten.view.default(view_612, _shape_param_900);  view_612 = _shape_param_900 = None
        permute_535: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(view_615, [0, 2, 1, 3]);  view_615 = None
        slice_665: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_535, 1, -256, 9223372036854775807);  permute_535 = None
        slice_666: "bf16[8, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_665, 3, -257, 9223372036854775807);  slice_665 = None
        where_90: "bf16[8, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_144, permute_534, slice_666);  convert_element_type_144 = permute_534 = slice_666 = None
        copy_137: "bf16[8, 256, 12, 257]" = torch.ops.aten.copy.default(slice_664, where_90);  slice_664 = where_90 = None
        slice_scatter_251: "bf16[8, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_662, copy_137, 3, -257, 9223372036854775807);  slice_662 = copy_137 = None
        slice_scatter_252: "bf16[8, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_532, slice_scatter_251, 1, -256, 9223372036854775807);  permute_532 = slice_scatter_251 = None
        permute_536: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_252, [0, 2, 1, 3]);  slice_scatter_252 = None
        permute_537: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_536, [0, 2, 1, 3]);  permute_536 = None
        full_126: "bf16[8, 4, 256, 513]" = torch.ops.aten.full.default(_shape_param_901, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_901 = None
        slice_667: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_126, 1, 0, -1)
        slice_668: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(full_126, 1, 0, -1)
        slice_669: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_668, 3, 256, 9223372036854775807);  slice_668 = None
        full_127: "bf16[8, 1024, 1, 1]" = torch.ops.aten.full.default(_shape_param_902, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_902 = None
        permute_538: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(full_127, [0, 2, 1, 3]);  full_127 = None
        view_616: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_538, _shape_param_903);  permute_538 = _shape_param_903 = None
        view_617: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_616, _shape_param_904);  view_616 = _shape_param_904 = None
        as_strided_57: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_617, _shape_param_905, _shape_param_906);  view_617 = _shape_param_905 = _shape_param_906 = None
        unsqueeze_206: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_57, 4);  as_strided_57 = None
        permute_539: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.permute.default(unsqueeze_206, [0, 1, 2, 4, 3]);  unsqueeze_206 = None
        full_128: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ne_11: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg6_1, 0);  arg6_1 = None
        unsqueeze_207: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(ne_11, 2);  ne_11 = None
        unsqueeze_208: "b8[8, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 3);  unsqueeze_207 = None
        convert_element_type_145: "bf16[8, 1024, 1, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze_208, torch.bfloat16)
        where_91: "bf16[8, 1024, 1, 1]" = torch.ops.aten.where.self(unsqueeze_208, full_128, convert_element_type_145);  unsqueeze_208 = full_128 = convert_element_type_145 = None
        permute_540: "bf16[8, 1, 1024, 1]" = torch.ops.aten.permute.default(where_91, [0, 2, 1, 3]);  where_91 = None
        view_618: "bf16[8, 1024, 1]" = torch.ops.aten.view.default(permute_540, _shape_param_907);  permute_540 = _shape_param_907 = None
        view_619: "bf16[8, 2, 512, 1]" = torch.ops.aten.view.default(view_618, _shape_param_908);  view_618 = _shape_param_908 = None
        as_strided_58: "bf16[8, 3, 512, 1]" = torch.ops.aten.as_strided.default(view_619, _shape_param_909, _shape_param_910);  view_619 = _shape_param_909 = _shape_param_910 = None
        unsqueeze_209: "bf16[8, 3, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(as_strided_58, 4);  as_strided_58 = None
        permute_541: "bf16[8, 3, 1, 512, 1]" = torch.ops.aten.permute.default(unsqueeze_209, [0, 1, 4, 2, 3]);  unsqueeze_209 = None
        mul_88: "bf16[8, 3, 512, 512, 1]" = torch.ops.aten.mul.Tensor(permute_539, permute_541);  permute_539 = permute_541 = None
        view_620: "bf16[8, 3, 512, 512]" = torch.ops.aten.view.default(mul_88, _shape_param_911);  mul_88 = _shape_param_911 = None
        constant_pad_nd_45: "bf16[8, 3, 513, 512]" = torch.ops.aten.constant_pad_nd.default(view_620, [0, 0, 0, 1], 0.0);  view_620 = None
        view_621: "bf16[8, 3, 512, 513]" = torch.ops.aten.view.default(constant_pad_nd_45, _shape_param_912);  constant_pad_nd_45 = _shape_param_912 = None
        slice_670: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_621, 2, 0, 256)
        slice_671: "bf16[8, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_670, 3, 0, 257);  slice_670 = None
        copy_138: "bf16[8, 3, 256, 257]" = torch.ops.aten.copy.default(slice_669, slice_671);  slice_669 = slice_671 = None
        slice_scatter_253: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_667, copy_138, 3, 256, 9223372036854775807);  slice_667 = copy_138 = None
        slice_scatter_254: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(full_126, slice_scatter_253, 1, 0, -1);  full_126 = slice_scatter_253 = None
        select_138: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_254, 1, -1)
        select_139: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_254, 1, -1)
        slice_672: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(select_139, 2, 256, 9223372036854775807);  select_139 = None
        select_140: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_621, 1, -1)
        slice_673: "bf16[8, 256, 513]" = torch.ops.aten.slice.Tensor(select_140, 1, 256, 9223372036854775807);  select_140 = None
        slice_674: "bf16[8, 256, 257]" = torch.ops.aten.slice.Tensor(slice_673, 2, 0, 257);  slice_673 = None
        copy_139: "bf16[8, 256, 257]" = torch.ops.aten.copy.default(slice_672, slice_674);  slice_672 = slice_674 = None
        slice_scatter_255: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_138, copy_139, 2, 256, 9223372036854775807);  select_138 = copy_139 = None
        select_scatter_46: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_254, slice_scatter_255, 1, -1);  slice_scatter_254 = slice_scatter_255 = None
        slice_675: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_46, 1, 1, 9223372036854775807)
        slice_676: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_46, 1, 1, 9223372036854775807)
        slice_677: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_676, 3, 0, 256);  slice_676 = None
        slice_678: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice.Tensor(view_621, 2, -257, -1)
        slice_679: "bf16[8, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_678, 3, 257, 9223372036854775807);  slice_678 = None
        copy_140: "bf16[8, 3, 256, 256]" = torch.ops.aten.copy.default(slice_677, slice_679);  slice_677 = slice_679 = None
        slice_scatter_256: "bf16[8, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_675, copy_140, 3, 0, 256);  slice_675 = copy_140 = None
        slice_scatter_257: "bf16[8, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_46, slice_scatter_256, 1, 1, 9223372036854775807);  select_scatter_46 = slice_scatter_256 = None
        select_141: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_257, 1, 0)
        slice_680: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_141, 1, 1, 256)
        select_142: "bf16[8, 256, 513]" = torch.ops.aten.select.int(slice_scatter_257, 1, 0)
        slice_681: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_142, 1, 1, 256);  select_142 = None
        slice_682: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_681, 2, 1, 256);  slice_681 = None
        select_143: "bf16[8, 512, 513]" = torch.ops.aten.select.int(view_621, 1, 0);  view_621 = None
        slice_683: "bf16[8, 255, 513]" = torch.ops.aten.slice.Tensor(select_143, 1, 0, 255);  select_143 = None
        slice_684: "bf16[8, 255, 255]" = torch.ops.aten.slice.Tensor(slice_683, 2, -255, 9223372036854775807);  slice_683 = None
        copy_141: "bf16[8, 255, 255]" = torch.ops.aten.copy.default(slice_682, slice_684);  slice_682 = slice_684 = None
        slice_scatter_258: "bf16[8, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_680, copy_141, 2, 1, 256);  slice_680 = copy_141 = None
        slice_scatter_259: "bf16[8, 256, 513]" = torch.ops.aten.slice_scatter.default(select_141, slice_scatter_258, 1, 1, 256);  select_141 = slice_scatter_258 = None
        select_scatter_47: "bf16[8, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_257, slice_scatter_259, 1, 0);  slice_scatter_257 = slice_scatter_259 = None
        view_622: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_47, _shape_param_913);  _shape_param_913 = None
        permute_542: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_622, [0, 2, 1, 3]);  view_622 = None
        slice_685: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_542, 1, 0, 256)
        view_623: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_47, _shape_param_914);  _shape_param_914 = None
        permute_543: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_623, [0, 2, 1, 3]);  view_623 = None
        slice_686: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_543, 1, 0, 256);  permute_543 = None
        slice_687: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_686, 3, 0, 257);  slice_686 = None
        iota_46: "i64[257]" = torch.ops.prims.iota.default(257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_210: "i64[1, 257]" = torch.ops.aten.unsqueeze.default(iota_46, -2);  iota_46 = None
        iota_47: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_211: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(iota_47, -1);  iota_47 = None
        sub_56: "i64[256, 257]" = torch.ops.aten.sub.Tensor(unsqueeze_210, unsqueeze_211);  unsqueeze_210 = unsqueeze_211 = None
        le_23: "b8[256, 257]" = torch.ops.aten.le.Scalar(sub_56, 0);  sub_56 = None
        full_129: "bf16[256, 257]" = torch.ops.aten.full.default(_shape_param_915, 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_915 = None
        scalar_tensor_23: "bf16[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where_92: "bf16[256, 257]" = torch.ops.aten.where.self(le_23, full_129, scalar_tensor_23);  le_23 = full_129 = scalar_tensor_23 = None
        rev_46: "bf16[256, 257]" = torch.ops.prims.rev.default(where_92, [0]);  where_92 = None
        unsqueeze_212: "bf16[1, 256, 257]" = torch.ops.aten.unsqueeze.default(rev_46, 0);  rev_46 = None
        unsqueeze_213: "bf16[1, 256, 1, 257]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 2);  unsqueeze_212 = None
        expand_46: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(unsqueeze_213, _shape_param_916);  _shape_param_916 = None
        convert_element_type_146: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_46, torch.bool);  expand_46 = None
        full_130: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_917, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_917 = None
        view_624: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(select_scatter_47, _shape_param_918);  select_scatter_47 = _shape_param_918 = None
        permute_544: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_624, [0, 2, 1, 3]);  view_624 = None
        slice_688: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_544, 1, 0, 256);  permute_544 = None
        slice_689: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_688, 3, 0, 257);  slice_688 = None
        where_93: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_146, full_130, slice_689);  convert_element_type_146 = full_130 = slice_689 = None
        copy_142: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_687, where_93);  slice_687 = where_93 = None
        slice_scatter_260: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_685, copy_142, 3, 0, 257);  slice_685 = copy_142 = None
        slice_scatter_261: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_542, slice_scatter_260, 1, 0, 256);  permute_542 = slice_scatter_260 = None
        permute_545: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_261, [0, 2, 1, 3]);  slice_scatter_261 = None
        view_625: "bf16[8, 4, 256, 513]" = torch.ops.aten.view.default(permute_545, _shape_param_919);  permute_545 = _shape_param_919 = None
        view_626: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_625, _shape_param_920);  _shape_param_920 = None
        permute_546: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_626, [0, 2, 1, 3]);  view_626 = None
        slice_690: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_546, 1, -256, 9223372036854775807)
        view_627: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_625, _shape_param_921);  _shape_param_921 = None
        permute_547: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_627, [0, 2, 1, 3]);  view_627 = None
        slice_691: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_547, 1, -256, 9223372036854775807);  permute_547 = None
        slice_692: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_691, 3, -257, 9223372036854775807);  slice_691 = None
        rev_47: "bf16[1, 256, 1, 257]" = torch.ops.prims.rev.default(unsqueeze_213, [1, 3]);  unsqueeze_213 = None
        expand_47: "bf16[8, 256, 1, 257]" = torch.ops.aten.expand.default(rev_47, _shape_param_922);  rev_47 = _shape_param_922 = None
        convert_element_type_147: "b8[8, 256, 1, 257]" = torch.ops.prims.convert_element_type.default(expand_47, torch.bool);  expand_47 = None
        full_131: "bf16[8, 256, 1, 257]" = torch.ops.aten.full.default(_shape_param_923, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_923 = None
        view_628: "bf16[8, 1, 1024, 513]" = torch.ops.aten.view.default(view_625, _shape_param_924);  view_625 = _shape_param_924 = None
        permute_548: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(view_628, [0, 2, 1, 3]);  view_628 = None
        slice_693: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice.Tensor(permute_548, 1, -256, 9223372036854775807);  permute_548 = None
        slice_694: "bf16[8, 256, 1, 257]" = torch.ops.aten.slice.Tensor(slice_693, 3, -257, 9223372036854775807);  slice_693 = None
        where_94: "bf16[8, 256, 1, 257]" = torch.ops.aten.where.self(convert_element_type_147, full_131, slice_694);  convert_element_type_147 = full_131 = slice_694 = None
        copy_143: "bf16[8, 256, 1, 257]" = torch.ops.aten.copy.default(slice_692, where_94);  slice_692 = where_94 = None
        slice_scatter_262: "bf16[8, 256, 1, 513]" = torch.ops.aten.slice_scatter.default(slice_690, copy_143, 3, -257, 9223372036854775807);  slice_690 = copy_143 = None
        slice_scatter_263: "bf16[8, 1024, 1, 513]" = torch.ops.aten.slice_scatter.default(permute_546, slice_scatter_262, 1, -256, 9223372036854775807);  permute_546 = slice_scatter_262 = None
        permute_549: "bf16[8, 1, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_263, [0, 2, 1, 3]);  slice_scatter_263 = None
        permute_550: "bf16[8, 1024, 1, 513]" = torch.ops.aten.permute.default(permute_549, [0, 2, 1, 3]);  permute_549 = None
        add_134: "bf16[8, 1024, 12, 513]" = torch.ops.aten.add.Tensor(permute_537, permute_550);  permute_537 = permute_550 = None
        permute_551: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(add_134, [0, 2, 1, 3]);  add_134 = None
        permute_552: "bf16[8, 1024, 12, 513]" = torch.ops.aten.permute.default(permute_551, [0, 2, 1, 3]);  permute_551 = None
        convert_element_type_148: "f32[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(permute_552, torch.float32);  permute_552 = None
        clone_114: "f32[8, 1024, 12, 513]" = torch.ops.aten.clone.default(convert_element_type_148, memory_format = torch.contiguous_format);  convert_element_type_148 = None
        amax_11: "f32[8, 1024, 12, 1]" = torch.ops.aten.amax.default(clone_114, [-1], True)
        sub_57: "f32[8, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(clone_114, amax_11);  clone_114 = amax_11 = None
        exp_11: "f32[8, 1024, 12, 513]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
        sum_12: "f32[8, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_23: "f32[8, 1024, 12, 513]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        where_95: "f32[8, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_199, full_121, div_23);  unsqueeze_199 = full_121 = div_23 = None
        convert_element_type_149: "bf16[8, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(where_95, torch.bfloat16);  where_95 = None
        permute_553: "bf16[8, 12, 1024, 513]" = torch.ops.aten.permute.default(convert_element_type_149, [0, 2, 1, 3]);  convert_element_type_149 = None
        clone_115: "bf16[8, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_553, memory_format = torch.contiguous_format);  permute_553 = None
        view_629: "bf16[96, 4, 256, 513]" = torch.ops.aten.view.default(clone_115, _shape_param_925);  clone_115 = _shape_param_925 = None
        constant_pad_nd_46: "bf16[96, 4, 256, 770]" = torch.ops.aten.constant_pad_nd.default(view_629, _shape_param_926, 0.0);  view_629 = _shape_param_926 = None
        view_630: "bf16[96, 4, 197120]" = torch.ops.aten.view.default(constant_pad_nd_46, _shape_param_927);  constant_pad_nd_46 = _shape_param_927 = None
        slice_695: "bf16[96, 4, 196864]" = torch.ops.aten.slice.Tensor(view_630, 2, 0, -256);  view_630 = None
        view_631: "bf16[96, 4, 256, 769]" = torch.ops.aten.view.default(slice_695, _shape_param_928);  slice_695 = _shape_param_928 = None
        slice_696: "bf16[96, 4, 256, 768]" = torch.ops.aten.slice.Tensor(view_631, 3, 0, -1);  view_631 = None
        unsqueeze_214: "bf16[96, 4, 256, 768, 1]" = torch.ops.aten.unsqueeze.default(slice_696, 4);  slice_696 = None
        view_632: "bf16[384, 256, 768]" = torch.ops.aten.view.default(unsqueeze_214, _shape_param_929);  unsqueeze_214 = _shape_param_929 = None
        clone_116: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format);  permute_517 = None
        view_633: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_116, _shape_param_930);  clone_116 = _shape_param_930 = None
        permute_554: "bf16[768, 768]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        mm_46: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_633, permute_554);  view_633 = permute_554 = None
        view_634: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_46, _shape_param_931);  mm_46 = _shape_param_931 = None
        add_135: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_634, arg184_1);  view_634 = arg184_1 = None
        view_635: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_135, _shape_param_932);  add_135 = _shape_param_932 = None
        permute_555: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_635, [1, 0, 2, 3]);  view_635 = None
        permute_556: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_555, [0, 2, 1, 3]);  permute_555 = None
        view_636: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_556, _shape_param_933);  permute_556 = _shape_param_933 = None
        constant_pad_nd_47: "bf16[96, 1536, 64]" = torch.ops.aten.constant_pad_nd.default(view_636, _shape_param_934, -1.0);  view_636 = _shape_param_934 = None
        as_strided_59: "bf16[96, 4, 768, 64]" = torch.ops.aten.as_strided.default(constant_pad_nd_47, _shape_param_935, _shape_param_936);  constant_pad_nd_47 = _shape_param_935 = _shape_param_936 = None
        unsqueeze_215: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_59, 4);  as_strided_59 = None
        clone_117: "bf16[96, 4, 768, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_215, memory_format = torch.contiguous_format);  unsqueeze_215 = None
        view_637: "bf16[384, 768, 64]" = torch.ops.aten.view.default(clone_117, _shape_param_937);  clone_117 = _shape_param_937 = None
        bmm_23: "bf16[384, 256, 64]" = torch.ops.aten.bmm.default(view_632, view_637);  view_632 = view_637 = None
        view_638: "bf16[96, 4, 256, 1, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_938);  bmm_23 = _shape_param_938 = None
        permute_557: "bf16[96, 4, 256, 64, 1]" = torch.ops.aten.permute.default(view_638, [0, 1, 2, 4, 3]);  view_638 = None
        view_639: "bf16[96, 4, 256, 64]" = torch.ops.aten.view.default(permute_557, _shape_param_939);  permute_557 = _shape_param_939 = None
        view_640: "bf16[8, 12, 1024, 64]" = torch.ops.aten.view.default(view_639, _shape_param_940);  view_639 = _shape_param_940 = None
        permute_558: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_640, [0, 2, 1, 3]);  view_640 = None
        permute_559: "bf16[1024, 8, 12, 64]" = torch.ops.aten.permute.default(permute_558, [1, 0, 2, 3]);  permute_558 = None
        clone_118: "bf16[1024, 8, 12, 64]" = torch.ops.aten.clone.default(permute_559, memory_format = torch.contiguous_format);  permute_559 = None
        view_641: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(clone_118, _shape_param_941);  clone_118 = _shape_param_941 = None
        permute_560: "bf16[8, 1024, 768]" = torch.ops.aten.permute.default(view_641, [1, 0, 2]);  view_641 = None
        clone_119: "bf16[8, 1024, 768]" = torch.ops.aten.clone.default(permute_560, memory_format = torch.contiguous_format);  permute_560 = None
        view_642: "bf16[8192, 768]" = torch.ops.aten.view.default(clone_119, _shape_param_942);  clone_119 = _shape_param_942 = None
        permute_561: "bf16[768, 768]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        mm_47: "bf16[8192, 768]" = torch.ops.aten.mm.default(view_642, permute_561);  view_642 = permute_561 = None
        view_643: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(mm_47, _shape_param_943);  mm_47 = _shape_param_943 = None
        add_136: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_643, arg186_1);  view_643 = arg186_1 = None
        add_137: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_136, convert_element_type_142);  add_136 = convert_element_type_142 = None
        convert_element_type_150: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_137, torch.float32);  add_137 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_150, [2], correction = 0, keepdim = True)
        getitem_44: "f32[8, 1024, 1]" = var_mean_22[0]
        getitem_45: "f32[8, 1024, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_58: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_150, getitem_45);  convert_element_type_150 = getitem_45 = None
        add_138: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        mul_89: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_22);  sub_58 = rsqrt_22 = None
        mul_90: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_89, arg187_1);  mul_89 = arg187_1 = None
        add_139: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_90, arg188_1);  mul_90 = arg188_1 = None
        convert_element_type_151: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_139, torch.bfloat16);  add_139 = None
        view_644: "bf16[8192, 768]" = torch.ops.aten.view.default(convert_element_type_151, _shape_param_944);  _shape_param_944 = None
        permute_562: "bf16[768, 3072]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_22: "bf16[8192, 3072]" = torch.ops.aten.addmm.default(arg190_1, view_644, permute_562);  arg190_1 = view_644 = permute_562 = None
        view_645: "bf16[8, 1024, 3072]" = torch.ops.aten.view.default(addmm_22, _shape_param_945);  addmm_22 = _shape_param_945 = None
        convert_element_type_152: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_645, torch.float32);  view_645 = None
        mul_91: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_152, 0.5)
        mul_92: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_152, 0.7071067811865476);  convert_element_type_152 = None
        erf_11: "f32[8, 1024, 3072]" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
        add_140: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_93: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_91, add_140);  mul_91 = add_140 = None
        convert_element_type_153: "bf16[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16);  mul_93 = None
        view_646: "bf16[8192, 3072]" = torch.ops.aten.view.default(convert_element_type_153, _shape_param_946);  convert_element_type_153 = _shape_param_946 = None
        permute_563: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_23: "bf16[8192, 768]" = torch.ops.aten.addmm.default(arg192_1, view_646, permute_563);  arg192_1 = view_646 = permute_563 = None
        view_647: "bf16[8, 1024, 768]" = torch.ops.aten.view.default(addmm_23, _shape_param_947);  addmm_23 = _shape_param_947 = None
        add_141: "bf16[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_647, convert_element_type_151);  view_647 = convert_element_type_151 = None
        convert_element_type_154: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_141, torch.float32);  add_141 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_154, [2], correction = 0, keepdim = True)
        getitem_46: "f32[8, 1024, 1]" = var_mean_23[0]
        getitem_47: "f32[8, 1024, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_59: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_154, getitem_47);  convert_element_type_154 = getitem_47 = None
        add_142: "f32[8, 1024, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[8, 1024, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        mul_94: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_23);  sub_59 = rsqrt_23 = None
        mul_95: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_94, arg193_1);  mul_94 = arg193_1 = None
        add_143: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_95, arg194_1);  mul_95 = arg194_1 = None
        convert_element_type_155: "bf16[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16);  add_143 = None
        return convert_element_type_155



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
