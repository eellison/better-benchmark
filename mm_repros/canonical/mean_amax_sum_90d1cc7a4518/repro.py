"""
Standalone repro captured via capture_hook.
Label: MT5ForConditionalGeneration
Pattern hash: 90d1cc7a4518
Shape hash: 1d787ece
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
    def forward(self, arg0_1: "i64[32, 128]", arg1_1: "bf16[250112, 512]", arg2_1: "i64[32, 128]", arg3_1: "bf16[512]", arg4_1: "bf16[384, 512]", arg5_1: "bf16[384, 512]", arg6_1: "bf16[32, 6]", arg7_1: "bf16[384, 512]", arg8_1: "bf16[512, 384]", arg9_1: "bf16[512]", arg10_1: "bf16[384, 512]", arg11_1: "bf16[512]", arg12_1: "bf16[384, 512]", arg13_1: "bf16[384, 512]", arg14_1: "bf16[32, 6]", arg15_1: "bf16[384, 512]", arg16_1: "bf16[512, 384]", arg17_1: "bf16[512]", arg18_1: "bf16[1024, 512]", arg19_1: "bf16[1024, 512]", arg20_1: "bf16[512, 1024]", arg21_1: "bf16[512]", arg22_1: "bf16[384, 512]", arg23_1: "bf16[384, 512]", arg24_1: "bf16[384, 512]", arg25_1: "bf16[512, 384]", arg26_1: "bf16[512]", arg27_1: "bf16[1024, 512]", arg28_1: "bf16[1024, 512]", arg29_1: "bf16[512, 1024]", arg30_1: "bf16[512]", arg31_1: "bf16[384, 512]", arg32_1: "bf16[384, 512]", arg33_1: "bf16[384, 512]", arg34_1: "bf16[512, 384]", arg35_1: "bf16[512]", arg36_1: "bf16[1024, 512]", arg37_1: "bf16[1024, 512]", arg38_1: "bf16[512, 1024]", arg39_1: "bf16[512]", arg40_1: "bf16[384, 512]", arg41_1: "bf16[384, 512]", arg42_1: "bf16[384, 512]", arg43_1: "bf16[512, 384]", arg44_1: "bf16[512]", arg45_1: "bf16[1024, 512]", arg46_1: "bf16[1024, 512]", arg47_1: "bf16[512, 1024]", arg48_1: "bf16[512]", arg49_1: "bf16[384, 512]", arg50_1: "bf16[384, 512]", arg51_1: "bf16[384, 512]", arg52_1: "bf16[512, 384]", arg53_1: "bf16[512]", arg54_1: "bf16[1024, 512]", arg55_1: "bf16[1024, 512]", arg56_1: "bf16[512, 1024]", arg57_1: "bf16[512]", arg58_1: "bf16[384, 512]", arg59_1: "bf16[384, 512]", arg60_1: "bf16[384, 512]", arg61_1: "bf16[512, 384]", arg62_1: "bf16[512]", arg63_1: "bf16[1024, 512]", arg64_1: "bf16[1024, 512]", arg65_1: "bf16[512, 1024]", arg66_1: "bf16[512]", arg67_1: "bf16[384, 512]", arg68_1: "bf16[384, 512]", arg69_1: "bf16[384, 512]", arg70_1: "bf16[512, 384]", arg71_1: "bf16[512]", arg72_1: "bf16[1024, 512]", arg73_1: "bf16[1024, 512]", arg74_1: "bf16[512, 1024]", arg75_1: "bf16[512]", arg76_1: "bf16[384, 512]", arg77_1: "bf16[384, 512]", arg78_1: "bf16[384, 512]", arg79_1: "bf16[512, 384]", arg80_1: "bf16[512]", arg81_1: "bf16[1024, 512]", arg82_1: "bf16[1024, 512]", arg83_1: "bf16[512, 1024]", arg84_1: "bf16[512]", arg85_1: "bf16[384, 512]", arg86_1: "bf16[384, 512]", arg87_1: "bf16[512, 384]", arg88_1: "bf16[512]", arg89_1: "bf16[1024, 512]", arg90_1: "bf16[1024, 512]", arg91_1: "bf16[512, 1024]", arg92_1: "bf16[512]", arg93_1: "bf16[384, 512]", arg94_1: "bf16[384, 512]", arg95_1: "bf16[384, 512]", arg96_1: "bf16[512, 384]", arg97_1: "bf16[512]", arg98_1: "bf16[384, 512]", arg99_1: "bf16[384, 512]", arg100_1: "bf16[384, 512]", arg101_1: "bf16[512, 384]", arg102_1: "bf16[512]", arg103_1: "bf16[1024, 512]", arg104_1: "bf16[1024, 512]", arg105_1: "bf16[512, 1024]", arg106_1: "bf16[512]", arg107_1: "bf16[384, 512]", arg108_1: "bf16[384, 512]", arg109_1: "bf16[384, 512]", arg110_1: "bf16[512, 384]", arg111_1: "bf16[512]", arg112_1: "bf16[384, 512]", arg113_1: "bf16[384, 512]", arg114_1: "bf16[384, 512]", arg115_1: "bf16[512, 384]", arg116_1: "bf16[512]", arg117_1: "bf16[1024, 512]", arg118_1: "bf16[1024, 512]", arg119_1: "bf16[512, 1024]", arg120_1: "bf16[512]", arg121_1: "bf16[384, 512]", arg122_1: "bf16[384, 512]", arg123_1: "bf16[384, 512]", arg124_1: "bf16[512, 384]", arg125_1: "bf16[512]", arg126_1: "bf16[384, 512]", arg127_1: "bf16[384, 512]", arg128_1: "bf16[384, 512]", arg129_1: "bf16[512, 384]", arg130_1: "bf16[512]", arg131_1: "bf16[1024, 512]", arg132_1: "bf16[1024, 512]", arg133_1: "bf16[512, 1024]", arg134_1: "bf16[512]", arg135_1: "bf16[384, 512]", arg136_1: "bf16[384, 512]", arg137_1: "bf16[384, 512]", arg138_1: "bf16[512, 384]", arg139_1: "bf16[512]", arg140_1: "bf16[384, 512]", arg141_1: "bf16[384, 512]", arg142_1: "bf16[384, 512]", arg143_1: "bf16[512, 384]", arg144_1: "bf16[512]", arg145_1: "bf16[1024, 512]", arg146_1: "bf16[1024, 512]", arg147_1: "bf16[512, 1024]", arg148_1: "bf16[512]", arg149_1: "bf16[384, 512]", arg150_1: "bf16[384, 512]", arg151_1: "bf16[384, 512]", arg152_1: "bf16[512, 384]", arg153_1: "bf16[512]", arg154_1: "bf16[384, 512]", arg155_1: "bf16[384, 512]", arg156_1: "bf16[384, 512]", arg157_1: "bf16[512, 384]", arg158_1: "bf16[512]", arg159_1: "bf16[1024, 512]", arg160_1: "bf16[1024, 512]", arg161_1: "bf16[512, 1024]", arg162_1: "bf16[512]", arg163_1: "bf16[384, 512]", arg164_1: "bf16[384, 512]", arg165_1: "bf16[384, 512]", arg166_1: "bf16[512, 384]", arg167_1: "bf16[512]", arg168_1: "bf16[384, 512]", arg169_1: "bf16[384, 512]", arg170_1: "bf16[384, 512]", arg171_1: "bf16[512, 384]", arg172_1: "bf16[512]", arg173_1: "bf16[1024, 512]", arg174_1: "bf16[1024, 512]", arg175_1: "bf16[512, 1024]", arg176_1: "bf16[512]", arg177_1: "bf16[384, 512]", arg178_1: "bf16[384, 512]", arg179_1: "bf16[384, 512]", arg180_1: "bf16[512, 384]", arg181_1: "bf16[512]", arg182_1: "bf16[384, 512]", arg183_1: "bf16[384, 512]", arg184_1: "bf16[384, 512]", arg185_1: "bf16[512, 384]", arg186_1: "bf16[512]", arg187_1: "bf16[1024, 512]", arg188_1: "bf16[1024, 512]", arg189_1: "bf16[512, 1024]", arg190_1: "bf16[512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631):
        # No stacktrace found for following nodes
        view: "i64[4096]" = torch.ops.aten.view.default(arg0_1, [-1]);  arg0_1 = None
        ne: "b8[4096]" = torch.ops.aten.ne.Scalar(view, -100)
        embedding: "bf16[32, 128, 512]" = torch.ops.aten.embedding.default(arg1_1, arg2_1)
        convert_element_type: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32)
        pow_1: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2);  convert_element_type = None
        mean: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(embedding, rsqrt);  rsqrt = None
        convert_element_type_1: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg3_1, convert_element_type_1);  arg3_1 = convert_element_type_1 = None
        view_1: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_0);  _shape_param_0 = None
        permute: "bf16[512, 384]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_1, permute);  view_1 = permute = None
        view_2: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        view_3: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_2, _shape_param_2);  view_2 = _shape_param_2 = None
        permute_1: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        expand: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_1, _shape_param_3);  permute_1 = _shape_param_3 = None
        clone: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_4: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        view_5: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_5);  _shape_param_5 = None
        permute_2: "bf16[512, 384]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_1: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_5, permute_2);  view_5 = permute_2 = None
        view_6: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_1, _shape_param_6);  mm_1 = _shape_param_6 = None
        view_7: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_6, _shape_param_7);  view_6 = _shape_param_7 = None
        permute_3: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        permute_4: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand_1: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_4, _shape_param_8);  permute_4 = _shape_param_8 = None
        clone_1: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_8: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_9);  clone_1 = _shape_param_9 = None
        bmm: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_4, view_8);  view_4 = view_8 = None
        view_9: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_10);  bmm = _shape_param_10 = None
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        iota_1: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        add_1: "i64[128, 1]" = torch.ops.aten.add.Tensor(unsqueeze_1, 0);  unsqueeze_1 = None
        sub: "i64[128, 128]" = torch.ops.aten.sub.Tensor(unsqueeze, add_1);  unsqueeze = add_1 = None
        full: "i64[128, 128]" = torch.ops.aten.full.default(_shape_param_11, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_11 = None
        minimum: "i64[128, 128]" = torch.ops.aten.minimum.default(sub, full);  sub = full = None
        neg: "i64[128, 128]" = torch.ops.aten.neg.default(minimum);  minimum = None
        lt: "b8[128, 128]" = torch.ops.aten.lt.Scalar(neg, 16)
        convert_element_type_2: "f32[128, 128]" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
        div: "f32[128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_2, 16);  convert_element_type_2 = None
        log: "f32[128, 128]" = torch.ops.aten.log.default(div);  div = None
        div_1: "f32[128, 128]" = torch.ops.aten.div.Tensor(log, 2.0794415416798357);  log = None
        mul_2: "f32[128, 128]" = torch.ops.aten.mul.Tensor(div_1, 16);  div_1 = None
        convert_element_type_3: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(mul_2, torch.int64);  mul_2 = None
        add_2: "i64[128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_3, 16);  convert_element_type_3 = None
        full_1: "i64[128, 128]" = torch.ops.aten.full.default(_shape_param_12, 31, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_12 = None
        minimum_1: "i64[128, 128]" = torch.ops.aten.minimum.default(add_2, full_1);  add_2 = full_1 = None
        where: "i64[128, 128]" = torch.ops.aten.where.self(lt, neg, minimum_1);  lt = neg = minimum_1 = None
        add_3: "i64[128, 128]" = torch.ops.aten.add.Tensor(where, 0);  where = None
        embedding_1: "bf16[128, 128, 6]" = torch.ops.aten.embedding.default(arg6_1, add_3);  arg6_1 = add_3 = None
        permute_5: "bf16[6, 128, 128]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
        unsqueeze_2: "bf16[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_5, 0);  permute_5 = None
        iota_2: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_4: "i64[128]" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None
        unsqueeze_3: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_4, 0);  add_4 = None
        unsqueeze_4: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        iota_3: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_5: "i64[128]" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None
        unsqueeze_6: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_5, 0);  add_5 = None
        unsqueeze_7: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        unsqueeze_8: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_8);  unsqueeze_5 = unsqueeze_8 = None
        expand_2: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(le, _shape_param_13);  le = _shape_param_13 = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_3: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_2, full_2, full_3);  expand_2 = full_2 = full_3 = None
        add_6: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_2, where_1);  unsqueeze_2 = where_1 = None
        add_7: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_9, add_6);  view_9 = None
        convert_element_type_4: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_7, torch.float32);  add_7 = None
        amax: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_4, [-1], True)
        sub_1: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_4, amax);  convert_element_type_4 = amax = None
        exp: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_2: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_5: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        expand_3: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_5, _shape_param_14);  convert_element_type_5 = _shape_param_14 = None
        view_10: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_3, _shape_param_15);  expand_3 = _shape_param_15 = None
        view_11: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_1, _shape_param_16);  mul_1 = _shape_param_16 = None
        permute_6: "bf16[512, 384]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_2: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_11, permute_6);  view_11 = permute_6 = None
        view_12: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_2, _shape_param_17);  mm_2 = _shape_param_17 = None
        view_13: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_12, _shape_param_18);  view_12 = _shape_param_18 = None
        permute_7: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        expand_4: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_7, _shape_param_19);  permute_7 = _shape_param_19 = None
        clone_2: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_14: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_20);  clone_2 = _shape_param_20 = None
        bmm_1: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_10, view_14);  view_10 = view_14 = None
        view_15: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_21);  bmm_1 = _shape_param_21 = None
        permute_8: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        clone_3: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_16: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_3, _shape_param_22);  clone_3 = _shape_param_22 = None
        view_17: "bf16[4096, 384]" = torch.ops.aten.view.default(view_16, _shape_param_23);  view_16 = _shape_param_23 = None
        permute_9: "bf16[384, 512]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        mm_3: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_17, permute_9);  view_17 = permute_9 = None
        view_18: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_3, _shape_param_24);  mm_3 = _shape_param_24 = None
        add_8: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(embedding, view_18);  embedding = view_18 = None
        convert_element_type_6: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_8, torch.float32)
        pow_2: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_6, 2);  convert_element_type_6 = None
        mean_1: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_9: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_8, rsqrt_1);  rsqrt_1 = None
        convert_element_type_7: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        mul_4: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg9_1, convert_element_type_7);  arg9_1 = convert_element_type_7 = None
        view_19: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_4, _shape_param_25);  mul_4 = _shape_param_25 = None
        permute_10: "bf16[512, 384]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm_4: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_19, permute_10);  view_19 = permute_10 = None
        view_20: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_4, _shape_param_26);  mm_4 = _shape_param_26 = None
        view_21: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_20, _shape_param_27);  view_20 = _shape_param_27 = None
        permute_11: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        expand_5: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_11, _shape_param_28);  permute_11 = _shape_param_28 = None
        clone_4: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_22: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_29);  clone_4 = _shape_param_29 = None
        embedding_2: "bf16[32, 128, 512]" = torch.ops.aten.embedding.default(arg1_1, arg2_1);  arg2_1 = None
        convert_element_type_8: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(embedding_2, torch.float32)
        pow_3: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_8, 2);  convert_element_type_8 = None
        mean_2: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        add_10: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_5: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(embedding_2, rsqrt_2);  rsqrt_2 = None
        convert_element_type_9: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        mul_6: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg11_1, convert_element_type_9);  arg11_1 = convert_element_type_9 = None
        view_23: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_6, _shape_param_30);  _shape_param_30 = None
        permute_12: "bf16[512, 384]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        mm_5: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_23, permute_12);  view_23 = permute_12 = None
        view_24: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_5, _shape_param_31);  mm_5 = _shape_param_31 = None
        view_25: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_24, _shape_param_32);  view_24 = _shape_param_32 = None
        permute_13: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        expand_6: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_13, _shape_param_33);  permute_13 = _shape_param_33 = None
        clone_5: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_26: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_5, _shape_param_34);  clone_5 = _shape_param_34 = None
        view_27: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_6, _shape_param_35);  _shape_param_35 = None
        permute_14: "bf16[512, 384]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        mm_6: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_27, permute_14);  view_27 = permute_14 = None
        view_28: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_6, _shape_param_36);  mm_6 = _shape_param_36 = None
        view_29: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_28, _shape_param_37);  view_28 = _shape_param_37 = None
        permute_15: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        permute_16: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        expand_7: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_16, _shape_param_38);  permute_16 = _shape_param_38 = None
        clone_6: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_30: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_6, _shape_param_39);  clone_6 = _shape_param_39 = None
        bmm_2: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_26, view_30);  view_26 = view_30 = None
        view_31: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_40);  bmm_2 = _shape_param_40 = None
        iota_4: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_9: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota_4, 0);  iota_4 = None
        iota_5: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_10: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(iota_5, 1);  iota_5 = None
        add_11: "i64[128, 1]" = torch.ops.aten.add.Tensor(unsqueeze_10, 0);  unsqueeze_10 = None
        sub_2: "i64[128, 128]" = torch.ops.aten.sub.Tensor(unsqueeze_9, add_11);  unsqueeze_9 = add_11 = None
        gt: "b8[128, 128]" = torch.ops.aten.gt.Scalar(sub_2, 0)
        convert_element_type_10: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(gt, torch.int64);  gt = None
        mul_7: "i64[128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 16);  convert_element_type_10 = None
        add_12: "i64[128, 128]" = torch.ops.aten.add.Tensor(mul_7, 0);  mul_7 = None
        abs_1: "i64[128, 128]" = torch.ops.aten.abs.default(sub_2);  sub_2 = None
        lt_1: "b8[128, 128]" = torch.ops.aten.lt.Scalar(abs_1, 8)
        convert_element_type_11: "f32[128, 128]" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
        div_3: "f32[128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_11, 8);  convert_element_type_11 = None
        log_1: "f32[128, 128]" = torch.ops.aten.log.default(div_3);  div_3 = None
        div_4: "f32[128, 128]" = torch.ops.aten.div.Tensor(log_1, 2.772588722239781);  log_1 = None
        mul_8: "f32[128, 128]" = torch.ops.aten.mul.Tensor(div_4, 8);  div_4 = None
        convert_element_type_12: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(mul_8, torch.int64);  mul_8 = None
        add_13: "i64[128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_12, 8);  convert_element_type_12 = None
        full_4: "i64[128, 128]" = torch.ops.aten.full.default(_shape_param_41, 15, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_41 = None
        minimum_2: "i64[128, 128]" = torch.ops.aten.minimum.default(add_13, full_4);  add_13 = full_4 = None
        where_2: "i64[128, 128]" = torch.ops.aten.where.self(lt_1, abs_1, minimum_2);  lt_1 = abs_1 = minimum_2 = None
        add_14: "i64[128, 128]" = torch.ops.aten.add.Tensor(add_12, where_2);  add_12 = where_2 = None
        embedding_3: "bf16[128, 128, 6]" = torch.ops.aten.embedding.default(arg14_1, add_14);  arg14_1 = add_14 = None
        permute_17: "bf16[6, 128, 128]" = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
        unsqueeze_11: "bf16[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_17, 0);  permute_17 = None
        full_5: "bf16[32, 1, 128, 128]" = torch.ops.aten.full.default(_shape_param_42, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_42 = None
        add_15: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_11, full_5);  unsqueeze_11 = full_5 = None
        add_16: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_31, add_15);  view_31 = None
        convert_element_type_13: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None
        amax_1: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_13, [-1], True)
        sub_3: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_13, amax_1);  convert_element_type_13 = amax_1 = None
        exp_1: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_2: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_5: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_14: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        expand_8: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_14, _shape_param_43);  convert_element_type_14 = _shape_param_43 = None
        view_32: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_8, _shape_param_44);  expand_8 = _shape_param_44 = None
        view_33: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_6, _shape_param_45);  mul_6 = _shape_param_45 = None
        permute_18: "bf16[512, 384]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        mm_7: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_33, permute_18);  view_33 = permute_18 = None
        view_34: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_7, _shape_param_46);  mm_7 = _shape_param_46 = None
        view_35: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_34, _shape_param_47);  view_34 = _shape_param_47 = None
        permute_19: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
        expand_9: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_19, _shape_param_48);  permute_19 = _shape_param_48 = None
        clone_7: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_36: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_7, _shape_param_49);  clone_7 = _shape_param_49 = None
        bmm_3: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_32, view_36);  view_32 = view_36 = None
        view_37: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_50);  bmm_3 = _shape_param_50 = None
        permute_20: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        clone_8: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
        view_38: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_8, _shape_param_51);  clone_8 = _shape_param_51 = None
        view_39: "bf16[4096, 384]" = torch.ops.aten.view.default(view_38, _shape_param_52);  view_38 = _shape_param_52 = None
        permute_21: "bf16[384, 512]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        mm_8: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_39, permute_21);  view_39 = permute_21 = None
        view_40: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_8, _shape_param_53);  mm_8 = _shape_param_53 = None
        add_17: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(embedding_2, view_40);  embedding_2 = view_40 = None
        convert_element_type_15: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_17, torch.float32)
        pow_4: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_15, 2);  convert_element_type_15 = None
        mean_3: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
        add_18: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_9: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_17, rsqrt_3);  rsqrt_3 = None
        convert_element_type_16: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        mul_10: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg17_1, convert_element_type_16);  arg17_1 = convert_element_type_16 = None
        view_41: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_10, _shape_param_54);  _shape_param_54 = None
        permute_22: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        mm_9: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_41, permute_22);  view_41 = permute_22 = None
        view_42: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_9, _shape_param_55);  mm_9 = _shape_param_55 = None
        mul_11: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_42, 0.5)
        pow_5: "bf16[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_42, 3.0)
        mul_12: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_19: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_42, mul_12);  view_42 = mul_12 = None
        mul_13: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_19, 0.7978845608028654);  add_19 = None
        tanh: "bf16[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_13);  mul_13 = None
        add_20: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_14: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_11, add_20);  mul_11 = add_20 = None
        view_43: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_10, _shape_param_56);  mul_10 = _shape_param_56 = None
        permute_23: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_10: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_43, permute_23);  view_43 = permute_23 = None
        view_44: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_10, _shape_param_57);  mm_10 = _shape_param_57 = None
        mul_15: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_14, view_44);  mul_14 = view_44 = None
        view_45: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_15, _shape_param_58);  mul_15 = _shape_param_58 = None
        permute_24: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        mm_11: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_45, permute_24);  view_45 = permute_24 = None
        view_46: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_11, _shape_param_59);  mm_11 = _shape_param_59 = None
        add_21: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_17, view_46);  add_17 = view_46 = None
        convert_element_type_17: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_21, torch.float32)
        pow_6: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_17, 2);  convert_element_type_17 = None
        mean_4: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None
        add_22: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
        rsqrt_4: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_16: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_21, rsqrt_4);  rsqrt_4 = None
        convert_element_type_18: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None
        mul_17: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg21_1, convert_element_type_18);  arg21_1 = convert_element_type_18 = None
        view_47: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_17, _shape_param_60);  _shape_param_60 = None
        permute_25: "bf16[512, 384]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_12: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_47, permute_25);  view_47 = permute_25 = None
        view_48: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_12, _shape_param_61);  mm_12 = _shape_param_61 = None
        view_49: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_48, _shape_param_62);  view_48 = _shape_param_62 = None
        permute_26: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
        expand_10: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_26, _shape_param_63);  permute_26 = _shape_param_63 = None
        clone_9: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_50: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_9, _shape_param_64);  clone_9 = _shape_param_64 = None
        view_51: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_17, _shape_param_65);  _shape_param_65 = None
        permute_27: "bf16[512, 384]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_13: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_51, permute_27);  view_51 = permute_27 = None
        view_52: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_13, _shape_param_66);  mm_13 = _shape_param_66 = None
        view_53: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_52, _shape_param_67);  view_52 = _shape_param_67 = None
        permute_28: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        permute_29: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_28, [0, 1, 3, 2]);  permute_28 = None
        expand_11: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_29, _shape_param_68);  permute_29 = _shape_param_68 = None
        clone_10: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_54: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_10, _shape_param_69);  clone_10 = _shape_param_69 = None
        bmm_4: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_50, view_54);  view_50 = view_54 = None
        view_55: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_4, _shape_param_70);  bmm_4 = _shape_param_70 = None
        add_23: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_55, add_15);  view_55 = None
        convert_element_type_19: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_23, torch.float32);  add_23 = None
        amax_2: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_19, [-1], True)
        sub_4: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_19, amax_2);  convert_element_type_19 = amax_2 = None
        exp_2: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_3: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_6: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_20: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        expand_12: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_20, _shape_param_71);  convert_element_type_20 = _shape_param_71 = None
        view_56: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_12, _shape_param_72);  expand_12 = _shape_param_72 = None
        view_57: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_17, _shape_param_73);  mul_17 = _shape_param_73 = None
        permute_30: "bf16[512, 384]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        mm_14: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_57, permute_30);  view_57 = permute_30 = None
        view_58: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_14, _shape_param_74);  mm_14 = _shape_param_74 = None
        view_59: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_58, _shape_param_75);  view_58 = _shape_param_75 = None
        permute_31: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3]);  view_59 = None
        expand_13: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_31, _shape_param_76);  permute_31 = _shape_param_76 = None
        clone_11: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_60: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_11, _shape_param_77);  clone_11 = _shape_param_77 = None
        bmm_5: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_56, view_60);  view_56 = view_60 = None
        view_61: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_5, _shape_param_78);  bmm_5 = _shape_param_78 = None
        permute_32: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
        clone_12: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_62: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_12, _shape_param_79);  clone_12 = _shape_param_79 = None
        view_63: "bf16[4096, 384]" = torch.ops.aten.view.default(view_62, _shape_param_80);  view_62 = _shape_param_80 = None
        permute_33: "bf16[384, 512]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        mm_15: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_63, permute_33);  view_63 = permute_33 = None
        view_64: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_15, _shape_param_81);  mm_15 = _shape_param_81 = None
        add_24: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_21, view_64);  add_21 = view_64 = None
        convert_element_type_21: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_24, torch.float32)
        pow_7: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_21, 2);  convert_element_type_21 = None
        mean_5: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
        add_25: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
        rsqrt_5: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_18: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_24, rsqrt_5);  rsqrt_5 = None
        convert_element_type_22: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        mul_19: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg26_1, convert_element_type_22);  arg26_1 = convert_element_type_22 = None
        view_65: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_19, _shape_param_82);  _shape_param_82 = None
        permute_34: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        mm_16: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_65, permute_34);  view_65 = permute_34 = None
        view_66: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_16, _shape_param_83);  mm_16 = _shape_param_83 = None
        mul_20: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_66, 0.5)
        pow_8: "bf16[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_66, 3.0)
        mul_21: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_26: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_66, mul_21);  view_66 = mul_21 = None
        mul_22: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_26, 0.7978845608028654);  add_26 = None
        tanh_1: "bf16[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_27: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_23: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_20, add_27);  mul_20 = add_27 = None
        view_67: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_19, _shape_param_84);  mul_19 = _shape_param_84 = None
        permute_35: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        mm_17: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_67, permute_35);  view_67 = permute_35 = None
        view_68: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_17, _shape_param_85);  mm_17 = _shape_param_85 = None
        mul_24: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_23, view_68);  mul_23 = view_68 = None
        view_69: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_24, _shape_param_86);  mul_24 = _shape_param_86 = None
        permute_36: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_18: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_69, permute_36);  view_69 = permute_36 = None
        view_70: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_18, _shape_param_87);  mm_18 = _shape_param_87 = None
        add_28: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_24, view_70);  add_24 = view_70 = None
        convert_element_type_23: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_28, torch.float32)
        pow_9: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_23, 2);  convert_element_type_23 = None
        mean_6: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None
        add_29: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
        rsqrt_6: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        mul_25: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_28, rsqrt_6);  rsqrt_6 = None
        convert_element_type_24: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None
        mul_26: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg30_1, convert_element_type_24);  arg30_1 = convert_element_type_24 = None
        view_71: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_26, _shape_param_88);  _shape_param_88 = None
        permute_37: "bf16[512, 384]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_19: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_71, permute_37);  view_71 = permute_37 = None
        view_72: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_19, _shape_param_89);  mm_19 = _shape_param_89 = None
        view_73: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_72, _shape_param_90);  view_72 = _shape_param_90 = None
        permute_38: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
        expand_14: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_38, _shape_param_91);  permute_38 = _shape_param_91 = None
        clone_13: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_74: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_13, _shape_param_92);  clone_13 = _shape_param_92 = None
        view_75: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_26, _shape_param_93);  _shape_param_93 = None
        permute_39: "bf16[512, 384]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        mm_20: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_75, permute_39);  view_75 = permute_39 = None
        view_76: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_20, _shape_param_94);  mm_20 = _shape_param_94 = None
        view_77: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_76, _shape_param_95);  view_76 = _shape_param_95 = None
        permute_40: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_77, [0, 2, 1, 3]);  view_77 = None
        permute_41: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_40, [0, 1, 3, 2]);  permute_40 = None
        expand_15: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_41, _shape_param_96);  permute_41 = _shape_param_96 = None
        clone_14: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_78: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_14, _shape_param_97);  clone_14 = _shape_param_97 = None
        bmm_6: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_74, view_78);  view_74 = view_78 = None
        view_79: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_98);  bmm_6 = _shape_param_98 = None
        add_30: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_79, add_15);  view_79 = None
        convert_element_type_25: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_30, torch.float32);  add_30 = None
        amax_3: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_25, [-1], True)
        sub_5: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_25, amax_3);  convert_element_type_25 = amax_3 = None
        exp_3: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_4: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_26: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        expand_16: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_26, _shape_param_99);  convert_element_type_26 = _shape_param_99 = None
        view_80: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_16, _shape_param_100);  expand_16 = _shape_param_100 = None
        view_81: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_26, _shape_param_101);  mul_26 = _shape_param_101 = None
        permute_42: "bf16[512, 384]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        mm_21: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_81, permute_42);  view_81 = permute_42 = None
        view_82: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_21, _shape_param_102);  mm_21 = _shape_param_102 = None
        view_83: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_82, _shape_param_103);  view_82 = _shape_param_103 = None
        permute_43: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
        expand_17: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_43, _shape_param_104);  permute_43 = _shape_param_104 = None
        clone_15: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_84: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_15, _shape_param_105);  clone_15 = _shape_param_105 = None
        bmm_7: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_80, view_84);  view_80 = view_84 = None
        view_85: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_106);  bmm_7 = _shape_param_106 = None
        permute_44: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        clone_16: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_86: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_16, _shape_param_107);  clone_16 = _shape_param_107 = None
        view_87: "bf16[4096, 384]" = torch.ops.aten.view.default(view_86, _shape_param_108);  view_86 = _shape_param_108 = None
        permute_45: "bf16[384, 512]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm_22: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_87, permute_45);  view_87 = permute_45 = None
        view_88: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_22, _shape_param_109);  mm_22 = _shape_param_109 = None
        add_31: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_28, view_88);  add_28 = view_88 = None
        convert_element_type_27: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_31, torch.float32)
        pow_10: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_27, 2);  convert_element_type_27 = None
        mean_7: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
        add_32: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
        rsqrt_7: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_27: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_31, rsqrt_7);  rsqrt_7 = None
        convert_element_type_28: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16);  mul_27 = None
        mul_28: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg35_1, convert_element_type_28);  arg35_1 = convert_element_type_28 = None
        view_89: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_28, _shape_param_110);  _shape_param_110 = None
        permute_46: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        mm_23: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_89, permute_46);  view_89 = permute_46 = None
        view_90: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_23, _shape_param_111);  mm_23 = _shape_param_111 = None
        mul_29: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_90, 0.5)
        pow_11: "bf16[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_90, 3.0)
        mul_30: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_33: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_90, mul_30);  view_90 = mul_30 = None
        mul_31: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_33, 0.7978845608028654);  add_33 = None
        tanh_2: "bf16[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_31);  mul_31 = None
        add_34: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_32: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_29, add_34);  mul_29 = add_34 = None
        view_91: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_28, _shape_param_112);  mul_28 = _shape_param_112 = None
        permute_47: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_24: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_91, permute_47);  view_91 = permute_47 = None
        view_92: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_24, _shape_param_113);  mm_24 = _shape_param_113 = None
        mul_33: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_32, view_92);  mul_32 = view_92 = None
        view_93: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_33, _shape_param_114);  mul_33 = _shape_param_114 = None
        permute_48: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        mm_25: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_93, permute_48);  view_93 = permute_48 = None
        view_94: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_25, _shape_param_115);  mm_25 = _shape_param_115 = None
        add_35: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_31, view_94);  add_31 = view_94 = None
        convert_element_type_29: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_35, torch.float32)
        pow_12: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_29, 2);  convert_element_type_29 = None
        mean_8: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None
        add_36: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
        rsqrt_8: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_34: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_35, rsqrt_8);  rsqrt_8 = None
        convert_element_type_30: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_34, torch.bfloat16);  mul_34 = None
        mul_35: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg39_1, convert_element_type_30);  arg39_1 = convert_element_type_30 = None
        view_95: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_35, _shape_param_116);  _shape_param_116 = None
        permute_49: "bf16[512, 384]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        mm_26: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_95, permute_49);  view_95 = permute_49 = None
        view_96: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_26, _shape_param_117);  mm_26 = _shape_param_117 = None
        view_97: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_96, _shape_param_118);  view_96 = _shape_param_118 = None
        permute_50: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
        expand_18: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_50, _shape_param_119);  permute_50 = _shape_param_119 = None
        clone_17: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_98: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_17, _shape_param_120);  clone_17 = _shape_param_120 = None
        view_99: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_35, _shape_param_121);  _shape_param_121 = None
        permute_51: "bf16[512, 384]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_27: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_99, permute_51);  view_99 = permute_51 = None
        view_100: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_27, _shape_param_122);  mm_27 = _shape_param_122 = None
        view_101: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_100, _shape_param_123);  view_100 = _shape_param_123 = None
        permute_52: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        permute_53: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_52, [0, 1, 3, 2]);  permute_52 = None
        expand_19: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_53, _shape_param_124);  permute_53 = _shape_param_124 = None
        clone_18: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_102: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_18, _shape_param_125);  clone_18 = _shape_param_125 = None
        bmm_8: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_98, view_102);  view_98 = view_102 = None
        view_103: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_8, _shape_param_126);  bmm_8 = _shape_param_126 = None
        add_37: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_103, add_15);  view_103 = None
        convert_element_type_31: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_37, torch.float32);  add_37 = None
        amax_4: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_31, [-1], True)
        sub_6: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_31, amax_4);  convert_element_type_31 = amax_4 = None
        exp_4: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_5: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_8: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_32: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        expand_20: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_32, _shape_param_127);  convert_element_type_32 = _shape_param_127 = None
        view_104: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_20, _shape_param_128);  expand_20 = _shape_param_128 = None
        view_105: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_35, _shape_param_129);  mul_35 = _shape_param_129 = None
        permute_54: "bf16[512, 384]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        mm_28: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_105, permute_54);  view_105 = permute_54 = None
        view_106: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_28, _shape_param_130);  mm_28 = _shape_param_130 = None
        view_107: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_106, _shape_param_131);  view_106 = _shape_param_131 = None
        permute_55: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
        expand_21: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_55, _shape_param_132);  permute_55 = _shape_param_132 = None
        clone_19: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_108: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_19, _shape_param_133);  clone_19 = _shape_param_133 = None
        bmm_9: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_104, view_108);  view_104 = view_108 = None
        view_109: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_9, _shape_param_134);  bmm_9 = _shape_param_134 = None
        permute_56: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
        clone_20: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_110: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_20, _shape_param_135);  clone_20 = _shape_param_135 = None
        view_111: "bf16[4096, 384]" = torch.ops.aten.view.default(view_110, _shape_param_136);  view_110 = _shape_param_136 = None
        permute_57: "bf16[384, 512]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        mm_29: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_111, permute_57);  view_111 = permute_57 = None
        view_112: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_29, _shape_param_137);  mm_29 = _shape_param_137 = None
        add_38: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_35, view_112);  add_35 = view_112 = None
        convert_element_type_33: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_38, torch.float32)
        pow_13: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_33, 2);  convert_element_type_33 = None
        mean_9: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
        add_39: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
        rsqrt_9: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_36: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_38, rsqrt_9);  rsqrt_9 = None
        convert_element_type_34: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_36, torch.bfloat16);  mul_36 = None
        mul_37: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg44_1, convert_element_type_34);  arg44_1 = convert_element_type_34 = None
        view_113: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_37, _shape_param_138);  _shape_param_138 = None
        permute_58: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        mm_30: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_113, permute_58);  view_113 = permute_58 = None
        view_114: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_30, _shape_param_139);  mm_30 = _shape_param_139 = None
        mul_38: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_114, 0.5)
        pow_14: "bf16[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_114, 3.0)
        mul_39: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_40: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_114, mul_39);  view_114 = mul_39 = None
        mul_40: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_40, 0.7978845608028654);  add_40 = None
        tanh_3: "bf16[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_40);  mul_40 = None
        add_41: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_41: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_38, add_41);  mul_38 = add_41 = None
        view_115: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_37, _shape_param_140);  mul_37 = _shape_param_140 = None
        permute_59: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        mm_31: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_115, permute_59);  view_115 = permute_59 = None
        view_116: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_31, _shape_param_141);  mm_31 = _shape_param_141 = None
        mul_42: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_41, view_116);  mul_41 = view_116 = None
        view_117: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_42, _shape_param_142);  mul_42 = _shape_param_142 = None
        permute_60: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        mm_32: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_117, permute_60);  view_117 = permute_60 = None
        view_118: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_32, _shape_param_143);  mm_32 = _shape_param_143 = None
        add_42: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_38, view_118);  add_38 = view_118 = None
        convert_element_type_35: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_42, torch.float32)
        pow_15: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_35, 2);  convert_element_type_35 = None
        mean_10: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None
        add_43: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
        rsqrt_10: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_43: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_42, rsqrt_10);  rsqrt_10 = None
        convert_element_type_36: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_43, torch.bfloat16);  mul_43 = None
        mul_44: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg48_1, convert_element_type_36);  arg48_1 = convert_element_type_36 = None
        view_119: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_44, _shape_param_144);  _shape_param_144 = None
        permute_61: "bf16[512, 384]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        mm_33: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_119, permute_61);  view_119 = permute_61 = None
        view_120: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_33, _shape_param_145);  mm_33 = _shape_param_145 = None
        view_121: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_120, _shape_param_146);  view_120 = _shape_param_146 = None
        permute_62: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
        expand_22: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_62, _shape_param_147);  permute_62 = _shape_param_147 = None
        clone_21: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_122: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_21, _shape_param_148);  clone_21 = _shape_param_148 = None
        view_123: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_44, _shape_param_149);  _shape_param_149 = None
        permute_63: "bf16[512, 384]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_34: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_123, permute_63);  view_123 = permute_63 = None
        view_124: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_34, _shape_param_150);  mm_34 = _shape_param_150 = None
        view_125: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_124, _shape_param_151);  view_124 = _shape_param_151 = None
        permute_64: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
        permute_65: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_64, [0, 1, 3, 2]);  permute_64 = None
        expand_23: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_65, _shape_param_152);  permute_65 = _shape_param_152 = None
        clone_22: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_126: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_22, _shape_param_153);  clone_22 = _shape_param_153 = None
        bmm_10: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_122, view_126);  view_122 = view_126 = None
        view_127: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_154);  bmm_10 = _shape_param_154 = None
        add_44: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_127, add_15);  view_127 = None
        convert_element_type_37: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_44, torch.float32);  add_44 = None
        amax_5: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_37, [-1], True)
        sub_7: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_37, amax_5);  convert_element_type_37 = amax_5 = None
        exp_5: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_6: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_9: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_38: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        expand_24: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_38, _shape_param_155);  convert_element_type_38 = _shape_param_155 = None
        view_128: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_24, _shape_param_156);  expand_24 = _shape_param_156 = None
        view_129: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_44, _shape_param_157);  mul_44 = _shape_param_157 = None
        permute_66: "bf16[512, 384]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        mm_35: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_129, permute_66);  view_129 = permute_66 = None
        view_130: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_35, _shape_param_158);  mm_35 = _shape_param_158 = None
        view_131: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_130, _shape_param_159);  view_130 = _shape_param_159 = None
        permute_67: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
        expand_25: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_67, _shape_param_160);  permute_67 = _shape_param_160 = None
        clone_23: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_132: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_23, _shape_param_161);  clone_23 = _shape_param_161 = None
        bmm_11: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_128, view_132);  view_128 = view_132 = None
        view_133: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_162);  bmm_11 = _shape_param_162 = None
        permute_68: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        clone_24: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_134: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_24, _shape_param_163);  clone_24 = _shape_param_163 = None
        view_135: "bf16[4096, 384]" = torch.ops.aten.view.default(view_134, _shape_param_164);  view_134 = _shape_param_164 = None
        permute_69: "bf16[384, 512]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        mm_36: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_135, permute_69);  view_135 = permute_69 = None
        view_136: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_36, _shape_param_165);  mm_36 = _shape_param_165 = None
        add_45: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_42, view_136);  add_42 = view_136 = None
        convert_element_type_39: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_45, torch.float32)
        pow_16: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_39, 2);  convert_element_type_39 = None
        mean_11: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
        add_46: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
        rsqrt_11: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_45: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_45, rsqrt_11);  rsqrt_11 = None
        convert_element_type_40: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        mul_46: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg53_1, convert_element_type_40);  arg53_1 = convert_element_type_40 = None
        view_137: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_46, _shape_param_166);  _shape_param_166 = None
        permute_70: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        mm_37: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_137, permute_70);  view_137 = permute_70 = None
        view_138: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_37, _shape_param_167);  mm_37 = _shape_param_167 = None
        mul_47: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_138, 0.5)
        pow_17: "bf16[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_138, 3.0)
        mul_48: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_47: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_138, mul_48);  view_138 = mul_48 = None
        mul_49: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_47, 0.7978845608028654);  add_47 = None
        tanh_4: "bf16[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_49);  mul_49 = None
        add_48: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_50: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_47, add_48);  mul_47 = add_48 = None
        view_139: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_46, _shape_param_168);  mul_46 = _shape_param_168 = None
        permute_71: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        mm_38: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_139, permute_71);  view_139 = permute_71 = None
        view_140: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_38, _shape_param_169);  mm_38 = _shape_param_169 = None
        mul_51: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_50, view_140);  mul_50 = view_140 = None
        view_141: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_51, _shape_param_170);  mul_51 = _shape_param_170 = None
        permute_72: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        mm_39: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_141, permute_72);  view_141 = permute_72 = None
        view_142: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_39, _shape_param_171);  mm_39 = _shape_param_171 = None
        add_49: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_45, view_142);  add_45 = view_142 = None
        convert_element_type_41: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_49, torch.float32)
        pow_18: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_41, 2);  convert_element_type_41 = None
        mean_12: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None
        add_50: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
        rsqrt_12: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_52: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_49, rsqrt_12);  rsqrt_12 = None
        convert_element_type_42: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None
        mul_53: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg57_1, convert_element_type_42);  arg57_1 = convert_element_type_42 = None
        view_143: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_53, _shape_param_172);  _shape_param_172 = None
        permute_73: "bf16[512, 384]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        mm_40: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_143, permute_73);  view_143 = permute_73 = None
        view_144: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_40, _shape_param_173);  mm_40 = _shape_param_173 = None
        view_145: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_144, _shape_param_174);  view_144 = _shape_param_174 = None
        permute_74: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
        expand_26: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_74, _shape_param_175);  permute_74 = _shape_param_175 = None
        clone_25: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_146: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_25, _shape_param_176);  clone_25 = _shape_param_176 = None
        view_147: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_53, _shape_param_177);  _shape_param_177 = None
        permute_75: "bf16[512, 384]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        mm_41: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_147, permute_75);  view_147 = permute_75 = None
        view_148: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_41, _shape_param_178);  mm_41 = _shape_param_178 = None
        view_149: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_148, _shape_param_179);  view_148 = _shape_param_179 = None
        permute_76: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
        permute_77: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_76, [0, 1, 3, 2]);  permute_76 = None
        expand_27: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_77, _shape_param_180);  permute_77 = _shape_param_180 = None
        clone_26: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_150: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_26, _shape_param_181);  clone_26 = _shape_param_181 = None
        bmm_12: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_146, view_150);  view_146 = view_150 = None
        view_151: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_12, _shape_param_182);  bmm_12 = _shape_param_182 = None
        add_51: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_151, add_15);  view_151 = None
        convert_element_type_43: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_51, torch.float32);  add_51 = None
        amax_6: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_43, [-1], True)
        sub_8: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_43, amax_6);  convert_element_type_43 = amax_6 = None
        exp_6: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_7: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_10: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_44: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        expand_28: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_44, _shape_param_183);  convert_element_type_44 = _shape_param_183 = None
        view_152: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_28, _shape_param_184);  expand_28 = _shape_param_184 = None
        view_153: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_53, _shape_param_185);  mul_53 = _shape_param_185 = None
        permute_78: "bf16[512, 384]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        mm_42: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_153, permute_78);  view_153 = permute_78 = None
        view_154: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_42, _shape_param_186);  mm_42 = _shape_param_186 = None
        view_155: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_154, _shape_param_187);  view_154 = _shape_param_187 = None
        permute_79: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None
        expand_29: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_79, _shape_param_188);  permute_79 = _shape_param_188 = None
        clone_27: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_156: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_27, _shape_param_189);  clone_27 = _shape_param_189 = None
        bmm_13: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_152, view_156);  view_152 = view_156 = None
        view_157: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_13, _shape_param_190);  bmm_13 = _shape_param_190 = None
        permute_80: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
        clone_28: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_158: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_28, _shape_param_191);  clone_28 = _shape_param_191 = None
        view_159: "bf16[4096, 384]" = torch.ops.aten.view.default(view_158, _shape_param_192);  view_158 = _shape_param_192 = None
        permute_81: "bf16[384, 512]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        mm_43: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_159, permute_81);  view_159 = permute_81 = None
        view_160: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_43, _shape_param_193);  mm_43 = _shape_param_193 = None
        add_52: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_49, view_160);  add_49 = view_160 = None
        convert_element_type_45: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_52, torch.float32)
        pow_19: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_45, 2);  convert_element_type_45 = None
        mean_13: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
        add_53: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
        rsqrt_13: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_54: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_52, rsqrt_13);  rsqrt_13 = None
        convert_element_type_46: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None
        mul_55: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg62_1, convert_element_type_46);  arg62_1 = convert_element_type_46 = None
        view_161: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_55, _shape_param_194);  _shape_param_194 = None
        permute_82: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_44: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_161, permute_82);  view_161 = permute_82 = None
        view_162: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_44, _shape_param_195);  mm_44 = _shape_param_195 = None
        mul_56: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_162, 0.5)
        pow_20: "bf16[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_162, 3.0)
        mul_57: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_54: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_162, mul_57);  view_162 = mul_57 = None
        mul_58: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_54, 0.7978845608028654);  add_54 = None
        tanh_5: "bf16[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_55: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_59: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_56, add_55);  mul_56 = add_55 = None
        view_163: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_55, _shape_param_196);  mul_55 = _shape_param_196 = None
        permute_83: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        mm_45: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_163, permute_83);  view_163 = permute_83 = None
        view_164: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_45, _shape_param_197);  mm_45 = _shape_param_197 = None
        mul_60: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_59, view_164);  mul_59 = view_164 = None
        view_165: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_60, _shape_param_198);  mul_60 = _shape_param_198 = None
        permute_84: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        mm_46: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_165, permute_84);  view_165 = permute_84 = None
        view_166: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_46, _shape_param_199);  mm_46 = _shape_param_199 = None
        add_56: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_52, view_166);  add_52 = view_166 = None
        convert_element_type_47: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_56, torch.float32)
        pow_21: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_47, 2);  convert_element_type_47 = None
        mean_14: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None
        add_57: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
        rsqrt_14: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        mul_61: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_56, rsqrt_14);  rsqrt_14 = None
        convert_element_type_48: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None
        mul_62: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg66_1, convert_element_type_48);  arg66_1 = convert_element_type_48 = None
        view_167: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_62, _shape_param_200);  _shape_param_200 = None
        permute_85: "bf16[512, 384]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        mm_47: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_167, permute_85);  view_167 = permute_85 = None
        view_168: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_47, _shape_param_201);  mm_47 = _shape_param_201 = None
        view_169: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_168, _shape_param_202);  view_168 = _shape_param_202 = None
        permute_86: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
        expand_30: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_86, _shape_param_203);  permute_86 = _shape_param_203 = None
        clone_29: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_170: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_29, _shape_param_204);  clone_29 = _shape_param_204 = None
        view_171: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_62, _shape_param_205);  _shape_param_205 = None
        permute_87: "bf16[512, 384]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        mm_48: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_171, permute_87);  view_171 = permute_87 = None
        view_172: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_48, _shape_param_206);  mm_48 = _shape_param_206 = None
        view_173: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_172, _shape_param_207);  view_172 = _shape_param_207 = None
        permute_88: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
        permute_89: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_88, [0, 1, 3, 2]);  permute_88 = None
        expand_31: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_89, _shape_param_208);  permute_89 = _shape_param_208 = None
        clone_30: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_174: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_30, _shape_param_209);  clone_30 = _shape_param_209 = None
        bmm_14: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_170, view_174);  view_170 = view_174 = None
        view_175: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_14, _shape_param_210);  bmm_14 = _shape_param_210 = None
        add_58: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_175, add_15);  view_175 = None
        convert_element_type_49: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_58, torch.float32);  add_58 = None
        amax_7: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_49, [-1], True)
        sub_9: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_49, amax_7);  convert_element_type_49 = amax_7 = None
        exp_7: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_8: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_11: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_50: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        expand_32: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_50, _shape_param_211);  convert_element_type_50 = _shape_param_211 = None
        view_176: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_32, _shape_param_212);  expand_32 = _shape_param_212 = None
        view_177: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_62, _shape_param_213);  mul_62 = _shape_param_213 = None
        permute_90: "bf16[512, 384]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        mm_49: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_177, permute_90);  view_177 = permute_90 = None
        view_178: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_49, _shape_param_214);  mm_49 = _shape_param_214 = None
        view_179: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_178, _shape_param_215);  view_178 = _shape_param_215 = None
        permute_91: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_179, [0, 2, 1, 3]);  view_179 = None
        expand_33: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_91, _shape_param_216);  permute_91 = _shape_param_216 = None
        clone_31: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_180: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_31, _shape_param_217);  clone_31 = _shape_param_217 = None
        bmm_15: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_176, view_180);  view_176 = view_180 = None
        view_181: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_218);  bmm_15 = _shape_param_218 = None
        permute_92: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        clone_32: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_182: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_32, _shape_param_219);  clone_32 = _shape_param_219 = None
        view_183: "bf16[4096, 384]" = torch.ops.aten.view.default(view_182, _shape_param_220);  view_182 = _shape_param_220 = None
        permute_93: "bf16[384, 512]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
        mm_50: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_183, permute_93);  view_183 = permute_93 = None
        view_184: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_50, _shape_param_221);  mm_50 = _shape_param_221 = None
        add_59: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_56, view_184);  add_56 = view_184 = None
        convert_element_type_51: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_59, torch.float32)
        pow_22: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_51, 2);  convert_element_type_51 = None
        mean_15: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
        add_60: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
        rsqrt_15: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_63: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_59, rsqrt_15);  rsqrt_15 = None
        convert_element_type_52: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16);  mul_63 = None
        mul_64: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg71_1, convert_element_type_52);  arg71_1 = convert_element_type_52 = None
        view_185: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_64, _shape_param_222);  _shape_param_222 = None
        permute_94: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
        mm_51: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_185, permute_94);  view_185 = permute_94 = None
        view_186: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_51, _shape_param_223);  mm_51 = _shape_param_223 = None
        mul_65: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_186, 0.5)
        pow_23: "bf16[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_186, 3.0)
        mul_66: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_61: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_186, mul_66);  view_186 = mul_66 = None
        mul_67: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_61, 0.7978845608028654);  add_61 = None
        tanh_6: "bf16[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_67);  mul_67 = None
        add_62: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_68: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_65, add_62);  mul_65 = add_62 = None
        view_187: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_64, _shape_param_224);  mul_64 = _shape_param_224 = None
        permute_95: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        mm_52: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_187, permute_95);  view_187 = permute_95 = None
        view_188: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_52, _shape_param_225);  mm_52 = _shape_param_225 = None
        mul_69: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_68, view_188);  mul_68 = view_188 = None
        view_189: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_69, _shape_param_226);  mul_69 = _shape_param_226 = None
        permute_96: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        mm_53: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_189, permute_96);  view_189 = permute_96 = None
        view_190: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_53, _shape_param_227);  mm_53 = _shape_param_227 = None
        add_63: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_59, view_190);  add_59 = view_190 = None
        convert_element_type_53: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_63, torch.float32)
        pow_24: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_53, 2);  convert_element_type_53 = None
        mean_16: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None
        add_64: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
        rsqrt_16: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_70: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_63, rsqrt_16);  rsqrt_16 = None
        convert_element_type_54: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None
        mul_71: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg75_1, convert_element_type_54);  arg75_1 = convert_element_type_54 = None
        view_191: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_71, _shape_param_228);  _shape_param_228 = None
        permute_97: "bf16[512, 384]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        mm_54: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_191, permute_97);  view_191 = permute_97 = None
        view_192: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_54, _shape_param_229);  mm_54 = _shape_param_229 = None
        view_193: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_192, _shape_param_230);  view_192 = _shape_param_230 = None
        permute_98: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
        expand_34: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_98, _shape_param_231);  permute_98 = _shape_param_231 = None
        clone_33: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_194: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_33, _shape_param_232);  clone_33 = _shape_param_232 = None
        view_195: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_71, _shape_param_233);  _shape_param_233 = None
        permute_99: "bf16[512, 384]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        mm_55: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_195, permute_99);  view_195 = permute_99 = None
        view_196: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_55, _shape_param_234);  mm_55 = _shape_param_234 = None
        view_197: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_196, _shape_param_235);  view_196 = _shape_param_235 = None
        permute_100: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_197, [0, 2, 1, 3]);  view_197 = None
        permute_101: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2]);  permute_100 = None
        expand_35: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_101, _shape_param_236);  permute_101 = _shape_param_236 = None
        clone_34: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_198: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_34, _shape_param_237);  clone_34 = _shape_param_237 = None
        bmm_16: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_194, view_198);  view_194 = view_198 = None
        view_199: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_16, _shape_param_238);  bmm_16 = _shape_param_238 = None
        add_65: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_199, add_15);  view_199 = add_15 = None
        convert_element_type_55: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_65, torch.float32);  add_65 = None
        amax_8: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_55, [-1], True)
        sub_10: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_55, amax_8);  convert_element_type_55 = amax_8 = None
        exp_8: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_9: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_12: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_56: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None
        expand_36: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_56, _shape_param_239);  convert_element_type_56 = _shape_param_239 = None
        view_200: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_36, _shape_param_240);  expand_36 = _shape_param_240 = None
        view_201: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_71, _shape_param_241);  mul_71 = _shape_param_241 = None
        permute_102: "bf16[512, 384]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        mm_56: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_201, permute_102);  view_201 = permute_102 = None
        view_202: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_56, _shape_param_242);  mm_56 = _shape_param_242 = None
        view_203: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_202, _shape_param_243);  view_202 = _shape_param_243 = None
        permute_103: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
        expand_37: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_103, _shape_param_244);  permute_103 = _shape_param_244 = None
        clone_35: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_204: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_35, _shape_param_245);  clone_35 = _shape_param_245 = None
        bmm_17: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_200, view_204);  view_200 = view_204 = None
        view_205: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_17, _shape_param_246);  bmm_17 = _shape_param_246 = None
        permute_104: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
        clone_36: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_206: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_36, _shape_param_247);  clone_36 = _shape_param_247 = None
        view_207: "bf16[4096, 384]" = torch.ops.aten.view.default(view_206, _shape_param_248);  view_206 = _shape_param_248 = None
        permute_105: "bf16[384, 512]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        mm_57: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_207, permute_105);  view_207 = permute_105 = None
        view_208: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_57, _shape_param_249);  mm_57 = _shape_param_249 = None
        add_66: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_63, view_208);  add_63 = view_208 = None
        convert_element_type_57: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_66, torch.float32)
        pow_25: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_57, 2);  convert_element_type_57 = None
        mean_17: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
        add_67: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
        rsqrt_17: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_72: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_66, rsqrt_17);  rsqrt_17 = None
        convert_element_type_58: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None
        mul_73: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg80_1, convert_element_type_58);  arg80_1 = convert_element_type_58 = None
        view_209: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_73, _shape_param_250);  _shape_param_250 = None
        permute_106: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        mm_58: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_209, permute_106);  view_209 = permute_106 = None
        view_210: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_58, _shape_param_251);  mm_58 = _shape_param_251 = None
        mul_74: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_210, 0.5)
        pow_26: "bf16[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_210, 3.0)
        mul_75: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_26, 0.044715);  pow_26 = None
        add_68: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_210, mul_75);  view_210 = mul_75 = None
        mul_76: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_68, 0.7978845608028654);  add_68 = None
        tanh_7: "bf16[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_76);  mul_76 = None
        add_69: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_77: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_74, add_69);  mul_74 = add_69 = None
        view_211: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_73, _shape_param_252);  mul_73 = _shape_param_252 = None
        permute_107: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
        mm_59: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_211, permute_107);  view_211 = permute_107 = None
        view_212: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_59, _shape_param_253);  mm_59 = _shape_param_253 = None
        mul_78: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_77, view_212);  mul_77 = view_212 = None
        view_213: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_78, _shape_param_254);  mul_78 = _shape_param_254 = None
        permute_108: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        mm_60: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_213, permute_108);  view_213 = permute_108 = None
        view_214: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_60, _shape_param_255);  mm_60 = _shape_param_255 = None
        add_70: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_66, view_214);  add_66 = view_214 = None
        convert_element_type_59: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_70, torch.float32)
        pow_27: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_59, 2);  convert_element_type_59 = None
        mean_18: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
        add_71: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
        rsqrt_18: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_79: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_70, rsqrt_18);  add_70 = rsqrt_18 = None
        convert_element_type_60: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None
        mul_80: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg84_1, convert_element_type_60);  arg84_1 = convert_element_type_60 = None
        view_215: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_80, _shape_param_256);  _shape_param_256 = None
        permute_109: "bf16[512, 384]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_61: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_215, permute_109);  view_215 = permute_109 = None
        view_216: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_61, _shape_param_257);  mm_61 = _shape_param_257 = None
        view_217: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_216, _shape_param_258);  view_216 = _shape_param_258 = None
        permute_110: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_217, [0, 2, 1, 3]);  view_217 = None
        permute_111: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_110, [0, 1, 3, 2]);  permute_110 = None
        expand_38: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_111, _shape_param_259);  permute_111 = _shape_param_259 = None
        clone_37: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_218: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_37, _shape_param_260);  clone_37 = _shape_param_260 = None
        bmm_18: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_22, view_218);  view_22 = view_218 = None
        view_219: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_18, _shape_param_261);  bmm_18 = _shape_param_261 = None
        convert_element_type_61: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_219, torch.float32);  view_219 = None
        amax_9: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_61, [-1], True)
        sub_11: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_61, amax_9);  convert_element_type_61 = amax_9 = None
        exp_9: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_10: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_13: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_62: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None
        expand_39: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_62, _shape_param_262);  convert_element_type_62 = _shape_param_262 = None
        view_220: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_39, _shape_param_263);  expand_39 = _shape_param_263 = None
        view_221: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_80, _shape_param_264);  _shape_param_264 = None
        permute_112: "bf16[512, 384]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        mm_62: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_221, permute_112);  view_221 = permute_112 = None
        view_222: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_62, _shape_param_265);  mm_62 = _shape_param_265 = None
        view_223: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_222, _shape_param_266);  view_222 = _shape_param_266 = None
        permute_113: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_223, [0, 2, 1, 3]);  view_223 = None
        expand_40: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_113, _shape_param_267);  permute_113 = _shape_param_267 = None
        clone_38: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_224: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_38, _shape_param_268);  clone_38 = _shape_param_268 = None
        bmm_19: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_220, view_224);  view_220 = view_224 = None
        view_225: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_269);  bmm_19 = _shape_param_269 = None
        permute_114: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
        clone_39: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        view_226: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_39, _shape_param_270);  clone_39 = _shape_param_270 = None
        view_227: "bf16[4096, 384]" = torch.ops.aten.view.default(view_226, _shape_param_271);  view_226 = _shape_param_271 = None
        permute_115: "bf16[384, 512]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        mm_63: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_227, permute_115);  view_227 = permute_115 = None
        view_228: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_63, _shape_param_272);  mm_63 = _shape_param_272 = None
        add_72: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_8, view_228);  add_8 = view_228 = None
        convert_element_type_63: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_72, torch.float32)
        pow_28: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_63, 2);  convert_element_type_63 = None
        mean_19: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None
        add_73: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
        rsqrt_19: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_81: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_72, rsqrt_19);  rsqrt_19 = None
        convert_element_type_64: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_81, torch.bfloat16);  mul_81 = None
        mul_82: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg88_1, convert_element_type_64);  arg88_1 = convert_element_type_64 = None
        view_229: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_82, _shape_param_273);  _shape_param_273 = None
        permute_116: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        mm_64: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_229, permute_116);  view_229 = permute_116 = None
        view_230: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_64, _shape_param_274);  mm_64 = _shape_param_274 = None
        mul_83: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_230, 0.5)
        pow_29: "bf16[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_230, 3.0)
        mul_84: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
        add_74: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_230, mul_84);  view_230 = mul_84 = None
        mul_85: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_74, 0.7978845608028654);  add_74 = None
        tanh_8: "bf16[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_85);  mul_85 = None
        add_75: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_86: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_83, add_75);  mul_83 = add_75 = None
        view_231: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_82, _shape_param_275);  mul_82 = _shape_param_275 = None
        permute_117: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        mm_65: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_231, permute_117);  view_231 = permute_117 = None
        view_232: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_65, _shape_param_276);  mm_65 = _shape_param_276 = None
        mul_87: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_86, view_232);  mul_86 = view_232 = None
        view_233: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_87, _shape_param_277);  mul_87 = _shape_param_277 = None
        permute_118: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        mm_66: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_233, permute_118);  view_233 = permute_118 = None
        view_234: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_66, _shape_param_278);  mm_66 = _shape_param_278 = None
        add_76: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_72, view_234);  add_72 = view_234 = None
        convert_element_type_65: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_76, torch.float32)
        pow_30: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_65, 2);  convert_element_type_65 = None
        mean_20: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None
        add_77: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
        rsqrt_20: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_88: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_76, rsqrt_20);  rsqrt_20 = None
        convert_element_type_66: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_88, torch.bfloat16);  mul_88 = None
        mul_89: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg92_1, convert_element_type_66);  arg92_1 = convert_element_type_66 = None
        view_235: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_89, _shape_param_279);  _shape_param_279 = None
        permute_119: "bf16[512, 384]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        mm_67: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_235, permute_119);  view_235 = permute_119 = None
        view_236: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_67, _shape_param_280);  mm_67 = _shape_param_280 = None
        view_237: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_236, _shape_param_281);  view_236 = _shape_param_281 = None
        permute_120: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None
        expand_41: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_120, _shape_param_282);  permute_120 = _shape_param_282 = None
        clone_40: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_238: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_283);  clone_40 = _shape_param_283 = None
        view_239: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_89, _shape_param_284);  _shape_param_284 = None
        permute_121: "bf16[512, 384]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        mm_68: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_239, permute_121);  view_239 = permute_121 = None
        view_240: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_68, _shape_param_285);  mm_68 = _shape_param_285 = None
        view_241: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_240, _shape_param_286);  view_240 = _shape_param_286 = None
        permute_122: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_241, [0, 2, 1, 3]);  view_241 = None
        permute_123: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_122, [0, 1, 3, 2]);  permute_122 = None
        expand_42: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_123, _shape_param_287);  permute_123 = _shape_param_287 = None
        clone_41: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_242: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_41, _shape_param_288);  clone_41 = _shape_param_288 = None
        bmm_20: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_238, view_242);  view_238 = view_242 = None
        view_243: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_20, _shape_param_289);  bmm_20 = _shape_param_289 = None
        add_78: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_243, add_6);  view_243 = None
        convert_element_type_67: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_78, torch.float32);  add_78 = None
        amax_10: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_67, [-1], True)
        sub_12: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_67, amax_10);  convert_element_type_67 = amax_10 = None
        exp_10: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_11: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_14: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_68: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None
        expand_43: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_68, _shape_param_290);  convert_element_type_68 = _shape_param_290 = None
        view_244: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_43, _shape_param_291);  expand_43 = _shape_param_291 = None
        view_245: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_89, _shape_param_292);  mul_89 = _shape_param_292 = None
        permute_124: "bf16[512, 384]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        mm_69: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_245, permute_124);  view_245 = permute_124 = None
        view_246: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_69, _shape_param_293);  mm_69 = _shape_param_293 = None
        view_247: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_246, _shape_param_294);  view_246 = _shape_param_294 = None
        permute_125: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
        expand_44: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_125, _shape_param_295);  permute_125 = _shape_param_295 = None
        clone_42: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_248: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_42, _shape_param_296);  clone_42 = _shape_param_296 = None
        bmm_21: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_244, view_248);  view_244 = view_248 = None
        view_249: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_21, _shape_param_297);  bmm_21 = _shape_param_297 = None
        permute_126: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
        clone_43: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_250: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_43, _shape_param_298);  clone_43 = _shape_param_298 = None
        view_251: "bf16[4096, 384]" = torch.ops.aten.view.default(view_250, _shape_param_299);  view_250 = _shape_param_299 = None
        permute_127: "bf16[384, 512]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        mm_70: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_251, permute_127);  view_251 = permute_127 = None
        view_252: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_70, _shape_param_300);  mm_70 = _shape_param_300 = None
        add_79: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_76, view_252);  add_76 = view_252 = None
        convert_element_type_69: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_79, torch.float32)
        pow_31: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_69, 2);  convert_element_type_69 = None
        mean_21: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
        add_80: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
        rsqrt_21: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_90: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_79, rsqrt_21);  rsqrt_21 = None
        convert_element_type_70: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16);  mul_90 = None
        mul_91: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg97_1, convert_element_type_70);  arg97_1 = convert_element_type_70 = None
        view_253: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_91, _shape_param_301);  mul_91 = _shape_param_301 = None
        permute_128: "bf16[512, 384]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        mm_71: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_253, permute_128);  view_253 = permute_128 = None
        view_254: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_71, _shape_param_302);  mm_71 = _shape_param_302 = None
        view_255: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_254, _shape_param_303);  view_254 = _shape_param_303 = None
        permute_129: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
        expand_45: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_129, _shape_param_304);  permute_129 = _shape_param_304 = None
        clone_44: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_256: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_44, _shape_param_305);  clone_44 = _shape_param_305 = None
        view_257: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_80, _shape_param_306);  _shape_param_306 = None
        permute_130: "bf16[512, 384]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        mm_72: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_257, permute_130);  view_257 = permute_130 = None
        view_258: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_72, _shape_param_307);  mm_72 = _shape_param_307 = None
        view_259: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_258, _shape_param_308);  view_258 = _shape_param_308 = None
        permute_131: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
        permute_132: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2]);  permute_131 = None
        expand_46: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_132, _shape_param_309);  permute_132 = _shape_param_309 = None
        clone_45: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_260: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_45, _shape_param_310);  clone_45 = _shape_param_310 = None
        bmm_22: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_256, view_260);  view_256 = view_260 = None
        view_261: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_311);  bmm_22 = _shape_param_311 = None
        convert_element_type_71: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        amax_11: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_71, [-1], True)
        sub_13: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_71, amax_11);  convert_element_type_71 = amax_11 = None
        exp_11: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_12: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_15: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_72: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None
        expand_47: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_72, _shape_param_312);  convert_element_type_72 = _shape_param_312 = None
        view_262: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_47, _shape_param_313);  expand_47 = _shape_param_313 = None
        view_263: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_80, _shape_param_314);  _shape_param_314 = None
        permute_133: "bf16[512, 384]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
        mm_73: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_263, permute_133);  view_263 = permute_133 = None
        view_264: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_73, _shape_param_315);  mm_73 = _shape_param_315 = None
        view_265: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_264, _shape_param_316);  view_264 = _shape_param_316 = None
        permute_134: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
        expand_48: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_134, _shape_param_317);  permute_134 = _shape_param_317 = None
        clone_46: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_266: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_46, _shape_param_318);  clone_46 = _shape_param_318 = None
        bmm_23: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_262, view_266);  view_262 = view_266 = None
        view_267: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_319);  bmm_23 = _shape_param_319 = None
        permute_135: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
        clone_47: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_268: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_47, _shape_param_320);  clone_47 = _shape_param_320 = None
        view_269: "bf16[4096, 384]" = torch.ops.aten.view.default(view_268, _shape_param_321);  view_268 = _shape_param_321 = None
        permute_136: "bf16[384, 512]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        mm_74: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_269, permute_136);  view_269 = permute_136 = None
        view_270: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_74, _shape_param_322);  mm_74 = _shape_param_322 = None
        add_81: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_79, view_270);  add_79 = view_270 = None
        convert_element_type_73: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_81, torch.float32)
        pow_32: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_73, 2);  convert_element_type_73 = None
        mean_22: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None
        add_82: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
        rsqrt_22: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_92: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_81, rsqrt_22);  rsqrt_22 = None
        convert_element_type_74: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_92, torch.bfloat16);  mul_92 = None
        mul_93: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg102_1, convert_element_type_74);  arg102_1 = convert_element_type_74 = None
        view_271: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_93, _shape_param_323);  _shape_param_323 = None
        permute_137: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        mm_75: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_271, permute_137);  view_271 = permute_137 = None
        view_272: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_75, _shape_param_324);  mm_75 = _shape_param_324 = None
        mul_94: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_272, 0.5)
        pow_33: "bf16[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_272, 3.0)
        mul_95: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
        add_83: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_272, mul_95);  view_272 = mul_95 = None
        mul_96: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_83, 0.7978845608028654);  add_83 = None
        tanh_9: "bf16[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_96);  mul_96 = None
        add_84: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_97: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_94, add_84);  mul_94 = add_84 = None
        view_273: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_93, _shape_param_325);  mul_93 = _shape_param_325 = None
        permute_138: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
        mm_76: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_273, permute_138);  view_273 = permute_138 = None
        view_274: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_76, _shape_param_326);  mm_76 = _shape_param_326 = None
        mul_98: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_97, view_274);  mul_97 = view_274 = None
        view_275: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_98, _shape_param_327);  mul_98 = _shape_param_327 = None
        permute_139: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        mm_77: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_275, permute_139);  view_275 = permute_139 = None
        view_276: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_77, _shape_param_328);  mm_77 = _shape_param_328 = None
        add_85: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_81, view_276);  add_81 = view_276 = None
        convert_element_type_75: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_85, torch.float32)
        pow_34: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_75, 2);  convert_element_type_75 = None
        mean_23: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None
        add_86: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
        rsqrt_23: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_99: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_85, rsqrt_23);  rsqrt_23 = None
        convert_element_type_76: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_99, torch.bfloat16);  mul_99 = None
        mul_100: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg106_1, convert_element_type_76);  arg106_1 = convert_element_type_76 = None
        view_277: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_100, _shape_param_329);  _shape_param_329 = None
        permute_140: "bf16[512, 384]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        mm_78: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_277, permute_140);  view_277 = permute_140 = None
        view_278: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_78, _shape_param_330);  mm_78 = _shape_param_330 = None
        view_279: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_278, _shape_param_331);  view_278 = _shape_param_331 = None
        permute_141: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None
        expand_49: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_141, _shape_param_332);  permute_141 = _shape_param_332 = None
        clone_48: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_280: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_48, _shape_param_333);  clone_48 = _shape_param_333 = None
        view_281: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_100, _shape_param_334);  _shape_param_334 = None
        permute_142: "bf16[512, 384]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        mm_79: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_281, permute_142);  view_281 = permute_142 = None
        view_282: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_79, _shape_param_335);  mm_79 = _shape_param_335 = None
        view_283: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_282, _shape_param_336);  view_282 = _shape_param_336 = None
        permute_143: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
        permute_144: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_143, [0, 1, 3, 2]);  permute_143 = None
        expand_50: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_144, _shape_param_337);  permute_144 = _shape_param_337 = None
        clone_49: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_284: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_49, _shape_param_338);  clone_49 = _shape_param_338 = None
        bmm_24: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_280, view_284);  view_280 = view_284 = None
        view_285: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_24, _shape_param_339);  bmm_24 = _shape_param_339 = None
        add_87: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_285, add_6);  view_285 = None
        convert_element_type_77: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_87, torch.float32);  add_87 = None
        amax_12: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_77, [-1], True)
        sub_14: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_77, amax_12);  convert_element_type_77 = amax_12 = None
        exp_12: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_13: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_16: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_78: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None
        expand_51: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_78, _shape_param_340);  convert_element_type_78 = _shape_param_340 = None
        view_286: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_51, _shape_param_341);  expand_51 = _shape_param_341 = None
        view_287: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_100, _shape_param_342);  mul_100 = _shape_param_342 = None
        permute_145: "bf16[512, 384]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        mm_80: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_287, permute_145);  view_287 = permute_145 = None
        view_288: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_80, _shape_param_343);  mm_80 = _shape_param_343 = None
        view_289: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_288, _shape_param_344);  view_288 = _shape_param_344 = None
        permute_146: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None
        expand_52: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_146, _shape_param_345);  permute_146 = _shape_param_345 = None
        clone_50: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_290: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_50, _shape_param_346);  clone_50 = _shape_param_346 = None
        bmm_25: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_286, view_290);  view_286 = view_290 = None
        view_291: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_25, _shape_param_347);  bmm_25 = _shape_param_347 = None
        permute_147: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        clone_51: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
        view_292: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_51, _shape_param_348);  clone_51 = _shape_param_348 = None
        view_293: "bf16[4096, 384]" = torch.ops.aten.view.default(view_292, _shape_param_349);  view_292 = _shape_param_349 = None
        permute_148: "bf16[384, 512]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        mm_81: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_293, permute_148);  view_293 = permute_148 = None
        view_294: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_81, _shape_param_350);  mm_81 = _shape_param_350 = None
        add_88: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_85, view_294);  add_85 = view_294 = None
        convert_element_type_79: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_88, torch.float32)
        pow_35: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_79, 2);  convert_element_type_79 = None
        mean_24: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None
        add_89: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
        rsqrt_24: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_101: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_88, rsqrt_24);  rsqrt_24 = None
        convert_element_type_80: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_101, torch.bfloat16);  mul_101 = None
        mul_102: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg111_1, convert_element_type_80);  arg111_1 = convert_element_type_80 = None
        view_295: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_102, _shape_param_351);  mul_102 = _shape_param_351 = None
        permute_149: "bf16[512, 384]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        mm_82: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_295, permute_149);  view_295 = permute_149 = None
        view_296: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_82, _shape_param_352);  mm_82 = _shape_param_352 = None
        view_297: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_296, _shape_param_353);  view_296 = _shape_param_353 = None
        permute_150: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_297, [0, 2, 1, 3]);  view_297 = None
        expand_53: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_150, _shape_param_354);  permute_150 = _shape_param_354 = None
        clone_52: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_298: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_52, _shape_param_355);  clone_52 = _shape_param_355 = None
        view_299: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_80, _shape_param_356);  _shape_param_356 = None
        permute_151: "bf16[512, 384]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        mm_83: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_299, permute_151);  view_299 = permute_151 = None
        view_300: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_83, _shape_param_357);  mm_83 = _shape_param_357 = None
        view_301: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_300, _shape_param_358);  view_300 = _shape_param_358 = None
        permute_152: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_301, [0, 2, 1, 3]);  view_301 = None
        permute_153: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_152, [0, 1, 3, 2]);  permute_152 = None
        expand_54: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_153, _shape_param_359);  permute_153 = _shape_param_359 = None
        clone_53: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_302: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_53, _shape_param_360);  clone_53 = _shape_param_360 = None
        bmm_26: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_298, view_302);  view_298 = view_302 = None
        view_303: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_26, _shape_param_361);  bmm_26 = _shape_param_361 = None
        convert_element_type_81: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_303, torch.float32);  view_303 = None
        amax_13: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_81, [-1], True)
        sub_15: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_81, amax_13);  convert_element_type_81 = amax_13 = None
        exp_13: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_14: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_17: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_82: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None
        expand_55: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_82, _shape_param_362);  convert_element_type_82 = _shape_param_362 = None
        view_304: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_55, _shape_param_363);  expand_55 = _shape_param_363 = None
        view_305: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_80, _shape_param_364);  _shape_param_364 = None
        permute_154: "bf16[512, 384]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
        mm_84: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_305, permute_154);  view_305 = permute_154 = None
        view_306: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_84, _shape_param_365);  mm_84 = _shape_param_365 = None
        view_307: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_306, _shape_param_366);  view_306 = _shape_param_366 = None
        permute_155: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
        expand_56: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_155, _shape_param_367);  permute_155 = _shape_param_367 = None
        clone_54: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_308: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_54, _shape_param_368);  clone_54 = _shape_param_368 = None
        bmm_27: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_304, view_308);  view_304 = view_308 = None
        view_309: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_27, _shape_param_369);  bmm_27 = _shape_param_369 = None
        permute_156: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_309, [0, 2, 1, 3]);  view_309 = None
        clone_55: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
        view_310: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_55, _shape_param_370);  clone_55 = _shape_param_370 = None
        view_311: "bf16[4096, 384]" = torch.ops.aten.view.default(view_310, _shape_param_371);  view_310 = _shape_param_371 = None
        permute_157: "bf16[384, 512]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        mm_85: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_311, permute_157);  view_311 = permute_157 = None
        view_312: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_85, _shape_param_372);  mm_85 = _shape_param_372 = None
        add_90: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_88, view_312);  add_88 = view_312 = None
        convert_element_type_83: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_90, torch.float32)
        pow_36: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_83, 2);  convert_element_type_83 = None
        mean_25: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None
        add_91: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
        rsqrt_25: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        mul_103: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_90, rsqrt_25);  rsqrt_25 = None
        convert_element_type_84: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_103, torch.bfloat16);  mul_103 = None
        mul_104: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg116_1, convert_element_type_84);  arg116_1 = convert_element_type_84 = None
        view_313: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_104, _shape_param_373);  _shape_param_373 = None
        permute_158: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_86: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_313, permute_158);  view_313 = permute_158 = None
        view_314: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_86, _shape_param_374);  mm_86 = _shape_param_374 = None
        mul_105: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_314, 0.5)
        pow_37: "bf16[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_314, 3.0)
        mul_106: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_37, 0.044715);  pow_37 = None
        add_92: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_314, mul_106);  view_314 = mul_106 = None
        mul_107: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_92, 0.7978845608028654);  add_92 = None
        tanh_10: "bf16[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_107);  mul_107 = None
        add_93: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_108: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_105, add_93);  mul_105 = add_93 = None
        view_315: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_104, _shape_param_375);  mul_104 = _shape_param_375 = None
        permute_159: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        mm_87: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_315, permute_159);  view_315 = permute_159 = None
        view_316: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_87, _shape_param_376);  mm_87 = _shape_param_376 = None
        mul_109: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_108, view_316);  mul_108 = view_316 = None
        view_317: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_109, _shape_param_377);  mul_109 = _shape_param_377 = None
        permute_160: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        mm_88: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_317, permute_160);  view_317 = permute_160 = None
        view_318: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_88, _shape_param_378);  mm_88 = _shape_param_378 = None
        add_94: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_90, view_318);  add_90 = view_318 = None
        convert_element_type_85: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_94, torch.float32)
        pow_38: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_85, 2);  convert_element_type_85 = None
        mean_26: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None
        add_95: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
        rsqrt_26: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        mul_110: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_94, rsqrt_26);  rsqrt_26 = None
        convert_element_type_86: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_110, torch.bfloat16);  mul_110 = None
        mul_111: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg120_1, convert_element_type_86);  arg120_1 = convert_element_type_86 = None
        view_319: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_111, _shape_param_379);  _shape_param_379 = None
        permute_161: "bf16[512, 384]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        mm_89: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_319, permute_161);  view_319 = permute_161 = None
        view_320: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_89, _shape_param_380);  mm_89 = _shape_param_380 = None
        view_321: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_320, _shape_param_381);  view_320 = _shape_param_381 = None
        permute_162: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_321, [0, 2, 1, 3]);  view_321 = None
        expand_57: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_162, _shape_param_382);  permute_162 = _shape_param_382 = None
        clone_56: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_322: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_56, _shape_param_383);  clone_56 = _shape_param_383 = None
        view_323: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_111, _shape_param_384);  _shape_param_384 = None
        permute_163: "bf16[512, 384]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        mm_90: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_323, permute_163);  view_323 = permute_163 = None
        view_324: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_90, _shape_param_385);  mm_90 = _shape_param_385 = None
        view_325: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_324, _shape_param_386);  view_324 = _shape_param_386 = None
        permute_164: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_325, [0, 2, 1, 3]);  view_325 = None
        permute_165: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_164, [0, 1, 3, 2]);  permute_164 = None
        expand_58: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_165, _shape_param_387);  permute_165 = _shape_param_387 = None
        clone_57: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_326: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_57, _shape_param_388);  clone_57 = _shape_param_388 = None
        bmm_28: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_322, view_326);  view_322 = view_326 = None
        view_327: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_28, _shape_param_389);  bmm_28 = _shape_param_389 = None
        add_96: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_327, add_6);  view_327 = None
        convert_element_type_87: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_96, torch.float32);  add_96 = None
        amax_14: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_87, [-1], True)
        sub_16: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_87, amax_14);  convert_element_type_87 = amax_14 = None
        exp_14: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_15: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_18: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_88: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None
        expand_59: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_88, _shape_param_390);  convert_element_type_88 = _shape_param_390 = None
        view_328: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_59, _shape_param_391);  expand_59 = _shape_param_391 = None
        view_329: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_111, _shape_param_392);  mul_111 = _shape_param_392 = None
        permute_166: "bf16[512, 384]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        mm_91: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_329, permute_166);  view_329 = permute_166 = None
        view_330: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_91, _shape_param_393);  mm_91 = _shape_param_393 = None
        view_331: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_330, _shape_param_394);  view_330 = _shape_param_394 = None
        permute_167: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None
        expand_60: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_167, _shape_param_395);  permute_167 = _shape_param_395 = None
        clone_58: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_332: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_58, _shape_param_396);  clone_58 = _shape_param_396 = None
        bmm_29: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_328, view_332);  view_328 = view_332 = None
        view_333: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_29, _shape_param_397);  bmm_29 = _shape_param_397 = None
        permute_168: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_333, [0, 2, 1, 3]);  view_333 = None
        clone_59: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
        view_334: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_59, _shape_param_398);  clone_59 = _shape_param_398 = None
        view_335: "bf16[4096, 384]" = torch.ops.aten.view.default(view_334, _shape_param_399);  view_334 = _shape_param_399 = None
        permute_169: "bf16[384, 512]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
        mm_92: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_335, permute_169);  view_335 = permute_169 = None
        view_336: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_92, _shape_param_400);  mm_92 = _shape_param_400 = None
        add_97: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_94, view_336);  add_94 = view_336 = None
        convert_element_type_89: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_97, torch.float32)
        pow_39: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_89, 2);  convert_element_type_89 = None
        mean_27: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None
        add_98: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
        rsqrt_27: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_112: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_97, rsqrt_27);  rsqrt_27 = None
        convert_element_type_90: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None
        mul_113: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg125_1, convert_element_type_90);  arg125_1 = convert_element_type_90 = None
        view_337: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_113, _shape_param_401);  mul_113 = _shape_param_401 = None
        permute_170: "bf16[512, 384]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
        mm_93: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_337, permute_170);  view_337 = permute_170 = None
        view_338: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_93, _shape_param_402);  mm_93 = _shape_param_402 = None
        view_339: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_338, _shape_param_403);  view_338 = _shape_param_403 = None
        permute_171: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
        expand_61: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_171, _shape_param_404);  permute_171 = _shape_param_404 = None
        clone_60: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_340: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_60, _shape_param_405);  clone_60 = _shape_param_405 = None
        view_341: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_80, _shape_param_406);  _shape_param_406 = None
        permute_172: "bf16[512, 384]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        mm_94: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_341, permute_172);  view_341 = permute_172 = None
        view_342: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_94, _shape_param_407);  mm_94 = _shape_param_407 = None
        view_343: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_342, _shape_param_408);  view_342 = _shape_param_408 = None
        permute_173: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None
        permute_174: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_173, [0, 1, 3, 2]);  permute_173 = None
        expand_62: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_174, _shape_param_409);  permute_174 = _shape_param_409 = None
        clone_61: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_344: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_61, _shape_param_410);  clone_61 = _shape_param_410 = None
        bmm_30: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_340, view_344);  view_340 = view_344 = None
        view_345: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_30, _shape_param_411);  bmm_30 = _shape_param_411 = None
        convert_element_type_91: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_345, torch.float32);  view_345 = None
        amax_15: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_91, [-1], True)
        sub_17: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_91, amax_15);  convert_element_type_91 = amax_15 = None
        exp_15: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_16: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_19: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_92: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None
        expand_63: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_92, _shape_param_412);  convert_element_type_92 = _shape_param_412 = None
        view_346: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_63, _shape_param_413);  expand_63 = _shape_param_413 = None
        view_347: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_80, _shape_param_414);  _shape_param_414 = None
        permute_175: "bf16[512, 384]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        mm_95: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_347, permute_175);  view_347 = permute_175 = None
        view_348: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_95, _shape_param_415);  mm_95 = _shape_param_415 = None
        view_349: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_348, _shape_param_416);  view_348 = _shape_param_416 = None
        permute_176: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_349, [0, 2, 1, 3]);  view_349 = None
        expand_64: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_176, _shape_param_417);  permute_176 = _shape_param_417 = None
        clone_62: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_350: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_62, _shape_param_418);  clone_62 = _shape_param_418 = None
        bmm_31: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_346, view_350);  view_346 = view_350 = None
        view_351: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_31, _shape_param_419);  bmm_31 = _shape_param_419 = None
        permute_177: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None
        clone_63: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_352: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_63, _shape_param_420);  clone_63 = _shape_param_420 = None
        view_353: "bf16[4096, 384]" = torch.ops.aten.view.default(view_352, _shape_param_421);  view_352 = _shape_param_421 = None
        permute_178: "bf16[384, 512]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        mm_96: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_353, permute_178);  view_353 = permute_178 = None
        view_354: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_96, _shape_param_422);  mm_96 = _shape_param_422 = None
        add_99: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_97, view_354);  add_97 = view_354 = None
        convert_element_type_93: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_99, torch.float32)
        pow_40: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_93, 2);  convert_element_type_93 = None
        mean_28: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None
        add_100: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
        rsqrt_28: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_114: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_99, rsqrt_28);  rsqrt_28 = None
        convert_element_type_94: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_114, torch.bfloat16);  mul_114 = None
        mul_115: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg130_1, convert_element_type_94);  arg130_1 = convert_element_type_94 = None
        view_355: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_115, _shape_param_423);  _shape_param_423 = None
        permute_179: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        mm_97: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_355, permute_179);  view_355 = permute_179 = None
        view_356: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_97, _shape_param_424);  mm_97 = _shape_param_424 = None
        mul_116: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_356, 0.5)
        pow_41: "bf16[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_356, 3.0)
        mul_117: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_41, 0.044715);  pow_41 = None
        add_101: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_356, mul_117);  view_356 = mul_117 = None
        mul_118: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_101, 0.7978845608028654);  add_101 = None
        tanh_11: "bf16[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_102: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_119: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_116, add_102);  mul_116 = add_102 = None
        view_357: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_115, _shape_param_425);  mul_115 = _shape_param_425 = None
        permute_180: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        mm_98: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_357, permute_180);  view_357 = permute_180 = None
        view_358: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_98, _shape_param_426);  mm_98 = _shape_param_426 = None
        mul_120: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_119, view_358);  mul_119 = view_358 = None
        view_359: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_120, _shape_param_427);  mul_120 = _shape_param_427 = None
        permute_181: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        mm_99: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_359, permute_181);  view_359 = permute_181 = None
        view_360: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_99, _shape_param_428);  mm_99 = _shape_param_428 = None
        add_103: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_99, view_360);  add_99 = view_360 = None
        convert_element_type_95: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_103, torch.float32)
        pow_42: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_95, 2);  convert_element_type_95 = None
        mean_29: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None
        add_104: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
        rsqrt_29: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_121: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_103, rsqrt_29);  rsqrt_29 = None
        convert_element_type_96: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_121, torch.bfloat16);  mul_121 = None
        mul_122: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg134_1, convert_element_type_96);  arg134_1 = convert_element_type_96 = None
        view_361: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_122, _shape_param_429);  _shape_param_429 = None
        permute_182: "bf16[512, 384]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        mm_100: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_361, permute_182);  view_361 = permute_182 = None
        view_362: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_100, _shape_param_430);  mm_100 = _shape_param_430 = None
        view_363: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_362, _shape_param_431);  view_362 = _shape_param_431 = None
        permute_183: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_363, [0, 2, 1, 3]);  view_363 = None
        expand_65: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_183, _shape_param_432);  permute_183 = _shape_param_432 = None
        clone_64: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_364: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_64, _shape_param_433);  clone_64 = _shape_param_433 = None
        view_365: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_122, _shape_param_434);  _shape_param_434 = None
        permute_184: "bf16[512, 384]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        mm_101: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_365, permute_184);  view_365 = permute_184 = None
        view_366: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_101, _shape_param_435);  mm_101 = _shape_param_435 = None
        view_367: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_366, _shape_param_436);  view_366 = _shape_param_436 = None
        permute_185: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None
        permute_186: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_185, [0, 1, 3, 2]);  permute_185 = None
        expand_66: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_186, _shape_param_437);  permute_186 = _shape_param_437 = None
        clone_65: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_368: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_65, _shape_param_438);  clone_65 = _shape_param_438 = None
        bmm_32: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_364, view_368);  view_364 = view_368 = None
        view_369: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_32, _shape_param_439);  bmm_32 = _shape_param_439 = None
        add_105: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_369, add_6);  view_369 = None
        convert_element_type_97: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_105, torch.float32);  add_105 = None
        amax_16: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_97, [-1], True)
        sub_18: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_97, amax_16);  convert_element_type_97 = amax_16 = None
        exp_16: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_17: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_20: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_98: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None
        expand_67: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_98, _shape_param_440);  convert_element_type_98 = _shape_param_440 = None
        view_370: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_67, _shape_param_441);  expand_67 = _shape_param_441 = None
        view_371: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_122, _shape_param_442);  mul_122 = _shape_param_442 = None
        permute_187: "bf16[512, 384]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        mm_102: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_371, permute_187);  view_371 = permute_187 = None
        view_372: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_102, _shape_param_443);  mm_102 = _shape_param_443 = None
        view_373: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_372, _shape_param_444);  view_372 = _shape_param_444 = None
        permute_188: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
        expand_68: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_188, _shape_param_445);  permute_188 = _shape_param_445 = None
        clone_66: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_374: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_66, _shape_param_446);  clone_66 = _shape_param_446 = None
        bmm_33: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_370, view_374);  view_370 = view_374 = None
        view_375: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_33, _shape_param_447);  bmm_33 = _shape_param_447 = None
        permute_189: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None
        clone_67: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_376: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_67, _shape_param_448);  clone_67 = _shape_param_448 = None
        view_377: "bf16[4096, 384]" = torch.ops.aten.view.default(view_376, _shape_param_449);  view_376 = _shape_param_449 = None
        permute_190: "bf16[384, 512]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
        mm_103: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_377, permute_190);  view_377 = permute_190 = None
        view_378: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_103, _shape_param_450);  mm_103 = _shape_param_450 = None
        add_106: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_103, view_378);  add_103 = view_378 = None
        convert_element_type_99: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_106, torch.float32)
        pow_43: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_99, 2);  convert_element_type_99 = None
        mean_30: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None
        add_107: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
        rsqrt_30: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        mul_123: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_106, rsqrt_30);  rsqrt_30 = None
        convert_element_type_100: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_123, torch.bfloat16);  mul_123 = None
        mul_124: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg139_1, convert_element_type_100);  arg139_1 = convert_element_type_100 = None
        view_379: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_124, _shape_param_451);  mul_124 = _shape_param_451 = None
        permute_191: "bf16[512, 384]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        mm_104: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_379, permute_191);  view_379 = permute_191 = None
        view_380: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_104, _shape_param_452);  mm_104 = _shape_param_452 = None
        view_381: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_380, _shape_param_453);  view_380 = _shape_param_453 = None
        permute_192: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_381, [0, 2, 1, 3]);  view_381 = None
        expand_69: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_192, _shape_param_454);  permute_192 = _shape_param_454 = None
        clone_68: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_382: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_68, _shape_param_455);  clone_68 = _shape_param_455 = None
        view_383: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_80, _shape_param_456);  _shape_param_456 = None
        permute_193: "bf16[512, 384]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        mm_105: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_383, permute_193);  view_383 = permute_193 = None
        view_384: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_105, _shape_param_457);  mm_105 = _shape_param_457 = None
        view_385: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_384, _shape_param_458);  view_384 = _shape_param_458 = None
        permute_194: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_385, [0, 2, 1, 3]);  view_385 = None
        permute_195: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_194, [0, 1, 3, 2]);  permute_194 = None
        expand_70: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_195, _shape_param_459);  permute_195 = _shape_param_459 = None
        clone_69: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_386: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_69, _shape_param_460);  clone_69 = _shape_param_460 = None
        bmm_34: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_382, view_386);  view_382 = view_386 = None
        view_387: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_34, _shape_param_461);  bmm_34 = _shape_param_461 = None
        convert_element_type_101: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_387, torch.float32);  view_387 = None
        amax_17: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_101, [-1], True)
        sub_19: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_101, amax_17);  convert_element_type_101 = amax_17 = None
        exp_17: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_18: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_21: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_102: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None
        expand_71: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_102, _shape_param_462);  convert_element_type_102 = _shape_param_462 = None
        view_388: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_71, _shape_param_463);  expand_71 = _shape_param_463 = None
        view_389: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_80, _shape_param_464);  _shape_param_464 = None
        permute_196: "bf16[512, 384]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        mm_106: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_389, permute_196);  view_389 = permute_196 = None
        view_390: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_106, _shape_param_465);  mm_106 = _shape_param_465 = None
        view_391: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_390, _shape_param_466);  view_390 = _shape_param_466 = None
        permute_197: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_391, [0, 2, 1, 3]);  view_391 = None
        expand_72: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_197, _shape_param_467);  permute_197 = _shape_param_467 = None
        clone_70: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_392: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_70, _shape_param_468);  clone_70 = _shape_param_468 = None
        bmm_35: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_388, view_392);  view_388 = view_392 = None
        view_393: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_35, _shape_param_469);  bmm_35 = _shape_param_469 = None
        permute_198: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
        clone_71: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
        view_394: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_71, _shape_param_470);  clone_71 = _shape_param_470 = None
        view_395: "bf16[4096, 384]" = torch.ops.aten.view.default(view_394, _shape_param_471);  view_394 = _shape_param_471 = None
        permute_199: "bf16[384, 512]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        mm_107: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_395, permute_199);  view_395 = permute_199 = None
        view_396: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_107, _shape_param_472);  mm_107 = _shape_param_472 = None
        add_108: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_106, view_396);  add_106 = view_396 = None
        convert_element_type_103: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_108, torch.float32)
        pow_44: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_103, 2);  convert_element_type_103 = None
        mean_31: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None
        add_109: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
        rsqrt_31: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        mul_125: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_108, rsqrt_31);  rsqrt_31 = None
        convert_element_type_104: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_125, torch.bfloat16);  mul_125 = None
        mul_126: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg144_1, convert_element_type_104);  arg144_1 = convert_element_type_104 = None
        view_397: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_126, _shape_param_473);  _shape_param_473 = None
        permute_200: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        mm_108: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_397, permute_200);  view_397 = permute_200 = None
        view_398: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_108, _shape_param_474);  mm_108 = _shape_param_474 = None
        mul_127: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_398, 0.5)
        pow_45: "bf16[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_398, 3.0)
        mul_128: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_45, 0.044715);  pow_45 = None
        add_110: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_398, mul_128);  view_398 = mul_128 = None
        mul_129: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_110, 0.7978845608028654);  add_110 = None
        tanh_12: "bf16[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_129);  mul_129 = None
        add_111: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_130: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_127, add_111);  mul_127 = add_111 = None
        view_399: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_126, _shape_param_475);  mul_126 = _shape_param_475 = None
        permute_201: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        mm_109: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_399, permute_201);  view_399 = permute_201 = None
        view_400: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_109, _shape_param_476);  mm_109 = _shape_param_476 = None
        mul_131: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_130, view_400);  mul_130 = view_400 = None
        view_401: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_131, _shape_param_477);  mul_131 = _shape_param_477 = None
        permute_202: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        mm_110: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_401, permute_202);  view_401 = permute_202 = None
        view_402: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_110, _shape_param_478);  mm_110 = _shape_param_478 = None
        add_112: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_108, view_402);  add_108 = view_402 = None
        convert_element_type_105: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_112, torch.float32)
        pow_46: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_105, 2);  convert_element_type_105 = None
        mean_32: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None
        add_113: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None
        rsqrt_32: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        mul_132: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_112, rsqrt_32);  rsqrt_32 = None
        convert_element_type_106: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_132, torch.bfloat16);  mul_132 = None
        mul_133: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg148_1, convert_element_type_106);  arg148_1 = convert_element_type_106 = None
        view_403: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_133, _shape_param_479);  _shape_param_479 = None
        permute_203: "bf16[512, 384]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        mm_111: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_403, permute_203);  view_403 = permute_203 = None
        view_404: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_111, _shape_param_480);  mm_111 = _shape_param_480 = None
        view_405: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_404, _shape_param_481);  view_404 = _shape_param_481 = None
        permute_204: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_405, [0, 2, 1, 3]);  view_405 = None
        expand_73: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_204, _shape_param_482);  permute_204 = _shape_param_482 = None
        clone_72: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_406: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_72, _shape_param_483);  clone_72 = _shape_param_483 = None
        view_407: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_133, _shape_param_484);  _shape_param_484 = None
        permute_205: "bf16[512, 384]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        mm_112: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_407, permute_205);  view_407 = permute_205 = None
        view_408: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_112, _shape_param_485);  mm_112 = _shape_param_485 = None
        view_409: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_408, _shape_param_486);  view_408 = _shape_param_486 = None
        permute_206: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None
        permute_207: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_206, [0, 1, 3, 2]);  permute_206 = None
        expand_74: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_207, _shape_param_487);  permute_207 = _shape_param_487 = None
        clone_73: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_410: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_73, _shape_param_488);  clone_73 = _shape_param_488 = None
        bmm_36: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_406, view_410);  view_406 = view_410 = None
        view_411: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_36, _shape_param_489);  bmm_36 = _shape_param_489 = None
        add_114: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_411, add_6);  view_411 = None
        convert_element_type_107: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_114, torch.float32);  add_114 = None
        amax_18: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_107, [-1], True)
        sub_20: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_107, amax_18);  convert_element_type_107 = amax_18 = None
        exp_18: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_19: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_22: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_108: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None
        expand_75: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_108, _shape_param_490);  convert_element_type_108 = _shape_param_490 = None
        view_412: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_75, _shape_param_491);  expand_75 = _shape_param_491 = None
        view_413: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_133, _shape_param_492);  mul_133 = _shape_param_492 = None
        permute_208: "bf16[512, 384]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        mm_113: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_413, permute_208);  view_413 = permute_208 = None
        view_414: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_113, _shape_param_493);  mm_113 = _shape_param_493 = None
        view_415: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_414, _shape_param_494);  view_414 = _shape_param_494 = None
        permute_209: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_415, [0, 2, 1, 3]);  view_415 = None
        expand_76: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_209, _shape_param_495);  permute_209 = _shape_param_495 = None
        clone_74: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_416: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_74, _shape_param_496);  clone_74 = _shape_param_496 = None
        bmm_37: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_412, view_416);  view_412 = view_416 = None
        view_417: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_37, _shape_param_497);  bmm_37 = _shape_param_497 = None
        permute_210: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_417, [0, 2, 1, 3]);  view_417 = None
        clone_75: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_418: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_75, _shape_param_498);  clone_75 = _shape_param_498 = None
        view_419: "bf16[4096, 384]" = torch.ops.aten.view.default(view_418, _shape_param_499);  view_418 = _shape_param_499 = None
        permute_211: "bf16[384, 512]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        mm_114: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_419, permute_211);  view_419 = permute_211 = None
        view_420: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_114, _shape_param_500);  mm_114 = _shape_param_500 = None
        add_115: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_112, view_420);  add_112 = view_420 = None
        convert_element_type_109: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_115, torch.float32)
        pow_47: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_109, 2);  convert_element_type_109 = None
        mean_33: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None
        add_116: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None
        rsqrt_33: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_134: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_115, rsqrt_33);  rsqrt_33 = None
        convert_element_type_110: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_134, torch.bfloat16);  mul_134 = None
        mul_135: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg153_1, convert_element_type_110);  arg153_1 = convert_element_type_110 = None
        view_421: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_135, _shape_param_501);  mul_135 = _shape_param_501 = None
        permute_212: "bf16[512, 384]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        mm_115: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_421, permute_212);  view_421 = permute_212 = None
        view_422: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_115, _shape_param_502);  mm_115 = _shape_param_502 = None
        view_423: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_422, _shape_param_503);  view_422 = _shape_param_503 = None
        permute_213: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_423, [0, 2, 1, 3]);  view_423 = None
        expand_77: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_213, _shape_param_504);  permute_213 = _shape_param_504 = None
        clone_76: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_424: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_76, _shape_param_505);  clone_76 = _shape_param_505 = None
        view_425: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_80, _shape_param_506);  _shape_param_506 = None
        permute_214: "bf16[512, 384]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        mm_116: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_425, permute_214);  view_425 = permute_214 = None
        view_426: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_116, _shape_param_507);  mm_116 = _shape_param_507 = None
        view_427: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_426, _shape_param_508);  view_426 = _shape_param_508 = None
        permute_215: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
        permute_216: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_215, [0, 1, 3, 2]);  permute_215 = None
        expand_78: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_216, _shape_param_509);  permute_216 = _shape_param_509 = None
        clone_77: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
        view_428: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_77, _shape_param_510);  clone_77 = _shape_param_510 = None
        bmm_38: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_424, view_428);  view_424 = view_428 = None
        view_429: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_38, _shape_param_511);  bmm_38 = _shape_param_511 = None
        convert_element_type_111: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_429, torch.float32);  view_429 = None
        amax_19: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_111, [-1], True)
        sub_21: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_111, amax_19);  convert_element_type_111 = amax_19 = None
        exp_19: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_20: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_23: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        convert_element_type_112: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None
        expand_79: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_112, _shape_param_512);  convert_element_type_112 = _shape_param_512 = None
        view_430: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_79, _shape_param_513);  expand_79 = _shape_param_513 = None
        view_431: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_80, _shape_param_514);  _shape_param_514 = None
        permute_217: "bf16[512, 384]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        mm_117: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_431, permute_217);  view_431 = permute_217 = None
        view_432: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_117, _shape_param_515);  mm_117 = _shape_param_515 = None
        view_433: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_432, _shape_param_516);  view_432 = _shape_param_516 = None
        permute_218: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_433, [0, 2, 1, 3]);  view_433 = None
        expand_80: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_218, _shape_param_517);  permute_218 = _shape_param_517 = None
        clone_78: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_434: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_78, _shape_param_518);  clone_78 = _shape_param_518 = None
        bmm_39: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_430, view_434);  view_430 = view_434 = None
        view_435: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_39, _shape_param_519);  bmm_39 = _shape_param_519 = None
        permute_219: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_435, [0, 2, 1, 3]);  view_435 = None
        clone_79: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
        view_436: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_79, _shape_param_520);  clone_79 = _shape_param_520 = None
        view_437: "bf16[4096, 384]" = torch.ops.aten.view.default(view_436, _shape_param_521);  view_436 = _shape_param_521 = None
        permute_220: "bf16[384, 512]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        mm_118: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_437, permute_220);  view_437 = permute_220 = None
        view_438: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_118, _shape_param_522);  mm_118 = _shape_param_522 = None
        add_117: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_115, view_438);  add_115 = view_438 = None
        convert_element_type_113: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_117, torch.float32)
        pow_48: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_113, 2);  convert_element_type_113 = None
        mean_34: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None
        add_118: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None
        rsqrt_34: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_136: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_117, rsqrt_34);  rsqrt_34 = None
        convert_element_type_114: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_136, torch.bfloat16);  mul_136 = None
        mul_137: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg158_1, convert_element_type_114);  arg158_1 = convert_element_type_114 = None
        view_439: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_137, _shape_param_523);  _shape_param_523 = None
        permute_221: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        mm_119: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_439, permute_221);  view_439 = permute_221 = None
        view_440: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_119, _shape_param_524);  mm_119 = _shape_param_524 = None
        mul_138: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_440, 0.5)
        pow_49: "bf16[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_440, 3.0)
        mul_139: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_49, 0.044715);  pow_49 = None
        add_119: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_440, mul_139);  view_440 = mul_139 = None
        mul_140: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_119, 0.7978845608028654);  add_119 = None
        tanh_13: "bf16[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_140);  mul_140 = None
        add_120: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_141: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_138, add_120);  mul_138 = add_120 = None
        view_441: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_137, _shape_param_525);  mul_137 = _shape_param_525 = None
        permute_222: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        mm_120: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_441, permute_222);  view_441 = permute_222 = None
        view_442: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_120, _shape_param_526);  mm_120 = _shape_param_526 = None
        mul_142: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_141, view_442);  mul_141 = view_442 = None
        view_443: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_142, _shape_param_527);  mul_142 = _shape_param_527 = None
        permute_223: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        mm_121: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_443, permute_223);  view_443 = permute_223 = None
        view_444: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_121, _shape_param_528);  mm_121 = _shape_param_528 = None
        add_121: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_117, view_444);  add_117 = view_444 = None
        convert_element_type_115: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_121, torch.float32)
        pow_50: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_115, 2);  convert_element_type_115 = None
        mean_35: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None
        add_122: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None
        rsqrt_35: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        mul_143: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_121, rsqrt_35);  rsqrt_35 = None
        convert_element_type_116: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_143, torch.bfloat16);  mul_143 = None
        mul_144: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg162_1, convert_element_type_116);  arg162_1 = convert_element_type_116 = None
        view_445: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_144, _shape_param_529);  _shape_param_529 = None
        permute_224: "bf16[512, 384]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        mm_122: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_445, permute_224);  view_445 = permute_224 = None
        view_446: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_122, _shape_param_530);  mm_122 = _shape_param_530 = None
        view_447: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_446, _shape_param_531);  view_446 = _shape_param_531 = None
        permute_225: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None
        expand_81: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_225, _shape_param_532);  permute_225 = _shape_param_532 = None
        clone_80: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_448: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_80, _shape_param_533);  clone_80 = _shape_param_533 = None
        view_449: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_144, _shape_param_534);  _shape_param_534 = None
        permute_226: "bf16[512, 384]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        mm_123: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_449, permute_226);  view_449 = permute_226 = None
        view_450: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_123, _shape_param_535);  mm_123 = _shape_param_535 = None
        view_451: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_450, _shape_param_536);  view_450 = _shape_param_536 = None
        permute_227: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_451, [0, 2, 1, 3]);  view_451 = None
        permute_228: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_227, [0, 1, 3, 2]);  permute_227 = None
        expand_82: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_228, _shape_param_537);  permute_228 = _shape_param_537 = None
        clone_81: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_452: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_81, _shape_param_538);  clone_81 = _shape_param_538 = None
        bmm_40: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_448, view_452);  view_448 = view_452 = None
        view_453: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_40, _shape_param_539);  bmm_40 = _shape_param_539 = None
        add_123: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_453, add_6);  view_453 = None
        convert_element_type_117: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_123, torch.float32);  add_123 = None
        amax_20: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_117, [-1], True)
        sub_22: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_117, amax_20);  convert_element_type_117 = amax_20 = None
        exp_20: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_21: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_24: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_118: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None
        expand_83: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_118, _shape_param_540);  convert_element_type_118 = _shape_param_540 = None
        view_454: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_83, _shape_param_541);  expand_83 = _shape_param_541 = None
        view_455: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_144, _shape_param_542);  mul_144 = _shape_param_542 = None
        permute_229: "bf16[512, 384]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        mm_124: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_455, permute_229);  view_455 = permute_229 = None
        view_456: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_124, _shape_param_543);  mm_124 = _shape_param_543 = None
        view_457: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_456, _shape_param_544);  view_456 = _shape_param_544 = None
        permute_230: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_457, [0, 2, 1, 3]);  view_457 = None
        expand_84: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_230, _shape_param_545);  permute_230 = _shape_param_545 = None
        clone_82: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_458: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_82, _shape_param_546);  clone_82 = _shape_param_546 = None
        bmm_41: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_454, view_458);  view_454 = view_458 = None
        view_459: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_41, _shape_param_547);  bmm_41 = _shape_param_547 = None
        permute_231: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
        clone_83: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
        view_460: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_83, _shape_param_548);  clone_83 = _shape_param_548 = None
        view_461: "bf16[4096, 384]" = torch.ops.aten.view.default(view_460, _shape_param_549);  view_460 = _shape_param_549 = None
        permute_232: "bf16[384, 512]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        mm_125: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_461, permute_232);  view_461 = permute_232 = None
        view_462: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_125, _shape_param_550);  mm_125 = _shape_param_550 = None
        add_124: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_121, view_462);  add_121 = view_462 = None
        convert_element_type_119: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_124, torch.float32)
        pow_51: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_119, 2);  convert_element_type_119 = None
        mean_36: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None
        add_125: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None
        rsqrt_36: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
        mul_145: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_124, rsqrt_36);  rsqrt_36 = None
        convert_element_type_120: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_145, torch.bfloat16);  mul_145 = None
        mul_146: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg167_1, convert_element_type_120);  arg167_1 = convert_element_type_120 = None
        view_463: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_146, _shape_param_551);  mul_146 = _shape_param_551 = None
        permute_233: "bf16[512, 384]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        mm_126: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_463, permute_233);  view_463 = permute_233 = None
        view_464: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_126, _shape_param_552);  mm_126 = _shape_param_552 = None
        view_465: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_464, _shape_param_553);  view_464 = _shape_param_553 = None
        permute_234: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_465, [0, 2, 1, 3]);  view_465 = None
        expand_85: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_234, _shape_param_554);  permute_234 = _shape_param_554 = None
        clone_84: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_466: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_84, _shape_param_555);  clone_84 = _shape_param_555 = None
        view_467: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_80, _shape_param_556);  _shape_param_556 = None
        permute_235: "bf16[512, 384]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        mm_127: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_467, permute_235);  view_467 = permute_235 = None
        view_468: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_127, _shape_param_557);  mm_127 = _shape_param_557 = None
        view_469: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_468, _shape_param_558);  view_468 = _shape_param_558 = None
        permute_236: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_469, [0, 2, 1, 3]);  view_469 = None
        permute_237: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_236, [0, 1, 3, 2]);  permute_236 = None
        expand_86: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_237, _shape_param_559);  permute_237 = _shape_param_559 = None
        clone_85: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
        view_470: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_85, _shape_param_560);  clone_85 = _shape_param_560 = None
        bmm_42: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_466, view_470);  view_466 = view_470 = None
        view_471: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_42, _shape_param_561);  bmm_42 = _shape_param_561 = None
        convert_element_type_121: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_471, torch.float32);  view_471 = None
        amax_21: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_121, [-1], True)
        sub_23: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_121, amax_21);  convert_element_type_121 = amax_21 = None
        exp_21: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_22: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_25: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        convert_element_type_122: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None
        expand_87: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_122, _shape_param_562);  convert_element_type_122 = _shape_param_562 = None
        view_472: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_87, _shape_param_563);  expand_87 = _shape_param_563 = None
        view_473: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_80, _shape_param_564);  _shape_param_564 = None
        permute_238: "bf16[512, 384]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        mm_128: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_473, permute_238);  view_473 = permute_238 = None
        view_474: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_128, _shape_param_565);  mm_128 = _shape_param_565 = None
        view_475: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_474, _shape_param_566);  view_474 = _shape_param_566 = None
        permute_239: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_475, [0, 2, 1, 3]);  view_475 = None
        expand_88: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_239, _shape_param_567);  permute_239 = _shape_param_567 = None
        clone_86: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_476: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_86, _shape_param_568);  clone_86 = _shape_param_568 = None
        bmm_43: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_472, view_476);  view_472 = view_476 = None
        view_477: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_43, _shape_param_569);  bmm_43 = _shape_param_569 = None
        permute_240: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_477, [0, 2, 1, 3]);  view_477 = None
        clone_87: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
        view_478: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_87, _shape_param_570);  clone_87 = _shape_param_570 = None
        view_479: "bf16[4096, 384]" = torch.ops.aten.view.default(view_478, _shape_param_571);  view_478 = _shape_param_571 = None
        permute_241: "bf16[384, 512]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        mm_129: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_479, permute_241);  view_479 = permute_241 = None
        view_480: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_129, _shape_param_572);  mm_129 = _shape_param_572 = None
        add_126: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_124, view_480);  add_124 = view_480 = None
        convert_element_type_123: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_126, torch.float32)
        pow_52: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_123, 2);  convert_element_type_123 = None
        mean_37: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None
        add_127: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None
        rsqrt_37: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        mul_147: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_126, rsqrt_37);  rsqrt_37 = None
        convert_element_type_124: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_147, torch.bfloat16);  mul_147 = None
        mul_148: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg172_1, convert_element_type_124);  arg172_1 = convert_element_type_124 = None
        view_481: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_148, _shape_param_573);  _shape_param_573 = None
        permute_242: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        mm_130: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_481, permute_242);  view_481 = permute_242 = None
        view_482: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_130, _shape_param_574);  mm_130 = _shape_param_574 = None
        mul_149: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_482, 0.5)
        pow_53: "bf16[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_482, 3.0)
        mul_150: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_53, 0.044715);  pow_53 = None
        add_128: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_482, mul_150);  view_482 = mul_150 = None
        mul_151: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_128, 0.7978845608028654);  add_128 = None
        tanh_14: "bf16[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_151);  mul_151 = None
        add_129: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_152: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_149, add_129);  mul_149 = add_129 = None
        view_483: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_148, _shape_param_575);  mul_148 = _shape_param_575 = None
        permute_243: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        mm_131: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_483, permute_243);  view_483 = permute_243 = None
        view_484: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_131, _shape_param_576);  mm_131 = _shape_param_576 = None
        mul_153: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_152, view_484);  mul_152 = view_484 = None
        view_485: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_153, _shape_param_577);  mul_153 = _shape_param_577 = None
        permute_244: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        mm_132: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_485, permute_244);  view_485 = permute_244 = None
        view_486: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_132, _shape_param_578);  mm_132 = _shape_param_578 = None
        add_130: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_126, view_486);  add_126 = view_486 = None
        convert_element_type_125: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_130, torch.float32)
        pow_54: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_125, 2);  convert_element_type_125 = None
        mean_38: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None
        add_131: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None
        rsqrt_38: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        mul_154: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_130, rsqrt_38);  rsqrt_38 = None
        convert_element_type_126: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_154, torch.bfloat16);  mul_154 = None
        mul_155: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg176_1, convert_element_type_126);  arg176_1 = convert_element_type_126 = None
        view_487: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_155, _shape_param_579);  _shape_param_579 = None
        permute_245: "bf16[512, 384]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        mm_133: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_487, permute_245);  view_487 = permute_245 = None
        view_488: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_133, _shape_param_580);  mm_133 = _shape_param_580 = None
        view_489: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_488, _shape_param_581);  view_488 = _shape_param_581 = None
        permute_246: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None
        expand_89: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_246, _shape_param_582);  permute_246 = _shape_param_582 = None
        clone_88: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_490: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_88, _shape_param_583);  clone_88 = _shape_param_583 = None
        view_491: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_155, _shape_param_584);  _shape_param_584 = None
        permute_247: "bf16[512, 384]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        mm_134: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_491, permute_247);  view_491 = permute_247 = None
        view_492: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_134, _shape_param_585);  mm_134 = _shape_param_585 = None
        view_493: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_492, _shape_param_586);  view_492 = _shape_param_586 = None
        permute_248: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
        permute_249: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_248, [0, 1, 3, 2]);  permute_248 = None
        expand_90: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_249, _shape_param_587);  permute_249 = _shape_param_587 = None
        clone_89: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_494: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_89, _shape_param_588);  clone_89 = _shape_param_588 = None
        bmm_44: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_490, view_494);  view_490 = view_494 = None
        view_495: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_44, _shape_param_589);  bmm_44 = _shape_param_589 = None
        add_132: "bf16[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_495, add_6);  view_495 = add_6 = None
        convert_element_type_127: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_132, torch.float32);  add_132 = None
        amax_22: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_127, [-1], True)
        sub_24: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_127, amax_22);  convert_element_type_127 = amax_22 = None
        exp_22: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
        sum_23: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_26: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_128: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_26, torch.bfloat16);  div_26 = None
        expand_91: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_128, _shape_param_590);  convert_element_type_128 = _shape_param_590 = None
        view_496: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_91, _shape_param_591);  expand_91 = _shape_param_591 = None
        view_497: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_155, _shape_param_592);  mul_155 = _shape_param_592 = None
        permute_250: "bf16[512, 384]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        mm_135: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_497, permute_250);  view_497 = permute_250 = None
        view_498: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_135, _shape_param_593);  mm_135 = _shape_param_593 = None
        view_499: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_498, _shape_param_594);  view_498 = _shape_param_594 = None
        permute_251: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_499, [0, 2, 1, 3]);  view_499 = None
        expand_92: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_251, _shape_param_595);  permute_251 = _shape_param_595 = None
        clone_90: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_500: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_90, _shape_param_596);  clone_90 = _shape_param_596 = None
        bmm_45: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_496, view_500);  view_496 = view_500 = None
        view_501: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_45, _shape_param_597);  bmm_45 = _shape_param_597 = None
        permute_252: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_501, [0, 2, 1, 3]);  view_501 = None
        clone_91: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None
        view_502: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_91, _shape_param_598);  clone_91 = _shape_param_598 = None
        view_503: "bf16[4096, 384]" = torch.ops.aten.view.default(view_502, _shape_param_599);  view_502 = _shape_param_599 = None
        permute_253: "bf16[384, 512]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
        mm_136: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_503, permute_253);  view_503 = permute_253 = None
        view_504: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_136, _shape_param_600);  mm_136 = _shape_param_600 = None
        add_133: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_130, view_504);  add_130 = view_504 = None
        convert_element_type_129: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_133, torch.float32)
        pow_55: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_129, 2);  convert_element_type_129 = None
        mean_39: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None
        add_134: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None
        rsqrt_39: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        mul_156: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_133, rsqrt_39);  rsqrt_39 = None
        convert_element_type_130: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_156, torch.bfloat16);  mul_156 = None
        mul_157: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg181_1, convert_element_type_130);  arg181_1 = convert_element_type_130 = None
        view_505: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_157, _shape_param_601);  mul_157 = _shape_param_601 = None
        permute_254: "bf16[512, 384]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        mm_137: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_505, permute_254);  view_505 = permute_254 = None
        view_506: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_137, _shape_param_602);  mm_137 = _shape_param_602 = None
        view_507: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_506, _shape_param_603);  view_506 = _shape_param_603 = None
        permute_255: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_507, [0, 2, 1, 3]);  view_507 = None
        expand_93: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_255, _shape_param_604);  permute_255 = _shape_param_604 = None
        clone_92: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_508: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_92, _shape_param_605);  clone_92 = _shape_param_605 = None
        view_509: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_80, _shape_param_606);  _shape_param_606 = None
        permute_256: "bf16[512, 384]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        mm_138: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_509, permute_256);  view_509 = permute_256 = None
        view_510: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_138, _shape_param_607);  mm_138 = _shape_param_607 = None
        view_511: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_510, _shape_param_608);  view_510 = _shape_param_608 = None
        permute_257: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_511, [0, 2, 1, 3]);  view_511 = None
        permute_258: "bf16[32, 6, 64, 128]" = torch.ops.aten.permute.default(permute_257, [0, 1, 3, 2]);  permute_257 = None
        expand_94: "bf16[32, 6, 64, 128]" = torch.ops.aten.expand.default(permute_258, _shape_param_609);  permute_258 = _shape_param_609 = None
        clone_93: "bf16[32, 6, 64, 128]" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
        view_512: "bf16[192, 64, 128]" = torch.ops.aten.view.default(clone_93, _shape_param_610);  clone_93 = _shape_param_610 = None
        bmm_46: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_508, view_512);  view_508 = view_512 = None
        view_513: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_46, _shape_param_611);  bmm_46 = _shape_param_611 = None
        convert_element_type_131: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_513, torch.float32);  view_513 = None
        amax_23: "f32[32, 6, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_131, [-1], True)
        sub_25: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_131, amax_23);  convert_element_type_131 = amax_23 = None
        exp_23: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_24: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_27: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        convert_element_type_132: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None
        expand_95: "bf16[32, 6, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_132, _shape_param_612);  convert_element_type_132 = _shape_param_612 = None
        view_514: "bf16[192, 128, 128]" = torch.ops.aten.view.default(expand_95, _shape_param_613);  expand_95 = _shape_param_613 = None
        view_515: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_80, _shape_param_614);  _shape_param_614 = None
        permute_259: "bf16[512, 384]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        mm_139: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_515, permute_259);  view_515 = permute_259 = None
        view_516: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_139, _shape_param_615);  mm_139 = _shape_param_615 = None
        view_517: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_516, _shape_param_616);  view_516 = _shape_param_616 = None
        permute_260: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_517, [0, 2, 1, 3]);  view_517 = None
        expand_96: "bf16[32, 6, 128, 64]" = torch.ops.aten.expand.default(permute_260, _shape_param_617);  permute_260 = _shape_param_617 = None
        clone_94: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_518: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_94, _shape_param_618);  clone_94 = _shape_param_618 = None
        bmm_47: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_514, view_518);  view_514 = view_518 = None
        view_519: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_47, _shape_param_619);  bmm_47 = _shape_param_619 = None
        permute_261: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_519, [0, 2, 1, 3]);  view_519 = None
        clone_95: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
        view_520: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_95, _shape_param_620);  clone_95 = _shape_param_620 = None
        view_521: "bf16[4096, 384]" = torch.ops.aten.view.default(view_520, _shape_param_621);  view_520 = _shape_param_621 = None
        permute_262: "bf16[384, 512]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        mm_140: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_521, permute_262);  view_521 = permute_262 = None
        view_522: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_140, _shape_param_622);  mm_140 = _shape_param_622 = None
        add_135: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_133, view_522);  add_133 = view_522 = None
        convert_element_type_133: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_135, torch.float32)
        pow_56: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_133, 2);  convert_element_type_133 = None
        mean_40: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None
        add_136: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None
        rsqrt_40: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        mul_158: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_135, rsqrt_40);  rsqrt_40 = None
        convert_element_type_134: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_158, torch.bfloat16);  mul_158 = None
        mul_159: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg186_1, convert_element_type_134);  arg186_1 = convert_element_type_134 = None
        view_523: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_159, _shape_param_623);  _shape_param_623 = None
        permute_263: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        mm_141: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_523, permute_263);  view_523 = permute_263 = None
        view_524: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_141, _shape_param_624);  mm_141 = _shape_param_624 = None
        mul_160: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_524, 0.5)
        pow_57: "bf16[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_524, 3.0)
        mul_161: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_57, 0.044715);  pow_57 = None
        add_137: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_524, mul_161);  view_524 = mul_161 = None
        mul_162: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_137, 0.7978845608028654);  add_137 = None
        tanh_15: "bf16[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_162);  mul_162 = None
        add_138: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_163: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_160, add_138);  mul_160 = add_138 = None
        view_525: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_159, _shape_param_625);  mul_159 = _shape_param_625 = None
        permute_264: "bf16[512, 1024]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        mm_142: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_525, permute_264);  view_525 = permute_264 = None
        view_526: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_142, _shape_param_626);  mm_142 = _shape_param_626 = None
        mul_164: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_163, view_526);  mul_163 = view_526 = None
        view_527: "bf16[4096, 1024]" = torch.ops.aten.view.default(mul_164, _shape_param_627);  mul_164 = _shape_param_627 = None
        permute_265: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        mm_143: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_527, permute_265);  view_527 = permute_265 = None
        view_528: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_143, _shape_param_628);  mm_143 = _shape_param_628 = None
        add_139: "bf16[32, 128, 512]" = torch.ops.aten.add.Tensor(add_135, view_528);  add_135 = view_528 = None
        convert_element_type_135: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_139, torch.float32)
        pow_58: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_135, 2);  convert_element_type_135 = None
        mean_41: "f32[32, 128, 1]" = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None
        add_140: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None
        rsqrt_41: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        mul_165: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_139, rsqrt_41);  add_139 = rsqrt_41 = None
        convert_element_type_136: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(mul_165, torch.bfloat16);  mul_165 = None
        mul_166: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg190_1, convert_element_type_136);  arg190_1 = convert_element_type_136 = None
        view_529: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_166, _shape_param_629);  mul_166 = _shape_param_629 = None
        permute_266: "bf16[512, 250112]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm_144: "bf16[4096, 250112]" = torch.ops.aten.mm.default(view_529, permute_266);  view_529 = permute_266 = None
        view_530: "bf16[32, 128, 250112]" = torch.ops.aten.view.default(mm_144, _shape_param_630);  mm_144 = _shape_param_630 = None
        view_531: "bf16[4096, 250112]" = torch.ops.aten.view.default(view_530, _shape_param_631);  _shape_param_631 = None
        convert_element_type_137: "f32[4096, 250112]" = torch.ops.prims.convert_element_type.default(view_531, torch.float32);  view_531 = None
        amax_24: "f32[4096, 1]" = torch.ops.aten.amax.default(convert_element_type_137, [1], True)
        sub_26: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(convert_element_type_137, amax_24);  convert_element_type_137 = amax_24 = None
        exp_24: "f32[4096, 250112]" = torch.ops.aten.exp.default(sub_26)
        sum_25: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log_2: "f32[4096, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_27: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(sub_26, log_2);  sub_26 = log_2 = None
        convert_element_type_138: "bf16[4096, 250112]" = torch.ops.prims.convert_element_type.default(sub_27, torch.bfloat16);  sub_27 = None
        ne_1: "b8[4096]" = torch.ops.aten.ne.Scalar(view, -100)
        full_6: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "i64[4096]" = torch.ops.aten.where.self(ne_1, view, full_6);  ne_1 = full_6 = None
        unsqueeze_12: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_3, 1);  where_3 = None
        gather: "bf16[4096, 1]" = torch.ops.aten.gather.default(convert_element_type_138, 1, unsqueeze_12);  convert_element_type_138 = unsqueeze_12 = None
        squeeze: "bf16[4096]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg_1: "bf16[4096]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_7: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[4096]" = torch.ops.aten.where.self(ne, neg_1, full_7);  ne = neg_1 = full_7 = None
        sum_26: "bf16[]" = torch.ops.aten.sum.default(where_4);  where_4 = None
        ne_2: "b8[4096]" = torch.ops.aten.ne.Scalar(view, -100);  view = None
        sum_27: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_139: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_27, torch.bfloat16);  sum_27 = None
        div_28: "bf16[]" = torch.ops.aten.div.Tensor(sum_26, convert_element_type_139);  sum_26 = convert_element_type_139 = None
        return (mul_80, view_530, div_28)



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
