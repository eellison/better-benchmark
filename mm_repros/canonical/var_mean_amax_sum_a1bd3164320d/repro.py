"""
Standalone repro captured via capture_hook.
Label: GPTJForCausalLM
Pattern hash: a1bd3164320d
Shape hash: 2ea4f686
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
    def forward(self, arg0_1: "f32[50400, 4096]", arg1_1: "i64[1, 128]", arg2_1: "i64[1, 128]", arg3_1: "i64[1, 128]", arg4_1: "f32[4096]", arg5_1: "f32[4096]", arg6_1: "f32[4096, 4096]", arg7_1: "f32[4096, 4096]", arg8_1: "f32[4096, 4096]", arg9_1: "f32[2048, 64]", arg10_1: "f32[4096, 4096]", arg11_1: "f32[16384]", arg12_1: "f32[16384, 4096]", arg13_1: "f32[4096]", arg14_1: "f32[4096, 16384]", arg15_1: "f32[4096]", arg16_1: "f32[4096]", arg17_1: "f32[4096, 4096]", arg18_1: "f32[4096, 4096]", arg19_1: "f32[4096, 4096]", arg20_1: "f32[2048, 64]", arg21_1: "f32[4096, 4096]", arg22_1: "f32[16384]", arg23_1: "f32[16384, 4096]", arg24_1: "f32[4096]", arg25_1: "f32[4096, 16384]", arg26_1: "f32[4096]", arg27_1: "f32[4096]", arg28_1: "f32[4096, 4096]", arg29_1: "f32[4096, 4096]", arg30_1: "f32[4096, 4096]", arg31_1: "f32[2048, 64]", arg32_1: "f32[4096, 4096]", arg33_1: "f32[16384]", arg34_1: "f32[16384, 4096]", arg35_1: "f32[4096]", arg36_1: "f32[4096, 16384]", arg37_1: "f32[4096]", arg38_1: "f32[4096]", arg39_1: "f32[4096, 4096]", arg40_1: "f32[4096, 4096]", arg41_1: "f32[4096, 4096]", arg42_1: "f32[2048, 64]", arg43_1: "f32[4096, 4096]", arg44_1: "f32[16384]", arg45_1: "f32[16384, 4096]", arg46_1: "f32[4096]", arg47_1: "f32[4096, 16384]", arg48_1: "f32[4096]", arg49_1: "f32[4096]", arg50_1: "f32[4096, 4096]", arg51_1: "f32[4096, 4096]", arg52_1: "f32[4096, 4096]", arg53_1: "f32[2048, 64]", arg54_1: "f32[4096, 4096]", arg55_1: "f32[16384]", arg56_1: "f32[16384, 4096]", arg57_1: "f32[4096]", arg58_1: "f32[4096, 16384]", arg59_1: "f32[4096]", arg60_1: "f32[4096]", arg61_1: "f32[4096, 4096]", arg62_1: "f32[4096, 4096]", arg63_1: "f32[4096, 4096]", arg64_1: "f32[2048, 64]", arg65_1: "f32[4096, 4096]", arg66_1: "f32[16384]", arg67_1: "f32[16384, 4096]", arg68_1: "f32[4096]", arg69_1: "f32[4096, 16384]", arg70_1: "f32[4096]", arg71_1: "f32[4096]", arg72_1: "f32[4096, 4096]", arg73_1: "f32[4096, 4096]", arg74_1: "f32[4096, 4096]", arg75_1: "f32[2048, 64]", arg76_1: "f32[4096, 4096]", arg77_1: "f32[16384]", arg78_1: "f32[16384, 4096]", arg79_1: "f32[4096]", arg80_1: "f32[4096, 16384]", arg81_1: "f32[4096]", arg82_1: "f32[4096]", arg83_1: "f32[4096, 4096]", arg84_1: "f32[4096, 4096]", arg85_1: "f32[4096, 4096]", arg86_1: "f32[2048, 64]", arg87_1: "f32[4096, 4096]", arg88_1: "f32[16384]", arg89_1: "f32[16384, 4096]", arg90_1: "f32[4096]", arg91_1: "f32[4096, 16384]", arg92_1: "f32[4096]", arg93_1: "f32[4096]", arg94_1: "f32[4096, 4096]", arg95_1: "f32[4096, 4096]", arg96_1: "f32[4096, 4096]", arg97_1: "f32[2048, 64]", arg98_1: "f32[4096, 4096]", arg99_1: "f32[16384]", arg100_1: "f32[16384, 4096]", arg101_1: "f32[4096]", arg102_1: "f32[4096, 16384]", arg103_1: "f32[4096]", arg104_1: "f32[4096]", arg105_1: "f32[4096, 4096]", arg106_1: "f32[4096, 4096]", arg107_1: "f32[4096, 4096]", arg108_1: "f32[2048, 64]", arg109_1: "f32[4096, 4096]", arg110_1: "f32[16384]", arg111_1: "f32[16384, 4096]", arg112_1: "f32[4096]", arg113_1: "f32[4096, 16384]", arg114_1: "f32[4096]", arg115_1: "f32[4096]", arg116_1: "f32[4096, 4096]", arg117_1: "f32[4096, 4096]", arg118_1: "f32[4096, 4096]", arg119_1: "f32[2048, 64]", arg120_1: "f32[4096, 4096]", arg121_1: "f32[16384]", arg122_1: "f32[16384, 4096]", arg123_1: "f32[4096]", arg124_1: "f32[4096, 16384]", arg125_1: "f32[4096]", arg126_1: "f32[4096]", arg127_1: "f32[4096, 4096]", arg128_1: "f32[4096, 4096]", arg129_1: "f32[4096, 4096]", arg130_1: "f32[2048, 64]", arg131_1: "f32[4096, 4096]", arg132_1: "f32[16384]", arg133_1: "f32[16384, 4096]", arg134_1: "f32[4096]", arg135_1: "f32[4096, 16384]", arg136_1: "f32[4096]", arg137_1: "f32[4096]", arg138_1: "f32[4096, 4096]", arg139_1: "f32[4096, 4096]", arg140_1: "f32[4096, 4096]", arg141_1: "f32[2048, 64]", arg142_1: "f32[4096, 4096]", arg143_1: "f32[16384]", arg144_1: "f32[16384, 4096]", arg145_1: "f32[4096]", arg146_1: "f32[4096, 16384]", arg147_1: "f32[4096]", arg148_1: "f32[4096]", arg149_1: "f32[4096, 4096]", arg150_1: "f32[4096, 4096]", arg151_1: "f32[4096, 4096]", arg152_1: "f32[2048, 64]", arg153_1: "f32[4096, 4096]", arg154_1: "f32[16384]", arg155_1: "f32[16384, 4096]", arg156_1: "f32[4096]", arg157_1: "f32[4096, 16384]", arg158_1: "f32[4096]", arg159_1: "f32[4096]", arg160_1: "f32[4096, 4096]", arg161_1: "f32[4096, 4096]", arg162_1: "f32[4096, 4096]", arg163_1: "f32[2048, 64]", arg164_1: "f32[4096, 4096]", arg165_1: "f32[16384]", arg166_1: "f32[16384, 4096]", arg167_1: "f32[4096]", arg168_1: "f32[4096, 16384]", arg169_1: "f32[4096]", arg170_1: "f32[4096]", arg171_1: "f32[4096, 4096]", arg172_1: "f32[4096, 4096]", arg173_1: "f32[4096, 4096]", arg174_1: "f32[2048, 64]", arg175_1: "f32[4096, 4096]", arg176_1: "f32[16384]", arg177_1: "f32[16384, 4096]", arg178_1: "f32[4096]", arg179_1: "f32[4096, 16384]", arg180_1: "f32[4096]", arg181_1: "f32[4096]", arg182_1: "f32[4096, 4096]", arg183_1: "f32[4096, 4096]", arg184_1: "f32[4096, 4096]", arg185_1: "f32[2048, 64]", arg186_1: "f32[4096, 4096]", arg187_1: "f32[16384]", arg188_1: "f32[16384, 4096]", arg189_1: "f32[4096]", arg190_1: "f32[4096, 16384]", arg191_1: "f32[4096]", arg192_1: "f32[4096]", arg193_1: "f32[4096, 4096]", arg194_1: "f32[4096, 4096]", arg195_1: "f32[4096, 4096]", arg196_1: "f32[2048, 64]", arg197_1: "f32[4096, 4096]", arg198_1: "f32[16384]", arg199_1: "f32[16384, 4096]", arg200_1: "f32[4096]", arg201_1: "f32[4096, 16384]", arg202_1: "f32[4096]", arg203_1: "f32[4096]", arg204_1: "f32[4096, 4096]", arg205_1: "f32[4096, 4096]", arg206_1: "f32[4096, 4096]", arg207_1: "f32[2048, 64]", arg208_1: "f32[4096, 4096]", arg209_1: "f32[16384]", arg210_1: "f32[16384, 4096]", arg211_1: "f32[4096]", arg212_1: "f32[4096, 16384]", arg213_1: "f32[4096]", arg214_1: "f32[4096]", arg215_1: "f32[4096, 4096]", arg216_1: "f32[4096, 4096]", arg217_1: "f32[4096, 4096]", arg218_1: "f32[2048, 64]", arg219_1: "f32[4096, 4096]", arg220_1: "f32[16384]", arg221_1: "f32[16384, 4096]", arg222_1: "f32[4096]", arg223_1: "f32[4096, 16384]", arg224_1: "f32[4096]", arg225_1: "f32[4096]", arg226_1: "f32[4096, 4096]", arg227_1: "f32[4096, 4096]", arg228_1: "f32[4096, 4096]", arg229_1: "f32[2048, 64]", arg230_1: "f32[4096, 4096]", arg231_1: "f32[16384]", arg232_1: "f32[16384, 4096]", arg233_1: "f32[4096]", arg234_1: "f32[4096, 16384]", arg235_1: "f32[4096]", arg236_1: "f32[4096]", arg237_1: "f32[4096, 4096]", arg238_1: "f32[4096, 4096]", arg239_1: "f32[4096, 4096]", arg240_1: "f32[2048, 64]", arg241_1: "f32[4096, 4096]", arg242_1: "f32[16384]", arg243_1: "f32[16384, 4096]", arg244_1: "f32[4096]", arg245_1: "f32[4096, 16384]", arg246_1: "f32[4096]", arg247_1: "f32[4096]", arg248_1: "f32[4096, 4096]", arg249_1: "f32[4096, 4096]", arg250_1: "f32[4096, 4096]", arg251_1: "f32[2048, 64]", arg252_1: "f32[4096, 4096]", arg253_1: "f32[16384]", arg254_1: "f32[16384, 4096]", arg255_1: "f32[4096]", arg256_1: "f32[4096, 16384]", arg257_1: "f32[4096]", arg258_1: "f32[4096]", arg259_1: "f32[4096, 4096]", arg260_1: "f32[4096, 4096]", arg261_1: "f32[4096, 4096]", arg262_1: "f32[2048, 64]", arg263_1: "f32[4096, 4096]", arg264_1: "f32[16384]", arg265_1: "f32[16384, 4096]", arg266_1: "f32[4096]", arg267_1: "f32[4096, 16384]", arg268_1: "f32[4096]", arg269_1: "f32[4096]", arg270_1: "f32[4096, 4096]", arg271_1: "f32[4096, 4096]", arg272_1: "f32[4096, 4096]", arg273_1: "f32[2048, 64]", arg274_1: "f32[4096, 4096]", arg275_1: "f32[16384]", arg276_1: "f32[16384, 4096]", arg277_1: "f32[4096]", arg278_1: "f32[4096, 16384]", arg279_1: "f32[4096]", arg280_1: "f32[4096]", arg281_1: "f32[4096, 4096]", arg282_1: "f32[4096, 4096]", arg283_1: "f32[4096, 4096]", arg284_1: "f32[2048, 64]", arg285_1: "f32[4096, 4096]", arg286_1: "f32[16384]", arg287_1: "f32[16384, 4096]", arg288_1: "f32[4096]", arg289_1: "f32[4096, 16384]", arg290_1: "f32[4096]", arg291_1: "f32[4096]", arg292_1: "f32[4096, 4096]", arg293_1: "f32[4096, 4096]", arg294_1: "f32[4096, 4096]", arg295_1: "f32[2048, 64]", arg296_1: "f32[4096, 4096]", arg297_1: "f32[16384]", arg298_1: "f32[16384, 4096]", arg299_1: "f32[4096]", arg300_1: "f32[4096, 16384]", arg301_1: "f32[4096]", arg302_1: "f32[4096]", arg303_1: "f32[4096, 4096]", arg304_1: "f32[4096, 4096]", arg305_1: "f32[4096, 4096]", arg306_1: "f32[2048, 64]", arg307_1: "f32[4096, 4096]", arg308_1: "f32[16384]", arg309_1: "f32[16384, 4096]", arg310_1: "f32[4096]", arg311_1: "f32[4096, 16384]", arg312_1: "f32[4096]", arg313_1: "f32[4096]", arg314_1: "f32[50400]", arg315_1: "f32[50400, 4096]", arg316_1: "i64[1, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631, _shape_param_632, _shape_param_633, _shape_param_634, _shape_param_635, _shape_param_636, _shape_param_637, _shape_param_638, _shape_param_639, _shape_param_640, _shape_param_641, _shape_param_642, _shape_param_643, _shape_param_644, _shape_param_645, _shape_param_646, _shape_param_647, _shape_param_648, _shape_param_649, _shape_param_650, _shape_param_651, _shape_param_652, _shape_param_653, _shape_param_654, _shape_param_655, _shape_param_656, _shape_param_657, _shape_param_658, _shape_param_659, _shape_param_660, _shape_param_661, _shape_param_662, _shape_param_663, _shape_param_664, _shape_param_665, _shape_param_666, _shape_param_667, _shape_param_668, _shape_param_669, _shape_param_670, _shape_param_671, _shape_param_672, _shape_param_673, _shape_param_674, _shape_param_675, _shape_param_676, _shape_param_677, _shape_param_678, _shape_param_679, _shape_param_680, _shape_param_681, _shape_param_682, _shape_param_683, _shape_param_684, _shape_param_685, _shape_param_686, _shape_param_687, _shape_param_688, _shape_param_689, _shape_param_690, _shape_param_691, _shape_param_692, _shape_param_693, _shape_param_694, _shape_param_695, _shape_param_696, _shape_param_697, _shape_param_698, _shape_param_699, _shape_param_700, _shape_param_701, _shape_param_702, _shape_param_703, _shape_param_704, _shape_param_705, _shape_param_706, _shape_param_707, _shape_param_708, _shape_param_709, _shape_param_710, _shape_param_711, _shape_param_712, _shape_param_713, _shape_param_714, _shape_param_715, _shape_param_716, _shape_param_717, _shape_param_718, _shape_param_719, _shape_param_720, _shape_param_721, _shape_param_722, _shape_param_723, _shape_param_724, _shape_param_725, _shape_param_726, _shape_param_727, _shape_param_728, _shape_param_729, _shape_param_730, _shape_param_731, _shape_param_732, _shape_param_733, _shape_param_734, _shape_param_735, _shape_param_736, _shape_param_737, _shape_param_738, _shape_param_739, _shape_param_740, _shape_param_741, _shape_param_742, _shape_param_743, _shape_param_744, _shape_param_745, _shape_param_746, _shape_param_747, _shape_param_748, _shape_param_749, _shape_param_750, _shape_param_751, _shape_param_752, _shape_param_753, _shape_param_754, _shape_param_755, _shape_param_756, _shape_param_757, _shape_param_758, _shape_param_759, _shape_param_760, _shape_param_761, _shape_param_762, _shape_param_763, _shape_param_764, _shape_param_765, _shape_param_766, _shape_param_767, _shape_param_768, _shape_param_769, _shape_param_770, _shape_param_771, _shape_param_772, _shape_param_773, _shape_param_774, _shape_param_775, _shape_param_776, _shape_param_777, _shape_param_778, _shape_param_779, _shape_param_780, _shape_param_781, _shape_param_782, _shape_param_783, _shape_param_784, _shape_param_785, _shape_param_786, _shape_param_787, _shape_param_788, _shape_param_789, _shape_param_790, _shape_param_791, _shape_param_792, _shape_param_793, _shape_param_794, _shape_param_795, _shape_param_796, _shape_param_797, _shape_param_798, _shape_param_799, _shape_param_800, _shape_param_801, _shape_param_802, _shape_param_803, _shape_param_804, _shape_param_805, _shape_param_806, _shape_param_807, _shape_param_808, _shape_param_809, _shape_param_810, _shape_param_811, _shape_param_812, _shape_param_813, _shape_param_814, _shape_param_815, _shape_param_816):
        # No stacktrace found for following nodes
        embedding: "f32[1, 128, 4096]" = torch.ops.aten.embedding.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        iota: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota, 1);  iota = None
        unsqueeze_1: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        unsqueeze_3: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        unsqueeze_4: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 3)
        unsqueeze_5: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        full: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_4)
        bitwise_and: "b8[1, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(full, le);  full = le = None
        index: "i64[1, 1, 128, 1]" = torch.ops.aten.index.Tensor(arg3_1, [unsqueeze_2, unsqueeze_4]);  unsqueeze_4 = None
        index_1: "i64[1, 1, 1, 128]" = torch.ops.aten.index.Tensor(arg3_1, [unsqueeze_2, unsqueeze_5]);  arg3_1 = unsqueeze_2 = unsqueeze_5 = None
        eq: "b8[1, 1, 128, 128]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None
        bitwise_and_1: "b8[1, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None
        expand: "b8[1, 1, 128, 128]" = torch.ops.aten.expand.default(bitwise_and_1, _shape_param_0);  bitwise_and_1 = _shape_param_0 = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_2: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[1, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_1, full_2);  expand = full_2 = None
        var_mean = torch.ops.aten.var_mean.correction(embedding, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 128, 1]" = var_mean[0]
        getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
        add: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(embedding, getitem_1)
        mul: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        convert_element_type_1: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        permute: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        view: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  convert_element_type_1 = _shape_param_1 = None
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(view, permute)
        view_1: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        convert_element_type_2: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        permute_1: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_2, [1, 0]);  convert_element_type_2 = None
        mm_1: "bf16[128, 4096]" = torch.ops.aten.mm.default(view, permute_1)
        view_2: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_1, _shape_param_3);  mm_1 = _shape_param_3 = None
        convert_element_type_3: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        permute_2: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0]);  convert_element_type_3 = None
        mm_2: "bf16[128, 4096]" = torch.ops.aten.mm.default(view, permute_2)
        view_3: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_2, _shape_param_4);  mm_2 = _shape_param_4 = None
        view_4: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_1, _shape_param_5);  view_1 = _shape_param_5 = None
        view_5: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_2, _shape_param_6);  view_2 = _shape_param_6 = None
        view_6: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_3, _shape_param_7);  view_3 = _shape_param_7 = None
        permute_3: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        repeat: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg9_1, [1, 1, 1]);  arg9_1 = None
        unsqueeze_6: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        repeat_1: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_6, _shape_param_8);  unsqueeze_6 = _shape_param_8 = None
        gather: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat, 1, repeat_1);  repeat = None
        convert_element_type_4: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather, torch.bfloat16);  gather = None
        split = torch.ops.aten.split.Tensor(convert_element_type_4, 32, -1);  convert_element_type_4 = None
        getitem_2: "bf16[1, 128, 32]" = split[0]
        getitem_3: "bf16[1, 128, 32]" = split[1];  split = None
        slice_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_5, 3, 0, 64)
        slice_2: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_5, 3, 64, 9223372036854775807);  view_5 = None
        slice_3: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_4, 3, 0, 64)
        slice_4: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_4, 3, 64, 9223372036854775807);  view_4 = None
        unsqueeze_7: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_2, 2);  getitem_2 = None
        unsqueeze_8: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 4);  unsqueeze_7 = None
        expand_1: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_8, _shape_param_9);  _shape_param_9 = None
        clone: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_7: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone, _shape_param_10);  clone = _shape_param_10 = None
        unsqueeze_9: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_3, 2);  getitem_3 = None
        unsqueeze_10: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 4);  unsqueeze_9 = None
        expand_2: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_10, _shape_param_11);  _shape_param_11 = None
        clone_1: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_8: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_12);  clone_1 = _shape_param_12 = None
        mul_2: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1, view_8)
        slice_5: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 9223372036854775807, 2)
        slice_6: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1, 3, 1, 9223372036854775807, 2);  slice_1 = None
        neg: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_6);  slice_6 = None
        unsqueeze_11: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg, 4);  neg = None
        unsqueeze_12: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_5, 4);  slice_5 = None
        cat: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_11, unsqueeze_12], -1);  unsqueeze_11 = unsqueeze_12 = None
        view_9: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat, _shape_param_13);  cat = _shape_param_13 = None
        mul_3: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_9, view_7);  view_9 = None
        add_2: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        mul_4: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_3, view_8);  view_8 = None
        slice_7: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 9223372036854775807, 2)
        slice_8: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_3, 3, 1, 9223372036854775807, 2);  slice_3 = None
        neg_1: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_8);  slice_8 = None
        unsqueeze_13: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_1, 4);  neg_1 = None
        unsqueeze_14: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_7, 4);  slice_7 = None
        cat_1: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_13, unsqueeze_14], -1);  unsqueeze_13 = unsqueeze_14 = None
        view_10: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_1, _shape_param_14);  cat_1 = _shape_param_14 = None
        mul_5: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_10, view_7);  view_10 = view_7 = None
        add_3: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        cat_2: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_2, slice_2], -1);  add_2 = slice_2 = None
        cat_3: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_3, slice_4], -1);  add_3 = slice_4 = None
        permute_4: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_2, [0, 2, 1, 3]);  cat_2 = None
        permute_5: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_3, [0, 2, 1, 3]);  cat_3 = None
        convert_element_type_5: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_4, torch.float32);  permute_4 = None
        permute_6: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_5, [0, 1, 3, 2]);  convert_element_type_5 = None
        convert_element_type_6: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_6, torch.bfloat16);  permute_6 = None
        convert_element_type_7: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_5, torch.bfloat16);  permute_5 = None
        expand_3: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_7, _shape_param_15);  convert_element_type_7 = _shape_param_15 = None
        view_11: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_3, _shape_param_16);  expand_3 = _shape_param_16 = None
        expand_4: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_6, _shape_param_17);  convert_element_type_6 = _shape_param_17 = None
        view_12: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_4, _shape_param_18);  expand_4 = _shape_param_18 = None
        bmm: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_11, view_12)
        view_13: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_19);  bmm = _shape_param_19 = None
        div: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_13, 16.0);  view_13 = None
        add_4: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div, where);  div = None
        amax: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_4, [-1], True)
        sub_1: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_4, amax);  add_4 = amax = None
        exp: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_8: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16)
        expand_5: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_8, _shape_param_20);  convert_element_type_8 = _shape_param_20 = None
        view_14: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_5, _shape_param_21);  expand_5 = _shape_param_21 = None
        expand_6: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_3, _shape_param_22);  permute_3 = _shape_param_22 = None
        view_15: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_6, _shape_param_23);  expand_6 = _shape_param_23 = None
        bmm_1: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_14, view_15)
        view_16: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_1, _shape_param_24);  bmm_1 = _shape_param_24 = None
        permute_7: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        clone_2: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_17: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_2, _shape_param_25);  clone_2 = _shape_param_25 = None
        convert_element_type_9: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        permute_8: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_9, [1, 0]);  convert_element_type_9 = None
        view_18: "bf16[128, 4096]" = torch.ops.aten.view.default(view_17, _shape_param_26);  view_17 = _shape_param_26 = None
        mm_3: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_18, permute_8)
        view_19: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_3, _shape_param_27);  mm_3 = _shape_param_27 = None
        convert_element_type_10: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.bfloat16);  arg11_1 = None
        convert_element_type_11: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.bfloat16);  arg12_1 = None
        permute_9: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_11, [1, 0]);  convert_element_type_11 = None
        addmm: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_10, view, permute_9);  convert_element_type_10 = None
        view_20: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm, _shape_param_28);  _shape_param_28 = None
        mul_6: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_20, 0.5)
        convert_element_type_12: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_20, torch.float32)
        pow_1: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_12, 3.0);  convert_element_type_12 = None
        mul_7: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_5: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_20, mul_7);  view_20 = mul_7 = None
        mul_8: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_5, 0.7978845608028654);  add_5 = None
        tanh: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
        add_6: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_9: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_6, add_6);  mul_6 = add_6 = None
        convert_element_type_13: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        convert_element_type_14: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.bfloat16);  arg14_1 = None
        convert_element_type_15: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        view_21: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_15, _shape_param_29);  convert_element_type_15 = _shape_param_29 = None
        permute_10: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_14, [1, 0]);  convert_element_type_14 = None
        addmm_1: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_13, view_21, permute_10);  convert_element_type_13 = None
        view_22: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_1, _shape_param_30);  addmm_1 = _shape_param_30 = None
        add_7: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_19, view_22);  view_19 = view_22 = None
        add_8: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_7, embedding)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_4: "f32[1, 128, 1]" = var_mean_1[0]
        getitem_5: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        add_9: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_2: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_8, getitem_5)
        mul_10: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_11: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_10, arg15_1);  mul_10 = arg15_1 = None
        add_10: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_11, arg16_1);  mul_11 = arg16_1 = None
        convert_element_type_16: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        convert_element_type_17: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None
        permute_11: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_16, [1, 0]);  convert_element_type_16 = None
        view_23: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_17, _shape_param_31);  convert_element_type_17 = _shape_param_31 = None
        mm_4: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_23, permute_11)
        view_24: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_4, _shape_param_32);  mm_4 = _shape_param_32 = None
        convert_element_type_18: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.bfloat16);  arg18_1 = None
        permute_12: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_18, [1, 0]);  convert_element_type_18 = None
        mm_5: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_23, permute_12)
        view_25: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_5, _shape_param_33);  mm_5 = _shape_param_33 = None
        convert_element_type_19: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg19_1, torch.bfloat16);  arg19_1 = None
        permute_13: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_19, [1, 0]);  convert_element_type_19 = None
        mm_6: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_23, permute_13)
        view_26: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_6, _shape_param_34);  mm_6 = _shape_param_34 = None
        view_27: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_24, _shape_param_35);  view_24 = _shape_param_35 = None
        view_28: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_25, _shape_param_36);  view_25 = _shape_param_36 = None
        view_29: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_26, _shape_param_37);  view_26 = _shape_param_37 = None
        permute_14: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        repeat_2: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg20_1, [1, 1, 1]);  arg20_1 = None
        gather_1: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_2, 1, repeat_1);  repeat_2 = None
        convert_element_type_20: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_1, torch.bfloat16);  gather_1 = None
        split_1 = torch.ops.aten.split.Tensor(convert_element_type_20, 32, -1);  convert_element_type_20 = None
        getitem_6: "bf16[1, 128, 32]" = split_1[0]
        getitem_7: "bf16[1, 128, 32]" = split_1[1];  split_1 = None
        slice_9: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_28, 3, 0, 64)
        slice_10: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_28, 3, 64, 9223372036854775807);  view_28 = None
        slice_11: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_27, 3, 0, 64)
        slice_12: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_27, 3, 64, 9223372036854775807);  view_27 = None
        unsqueeze_15: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_6, 2);  getitem_6 = None
        unsqueeze_16: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 4);  unsqueeze_15 = None
        expand_7: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_16, _shape_param_38);  _shape_param_38 = None
        clone_3: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_30: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_3, _shape_param_39);  clone_3 = _shape_param_39 = None
        unsqueeze_17: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_7, 2);  getitem_7 = None
        unsqueeze_18: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 4);  unsqueeze_17 = None
        expand_8: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_18, _shape_param_40);  _shape_param_40 = None
        clone_4: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_31: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_41);  clone_4 = _shape_param_41 = None
        mul_12: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_9, view_31)
        slice_13: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_9, 3, 0, 9223372036854775807, 2)
        slice_14: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_9, 3, 1, 9223372036854775807, 2);  slice_9 = None
        neg_2: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_14);  slice_14 = None
        unsqueeze_19: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_2, 4);  neg_2 = None
        unsqueeze_20: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_13, 4);  slice_13 = None
        cat_4: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_19, unsqueeze_20], -1);  unsqueeze_19 = unsqueeze_20 = None
        view_32: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_4, _shape_param_42);  cat_4 = _shape_param_42 = None
        mul_13: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_32, view_30);  view_32 = None
        add_11: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
        mul_14: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_11, view_31);  view_31 = None
        slice_15: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_11, 3, 0, 9223372036854775807, 2)
        slice_16: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_11, 3, 1, 9223372036854775807, 2);  slice_11 = None
        neg_3: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_16);  slice_16 = None
        unsqueeze_21: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_3, 4);  neg_3 = None
        unsqueeze_22: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_15, 4);  slice_15 = None
        cat_5: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_21, unsqueeze_22], -1);  unsqueeze_21 = unsqueeze_22 = None
        view_33: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_5, _shape_param_43);  cat_5 = _shape_param_43 = None
        mul_15: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_33, view_30);  view_33 = view_30 = None
        add_12: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None
        cat_6: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_11, slice_10], -1);  add_11 = slice_10 = None
        cat_7: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_12, slice_12], -1);  add_12 = slice_12 = None
        permute_15: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_6, [0, 2, 1, 3]);  cat_6 = None
        permute_16: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_7, [0, 2, 1, 3]);  cat_7 = None
        convert_element_type_21: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_15, torch.float32);  permute_15 = None
        permute_17: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_21, [0, 1, 3, 2]);  convert_element_type_21 = None
        convert_element_type_22: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_17, torch.bfloat16);  permute_17 = None
        convert_element_type_23: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_16, torch.bfloat16);  permute_16 = None
        expand_9: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_23, _shape_param_44);  convert_element_type_23 = _shape_param_44 = None
        view_34: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_9, _shape_param_45);  expand_9 = _shape_param_45 = None
        expand_10: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_22, _shape_param_46);  convert_element_type_22 = _shape_param_46 = None
        view_35: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_10, _shape_param_47);  expand_10 = _shape_param_47 = None
        bmm_2: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_34, view_35)
        view_36: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_48);  bmm_2 = _shape_param_48 = None
        div_2: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_36, 16.0);  view_36 = None
        add_13: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_2, where);  div_2 = None
        amax_1: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_13, [-1], True)
        sub_3: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_13, amax_1);  add_13 = amax_1 = None
        exp_1: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_2: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_24: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16)
        expand_11: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_24, _shape_param_49);  convert_element_type_24 = _shape_param_49 = None
        view_37: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_11, _shape_param_50);  expand_11 = _shape_param_50 = None
        expand_12: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_14, _shape_param_51);  permute_14 = _shape_param_51 = None
        view_38: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_12, _shape_param_52);  expand_12 = _shape_param_52 = None
        bmm_3: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_37, view_38)
        view_39: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_3, _shape_param_53);  bmm_3 = _shape_param_53 = None
        permute_18: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
        clone_5: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_40: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_5, _shape_param_54);  clone_5 = _shape_param_54 = None
        convert_element_type_25: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg21_1, torch.bfloat16);  arg21_1 = None
        permute_19: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_25, [1, 0]);  convert_element_type_25 = None
        view_41: "bf16[128, 4096]" = torch.ops.aten.view.default(view_40, _shape_param_55);  view_40 = _shape_param_55 = None
        mm_7: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_41, permute_19)
        view_42: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_7, _shape_param_56);  mm_7 = _shape_param_56 = None
        convert_element_type_26: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg22_1, torch.bfloat16);  arg22_1 = None
        convert_element_type_27: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg23_1, torch.bfloat16);  arg23_1 = None
        permute_20: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_27, [1, 0]);  convert_element_type_27 = None
        addmm_2: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_26, view_23, permute_20);  convert_element_type_26 = None
        view_43: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_2, _shape_param_57);  _shape_param_57 = None
        mul_16: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
        convert_element_type_28: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_43, torch.float32)
        pow_2: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_28, 3.0);  convert_element_type_28 = None
        mul_17: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_14: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_43, mul_17);  view_43 = mul_17 = None
        mul_18: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
        tanh_1: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
        add_15: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_19: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_16, add_15);  mul_16 = add_15 = None
        convert_element_type_29: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg24_1, torch.bfloat16);  arg24_1 = None
        convert_element_type_30: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg25_1, torch.bfloat16);  arg25_1 = None
        convert_element_type_31: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_19, torch.bfloat16);  mul_19 = None
        view_44: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_31, _shape_param_58);  convert_element_type_31 = _shape_param_58 = None
        permute_21: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_30, [1, 0]);  convert_element_type_30 = None
        addmm_3: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_29, view_44, permute_21);  convert_element_type_29 = None
        view_45: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_3, _shape_param_59);  addmm_3 = _shape_param_59 = None
        add_16: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_42, view_45);  view_42 = view_45 = None
        add_17: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_16, add_8);  add_16 = add_8 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_8: "f32[1, 128, 1]" = var_mean_2[0]
        getitem_9: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        add_18: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_4: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_17, getitem_9);  getitem_9 = None
        mul_20: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
        mul_21: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_20, arg26_1);  arg26_1 = None
        add_19: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_21, arg27_1);  mul_21 = arg27_1 = None
        convert_element_type_32: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg28_1, torch.bfloat16);  arg28_1 = None
        convert_element_type_33: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None
        permute_22: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_32, [1, 0]);  convert_element_type_32 = None
        view_46: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_33, _shape_param_60);  convert_element_type_33 = _shape_param_60 = None
        mm_8: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_46, permute_22)
        view_47: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_8, _shape_param_61);  mm_8 = _shape_param_61 = None
        convert_element_type_34: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg29_1, torch.bfloat16);  arg29_1 = None
        permute_23: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        mm_9: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_46, permute_23)
        view_48: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_9, _shape_param_62);  mm_9 = _shape_param_62 = None
        convert_element_type_35: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg30_1, torch.bfloat16);  arg30_1 = None
        permute_24: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_35, [1, 0]);  convert_element_type_35 = None
        mm_10: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_46, permute_24)
        view_49: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_10, _shape_param_63);  mm_10 = _shape_param_63 = None
        view_50: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_47, _shape_param_64);  view_47 = _shape_param_64 = None
        view_51: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_48, _shape_param_65);  view_48 = _shape_param_65 = None
        view_52: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_49, _shape_param_66);  view_49 = _shape_param_66 = None
        permute_25: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        repeat_3: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg31_1, [1, 1, 1]);  arg31_1 = None
        gather_2: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_3, 1, repeat_1);  repeat_3 = None
        convert_element_type_36: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_2, torch.bfloat16);  gather_2 = None
        split_2 = torch.ops.aten.split.Tensor(convert_element_type_36, 32, -1);  convert_element_type_36 = None
        getitem_10: "bf16[1, 128, 32]" = split_2[0]
        getitem_11: "bf16[1, 128, 32]" = split_2[1];  split_2 = None
        slice_17: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_51, 3, 0, 64)
        slice_18: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_51, 3, 64, 9223372036854775807);  view_51 = None
        slice_19: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_50, 3, 0, 64)
        slice_20: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_50, 3, 64, 9223372036854775807);  view_50 = None
        unsqueeze_23: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_10, 2);  getitem_10 = None
        unsqueeze_24: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, 4);  unsqueeze_23 = None
        expand_13: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_24, _shape_param_67);  _shape_param_67 = None
        clone_6: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_53: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_6, _shape_param_68);  clone_6 = _shape_param_68 = None
        unsqueeze_25: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_11, 2);  getitem_11 = None
        unsqueeze_26: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_25, 4);  unsqueeze_25 = None
        expand_14: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_26, _shape_param_69);  _shape_param_69 = None
        clone_7: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_54: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_7, _shape_param_70);  clone_7 = _shape_param_70 = None
        mul_22: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_17, view_54)
        slice_21: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_17, 3, 0, 9223372036854775807, 2)
        slice_22: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_17, 3, 1, 9223372036854775807, 2);  slice_17 = None
        neg_4: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_22);  slice_22 = None
        unsqueeze_27: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_4, 4);  neg_4 = None
        unsqueeze_28: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_21, 4);  slice_21 = None
        cat_8: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_27, unsqueeze_28], -1);  unsqueeze_27 = unsqueeze_28 = None
        view_55: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_8, _shape_param_71);  cat_8 = _shape_param_71 = None
        mul_23: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_55, view_53);  view_55 = None
        add_20: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        mul_24: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_19, view_54);  view_54 = None
        slice_23: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_19, 3, 0, 9223372036854775807, 2)
        slice_24: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_19, 3, 1, 9223372036854775807, 2);  slice_19 = None
        neg_5: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_24);  slice_24 = None
        unsqueeze_29: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_5, 4);  neg_5 = None
        unsqueeze_30: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_23, 4);  slice_23 = None
        cat_9: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_29, unsqueeze_30], -1);  unsqueeze_29 = unsqueeze_30 = None
        view_56: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_9, _shape_param_72);  cat_9 = _shape_param_72 = None
        mul_25: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_56, view_53);  view_56 = view_53 = None
        add_21: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_24, mul_25);  mul_24 = mul_25 = None
        cat_10: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_20, slice_18], -1);  add_20 = slice_18 = None
        cat_11: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_21, slice_20], -1);  add_21 = slice_20 = None
        permute_26: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_10, [0, 2, 1, 3]);  cat_10 = None
        permute_27: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_11, [0, 2, 1, 3]);  cat_11 = None
        convert_element_type_37: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_26, torch.float32);  permute_26 = None
        permute_28: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_37, [0, 1, 3, 2]);  convert_element_type_37 = None
        convert_element_type_38: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_28, torch.bfloat16);  permute_28 = None
        convert_element_type_39: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_27, torch.bfloat16);  permute_27 = None
        expand_15: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_39, _shape_param_73);  convert_element_type_39 = _shape_param_73 = None
        view_57: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_15, _shape_param_74);  expand_15 = _shape_param_74 = None
        expand_16: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_38, _shape_param_75);  convert_element_type_38 = _shape_param_75 = None
        view_58: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_16, _shape_param_76);  expand_16 = _shape_param_76 = None
        bmm_4: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_57, view_58)
        view_59: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_4, _shape_param_77);  bmm_4 = _shape_param_77 = None
        div_4: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_59, 16.0);  view_59 = None
        add_22: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_4, where);  div_4 = None
        amax_2: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_22, [-1], True)
        sub_5: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_22, amax_2);  add_22 = amax_2 = None
        exp_2: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_3: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_5: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_40: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16)
        expand_17: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_40, _shape_param_78);  convert_element_type_40 = _shape_param_78 = None
        view_60: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_17, _shape_param_79);  expand_17 = _shape_param_79 = None
        expand_18: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_25, _shape_param_80);  permute_25 = _shape_param_80 = None
        view_61: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_18, _shape_param_81);  expand_18 = _shape_param_81 = None
        bmm_5: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_60, view_61)
        view_62: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_5, _shape_param_82);  bmm_5 = _shape_param_82 = None
        permute_29: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_62, [0, 2, 1, 3]);  view_62 = None
        clone_8: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_63: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_8, _shape_param_83);  clone_8 = _shape_param_83 = None
        convert_element_type_41: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg32_1, torch.bfloat16);  arg32_1 = None
        permute_30: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_41, [1, 0]);  convert_element_type_41 = None
        view_64: "bf16[128, 4096]" = torch.ops.aten.view.default(view_63, _shape_param_84);  view_63 = _shape_param_84 = None
        mm_11: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_64, permute_30)
        view_65: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_11, _shape_param_85);  mm_11 = _shape_param_85 = None
        convert_element_type_42: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg33_1, torch.bfloat16);  arg33_1 = None
        convert_element_type_43: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg34_1, torch.bfloat16);  arg34_1 = None
        permute_31: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_43, [1, 0]);  convert_element_type_43 = None
        addmm_4: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_42, view_46, permute_31);  convert_element_type_42 = None
        view_66: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_4, _shape_param_86);  _shape_param_86 = None
        mul_26: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_66, 0.5)
        convert_element_type_44: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_66, torch.float32)
        pow_3: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_44, 3.0);  convert_element_type_44 = None
        mul_27: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_23: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_66, mul_27);  view_66 = mul_27 = None
        mul_28: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_23, 0.7978845608028654);  add_23 = None
        tanh_2: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_28);  mul_28 = None
        add_24: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_29: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_26, add_24);  mul_26 = add_24 = None
        convert_element_type_45: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg35_1, torch.bfloat16);  arg35_1 = None
        convert_element_type_46: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg36_1, torch.bfloat16);  arg36_1 = None
        convert_element_type_47: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        view_67: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_47, _shape_param_87);  convert_element_type_47 = _shape_param_87 = None
        permute_32: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_46, [1, 0]);  convert_element_type_46 = None
        addmm_5: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_45, view_67, permute_32);  convert_element_type_45 = None
        view_68: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_5, _shape_param_88);  addmm_5 = _shape_param_88 = None
        add_25: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_65, view_68);  view_65 = view_68 = None
        add_26: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_25, add_17);  add_25 = add_17 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
        getitem_12: "f32[1, 128, 1]" = var_mean_3[0]
        getitem_13: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        add_27: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_6: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_26, getitem_13);  getitem_13 = None
        mul_30: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
        mul_31: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_30, arg37_1);  arg37_1 = None
        add_28: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_31, arg38_1);  mul_31 = arg38_1 = None
        convert_element_type_48: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg39_1, torch.bfloat16);  arg39_1 = None
        convert_element_type_49: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None
        permute_33: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_48, [1, 0]);  convert_element_type_48 = None
        view_69: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_49, _shape_param_89);  convert_element_type_49 = _shape_param_89 = None
        mm_12: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_69, permute_33)
        view_70: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_12, _shape_param_90);  mm_12 = _shape_param_90 = None
        convert_element_type_50: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg40_1, torch.bfloat16);  arg40_1 = None
        permute_34: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_50, [1, 0]);  convert_element_type_50 = None
        mm_13: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_69, permute_34)
        view_71: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_13, _shape_param_91);  mm_13 = _shape_param_91 = None
        convert_element_type_51: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg41_1, torch.bfloat16);  arg41_1 = None
        permute_35: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_51, [1, 0]);  convert_element_type_51 = None
        mm_14: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_69, permute_35)
        view_72: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_14, _shape_param_92);  mm_14 = _shape_param_92 = None
        view_73: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_70, _shape_param_93);  view_70 = _shape_param_93 = None
        view_74: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_71, _shape_param_94);  view_71 = _shape_param_94 = None
        view_75: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_72, _shape_param_95);  view_72 = _shape_param_95 = None
        permute_36: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        repeat_4: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg42_1, [1, 1, 1]);  arg42_1 = None
        gather_3: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_4, 1, repeat_1);  repeat_4 = None
        convert_element_type_52: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_3, torch.bfloat16);  gather_3 = None
        split_3 = torch.ops.aten.split.Tensor(convert_element_type_52, 32, -1);  convert_element_type_52 = None
        getitem_14: "bf16[1, 128, 32]" = split_3[0]
        getitem_15: "bf16[1, 128, 32]" = split_3[1];  split_3 = None
        slice_25: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_74, 3, 0, 64)
        slice_26: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_74, 3, 64, 9223372036854775807);  view_74 = None
        slice_27: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_73, 3, 0, 64)
        slice_28: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_73, 3, 64, 9223372036854775807);  view_73 = None
        unsqueeze_31: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_14, 2);  getitem_14 = None
        unsqueeze_32: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_31, 4);  unsqueeze_31 = None
        expand_19: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_32, _shape_param_96);  _shape_param_96 = None
        clone_9: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_76: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_9, _shape_param_97);  clone_9 = _shape_param_97 = None
        unsqueeze_33: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_15, 2);  getitem_15 = None
        unsqueeze_34: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_33, 4);  unsqueeze_33 = None
        expand_20: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_34, _shape_param_98);  _shape_param_98 = None
        clone_10: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_77: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_10, _shape_param_99);  clone_10 = _shape_param_99 = None
        mul_32: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_25, view_77)
        slice_29: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_25, 3, 0, 9223372036854775807, 2)
        slice_30: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_25, 3, 1, 9223372036854775807, 2);  slice_25 = None
        neg_6: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_30);  slice_30 = None
        unsqueeze_35: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_6, 4);  neg_6 = None
        unsqueeze_36: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_29, 4);  slice_29 = None
        cat_12: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_35, unsqueeze_36], -1);  unsqueeze_35 = unsqueeze_36 = None
        view_78: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_12, _shape_param_100);  cat_12 = _shape_param_100 = None
        mul_33: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_78, view_76);  view_78 = None
        add_29: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        mul_34: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_27, view_77);  view_77 = None
        slice_31: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_27, 3, 0, 9223372036854775807, 2)
        slice_32: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_27, 3, 1, 9223372036854775807, 2);  slice_27 = None
        neg_7: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_32);  slice_32 = None
        unsqueeze_37: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_7, 4);  neg_7 = None
        unsqueeze_38: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_31, 4);  slice_31 = None
        cat_13: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_37, unsqueeze_38], -1);  unsqueeze_37 = unsqueeze_38 = None
        view_79: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_13, _shape_param_101);  cat_13 = _shape_param_101 = None
        mul_35: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_79, view_76);  view_79 = view_76 = None
        add_30: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None
        cat_14: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_29, slice_26], -1);  add_29 = slice_26 = None
        cat_15: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_30, slice_28], -1);  add_30 = slice_28 = None
        permute_37: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_14, [0, 2, 1, 3]);  cat_14 = None
        permute_38: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_15, [0, 2, 1, 3]);  cat_15 = None
        convert_element_type_53: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_37, torch.float32);  permute_37 = None
        permute_39: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_53, [0, 1, 3, 2]);  convert_element_type_53 = None
        convert_element_type_54: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_39, torch.bfloat16);  permute_39 = None
        convert_element_type_55: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_38, torch.bfloat16);  permute_38 = None
        expand_21: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_55, _shape_param_102);  convert_element_type_55 = _shape_param_102 = None
        view_80: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_21, _shape_param_103);  expand_21 = _shape_param_103 = None
        expand_22: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_54, _shape_param_104);  convert_element_type_54 = _shape_param_104 = None
        view_81: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_22, _shape_param_105);  expand_22 = _shape_param_105 = None
        bmm_6: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_80, view_81)
        view_82: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_106);  bmm_6 = _shape_param_106 = None
        div_6: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_82, 16.0);  view_82 = None
        add_31: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_6, where);  div_6 = None
        amax_3: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_31, [-1], True)
        sub_7: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_31, amax_3);  add_31 = amax_3 = None
        exp_3: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_4: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_56: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16)
        expand_23: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_56, _shape_param_107);  convert_element_type_56 = _shape_param_107 = None
        view_83: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_23, _shape_param_108);  expand_23 = _shape_param_108 = None
        expand_24: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_36, _shape_param_109);  permute_36 = _shape_param_109 = None
        view_84: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_24, _shape_param_110);  expand_24 = _shape_param_110 = None
        bmm_7: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_83, view_84)
        view_85: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_7, _shape_param_111);  bmm_7 = _shape_param_111 = None
        permute_40: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        clone_11: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_86: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_11, _shape_param_112);  clone_11 = _shape_param_112 = None
        convert_element_type_57: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg43_1, torch.bfloat16);  arg43_1 = None
        permute_41: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_57, [1, 0]);  convert_element_type_57 = None
        view_87: "bf16[128, 4096]" = torch.ops.aten.view.default(view_86, _shape_param_113);  view_86 = _shape_param_113 = None
        mm_15: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_87, permute_41)
        view_88: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_15, _shape_param_114);  mm_15 = _shape_param_114 = None
        convert_element_type_58: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg44_1, torch.bfloat16);  arg44_1 = None
        convert_element_type_59: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg45_1, torch.bfloat16);  arg45_1 = None
        permute_42: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_59, [1, 0]);  convert_element_type_59 = None
        addmm_6: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_58, view_69, permute_42);  convert_element_type_58 = None
        view_89: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_6, _shape_param_115);  _shape_param_115 = None
        mul_36: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_89, 0.5)
        convert_element_type_60: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_89, torch.float32)
        pow_4: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_60, 3.0);  convert_element_type_60 = None
        mul_37: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_32: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_89, mul_37);  view_89 = mul_37 = None
        mul_38: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_32, 0.7978845608028654);  add_32 = None
        tanh_3: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_33: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_39: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_36, add_33);  mul_36 = add_33 = None
        convert_element_type_61: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg46_1, torch.bfloat16);  arg46_1 = None
        convert_element_type_62: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg47_1, torch.bfloat16);  arg47_1 = None
        convert_element_type_63: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None
        view_90: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_63, _shape_param_116);  convert_element_type_63 = _shape_param_116 = None
        permute_43: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_62, [1, 0]);  convert_element_type_62 = None
        addmm_7: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_61, view_90, permute_43);  convert_element_type_61 = None
        view_91: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_7, _shape_param_117);  addmm_7 = _shape_param_117 = None
        add_34: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_88, view_91);  view_88 = view_91 = None
        add_35: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_34, add_26);  add_34 = add_26 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_16: "f32[1, 128, 1]" = var_mean_4[0]
        getitem_17: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        add_36: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_8: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_35, getitem_17);  getitem_17 = None
        mul_40: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_4);  sub_8 = None
        mul_41: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_40, arg48_1);  arg48_1 = None
        add_37: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_41, arg49_1);  mul_41 = arg49_1 = None
        convert_element_type_64: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg50_1, torch.bfloat16);  arg50_1 = None
        convert_element_type_65: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16);  add_37 = None
        permute_44: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_64, [1, 0]);  convert_element_type_64 = None
        view_92: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_65, _shape_param_118);  convert_element_type_65 = _shape_param_118 = None
        mm_16: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_92, permute_44)
        view_93: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_16, _shape_param_119);  mm_16 = _shape_param_119 = None
        convert_element_type_66: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg51_1, torch.bfloat16);  arg51_1 = None
        permute_45: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_66, [1, 0]);  convert_element_type_66 = None
        mm_17: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_92, permute_45)
        view_94: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_17, _shape_param_120);  mm_17 = _shape_param_120 = None
        convert_element_type_67: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg52_1, torch.bfloat16);  arg52_1 = None
        permute_46: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_67, [1, 0]);  convert_element_type_67 = None
        mm_18: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_92, permute_46)
        view_95: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_18, _shape_param_121);  mm_18 = _shape_param_121 = None
        view_96: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_93, _shape_param_122);  view_93 = _shape_param_122 = None
        view_97: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_94, _shape_param_123);  view_94 = _shape_param_123 = None
        view_98: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_95, _shape_param_124);  view_95 = _shape_param_124 = None
        permute_47: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        repeat_5: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg53_1, [1, 1, 1]);  arg53_1 = None
        gather_4: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_5, 1, repeat_1);  repeat_5 = None
        convert_element_type_68: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_4, torch.bfloat16);  gather_4 = None
        split_4 = torch.ops.aten.split.Tensor(convert_element_type_68, 32, -1);  convert_element_type_68 = None
        getitem_18: "bf16[1, 128, 32]" = split_4[0]
        getitem_19: "bf16[1, 128, 32]" = split_4[1];  split_4 = None
        slice_33: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_97, 3, 0, 64)
        slice_34: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_97, 3, 64, 9223372036854775807);  view_97 = None
        slice_35: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_96, 3, 0, 64)
        slice_36: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_96, 3, 64, 9223372036854775807);  view_96 = None
        unsqueeze_39: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_18, 2);  getitem_18 = None
        unsqueeze_40: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_39, 4);  unsqueeze_39 = None
        expand_25: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_40, _shape_param_125);  _shape_param_125 = None
        clone_12: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_99: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_126);  clone_12 = _shape_param_126 = None
        unsqueeze_41: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_19, 2);  getitem_19 = None
        unsqueeze_42: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_41, 4);  unsqueeze_41 = None
        expand_26: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_42, _shape_param_127);  _shape_param_127 = None
        clone_13: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_100: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_13, _shape_param_128);  clone_13 = _shape_param_128 = None
        mul_42: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_33, view_100)
        slice_37: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_33, 3, 0, 9223372036854775807, 2)
        slice_38: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_33, 3, 1, 9223372036854775807, 2);  slice_33 = None
        neg_8: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_38);  slice_38 = None
        unsqueeze_43: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_8, 4);  neg_8 = None
        unsqueeze_44: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_37, 4);  slice_37 = None
        cat_16: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_43, unsqueeze_44], -1);  unsqueeze_43 = unsqueeze_44 = None
        view_101: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_16, _shape_param_129);  cat_16 = _shape_param_129 = None
        mul_43: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_101, view_99);  view_101 = None
        add_38: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_42, mul_43);  mul_42 = mul_43 = None
        mul_44: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_35, view_100);  view_100 = None
        slice_39: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_35, 3, 0, 9223372036854775807, 2)
        slice_40: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_35, 3, 1, 9223372036854775807, 2);  slice_35 = None
        neg_9: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_40);  slice_40 = None
        unsqueeze_45: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_9, 4);  neg_9 = None
        unsqueeze_46: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_39, 4);  slice_39 = None
        cat_17: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_45, unsqueeze_46], -1);  unsqueeze_45 = unsqueeze_46 = None
        view_102: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_17, _shape_param_130);  cat_17 = _shape_param_130 = None
        mul_45: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_102, view_99);  view_102 = view_99 = None
        add_39: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
        cat_18: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_38, slice_34], -1);  add_38 = slice_34 = None
        cat_19: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_39, slice_36], -1);  add_39 = slice_36 = None
        permute_48: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_18, [0, 2, 1, 3]);  cat_18 = None
        permute_49: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_19, [0, 2, 1, 3]);  cat_19 = None
        convert_element_type_69: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_48, torch.float32);  permute_48 = None
        permute_50: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_69, [0, 1, 3, 2]);  convert_element_type_69 = None
        convert_element_type_70: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_50, torch.bfloat16);  permute_50 = None
        convert_element_type_71: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_49, torch.bfloat16);  permute_49 = None
        expand_27: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_71, _shape_param_131);  convert_element_type_71 = _shape_param_131 = None
        view_103: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_27, _shape_param_132);  expand_27 = _shape_param_132 = None
        expand_28: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_70, _shape_param_133);  convert_element_type_70 = _shape_param_133 = None
        view_104: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_28, _shape_param_134);  expand_28 = _shape_param_134 = None
        bmm_8: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_103, view_104)
        view_105: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_8, _shape_param_135);  bmm_8 = _shape_param_135 = None
        div_8: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_105, 16.0);  view_105 = None
        add_40: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_8, where);  div_8 = None
        amax_4: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_40, [-1], True)
        sub_9: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_40, amax_4);  add_40 = amax_4 = None
        exp_4: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_5: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_9: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_72: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16)
        expand_29: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_72, _shape_param_136);  convert_element_type_72 = _shape_param_136 = None
        view_106: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_29, _shape_param_137);  expand_29 = _shape_param_137 = None
        expand_30: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_47, _shape_param_138);  permute_47 = _shape_param_138 = None
        view_107: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_30, _shape_param_139);  expand_30 = _shape_param_139 = None
        bmm_9: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_106, view_107)
        view_108: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_9, _shape_param_140);  bmm_9 = _shape_param_140 = None
        permute_51: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
        clone_14: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_109: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_14, _shape_param_141);  clone_14 = _shape_param_141 = None
        convert_element_type_73: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg54_1, torch.bfloat16);  arg54_1 = None
        permute_52: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_73, [1, 0]);  convert_element_type_73 = None
        view_110: "bf16[128, 4096]" = torch.ops.aten.view.default(view_109, _shape_param_142);  view_109 = _shape_param_142 = None
        mm_19: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_110, permute_52)
        view_111: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_19, _shape_param_143);  mm_19 = _shape_param_143 = None
        convert_element_type_74: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg55_1, torch.bfloat16);  arg55_1 = None
        convert_element_type_75: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg56_1, torch.bfloat16);  arg56_1 = None
        permute_53: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_75, [1, 0]);  convert_element_type_75 = None
        addmm_8: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_74, view_92, permute_53);  convert_element_type_74 = None
        view_112: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_8, _shape_param_144);  _shape_param_144 = None
        mul_46: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_112, 0.5)
        convert_element_type_76: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_112, torch.float32)
        pow_5: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_76, 3.0);  convert_element_type_76 = None
        mul_47: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_41: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_112, mul_47);  view_112 = mul_47 = None
        mul_48: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_41, 0.7978845608028654);  add_41 = None
        tanh_4: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
        add_42: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_49: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_46, add_42);  mul_46 = add_42 = None
        convert_element_type_77: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg57_1, torch.bfloat16);  arg57_1 = None
        convert_element_type_78: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg58_1, torch.bfloat16);  arg58_1 = None
        convert_element_type_79: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        view_113: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_79, _shape_param_145);  convert_element_type_79 = _shape_param_145 = None
        permute_54: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_78, [1, 0]);  convert_element_type_78 = None
        addmm_9: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_77, view_113, permute_54);  convert_element_type_77 = None
        view_114: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_9, _shape_param_146);  addmm_9 = _shape_param_146 = None
        add_43: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_111, view_114);  view_111 = view_114 = None
        add_44: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_43, add_35);  add_43 = add_35 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_20: "f32[1, 128, 1]" = var_mean_5[0]
        getitem_21: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        add_45: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_10: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_44, getitem_21);  getitem_21 = None
        mul_50: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_5);  sub_10 = None
        mul_51: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_50, arg59_1);  arg59_1 = None
        add_46: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_51, arg60_1);  mul_51 = arg60_1 = None
        convert_element_type_80: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg61_1, torch.bfloat16);  arg61_1 = None
        convert_element_type_81: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None
        permute_55: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_80, [1, 0]);  convert_element_type_80 = None
        view_115: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_81, _shape_param_147);  convert_element_type_81 = _shape_param_147 = None
        mm_20: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_115, permute_55)
        view_116: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_20, _shape_param_148);  mm_20 = _shape_param_148 = None
        convert_element_type_82: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg62_1, torch.bfloat16);  arg62_1 = None
        permute_56: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_82, [1, 0]);  convert_element_type_82 = None
        mm_21: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_115, permute_56)
        view_117: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_21, _shape_param_149);  mm_21 = _shape_param_149 = None
        convert_element_type_83: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg63_1, torch.bfloat16);  arg63_1 = None
        permute_57: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_83, [1, 0]);  convert_element_type_83 = None
        mm_22: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_115, permute_57)
        view_118: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_22, _shape_param_150);  mm_22 = _shape_param_150 = None
        view_119: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_116, _shape_param_151);  view_116 = _shape_param_151 = None
        view_120: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_117, _shape_param_152);  view_117 = _shape_param_152 = None
        view_121: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_118, _shape_param_153);  view_118 = _shape_param_153 = None
        permute_58: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
        repeat_6: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg64_1, [1, 1, 1]);  arg64_1 = None
        gather_5: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_6, 1, repeat_1);  repeat_6 = None
        convert_element_type_84: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_5, torch.bfloat16);  gather_5 = None
        split_5 = torch.ops.aten.split.Tensor(convert_element_type_84, 32, -1);  convert_element_type_84 = None
        getitem_22: "bf16[1, 128, 32]" = split_5[0]
        getitem_23: "bf16[1, 128, 32]" = split_5[1];  split_5 = None
        slice_41: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_120, 3, 0, 64)
        slice_42: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_120, 3, 64, 9223372036854775807);  view_120 = None
        slice_43: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_119, 3, 0, 64)
        slice_44: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_119, 3, 64, 9223372036854775807);  view_119 = None
        unsqueeze_47: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_22, 2);  getitem_22 = None
        unsqueeze_48: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_47, 4);  unsqueeze_47 = None
        expand_31: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_48, _shape_param_154);  _shape_param_154 = None
        clone_15: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_122: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_15, _shape_param_155);  clone_15 = _shape_param_155 = None
        unsqueeze_49: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_23, 2);  getitem_23 = None
        unsqueeze_50: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_49, 4);  unsqueeze_49 = None
        expand_32: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_50, _shape_param_156);  _shape_param_156 = None
        clone_16: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_123: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_157);  clone_16 = _shape_param_157 = None
        mul_52: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_41, view_123)
        slice_45: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_41, 3, 0, 9223372036854775807, 2)
        slice_46: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_41, 3, 1, 9223372036854775807, 2);  slice_41 = None
        neg_10: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_46);  slice_46 = None
        unsqueeze_51: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_10, 4);  neg_10 = None
        unsqueeze_52: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_45, 4);  slice_45 = None
        cat_20: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_51, unsqueeze_52], -1);  unsqueeze_51 = unsqueeze_52 = None
        view_124: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_20, _shape_param_158);  cat_20 = _shape_param_158 = None
        mul_53: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_124, view_122);  view_124 = None
        add_47: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
        mul_54: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_43, view_123);  view_123 = None
        slice_47: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_43, 3, 0, 9223372036854775807, 2)
        slice_48: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_43, 3, 1, 9223372036854775807, 2);  slice_43 = None
        neg_11: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_48);  slice_48 = None
        unsqueeze_53: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_11, 4);  neg_11 = None
        unsqueeze_54: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_47, 4);  slice_47 = None
        cat_21: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_53, unsqueeze_54], -1);  unsqueeze_53 = unsqueeze_54 = None
        view_125: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_21, _shape_param_159);  cat_21 = _shape_param_159 = None
        mul_55: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_125, view_122);  view_125 = view_122 = None
        add_48: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_54, mul_55);  mul_54 = mul_55 = None
        cat_22: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_47, slice_42], -1);  add_47 = slice_42 = None
        cat_23: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_48, slice_44], -1);  add_48 = slice_44 = None
        permute_59: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_22, [0, 2, 1, 3]);  cat_22 = None
        permute_60: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_23, [0, 2, 1, 3]);  cat_23 = None
        convert_element_type_85: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_59, torch.float32);  permute_59 = None
        permute_61: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_85, [0, 1, 3, 2]);  convert_element_type_85 = None
        convert_element_type_86: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_61, torch.bfloat16);  permute_61 = None
        convert_element_type_87: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_60, torch.bfloat16);  permute_60 = None
        expand_33: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_87, _shape_param_160);  convert_element_type_87 = _shape_param_160 = None
        view_126: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_33, _shape_param_161);  expand_33 = _shape_param_161 = None
        expand_34: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_86, _shape_param_162);  convert_element_type_86 = _shape_param_162 = None
        view_127: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_34, _shape_param_163);  expand_34 = _shape_param_163 = None
        bmm_10: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_126, view_127)
        view_128: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_164);  bmm_10 = _shape_param_164 = None
        div_10: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_128, 16.0);  view_128 = None
        add_49: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_10, where);  div_10 = None
        amax_5: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_49, [-1], True)
        sub_11: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_49, amax_5);  add_49 = amax_5 = None
        exp_5: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_6: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_11: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_88: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16)
        expand_35: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_88, _shape_param_165);  convert_element_type_88 = _shape_param_165 = None
        view_129: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_35, _shape_param_166);  expand_35 = _shape_param_166 = None
        expand_36: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_58, _shape_param_167);  permute_58 = _shape_param_167 = None
        view_130: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_36, _shape_param_168);  expand_36 = _shape_param_168 = None
        bmm_11: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_129, view_130)
        view_131: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_11, _shape_param_169);  bmm_11 = _shape_param_169 = None
        permute_62: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
        clone_17: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_132: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_17, _shape_param_170);  clone_17 = _shape_param_170 = None
        convert_element_type_89: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg65_1, torch.bfloat16);  arg65_1 = None
        permute_63: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_89, [1, 0]);  convert_element_type_89 = None
        view_133: "bf16[128, 4096]" = torch.ops.aten.view.default(view_132, _shape_param_171);  view_132 = _shape_param_171 = None
        mm_23: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_133, permute_63)
        view_134: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_23, _shape_param_172);  mm_23 = _shape_param_172 = None
        convert_element_type_90: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg66_1, torch.bfloat16);  arg66_1 = None
        convert_element_type_91: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg67_1, torch.bfloat16);  arg67_1 = None
        permute_64: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_91, [1, 0]);  convert_element_type_91 = None
        addmm_10: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_90, view_115, permute_64);  convert_element_type_90 = None
        view_135: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_10, _shape_param_173);  _shape_param_173 = None
        mul_56: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_135, 0.5)
        convert_element_type_92: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_135, torch.float32)
        pow_6: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_92, 3.0);  convert_element_type_92 = None
        mul_57: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_50: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_135, mul_57);  view_135 = mul_57 = None
        mul_58: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_50, 0.7978845608028654);  add_50 = None
        tanh_5: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_51: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_59: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_56, add_51);  mul_56 = add_51 = None
        convert_element_type_93: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg68_1, torch.bfloat16);  arg68_1 = None
        convert_element_type_94: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg69_1, torch.bfloat16);  arg69_1 = None
        convert_element_type_95: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_59, torch.bfloat16);  mul_59 = None
        view_136: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_95, _shape_param_174);  convert_element_type_95 = _shape_param_174 = None
        permute_65: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_94, [1, 0]);  convert_element_type_94 = None
        addmm_11: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_93, view_136, permute_65);  convert_element_type_93 = None
        view_137: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_11, _shape_param_175);  addmm_11 = _shape_param_175 = None
        add_52: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_134, view_137);  view_134 = view_137 = None
        add_53: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_52, add_44);  add_52 = add_44 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_24: "f32[1, 128, 1]" = var_mean_6[0]
        getitem_25: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        add_54: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_12: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_53, getitem_25);  getitem_25 = None
        mul_60: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_6);  sub_12 = None
        mul_61: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_60, arg70_1);  arg70_1 = None
        add_55: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_61, arg71_1);  mul_61 = arg71_1 = None
        convert_element_type_96: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg72_1, torch.bfloat16);  arg72_1 = None
        convert_element_type_97: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None
        permute_66: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_96, [1, 0]);  convert_element_type_96 = None
        view_138: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_97, _shape_param_176);  convert_element_type_97 = _shape_param_176 = None
        mm_24: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_138, permute_66)
        view_139: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_24, _shape_param_177);  mm_24 = _shape_param_177 = None
        convert_element_type_98: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg73_1, torch.bfloat16);  arg73_1 = None
        permute_67: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_98, [1, 0]);  convert_element_type_98 = None
        mm_25: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_138, permute_67)
        view_140: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_25, _shape_param_178);  mm_25 = _shape_param_178 = None
        convert_element_type_99: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg74_1, torch.bfloat16);  arg74_1 = None
        permute_68: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_99, [1, 0]);  convert_element_type_99 = None
        mm_26: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_138, permute_68)
        view_141: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_26, _shape_param_179);  mm_26 = _shape_param_179 = None
        view_142: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_139, _shape_param_180);  view_139 = _shape_param_180 = None
        view_143: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_140, _shape_param_181);  view_140 = _shape_param_181 = None
        view_144: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_141, _shape_param_182);  view_141 = _shape_param_182 = None
        permute_69: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        repeat_7: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg75_1, [1, 1, 1]);  arg75_1 = None
        gather_6: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_7, 1, repeat_1);  repeat_7 = None
        convert_element_type_100: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_6, torch.bfloat16);  gather_6 = None
        split_6 = torch.ops.aten.split.Tensor(convert_element_type_100, 32, -1);  convert_element_type_100 = None
        getitem_26: "bf16[1, 128, 32]" = split_6[0]
        getitem_27: "bf16[1, 128, 32]" = split_6[1];  split_6 = None
        slice_49: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_143, 3, 0, 64)
        slice_50: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_143, 3, 64, 9223372036854775807);  view_143 = None
        slice_51: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_142, 3, 0, 64)
        slice_52: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_142, 3, 64, 9223372036854775807);  view_142 = None
        unsqueeze_55: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_26, 2);  getitem_26 = None
        unsqueeze_56: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_55, 4);  unsqueeze_55 = None
        expand_37: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_56, _shape_param_183);  _shape_param_183 = None
        clone_18: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_145: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_18, _shape_param_184);  clone_18 = _shape_param_184 = None
        unsqueeze_57: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_27, 2);  getitem_27 = None
        unsqueeze_58: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_57, 4);  unsqueeze_57 = None
        expand_38: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_58, _shape_param_185);  _shape_param_185 = None
        clone_19: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_146: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_19, _shape_param_186);  clone_19 = _shape_param_186 = None
        mul_62: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_49, view_146)
        slice_53: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_49, 3, 0, 9223372036854775807, 2)
        slice_54: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_49, 3, 1, 9223372036854775807, 2);  slice_49 = None
        neg_12: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_54);  slice_54 = None
        unsqueeze_59: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_12, 4);  neg_12 = None
        unsqueeze_60: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_53, 4);  slice_53 = None
        cat_24: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_59, unsqueeze_60], -1);  unsqueeze_59 = unsqueeze_60 = None
        view_147: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_24, _shape_param_187);  cat_24 = _shape_param_187 = None
        mul_63: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_147, view_145);  view_147 = None
        add_56: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
        mul_64: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_51, view_146);  view_146 = None
        slice_55: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_51, 3, 0, 9223372036854775807, 2)
        slice_56: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_51, 3, 1, 9223372036854775807, 2);  slice_51 = None
        neg_13: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_56);  slice_56 = None
        unsqueeze_61: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_13, 4);  neg_13 = None
        unsqueeze_62: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_55, 4);  slice_55 = None
        cat_25: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_61, unsqueeze_62], -1);  unsqueeze_61 = unsqueeze_62 = None
        view_148: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_25, _shape_param_188);  cat_25 = _shape_param_188 = None
        mul_65: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_148, view_145);  view_148 = view_145 = None
        add_57: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        cat_26: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_56, slice_50], -1);  add_56 = slice_50 = None
        cat_27: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_57, slice_52], -1);  add_57 = slice_52 = None
        permute_70: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_26, [0, 2, 1, 3]);  cat_26 = None
        permute_71: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_27, [0, 2, 1, 3]);  cat_27 = None
        convert_element_type_101: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_70, torch.float32);  permute_70 = None
        permute_72: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_101, [0, 1, 3, 2]);  convert_element_type_101 = None
        convert_element_type_102: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_72, torch.bfloat16);  permute_72 = None
        convert_element_type_103: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_71, torch.bfloat16);  permute_71 = None
        expand_39: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_103, _shape_param_189);  convert_element_type_103 = _shape_param_189 = None
        view_149: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_39, _shape_param_190);  expand_39 = _shape_param_190 = None
        expand_40: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_102, _shape_param_191);  convert_element_type_102 = _shape_param_191 = None
        view_150: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_40, _shape_param_192);  expand_40 = _shape_param_192 = None
        bmm_12: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_149, view_150)
        view_151: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_12, _shape_param_193);  bmm_12 = _shape_param_193 = None
        div_12: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_151, 16.0);  view_151 = None
        add_58: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_12, where);  div_12 = None
        amax_6: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_58, [-1], True)
        sub_13: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_58, amax_6);  add_58 = amax_6 = None
        exp_6: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_7: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_13: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_104: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16)
        expand_41: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_104, _shape_param_194);  convert_element_type_104 = _shape_param_194 = None
        view_152: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_41, _shape_param_195);  expand_41 = _shape_param_195 = None
        expand_42: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_69, _shape_param_196);  permute_69 = _shape_param_196 = None
        view_153: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_42, _shape_param_197);  expand_42 = _shape_param_197 = None
        bmm_13: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_152, view_153)
        view_154: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_13, _shape_param_198);  bmm_13 = _shape_param_198 = None
        permute_73: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
        clone_20: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_155: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_20, _shape_param_199);  clone_20 = _shape_param_199 = None
        convert_element_type_105: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg76_1, torch.bfloat16);  arg76_1 = None
        permute_74: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_105, [1, 0]);  convert_element_type_105 = None
        view_156: "bf16[128, 4096]" = torch.ops.aten.view.default(view_155, _shape_param_200);  view_155 = _shape_param_200 = None
        mm_27: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_156, permute_74)
        view_157: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_27, _shape_param_201);  mm_27 = _shape_param_201 = None
        convert_element_type_106: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg77_1, torch.bfloat16);  arg77_1 = None
        convert_element_type_107: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg78_1, torch.bfloat16);  arg78_1 = None
        permute_75: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_107, [1, 0]);  convert_element_type_107 = None
        addmm_12: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_106, view_138, permute_75);  convert_element_type_106 = None
        view_158: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_12, _shape_param_202);  _shape_param_202 = None
        mul_66: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_158, 0.5)
        convert_element_type_108: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_158, torch.float32)
        pow_7: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_108, 3.0);  convert_element_type_108 = None
        mul_67: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_59: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_158, mul_67);  view_158 = mul_67 = None
        mul_68: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_59, 0.7978845608028654);  add_59 = None
        tanh_6: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_68);  mul_68 = None
        add_60: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_69: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_66, add_60);  mul_66 = add_60 = None
        convert_element_type_109: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg79_1, torch.bfloat16);  arg79_1 = None
        convert_element_type_110: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg80_1, torch.bfloat16);  arg80_1 = None
        convert_element_type_111: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        view_159: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_111, _shape_param_203);  convert_element_type_111 = _shape_param_203 = None
        permute_76: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_110, [1, 0]);  convert_element_type_110 = None
        addmm_13: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_109, view_159, permute_76);  convert_element_type_109 = None
        view_160: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_13, _shape_param_204);  addmm_13 = _shape_param_204 = None
        add_61: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_157, view_160);  view_157 = view_160 = None
        add_62: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_61, add_53);  add_61 = add_53 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
        getitem_28: "f32[1, 128, 1]" = var_mean_7[0]
        getitem_29: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        add_63: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        sub_14: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_62, getitem_29);  getitem_29 = None
        mul_70: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_7);  sub_14 = None
        mul_71: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_70, arg81_1);  arg81_1 = None
        add_64: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_71, arg82_1);  mul_71 = arg82_1 = None
        convert_element_type_112: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg83_1, torch.bfloat16);  arg83_1 = None
        convert_element_type_113: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16);  add_64 = None
        permute_77: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_112, [1, 0]);  convert_element_type_112 = None
        view_161: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_113, _shape_param_205);  convert_element_type_113 = _shape_param_205 = None
        mm_28: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_161, permute_77)
        view_162: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_28, _shape_param_206);  mm_28 = _shape_param_206 = None
        convert_element_type_114: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg84_1, torch.bfloat16);  arg84_1 = None
        permute_78: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_114, [1, 0]);  convert_element_type_114 = None
        mm_29: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_161, permute_78)
        view_163: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_29, _shape_param_207);  mm_29 = _shape_param_207 = None
        convert_element_type_115: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg85_1, torch.bfloat16);  arg85_1 = None
        permute_79: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_115, [1, 0]);  convert_element_type_115 = None
        mm_30: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_161, permute_79)
        view_164: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_30, _shape_param_208);  mm_30 = _shape_param_208 = None
        view_165: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_162, _shape_param_209);  view_162 = _shape_param_209 = None
        view_166: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_163, _shape_param_210);  view_163 = _shape_param_210 = None
        view_167: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_164, _shape_param_211);  view_164 = _shape_param_211 = None
        permute_80: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
        repeat_8: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg86_1, [1, 1, 1]);  arg86_1 = None
        gather_7: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_8, 1, repeat_1);  repeat_8 = None
        convert_element_type_116: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_7, torch.bfloat16);  gather_7 = None
        split_7 = torch.ops.aten.split.Tensor(convert_element_type_116, 32, -1);  convert_element_type_116 = None
        getitem_30: "bf16[1, 128, 32]" = split_7[0]
        getitem_31: "bf16[1, 128, 32]" = split_7[1];  split_7 = None
        slice_57: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_166, 3, 0, 64)
        slice_58: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_166, 3, 64, 9223372036854775807);  view_166 = None
        slice_59: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_165, 3, 0, 64)
        slice_60: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_165, 3, 64, 9223372036854775807);  view_165 = None
        unsqueeze_63: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_30, 2);  getitem_30 = None
        unsqueeze_64: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 4);  unsqueeze_63 = None
        expand_43: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_64, _shape_param_212);  _shape_param_212 = None
        clone_21: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_168: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_21, _shape_param_213);  clone_21 = _shape_param_213 = None
        unsqueeze_65: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_31, 2);  getitem_31 = None
        unsqueeze_66: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_65, 4);  unsqueeze_65 = None
        expand_44: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_66, _shape_param_214);  _shape_param_214 = None
        clone_22: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_169: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_22, _shape_param_215);  clone_22 = _shape_param_215 = None
        mul_72: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_57, view_169)
        slice_61: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_57, 3, 0, 9223372036854775807, 2)
        slice_62: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_57, 3, 1, 9223372036854775807, 2);  slice_57 = None
        neg_14: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_62);  slice_62 = None
        unsqueeze_67: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_14, 4);  neg_14 = None
        unsqueeze_68: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_61, 4);  slice_61 = None
        cat_28: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_67, unsqueeze_68], -1);  unsqueeze_67 = unsqueeze_68 = None
        view_170: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_28, _shape_param_216);  cat_28 = _shape_param_216 = None
        mul_73: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_170, view_168);  view_170 = None
        add_65: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
        mul_74: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_59, view_169);  view_169 = None
        slice_63: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_59, 3, 0, 9223372036854775807, 2)
        slice_64: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_59, 3, 1, 9223372036854775807, 2);  slice_59 = None
        neg_15: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_64);  slice_64 = None
        unsqueeze_69: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_15, 4);  neg_15 = None
        unsqueeze_70: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_63, 4);  slice_63 = None
        cat_29: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_69, unsqueeze_70], -1);  unsqueeze_69 = unsqueeze_70 = None
        view_171: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_29, _shape_param_217);  cat_29 = _shape_param_217 = None
        mul_75: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_171, view_168);  view_171 = view_168 = None
        add_66: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        cat_30: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_65, slice_58], -1);  add_65 = slice_58 = None
        cat_31: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_66, slice_60], -1);  add_66 = slice_60 = None
        permute_81: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_30, [0, 2, 1, 3]);  cat_30 = None
        permute_82: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_31, [0, 2, 1, 3]);  cat_31 = None
        convert_element_type_117: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_81, torch.float32);  permute_81 = None
        permute_83: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_117, [0, 1, 3, 2]);  convert_element_type_117 = None
        convert_element_type_118: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_83, torch.bfloat16);  permute_83 = None
        convert_element_type_119: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_82, torch.bfloat16);  permute_82 = None
        expand_45: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_119, _shape_param_218);  convert_element_type_119 = _shape_param_218 = None
        view_172: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_45, _shape_param_219);  expand_45 = _shape_param_219 = None
        expand_46: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_118, _shape_param_220);  convert_element_type_118 = _shape_param_220 = None
        view_173: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_46, _shape_param_221);  expand_46 = _shape_param_221 = None
        bmm_14: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_172, view_173)
        view_174: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_14, _shape_param_222);  bmm_14 = _shape_param_222 = None
        div_14: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_174, 16.0);  view_174 = None
        add_67: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_14, where);  div_14 = None
        amax_7: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_67, [-1], True)
        sub_15: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_67, amax_7);  add_67 = amax_7 = None
        exp_7: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_8: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_15: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_120: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16)
        expand_47: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_120, _shape_param_223);  convert_element_type_120 = _shape_param_223 = None
        view_175: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_47, _shape_param_224);  expand_47 = _shape_param_224 = None
        expand_48: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_80, _shape_param_225);  permute_80 = _shape_param_225 = None
        view_176: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_48, _shape_param_226);  expand_48 = _shape_param_226 = None
        bmm_15: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_175, view_176)
        view_177: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_15, _shape_param_227);  bmm_15 = _shape_param_227 = None
        permute_84: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None
        clone_23: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_178: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_23, _shape_param_228);  clone_23 = _shape_param_228 = None
        convert_element_type_121: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg87_1, torch.bfloat16);  arg87_1 = None
        permute_85: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_121, [1, 0]);  convert_element_type_121 = None
        view_179: "bf16[128, 4096]" = torch.ops.aten.view.default(view_178, _shape_param_229);  view_178 = _shape_param_229 = None
        mm_31: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_179, permute_85)
        view_180: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_31, _shape_param_230);  mm_31 = _shape_param_230 = None
        convert_element_type_122: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg88_1, torch.bfloat16);  arg88_1 = None
        convert_element_type_123: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg89_1, torch.bfloat16);  arg89_1 = None
        permute_86: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_123, [1, 0]);  convert_element_type_123 = None
        addmm_14: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_122, view_161, permute_86);  convert_element_type_122 = None
        view_181: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_14, _shape_param_231);  _shape_param_231 = None
        mul_76: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_181, 0.5)
        convert_element_type_124: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_181, torch.float32)
        pow_8: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_124, 3.0);  convert_element_type_124 = None
        mul_77: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_68: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_181, mul_77);  view_181 = mul_77 = None
        mul_78: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_68, 0.7978845608028654);  add_68 = None
        tanh_7: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_69: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_79: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_76, add_69);  mul_76 = add_69 = None
        convert_element_type_125: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg90_1, torch.bfloat16);  arg90_1 = None
        convert_element_type_126: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg91_1, torch.bfloat16);  arg91_1 = None
        convert_element_type_127: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None
        view_182: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_127, _shape_param_232);  convert_element_type_127 = _shape_param_232 = None
        permute_87: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_126, [1, 0]);  convert_element_type_126 = None
        addmm_15: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_125, view_182, permute_87);  convert_element_type_125 = None
        view_183: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_15, _shape_param_233);  addmm_15 = _shape_param_233 = None
        add_70: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_180, view_183);  view_180 = view_183 = None
        add_71: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_70, add_62);  add_70 = add_62 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_32: "f32[1, 128, 1]" = var_mean_8[0]
        getitem_33: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        add_72: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_16: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_71, getitem_33);  getitem_33 = None
        mul_80: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_8);  sub_16 = None
        mul_81: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_80, arg92_1);  arg92_1 = None
        add_73: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_81, arg93_1);  mul_81 = arg93_1 = None
        convert_element_type_128: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg94_1, torch.bfloat16);  arg94_1 = None
        convert_element_type_129: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16);  add_73 = None
        permute_88: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_128, [1, 0]);  convert_element_type_128 = None
        view_184: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_129, _shape_param_234);  convert_element_type_129 = _shape_param_234 = None
        mm_32: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_184, permute_88)
        view_185: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_32, _shape_param_235);  mm_32 = _shape_param_235 = None
        convert_element_type_130: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg95_1, torch.bfloat16);  arg95_1 = None
        permute_89: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_130, [1, 0]);  convert_element_type_130 = None
        mm_33: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_184, permute_89)
        view_186: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_33, _shape_param_236);  mm_33 = _shape_param_236 = None
        convert_element_type_131: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg96_1, torch.bfloat16);  arg96_1 = None
        permute_90: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_131, [1, 0]);  convert_element_type_131 = None
        mm_34: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_184, permute_90)
        view_187: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_34, _shape_param_237);  mm_34 = _shape_param_237 = None
        view_188: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_185, _shape_param_238);  view_185 = _shape_param_238 = None
        view_189: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_186, _shape_param_239);  view_186 = _shape_param_239 = None
        view_190: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_187, _shape_param_240);  view_187 = _shape_param_240 = None
        permute_91: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        repeat_9: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg97_1, [1, 1, 1]);  arg97_1 = None
        gather_8: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_9, 1, repeat_1);  repeat_9 = None
        convert_element_type_132: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_8, torch.bfloat16);  gather_8 = None
        split_8 = torch.ops.aten.split.Tensor(convert_element_type_132, 32, -1);  convert_element_type_132 = None
        getitem_34: "bf16[1, 128, 32]" = split_8[0]
        getitem_35: "bf16[1, 128, 32]" = split_8[1];  split_8 = None
        slice_65: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_189, 3, 0, 64)
        slice_66: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_189, 3, 64, 9223372036854775807);  view_189 = None
        slice_67: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_188, 3, 0, 64)
        slice_68: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_188, 3, 64, 9223372036854775807);  view_188 = None
        unsqueeze_71: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_34, 2);  getitem_34 = None
        unsqueeze_72: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_71, 4);  unsqueeze_71 = None
        expand_49: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_72, _shape_param_241);  _shape_param_241 = None
        clone_24: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_191: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_242);  clone_24 = _shape_param_242 = None
        unsqueeze_73: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_35, 2);  getitem_35 = None
        unsqueeze_74: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_73, 4);  unsqueeze_73 = None
        expand_50: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_74, _shape_param_243);  _shape_param_243 = None
        clone_25: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_192: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_25, _shape_param_244);  clone_25 = _shape_param_244 = None
        mul_82: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_65, view_192)
        slice_69: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_65, 3, 0, 9223372036854775807, 2)
        slice_70: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_65, 3, 1, 9223372036854775807, 2);  slice_65 = None
        neg_16: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_70);  slice_70 = None
        unsqueeze_75: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_16, 4);  neg_16 = None
        unsqueeze_76: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_69, 4);  slice_69 = None
        cat_32: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_75, unsqueeze_76], -1);  unsqueeze_75 = unsqueeze_76 = None
        view_193: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_32, _shape_param_245);  cat_32 = _shape_param_245 = None
        mul_83: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_193, view_191);  view_193 = None
        add_74: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None
        mul_84: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_67, view_192);  view_192 = None
        slice_71: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_67, 3, 0, 9223372036854775807, 2)
        slice_72: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_67, 3, 1, 9223372036854775807, 2);  slice_67 = None
        neg_17: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_72);  slice_72 = None
        unsqueeze_77: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_17, 4);  neg_17 = None
        unsqueeze_78: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_71, 4);  slice_71 = None
        cat_33: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_77, unsqueeze_78], -1);  unsqueeze_77 = unsqueeze_78 = None
        view_194: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_33, _shape_param_246);  cat_33 = _shape_param_246 = None
        mul_85: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_194, view_191);  view_194 = view_191 = None
        add_75: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None
        cat_34: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_74, slice_66], -1);  add_74 = slice_66 = None
        cat_35: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_75, slice_68], -1);  add_75 = slice_68 = None
        permute_92: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_34, [0, 2, 1, 3]);  cat_34 = None
        permute_93: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_35, [0, 2, 1, 3]);  cat_35 = None
        convert_element_type_133: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_92, torch.float32);  permute_92 = None
        permute_94: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_133, [0, 1, 3, 2]);  convert_element_type_133 = None
        convert_element_type_134: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_94, torch.bfloat16);  permute_94 = None
        convert_element_type_135: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_93, torch.bfloat16);  permute_93 = None
        expand_51: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_135, _shape_param_247);  convert_element_type_135 = _shape_param_247 = None
        view_195: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_51, _shape_param_248);  expand_51 = _shape_param_248 = None
        expand_52: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_134, _shape_param_249);  convert_element_type_134 = _shape_param_249 = None
        view_196: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_52, _shape_param_250);  expand_52 = _shape_param_250 = None
        bmm_16: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_195, view_196)
        view_197: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_16, _shape_param_251);  bmm_16 = _shape_param_251 = None
        div_16: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_197, 16.0);  view_197 = None
        add_76: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_16, where);  div_16 = None
        amax_8: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_76, [-1], True)
        sub_17: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_76, amax_8);  add_76 = amax_8 = None
        exp_8: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_9: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_17: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_136: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16)
        expand_53: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_136, _shape_param_252);  convert_element_type_136 = _shape_param_252 = None
        view_198: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_53, _shape_param_253);  expand_53 = _shape_param_253 = None
        expand_54: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_91, _shape_param_254);  permute_91 = _shape_param_254 = None
        view_199: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_54, _shape_param_255);  expand_54 = _shape_param_255 = None
        bmm_17: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_198, view_199)
        view_200: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_17, _shape_param_256);  bmm_17 = _shape_param_256 = None
        permute_95: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
        clone_26: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_201: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_26, _shape_param_257);  clone_26 = _shape_param_257 = None
        convert_element_type_137: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg98_1, torch.bfloat16);  arg98_1 = None
        permute_96: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_137, [1, 0]);  convert_element_type_137 = None
        view_202: "bf16[128, 4096]" = torch.ops.aten.view.default(view_201, _shape_param_258);  view_201 = _shape_param_258 = None
        mm_35: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_202, permute_96)
        view_203: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_35, _shape_param_259);  mm_35 = _shape_param_259 = None
        convert_element_type_138: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg99_1, torch.bfloat16);  arg99_1 = None
        convert_element_type_139: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg100_1, torch.bfloat16);  arg100_1 = None
        permute_97: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_139, [1, 0]);  convert_element_type_139 = None
        addmm_16: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_138, view_184, permute_97);  convert_element_type_138 = None
        view_204: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_16, _shape_param_260);  _shape_param_260 = None
        mul_86: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_204, 0.5)
        convert_element_type_140: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_204, torch.float32)
        pow_9: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_140, 3.0);  convert_element_type_140 = None
        mul_87: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_77: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_204, mul_87);  view_204 = mul_87 = None
        mul_88: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_77, 0.7978845608028654);  add_77 = None
        tanh_8: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_88);  mul_88 = None
        add_78: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_89: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_86, add_78);  mul_86 = add_78 = None
        convert_element_type_141: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg101_1, torch.bfloat16);  arg101_1 = None
        convert_element_type_142: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg102_1, torch.bfloat16);  arg102_1 = None
        convert_element_type_143: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None
        view_205: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_143, _shape_param_261);  convert_element_type_143 = _shape_param_261 = None
        permute_98: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_142, [1, 0]);  convert_element_type_142 = None
        addmm_17: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_141, view_205, permute_98);  convert_element_type_141 = None
        view_206: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_17, _shape_param_262);  addmm_17 = _shape_param_262 = None
        add_79: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_203, view_206);  view_203 = view_206 = None
        add_80: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_79, add_71);  add_79 = add_71 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
        getitem_36: "f32[1, 128, 1]" = var_mean_9[0]
        getitem_37: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        add_81: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_18: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_80, getitem_37);  getitem_37 = None
        mul_90: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_9);  sub_18 = None
        mul_91: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_90, arg103_1);  arg103_1 = None
        add_82: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_91, arg104_1);  mul_91 = arg104_1 = None
        convert_element_type_144: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg105_1, torch.bfloat16);  arg105_1 = None
        convert_element_type_145: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_82, torch.bfloat16);  add_82 = None
        permute_99: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_144, [1, 0]);  convert_element_type_144 = None
        view_207: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_145, _shape_param_263);  convert_element_type_145 = _shape_param_263 = None
        mm_36: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_207, permute_99)
        view_208: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_36, _shape_param_264);  mm_36 = _shape_param_264 = None
        convert_element_type_146: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg106_1, torch.bfloat16);  arg106_1 = None
        permute_100: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_146, [1, 0]);  convert_element_type_146 = None
        mm_37: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_207, permute_100)
        view_209: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_37, _shape_param_265);  mm_37 = _shape_param_265 = None
        convert_element_type_147: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg107_1, torch.bfloat16);  arg107_1 = None
        permute_101: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_147, [1, 0]);  convert_element_type_147 = None
        mm_38: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_207, permute_101)
        view_210: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_38, _shape_param_266);  mm_38 = _shape_param_266 = None
        view_211: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_208, _shape_param_267);  view_208 = _shape_param_267 = None
        view_212: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_209, _shape_param_268);  view_209 = _shape_param_268 = None
        view_213: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_210, _shape_param_269);  view_210 = _shape_param_269 = None
        permute_102: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
        repeat_10: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg108_1, [1, 1, 1]);  arg108_1 = None
        gather_9: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_10, 1, repeat_1);  repeat_10 = None
        convert_element_type_148: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_9, torch.bfloat16);  gather_9 = None
        split_9 = torch.ops.aten.split.Tensor(convert_element_type_148, 32, -1);  convert_element_type_148 = None
        getitem_38: "bf16[1, 128, 32]" = split_9[0]
        getitem_39: "bf16[1, 128, 32]" = split_9[1];  split_9 = None
        slice_73: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_212, 3, 0, 64)
        slice_74: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_212, 3, 64, 9223372036854775807);  view_212 = None
        slice_75: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_211, 3, 0, 64)
        slice_76: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_211, 3, 64, 9223372036854775807);  view_211 = None
        unsqueeze_79: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_38, 2);  getitem_38 = None
        unsqueeze_80: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_79, 4);  unsqueeze_79 = None
        expand_55: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_80, _shape_param_270);  _shape_param_270 = None
        clone_27: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_214: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_27, _shape_param_271);  clone_27 = _shape_param_271 = None
        unsqueeze_81: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_39, 2);  getitem_39 = None
        unsqueeze_82: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_81, 4);  unsqueeze_81 = None
        expand_56: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_82, _shape_param_272);  _shape_param_272 = None
        clone_28: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_215: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_273);  clone_28 = _shape_param_273 = None
        mul_92: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_73, view_215)
        slice_77: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_73, 3, 0, 9223372036854775807, 2)
        slice_78: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_73, 3, 1, 9223372036854775807, 2);  slice_73 = None
        neg_18: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_78);  slice_78 = None
        unsqueeze_83: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_18, 4);  neg_18 = None
        unsqueeze_84: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_77, 4);  slice_77 = None
        cat_36: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_83, unsqueeze_84], -1);  unsqueeze_83 = unsqueeze_84 = None
        view_216: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_36, _shape_param_274);  cat_36 = _shape_param_274 = None
        mul_93: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_216, view_214);  view_216 = None
        add_83: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        mul_94: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_75, view_215);  view_215 = None
        slice_79: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_75, 3, 0, 9223372036854775807, 2)
        slice_80: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_75, 3, 1, 9223372036854775807, 2);  slice_75 = None
        neg_19: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_80);  slice_80 = None
        unsqueeze_85: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_19, 4);  neg_19 = None
        unsqueeze_86: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_79, 4);  slice_79 = None
        cat_37: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_85, unsqueeze_86], -1);  unsqueeze_85 = unsqueeze_86 = None
        view_217: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_37, _shape_param_275);  cat_37 = _shape_param_275 = None
        mul_95: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_217, view_214);  view_217 = view_214 = None
        add_84: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
        cat_38: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_83, slice_74], -1);  add_83 = slice_74 = None
        cat_39: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_84, slice_76], -1);  add_84 = slice_76 = None
        permute_103: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_38, [0, 2, 1, 3]);  cat_38 = None
        permute_104: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_39, [0, 2, 1, 3]);  cat_39 = None
        convert_element_type_149: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_103, torch.float32);  permute_103 = None
        permute_105: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_149, [0, 1, 3, 2]);  convert_element_type_149 = None
        convert_element_type_150: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_105, torch.bfloat16);  permute_105 = None
        convert_element_type_151: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_104, torch.bfloat16);  permute_104 = None
        expand_57: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_151, _shape_param_276);  convert_element_type_151 = _shape_param_276 = None
        view_218: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_57, _shape_param_277);  expand_57 = _shape_param_277 = None
        expand_58: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_150, _shape_param_278);  convert_element_type_150 = _shape_param_278 = None
        view_219: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_58, _shape_param_279);  expand_58 = _shape_param_279 = None
        bmm_18: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_218, view_219)
        view_220: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_18, _shape_param_280);  bmm_18 = _shape_param_280 = None
        div_18: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_220, 16.0);  view_220 = None
        add_85: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_18, where);  div_18 = None
        amax_9: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_85, [-1], True)
        sub_19: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_85, amax_9);  add_85 = amax_9 = None
        exp_9: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_10: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_19: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_152: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16)
        expand_59: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_152, _shape_param_281);  convert_element_type_152 = _shape_param_281 = None
        view_221: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_59, _shape_param_282);  expand_59 = _shape_param_282 = None
        expand_60: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_102, _shape_param_283);  permute_102 = _shape_param_283 = None
        view_222: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_60, _shape_param_284);  expand_60 = _shape_param_284 = None
        bmm_19: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_221, view_222)
        view_223: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_19, _shape_param_285);  bmm_19 = _shape_param_285 = None
        permute_106: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_223, [0, 2, 1, 3]);  view_223 = None
        clone_29: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_224: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_29, _shape_param_286);  clone_29 = _shape_param_286 = None
        convert_element_type_153: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg109_1, torch.bfloat16);  arg109_1 = None
        permute_107: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_153, [1, 0]);  convert_element_type_153 = None
        view_225: "bf16[128, 4096]" = torch.ops.aten.view.default(view_224, _shape_param_287);  view_224 = _shape_param_287 = None
        mm_39: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_225, permute_107)
        view_226: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_39, _shape_param_288);  mm_39 = _shape_param_288 = None
        convert_element_type_154: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg110_1, torch.bfloat16);  arg110_1 = None
        convert_element_type_155: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg111_1, torch.bfloat16);  arg111_1 = None
        permute_108: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_155, [1, 0]);  convert_element_type_155 = None
        addmm_18: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_154, view_207, permute_108);  convert_element_type_154 = None
        view_227: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_18, _shape_param_289);  _shape_param_289 = None
        mul_96: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_227, 0.5)
        convert_element_type_156: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_227, torch.float32)
        pow_10: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_156, 3.0);  convert_element_type_156 = None
        mul_97: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_86: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_227, mul_97);  view_227 = mul_97 = None
        mul_98: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_86, 0.7978845608028654);  add_86 = None
        tanh_9: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_98);  mul_98 = None
        add_87: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_99: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_96, add_87);  mul_96 = add_87 = None
        convert_element_type_157: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg112_1, torch.bfloat16);  arg112_1 = None
        convert_element_type_158: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg113_1, torch.bfloat16);  arg113_1 = None
        convert_element_type_159: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_99, torch.bfloat16);  mul_99 = None
        view_228: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_159, _shape_param_290);  convert_element_type_159 = _shape_param_290 = None
        permute_109: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_158, [1, 0]);  convert_element_type_158 = None
        addmm_19: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_157, view_228, permute_109);  convert_element_type_157 = None
        view_229: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_19, _shape_param_291);  addmm_19 = _shape_param_291 = None
        add_88: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_226, view_229);  view_226 = view_229 = None
        add_89: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_88, add_80);  add_88 = add_80 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_40: "f32[1, 128, 1]" = var_mean_10[0]
        getitem_41: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        add_90: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_20: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_89, getitem_41);  getitem_41 = None
        mul_100: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_10);  sub_20 = None
        mul_101: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_100, arg114_1);  arg114_1 = None
        add_91: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_101, arg115_1);  mul_101 = arg115_1 = None
        convert_element_type_160: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg116_1, torch.bfloat16);  arg116_1 = None
        convert_element_type_161: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16);  add_91 = None
        permute_110: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_160, [1, 0]);  convert_element_type_160 = None
        view_230: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_161, _shape_param_292);  convert_element_type_161 = _shape_param_292 = None
        mm_40: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_230, permute_110)
        view_231: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_40, _shape_param_293);  mm_40 = _shape_param_293 = None
        convert_element_type_162: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg117_1, torch.bfloat16);  arg117_1 = None
        permute_111: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_162, [1, 0]);  convert_element_type_162 = None
        mm_41: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_230, permute_111)
        view_232: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_41, _shape_param_294);  mm_41 = _shape_param_294 = None
        convert_element_type_163: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg118_1, torch.bfloat16);  arg118_1 = None
        permute_112: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_163, [1, 0]);  convert_element_type_163 = None
        mm_42: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_230, permute_112)
        view_233: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_42, _shape_param_295);  mm_42 = _shape_param_295 = None
        view_234: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_231, _shape_param_296);  view_231 = _shape_param_296 = None
        view_235: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_232, _shape_param_297);  view_232 = _shape_param_297 = None
        view_236: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_233, _shape_param_298);  view_233 = _shape_param_298 = None
        permute_113: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        repeat_11: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg119_1, [1, 1, 1]);  arg119_1 = None
        gather_10: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_11, 1, repeat_1);  repeat_11 = None
        convert_element_type_164: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_10, torch.bfloat16);  gather_10 = None
        split_10 = torch.ops.aten.split.Tensor(convert_element_type_164, 32, -1);  convert_element_type_164 = None
        getitem_42: "bf16[1, 128, 32]" = split_10[0]
        getitem_43: "bf16[1, 128, 32]" = split_10[1];  split_10 = None
        slice_81: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_235, 3, 0, 64)
        slice_82: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_235, 3, 64, 9223372036854775807);  view_235 = None
        slice_83: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_234, 3, 0, 64)
        slice_84: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_234, 3, 64, 9223372036854775807);  view_234 = None
        unsqueeze_87: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_42, 2);  getitem_42 = None
        unsqueeze_88: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_87, 4);  unsqueeze_87 = None
        expand_61: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_88, _shape_param_299);  _shape_param_299 = None
        clone_30: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_237: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_30, _shape_param_300);  clone_30 = _shape_param_300 = None
        unsqueeze_89: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_43, 2);  getitem_43 = None
        unsqueeze_90: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_89, 4);  unsqueeze_89 = None
        expand_62: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_90, _shape_param_301);  _shape_param_301 = None
        clone_31: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_238: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_31, _shape_param_302);  clone_31 = _shape_param_302 = None
        mul_102: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_81, view_238)
        slice_85: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_81, 3, 0, 9223372036854775807, 2)
        slice_86: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_81, 3, 1, 9223372036854775807, 2);  slice_81 = None
        neg_20: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_86);  slice_86 = None
        unsqueeze_91: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_20, 4);  neg_20 = None
        unsqueeze_92: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_85, 4);  slice_85 = None
        cat_40: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_91, unsqueeze_92], -1);  unsqueeze_91 = unsqueeze_92 = None
        view_239: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_40, _shape_param_303);  cat_40 = _shape_param_303 = None
        mul_103: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_239, view_237);  view_239 = None
        add_92: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        mul_104: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_83, view_238);  view_238 = None
        slice_87: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_83, 3, 0, 9223372036854775807, 2)
        slice_88: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_83, 3, 1, 9223372036854775807, 2);  slice_83 = None
        neg_21: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_88);  slice_88 = None
        unsqueeze_93: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_21, 4);  neg_21 = None
        unsqueeze_94: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_87, 4);  slice_87 = None
        cat_41: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_93, unsqueeze_94], -1);  unsqueeze_93 = unsqueeze_94 = None
        view_240: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_41, _shape_param_304);  cat_41 = _shape_param_304 = None
        mul_105: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_240, view_237);  view_240 = view_237 = None
        add_93: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
        cat_42: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_92, slice_82], -1);  add_92 = slice_82 = None
        cat_43: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_93, slice_84], -1);  add_93 = slice_84 = None
        permute_114: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_42, [0, 2, 1, 3]);  cat_42 = None
        permute_115: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_43, [0, 2, 1, 3]);  cat_43 = None
        convert_element_type_165: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_114, torch.float32);  permute_114 = None
        permute_116: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_165, [0, 1, 3, 2]);  convert_element_type_165 = None
        convert_element_type_166: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_116, torch.bfloat16);  permute_116 = None
        convert_element_type_167: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_115, torch.bfloat16);  permute_115 = None
        expand_63: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_167, _shape_param_305);  convert_element_type_167 = _shape_param_305 = None
        view_241: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_63, _shape_param_306);  expand_63 = _shape_param_306 = None
        expand_64: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_166, _shape_param_307);  convert_element_type_166 = _shape_param_307 = None
        view_242: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_64, _shape_param_308);  expand_64 = _shape_param_308 = None
        bmm_20: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_241, view_242)
        view_243: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_20, _shape_param_309);  bmm_20 = _shape_param_309 = None
        div_20: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_243, 16.0);  view_243 = None
        add_94: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_20, where);  div_20 = None
        amax_10: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_94, [-1], True)
        sub_21: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_94, amax_10);  add_94 = amax_10 = None
        exp_10: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_11: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_21: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_168: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16)
        expand_65: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_168, _shape_param_310);  convert_element_type_168 = _shape_param_310 = None
        view_244: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_65, _shape_param_311);  expand_65 = _shape_param_311 = None
        expand_66: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_113, _shape_param_312);  permute_113 = _shape_param_312 = None
        view_245: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_66, _shape_param_313);  expand_66 = _shape_param_313 = None
        bmm_21: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_244, view_245)
        view_246: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_21, _shape_param_314);  bmm_21 = _shape_param_314 = None
        permute_117: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        clone_32: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_247: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_32, _shape_param_315);  clone_32 = _shape_param_315 = None
        convert_element_type_169: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg120_1, torch.bfloat16);  arg120_1 = None
        permute_118: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_169, [1, 0]);  convert_element_type_169 = None
        view_248: "bf16[128, 4096]" = torch.ops.aten.view.default(view_247, _shape_param_316);  view_247 = _shape_param_316 = None
        mm_43: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_248, permute_118)
        view_249: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_43, _shape_param_317);  mm_43 = _shape_param_317 = None
        convert_element_type_170: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg121_1, torch.bfloat16);  arg121_1 = None
        convert_element_type_171: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg122_1, torch.bfloat16);  arg122_1 = None
        permute_119: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_171, [1, 0]);  convert_element_type_171 = None
        addmm_20: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_170, view_230, permute_119);  convert_element_type_170 = None
        view_250: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_20, _shape_param_318);  _shape_param_318 = None
        mul_106: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_250, 0.5)
        convert_element_type_172: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_250, torch.float32)
        pow_11: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_172, 3.0);  convert_element_type_172 = None
        mul_107: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_95: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_250, mul_107);  view_250 = mul_107 = None
        mul_108: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_95, 0.7978845608028654);  add_95 = None
        tanh_10: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_108);  mul_108 = None
        add_96: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_109: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_106, add_96);  mul_106 = add_96 = None
        convert_element_type_173: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg123_1, torch.bfloat16);  arg123_1 = None
        convert_element_type_174: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg124_1, torch.bfloat16);  arg124_1 = None
        convert_element_type_175: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_109, torch.bfloat16);  mul_109 = None
        view_251: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_175, _shape_param_319);  convert_element_type_175 = _shape_param_319 = None
        permute_120: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_174, [1, 0]);  convert_element_type_174 = None
        addmm_21: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_173, view_251, permute_120);  convert_element_type_173 = None
        view_252: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_21, _shape_param_320);  addmm_21 = _shape_param_320 = None
        add_97: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_249, view_252);  view_249 = view_252 = None
        add_98: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_97, add_89);  add_97 = add_89 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
        getitem_44: "f32[1, 128, 1]" = var_mean_11[0]
        getitem_45: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        add_99: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_22: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_98, getitem_45);  getitem_45 = None
        mul_110: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_11);  sub_22 = None
        mul_111: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_110, arg125_1);  arg125_1 = None
        add_100: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_111, arg126_1);  mul_111 = arg126_1 = None
        convert_element_type_176: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg127_1, torch.bfloat16);  arg127_1 = None
        convert_element_type_177: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_100, torch.bfloat16);  add_100 = None
        permute_121: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_176, [1, 0]);  convert_element_type_176 = None
        view_253: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_177, _shape_param_321);  convert_element_type_177 = _shape_param_321 = None
        mm_44: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_253, permute_121)
        view_254: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_44, _shape_param_322);  mm_44 = _shape_param_322 = None
        convert_element_type_178: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg128_1, torch.bfloat16);  arg128_1 = None
        permute_122: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_178, [1, 0]);  convert_element_type_178 = None
        mm_45: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_253, permute_122)
        view_255: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_45, _shape_param_323);  mm_45 = _shape_param_323 = None
        convert_element_type_179: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg129_1, torch.bfloat16);  arg129_1 = None
        permute_123: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_179, [1, 0]);  convert_element_type_179 = None
        mm_46: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_253, permute_123)
        view_256: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_46, _shape_param_324);  mm_46 = _shape_param_324 = None
        view_257: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_254, _shape_param_325);  view_254 = _shape_param_325 = None
        view_258: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_255, _shape_param_326);  view_255 = _shape_param_326 = None
        view_259: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_256, _shape_param_327);  view_256 = _shape_param_327 = None
        permute_124: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
        repeat_12: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg130_1, [1, 1, 1]);  arg130_1 = None
        gather_11: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_12, 1, repeat_1);  repeat_12 = None
        convert_element_type_180: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_11, torch.bfloat16);  gather_11 = None
        split_11 = torch.ops.aten.split.Tensor(convert_element_type_180, 32, -1);  convert_element_type_180 = None
        getitem_46: "bf16[1, 128, 32]" = split_11[0]
        getitem_47: "bf16[1, 128, 32]" = split_11[1];  split_11 = None
        slice_89: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_258, 3, 0, 64)
        slice_90: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_258, 3, 64, 9223372036854775807);  view_258 = None
        slice_91: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_257, 3, 0, 64)
        slice_92: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_257, 3, 64, 9223372036854775807);  view_257 = None
        unsqueeze_95: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_46, 2);  getitem_46 = None
        unsqueeze_96: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_95, 4);  unsqueeze_95 = None
        expand_67: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_96, _shape_param_328);  _shape_param_328 = None
        clone_33: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_260: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_33, _shape_param_329);  clone_33 = _shape_param_329 = None
        unsqueeze_97: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_47, 2);  getitem_47 = None
        unsqueeze_98: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_97, 4);  unsqueeze_97 = None
        expand_68: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_98, _shape_param_330);  _shape_param_330 = None
        clone_34: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_261: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_34, _shape_param_331);  clone_34 = _shape_param_331 = None
        mul_112: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_89, view_261)
        slice_93: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_89, 3, 0, 9223372036854775807, 2)
        slice_94: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_89, 3, 1, 9223372036854775807, 2);  slice_89 = None
        neg_22: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_94);  slice_94 = None
        unsqueeze_99: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_22, 4);  neg_22 = None
        unsqueeze_100: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_93, 4);  slice_93 = None
        cat_44: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_99, unsqueeze_100], -1);  unsqueeze_99 = unsqueeze_100 = None
        view_262: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_44, _shape_param_332);  cat_44 = _shape_param_332 = None
        mul_113: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_262, view_260);  view_262 = None
        add_101: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
        mul_114: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_91, view_261);  view_261 = None
        slice_95: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_91, 3, 0, 9223372036854775807, 2)
        slice_96: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_91, 3, 1, 9223372036854775807, 2);  slice_91 = None
        neg_23: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_96);  slice_96 = None
        unsqueeze_101: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_23, 4);  neg_23 = None
        unsqueeze_102: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_95, 4);  slice_95 = None
        cat_45: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_101, unsqueeze_102], -1);  unsqueeze_101 = unsqueeze_102 = None
        view_263: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_45, _shape_param_333);  cat_45 = _shape_param_333 = None
        mul_115: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_263, view_260);  view_263 = view_260 = None
        add_102: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None
        cat_46: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_101, slice_90], -1);  add_101 = slice_90 = None
        cat_47: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_102, slice_92], -1);  add_102 = slice_92 = None
        permute_125: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_46, [0, 2, 1, 3]);  cat_46 = None
        permute_126: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_47, [0, 2, 1, 3]);  cat_47 = None
        convert_element_type_181: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_125, torch.float32);  permute_125 = None
        permute_127: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_181, [0, 1, 3, 2]);  convert_element_type_181 = None
        convert_element_type_182: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_127, torch.bfloat16);  permute_127 = None
        convert_element_type_183: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_126, torch.bfloat16);  permute_126 = None
        expand_69: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_183, _shape_param_334);  convert_element_type_183 = _shape_param_334 = None
        view_264: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_69, _shape_param_335);  expand_69 = _shape_param_335 = None
        expand_70: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_182, _shape_param_336);  convert_element_type_182 = _shape_param_336 = None
        view_265: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_70, _shape_param_337);  expand_70 = _shape_param_337 = None
        bmm_22: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_264, view_265)
        view_266: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_338);  bmm_22 = _shape_param_338 = None
        div_22: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_266, 16.0);  view_266 = None
        add_103: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_22, where);  div_22 = None
        amax_11: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_103, [-1], True)
        sub_23: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_103, amax_11);  add_103 = amax_11 = None
        exp_11: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_12: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_23: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_184: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16)
        expand_71: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_184, _shape_param_339);  convert_element_type_184 = _shape_param_339 = None
        view_267: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_71, _shape_param_340);  expand_71 = _shape_param_340 = None
        expand_72: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_124, _shape_param_341);  permute_124 = _shape_param_341 = None
        view_268: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_72, _shape_param_342);  expand_72 = _shape_param_342 = None
        bmm_23: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_267, view_268)
        view_269: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_23, _shape_param_343);  bmm_23 = _shape_param_343 = None
        permute_128: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
        clone_35: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_270: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_35, _shape_param_344);  clone_35 = _shape_param_344 = None
        convert_element_type_185: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg131_1, torch.bfloat16);  arg131_1 = None
        permute_129: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_185, [1, 0]);  convert_element_type_185 = None
        view_271: "bf16[128, 4096]" = torch.ops.aten.view.default(view_270, _shape_param_345);  view_270 = _shape_param_345 = None
        mm_47: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_271, permute_129)
        view_272: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_47, _shape_param_346);  mm_47 = _shape_param_346 = None
        convert_element_type_186: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg132_1, torch.bfloat16);  arg132_1 = None
        convert_element_type_187: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg133_1, torch.bfloat16);  arg133_1 = None
        permute_130: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_187, [1, 0]);  convert_element_type_187 = None
        addmm_22: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_186, view_253, permute_130);  convert_element_type_186 = None
        view_273: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_22, _shape_param_347);  _shape_param_347 = None
        mul_116: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_273, 0.5)
        convert_element_type_188: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_273, torch.float32)
        pow_12: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_188, 3.0);  convert_element_type_188 = None
        mul_117: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_104: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_273, mul_117);  view_273 = mul_117 = None
        mul_118: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_104, 0.7978845608028654);  add_104 = None
        tanh_11: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_105: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_119: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_116, add_105);  mul_116 = add_105 = None
        convert_element_type_189: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg134_1, torch.bfloat16);  arg134_1 = None
        convert_element_type_190: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg135_1, torch.bfloat16);  arg135_1 = None
        convert_element_type_191: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_119, torch.bfloat16);  mul_119 = None
        view_274: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_191, _shape_param_348);  convert_element_type_191 = _shape_param_348 = None
        permute_131: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_190, [1, 0]);  convert_element_type_190 = None
        addmm_23: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_189, view_274, permute_131);  convert_element_type_189 = None
        view_275: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_23, _shape_param_349);  addmm_23 = _shape_param_349 = None
        add_106: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_272, view_275);  view_272 = view_275 = None
        add_107: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_106, add_98);  add_106 = add_98 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
        getitem_48: "f32[1, 128, 1]" = var_mean_12[0]
        getitem_49: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        add_108: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        sub_24: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_107, getitem_49);  getitem_49 = None
        mul_120: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_12);  sub_24 = None
        mul_121: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_120, arg136_1);  arg136_1 = None
        add_109: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_121, arg137_1);  mul_121 = arg137_1 = None
        convert_element_type_192: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg138_1, torch.bfloat16);  arg138_1 = None
        convert_element_type_193: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16);  add_109 = None
        permute_132: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_192, [1, 0]);  convert_element_type_192 = None
        view_276: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_193, _shape_param_350);  convert_element_type_193 = _shape_param_350 = None
        mm_48: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_276, permute_132)
        view_277: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_48, _shape_param_351);  mm_48 = _shape_param_351 = None
        convert_element_type_194: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg139_1, torch.bfloat16);  arg139_1 = None
        permute_133: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_194, [1, 0]);  convert_element_type_194 = None
        mm_49: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_276, permute_133)
        view_278: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_49, _shape_param_352);  mm_49 = _shape_param_352 = None
        convert_element_type_195: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg140_1, torch.bfloat16);  arg140_1 = None
        permute_134: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_195, [1, 0]);  convert_element_type_195 = None
        mm_50: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_276, permute_134)
        view_279: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_50, _shape_param_353);  mm_50 = _shape_param_353 = None
        view_280: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_277, _shape_param_354);  view_277 = _shape_param_354 = None
        view_281: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_278, _shape_param_355);  view_278 = _shape_param_355 = None
        view_282: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_279, _shape_param_356);  view_279 = _shape_param_356 = None
        permute_135: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
        repeat_13: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg141_1, [1, 1, 1]);  arg141_1 = None
        gather_12: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_13, 1, repeat_1);  repeat_13 = None
        convert_element_type_196: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_12, torch.bfloat16);  gather_12 = None
        split_12 = torch.ops.aten.split.Tensor(convert_element_type_196, 32, -1);  convert_element_type_196 = None
        getitem_50: "bf16[1, 128, 32]" = split_12[0]
        getitem_51: "bf16[1, 128, 32]" = split_12[1];  split_12 = None
        slice_97: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_281, 3, 0, 64)
        slice_98: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_281, 3, 64, 9223372036854775807);  view_281 = None
        slice_99: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_280, 3, 0, 64)
        slice_100: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_280, 3, 64, 9223372036854775807);  view_280 = None
        unsqueeze_103: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_50, 2);  getitem_50 = None
        unsqueeze_104: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 4);  unsqueeze_103 = None
        expand_73: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_104, _shape_param_357);  _shape_param_357 = None
        clone_36: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_283: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_358);  clone_36 = _shape_param_358 = None
        unsqueeze_105: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_51, 2);  getitem_51 = None
        unsqueeze_106: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 4);  unsqueeze_105 = None
        expand_74: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_106, _shape_param_359);  _shape_param_359 = None
        clone_37: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_284: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_37, _shape_param_360);  clone_37 = _shape_param_360 = None
        mul_122: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_97, view_284)
        slice_101: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_97, 3, 0, 9223372036854775807, 2)
        slice_102: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_97, 3, 1, 9223372036854775807, 2);  slice_97 = None
        neg_24: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_102);  slice_102 = None
        unsqueeze_107: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_24, 4);  neg_24 = None
        unsqueeze_108: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_101, 4);  slice_101 = None
        cat_48: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_107, unsqueeze_108], -1);  unsqueeze_107 = unsqueeze_108 = None
        view_285: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_48, _shape_param_361);  cat_48 = _shape_param_361 = None
        mul_123: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_285, view_283);  view_285 = None
        add_110: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None
        mul_124: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_99, view_284);  view_284 = None
        slice_103: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_99, 3, 0, 9223372036854775807, 2)
        slice_104: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_99, 3, 1, 9223372036854775807, 2);  slice_99 = None
        neg_25: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_104);  slice_104 = None
        unsqueeze_109: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_25, 4);  neg_25 = None
        unsqueeze_110: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_103, 4);  slice_103 = None
        cat_49: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_109, unsqueeze_110], -1);  unsqueeze_109 = unsqueeze_110 = None
        view_286: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_49, _shape_param_362);  cat_49 = _shape_param_362 = None
        mul_125: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_286, view_283);  view_286 = view_283 = None
        add_111: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
        cat_50: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_110, slice_98], -1);  add_110 = slice_98 = None
        cat_51: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_111, slice_100], -1);  add_111 = slice_100 = None
        permute_136: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_50, [0, 2, 1, 3]);  cat_50 = None
        permute_137: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_51, [0, 2, 1, 3]);  cat_51 = None
        convert_element_type_197: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_136, torch.float32);  permute_136 = None
        permute_138: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_197, [0, 1, 3, 2]);  convert_element_type_197 = None
        convert_element_type_198: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_138, torch.bfloat16);  permute_138 = None
        convert_element_type_199: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_137, torch.bfloat16);  permute_137 = None
        expand_75: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_199, _shape_param_363);  convert_element_type_199 = _shape_param_363 = None
        view_287: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_75, _shape_param_364);  expand_75 = _shape_param_364 = None
        expand_76: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_198, _shape_param_365);  convert_element_type_198 = _shape_param_365 = None
        view_288: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_76, _shape_param_366);  expand_76 = _shape_param_366 = None
        bmm_24: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_287, view_288)
        view_289: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_24, _shape_param_367);  bmm_24 = _shape_param_367 = None
        div_24: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_289, 16.0);  view_289 = None
        add_112: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_24, where);  div_24 = None
        amax_12: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_112, [-1], True)
        sub_25: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_112, amax_12);  add_112 = amax_12 = None
        exp_12: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_13: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_25: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_200: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16)
        expand_77: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_200, _shape_param_368);  convert_element_type_200 = _shape_param_368 = None
        view_290: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_77, _shape_param_369);  expand_77 = _shape_param_369 = None
        expand_78: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_135, _shape_param_370);  permute_135 = _shape_param_370 = None
        view_291: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_78, _shape_param_371);  expand_78 = _shape_param_371 = None
        bmm_25: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_290, view_291)
        view_292: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_25, _shape_param_372);  bmm_25 = _shape_param_372 = None
        permute_139: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
        clone_38: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        view_293: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_38, _shape_param_373);  clone_38 = _shape_param_373 = None
        convert_element_type_201: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg142_1, torch.bfloat16);  arg142_1 = None
        permute_140: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_201, [1, 0]);  convert_element_type_201 = None
        view_294: "bf16[128, 4096]" = torch.ops.aten.view.default(view_293, _shape_param_374);  view_293 = _shape_param_374 = None
        mm_51: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_294, permute_140)
        view_295: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_51, _shape_param_375);  mm_51 = _shape_param_375 = None
        convert_element_type_202: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg143_1, torch.bfloat16);  arg143_1 = None
        convert_element_type_203: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg144_1, torch.bfloat16);  arg144_1 = None
        permute_141: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_203, [1, 0]);  convert_element_type_203 = None
        addmm_24: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_202, view_276, permute_141);  convert_element_type_202 = None
        view_296: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_24, _shape_param_376);  _shape_param_376 = None
        mul_126: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_296, 0.5)
        convert_element_type_204: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_296, torch.float32)
        pow_13: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_204, 3.0);  convert_element_type_204 = None
        mul_127: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_113: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_296, mul_127);  view_296 = mul_127 = None
        mul_128: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_113, 0.7978845608028654);  add_113 = None
        tanh_12: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_128);  mul_128 = None
        add_114: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_129: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_126, add_114);  mul_126 = add_114 = None
        convert_element_type_205: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg145_1, torch.bfloat16);  arg145_1 = None
        convert_element_type_206: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg146_1, torch.bfloat16);  arg146_1 = None
        convert_element_type_207: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_129, torch.bfloat16);  mul_129 = None
        view_297: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_207, _shape_param_377);  convert_element_type_207 = _shape_param_377 = None
        permute_142: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_206, [1, 0]);  convert_element_type_206 = None
        addmm_25: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_205, view_297, permute_142);  convert_element_type_205 = None
        view_298: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_25, _shape_param_378);  addmm_25 = _shape_param_378 = None
        add_115: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_295, view_298);  view_295 = view_298 = None
        add_116: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_115, add_107);  add_115 = add_107 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_116, [2], correction = 0, keepdim = True)
        getitem_52: "f32[1, 128, 1]" = var_mean_13[0]
        getitem_53: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None
        add_117: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        sub_26: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_116, getitem_53);  getitem_53 = None
        mul_130: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_13);  sub_26 = None
        mul_131: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_130, arg147_1);  arg147_1 = None
        add_118: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_131, arg148_1);  mul_131 = arg148_1 = None
        convert_element_type_208: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg149_1, torch.bfloat16);  arg149_1 = None
        convert_element_type_209: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_118, torch.bfloat16);  add_118 = None
        permute_143: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_208, [1, 0]);  convert_element_type_208 = None
        view_299: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_209, _shape_param_379);  convert_element_type_209 = _shape_param_379 = None
        mm_52: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_299, permute_143)
        view_300: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_52, _shape_param_380);  mm_52 = _shape_param_380 = None
        convert_element_type_210: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg150_1, torch.bfloat16);  arg150_1 = None
        permute_144: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_210, [1, 0]);  convert_element_type_210 = None
        mm_53: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_299, permute_144)
        view_301: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_53, _shape_param_381);  mm_53 = _shape_param_381 = None
        convert_element_type_211: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg151_1, torch.bfloat16);  arg151_1 = None
        permute_145: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_211, [1, 0]);  convert_element_type_211 = None
        mm_54: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_299, permute_145)
        view_302: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_54, _shape_param_382);  mm_54 = _shape_param_382 = None
        view_303: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_300, _shape_param_383);  view_300 = _shape_param_383 = None
        view_304: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_301, _shape_param_384);  view_301 = _shape_param_384 = None
        view_305: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_302, _shape_param_385);  view_302 = _shape_param_385 = None
        permute_146: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_305, [0, 2, 1, 3]);  view_305 = None
        repeat_14: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg152_1, [1, 1, 1]);  arg152_1 = None
        gather_13: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_14, 1, repeat_1);  repeat_14 = None
        convert_element_type_212: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_13, torch.bfloat16);  gather_13 = None
        split_13 = torch.ops.aten.split.Tensor(convert_element_type_212, 32, -1);  convert_element_type_212 = None
        getitem_54: "bf16[1, 128, 32]" = split_13[0]
        getitem_55: "bf16[1, 128, 32]" = split_13[1];  split_13 = None
        slice_105: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_304, 3, 0, 64)
        slice_106: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_304, 3, 64, 9223372036854775807);  view_304 = None
        slice_107: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_303, 3, 0, 64)
        slice_108: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_303, 3, 64, 9223372036854775807);  view_303 = None
        unsqueeze_111: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_54, 2);  getitem_54 = None
        unsqueeze_112: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 4);  unsqueeze_111 = None
        expand_79: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_112, _shape_param_386);  _shape_param_386 = None
        clone_39: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_306: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_39, _shape_param_387);  clone_39 = _shape_param_387 = None
        unsqueeze_113: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_55, 2);  getitem_55 = None
        unsqueeze_114: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 4);  unsqueeze_113 = None
        expand_80: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_114, _shape_param_388);  _shape_param_388 = None
        clone_40: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_307: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_389);  clone_40 = _shape_param_389 = None
        mul_132: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_105, view_307)
        slice_109: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_105, 3, 0, 9223372036854775807, 2)
        slice_110: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_105, 3, 1, 9223372036854775807, 2);  slice_105 = None
        neg_26: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_110);  slice_110 = None
        unsqueeze_115: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_26, 4);  neg_26 = None
        unsqueeze_116: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_109, 4);  slice_109 = None
        cat_52: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_115, unsqueeze_116], -1);  unsqueeze_115 = unsqueeze_116 = None
        view_308: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_52, _shape_param_390);  cat_52 = _shape_param_390 = None
        mul_133: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_308, view_306);  view_308 = None
        add_119: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
        mul_134: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_107, view_307);  view_307 = None
        slice_111: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_107, 3, 0, 9223372036854775807, 2)
        slice_112: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_107, 3, 1, 9223372036854775807, 2);  slice_107 = None
        neg_27: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_112);  slice_112 = None
        unsqueeze_117: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_27, 4);  neg_27 = None
        unsqueeze_118: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_111, 4);  slice_111 = None
        cat_53: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_117, unsqueeze_118], -1);  unsqueeze_117 = unsqueeze_118 = None
        view_309: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_53, _shape_param_391);  cat_53 = _shape_param_391 = None
        mul_135: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_309, view_306);  view_309 = view_306 = None
        add_120: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        cat_54: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_119, slice_106], -1);  add_119 = slice_106 = None
        cat_55: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_120, slice_108], -1);  add_120 = slice_108 = None
        permute_147: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_54, [0, 2, 1, 3]);  cat_54 = None
        permute_148: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_55, [0, 2, 1, 3]);  cat_55 = None
        convert_element_type_213: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_147, torch.float32);  permute_147 = None
        permute_149: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_213, [0, 1, 3, 2]);  convert_element_type_213 = None
        convert_element_type_214: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_149, torch.bfloat16);  permute_149 = None
        convert_element_type_215: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_148, torch.bfloat16);  permute_148 = None
        expand_81: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_215, _shape_param_392);  convert_element_type_215 = _shape_param_392 = None
        view_310: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_81, _shape_param_393);  expand_81 = _shape_param_393 = None
        expand_82: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_214, _shape_param_394);  convert_element_type_214 = _shape_param_394 = None
        view_311: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_82, _shape_param_395);  expand_82 = _shape_param_395 = None
        bmm_26: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_310, view_311)
        view_312: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_26, _shape_param_396);  bmm_26 = _shape_param_396 = None
        div_26: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_312, 16.0);  view_312 = None
        add_121: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_26, where);  div_26 = None
        amax_13: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_121, [-1], True)
        sub_27: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_121, amax_13);  add_121 = amax_13 = None
        exp_13: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_14: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_27: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_216: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16)
        expand_83: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_216, _shape_param_397);  convert_element_type_216 = _shape_param_397 = None
        view_313: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_83, _shape_param_398);  expand_83 = _shape_param_398 = None
        expand_84: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_146, _shape_param_399);  permute_146 = _shape_param_399 = None
        view_314: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_84, _shape_param_400);  expand_84 = _shape_param_400 = None
        bmm_27: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_313, view_314)
        view_315: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_27, _shape_param_401);  bmm_27 = _shape_param_401 = None
        permute_150: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
        clone_41: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_316: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_41, _shape_param_402);  clone_41 = _shape_param_402 = None
        convert_element_type_217: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg153_1, torch.bfloat16);  arg153_1 = None
        permute_151: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_217, [1, 0]);  convert_element_type_217 = None
        view_317: "bf16[128, 4096]" = torch.ops.aten.view.default(view_316, _shape_param_403);  view_316 = _shape_param_403 = None
        mm_55: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_317, permute_151)
        view_318: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_55, _shape_param_404);  mm_55 = _shape_param_404 = None
        convert_element_type_218: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg154_1, torch.bfloat16);  arg154_1 = None
        convert_element_type_219: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg155_1, torch.bfloat16);  arg155_1 = None
        permute_152: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_219, [1, 0]);  convert_element_type_219 = None
        addmm_26: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_218, view_299, permute_152);  convert_element_type_218 = None
        view_319: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_26, _shape_param_405);  _shape_param_405 = None
        mul_136: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_319, 0.5)
        convert_element_type_220: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_319, torch.float32)
        pow_14: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_220, 3.0);  convert_element_type_220 = None
        mul_137: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_122: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_319, mul_137);  view_319 = mul_137 = None
        mul_138: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_122, 0.7978845608028654);  add_122 = None
        tanh_13: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_138);  mul_138 = None
        add_123: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_139: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_136, add_123);  mul_136 = add_123 = None
        convert_element_type_221: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg156_1, torch.bfloat16);  arg156_1 = None
        convert_element_type_222: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg157_1, torch.bfloat16);  arg157_1 = None
        convert_element_type_223: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_139, torch.bfloat16);  mul_139 = None
        view_320: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_223, _shape_param_406);  convert_element_type_223 = _shape_param_406 = None
        permute_153: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_222, [1, 0]);  convert_element_type_222 = None
        addmm_27: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_221, view_320, permute_153);  convert_element_type_221 = None
        view_321: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_27, _shape_param_407);  addmm_27 = _shape_param_407 = None
        add_124: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_318, view_321);  view_318 = view_321 = None
        add_125: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_124, add_116);  add_124 = add_116 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
        getitem_56: "f32[1, 128, 1]" = var_mean_14[0]
        getitem_57: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None
        add_126: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_28: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_125, getitem_57);  getitem_57 = None
        mul_140: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_14);  sub_28 = None
        mul_141: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_140, arg158_1);  arg158_1 = None
        add_127: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_141, arg159_1);  mul_141 = arg159_1 = None
        convert_element_type_224: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg160_1, torch.bfloat16);  arg160_1 = None
        convert_element_type_225: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_127, torch.bfloat16);  add_127 = None
        permute_154: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_224, [1, 0]);  convert_element_type_224 = None
        view_322: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_225, _shape_param_408);  convert_element_type_225 = _shape_param_408 = None
        mm_56: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_322, permute_154)
        view_323: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_56, _shape_param_409);  mm_56 = _shape_param_409 = None
        convert_element_type_226: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg161_1, torch.bfloat16);  arg161_1 = None
        permute_155: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_226, [1, 0]);  convert_element_type_226 = None
        mm_57: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_322, permute_155)
        view_324: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_57, _shape_param_410);  mm_57 = _shape_param_410 = None
        convert_element_type_227: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg162_1, torch.bfloat16);  arg162_1 = None
        permute_156: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_227, [1, 0]);  convert_element_type_227 = None
        mm_58: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_322, permute_156)
        view_325: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_58, _shape_param_411);  mm_58 = _shape_param_411 = None
        view_326: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_323, _shape_param_412);  view_323 = _shape_param_412 = None
        view_327: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_324, _shape_param_413);  view_324 = _shape_param_413 = None
        view_328: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_325, _shape_param_414);  view_325 = _shape_param_414 = None
        permute_157: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None
        repeat_15: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg163_1, [1, 1, 1]);  arg163_1 = None
        gather_14: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_15, 1, repeat_1);  repeat_15 = None
        convert_element_type_228: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_14, torch.bfloat16);  gather_14 = None
        split_14 = torch.ops.aten.split.Tensor(convert_element_type_228, 32, -1);  convert_element_type_228 = None
        getitem_58: "bf16[1, 128, 32]" = split_14[0]
        getitem_59: "bf16[1, 128, 32]" = split_14[1];  split_14 = None
        slice_113: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_327, 3, 0, 64)
        slice_114: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_327, 3, 64, 9223372036854775807);  view_327 = None
        slice_115: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_326, 3, 0, 64)
        slice_116: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_326, 3, 64, 9223372036854775807);  view_326 = None
        unsqueeze_119: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_58, 2);  getitem_58 = None
        unsqueeze_120: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 4);  unsqueeze_119 = None
        expand_85: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_120, _shape_param_415);  _shape_param_415 = None
        clone_42: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_329: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_42, _shape_param_416);  clone_42 = _shape_param_416 = None
        unsqueeze_121: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_59, 2);  getitem_59 = None
        unsqueeze_122: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 4);  unsqueeze_121 = None
        expand_86: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_122, _shape_param_417);  _shape_param_417 = None
        clone_43: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
        view_330: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_43, _shape_param_418);  clone_43 = _shape_param_418 = None
        mul_142: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_113, view_330)
        slice_117: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_113, 3, 0, 9223372036854775807, 2)
        slice_118: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_113, 3, 1, 9223372036854775807, 2);  slice_113 = None
        neg_28: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_118);  slice_118 = None
        unsqueeze_123: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_28, 4);  neg_28 = None
        unsqueeze_124: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_117, 4);  slice_117 = None
        cat_56: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_123, unsqueeze_124], -1);  unsqueeze_123 = unsqueeze_124 = None
        view_331: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_56, _shape_param_419);  cat_56 = _shape_param_419 = None
        mul_143: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_331, view_329);  view_331 = None
        add_128: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
        mul_144: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_115, view_330);  view_330 = None
        slice_119: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_115, 3, 0, 9223372036854775807, 2)
        slice_120: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_115, 3, 1, 9223372036854775807, 2);  slice_115 = None
        neg_29: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_120);  slice_120 = None
        unsqueeze_125: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_29, 4);  neg_29 = None
        unsqueeze_126: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_119, 4);  slice_119 = None
        cat_57: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_125, unsqueeze_126], -1);  unsqueeze_125 = unsqueeze_126 = None
        view_332: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_57, _shape_param_420);  cat_57 = _shape_param_420 = None
        mul_145: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_332, view_329);  view_332 = view_329 = None
        add_129: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        cat_58: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_128, slice_114], -1);  add_128 = slice_114 = None
        cat_59: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_129, slice_116], -1);  add_129 = slice_116 = None
        permute_158: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_58, [0, 2, 1, 3]);  cat_58 = None
        permute_159: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_59, [0, 2, 1, 3]);  cat_59 = None
        convert_element_type_229: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_158, torch.float32);  permute_158 = None
        permute_160: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_229, [0, 1, 3, 2]);  convert_element_type_229 = None
        convert_element_type_230: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_160, torch.bfloat16);  permute_160 = None
        convert_element_type_231: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_159, torch.bfloat16);  permute_159 = None
        expand_87: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_231, _shape_param_421);  convert_element_type_231 = _shape_param_421 = None
        view_333: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_87, _shape_param_422);  expand_87 = _shape_param_422 = None
        expand_88: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_230, _shape_param_423);  convert_element_type_230 = _shape_param_423 = None
        view_334: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_88, _shape_param_424);  expand_88 = _shape_param_424 = None
        bmm_28: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_333, view_334)
        view_335: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_28, _shape_param_425);  bmm_28 = _shape_param_425 = None
        div_28: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_335, 16.0);  view_335 = None
        add_130: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_28, where);  div_28 = None
        amax_14: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_130, [-1], True)
        sub_29: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_130, amax_14);  add_130 = amax_14 = None
        exp_14: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
        sum_15: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_29: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_232: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_29, torch.bfloat16)
        expand_89: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_232, _shape_param_426);  convert_element_type_232 = _shape_param_426 = None
        view_336: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_89, _shape_param_427);  expand_89 = _shape_param_427 = None
        expand_90: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_157, _shape_param_428);  permute_157 = _shape_param_428 = None
        view_337: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_90, _shape_param_429);  expand_90 = _shape_param_429 = None
        bmm_29: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_336, view_337)
        view_338: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_29, _shape_param_430);  bmm_29 = _shape_param_430 = None
        permute_161: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
        clone_44: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_339: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_44, _shape_param_431);  clone_44 = _shape_param_431 = None
        convert_element_type_233: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg164_1, torch.bfloat16);  arg164_1 = None
        permute_162: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_233, [1, 0]);  convert_element_type_233 = None
        view_340: "bf16[128, 4096]" = torch.ops.aten.view.default(view_339, _shape_param_432);  view_339 = _shape_param_432 = None
        mm_59: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_340, permute_162)
        view_341: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_59, _shape_param_433);  mm_59 = _shape_param_433 = None
        convert_element_type_234: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg165_1, torch.bfloat16);  arg165_1 = None
        convert_element_type_235: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg166_1, torch.bfloat16);  arg166_1 = None
        permute_163: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_235, [1, 0]);  convert_element_type_235 = None
        addmm_28: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_234, view_322, permute_163);  convert_element_type_234 = None
        view_342: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_28, _shape_param_434);  _shape_param_434 = None
        mul_146: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_342, 0.5)
        convert_element_type_236: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_342, torch.float32)
        pow_15: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_236, 3.0);  convert_element_type_236 = None
        mul_147: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_131: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_342, mul_147);  view_342 = mul_147 = None
        mul_148: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_131, 0.7978845608028654);  add_131 = None
        tanh_14: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_148);  mul_148 = None
        add_132: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_149: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_146, add_132);  mul_146 = add_132 = None
        convert_element_type_237: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg167_1, torch.bfloat16);  arg167_1 = None
        convert_element_type_238: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg168_1, torch.bfloat16);  arg168_1 = None
        convert_element_type_239: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_149, torch.bfloat16);  mul_149 = None
        view_343: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_239, _shape_param_435);  convert_element_type_239 = _shape_param_435 = None
        permute_164: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_238, [1, 0]);  convert_element_type_238 = None
        addmm_29: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_237, view_343, permute_164);  convert_element_type_237 = None
        view_344: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_29, _shape_param_436);  addmm_29 = _shape_param_436 = None
        add_133: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_341, view_344);  view_341 = view_344 = None
        add_134: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_133, add_125);  add_133 = add_125 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_134, [2], correction = 0, keepdim = True)
        getitem_60: "f32[1, 128, 1]" = var_mean_15[0]
        getitem_61: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None
        add_135: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
        sub_30: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_134, getitem_61);  getitem_61 = None
        mul_150: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_15);  sub_30 = None
        mul_151: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_150, arg169_1);  arg169_1 = None
        add_136: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_151, arg170_1);  mul_151 = arg170_1 = None
        convert_element_type_240: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg171_1, torch.bfloat16);  arg171_1 = None
        convert_element_type_241: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_136, torch.bfloat16);  add_136 = None
        permute_165: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_240, [1, 0]);  convert_element_type_240 = None
        view_345: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_241, _shape_param_437);  convert_element_type_241 = _shape_param_437 = None
        mm_60: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_345, permute_165)
        view_346: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_60, _shape_param_438);  mm_60 = _shape_param_438 = None
        convert_element_type_242: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg172_1, torch.bfloat16);  arg172_1 = None
        permute_166: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_242, [1, 0]);  convert_element_type_242 = None
        mm_61: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_345, permute_166)
        view_347: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_61, _shape_param_439);  mm_61 = _shape_param_439 = None
        convert_element_type_243: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg173_1, torch.bfloat16);  arg173_1 = None
        permute_167: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_243, [1, 0]);  convert_element_type_243 = None
        mm_62: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_345, permute_167)
        view_348: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_62, _shape_param_440);  mm_62 = _shape_param_440 = None
        view_349: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_346, _shape_param_441);  view_346 = _shape_param_441 = None
        view_350: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_347, _shape_param_442);  view_347 = _shape_param_442 = None
        view_351: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_348, _shape_param_443);  view_348 = _shape_param_443 = None
        permute_168: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None
        repeat_16: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg174_1, [1, 1, 1]);  arg174_1 = None
        gather_15: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_16, 1, repeat_1);  repeat_16 = None
        convert_element_type_244: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_15, torch.bfloat16);  gather_15 = None
        split_15 = torch.ops.aten.split.Tensor(convert_element_type_244, 32, -1);  convert_element_type_244 = None
        getitem_62: "bf16[1, 128, 32]" = split_15[0]
        getitem_63: "bf16[1, 128, 32]" = split_15[1];  split_15 = None
        slice_121: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_350, 3, 0, 64)
        slice_122: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_350, 3, 64, 9223372036854775807);  view_350 = None
        slice_123: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_349, 3, 0, 64)
        slice_124: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_349, 3, 64, 9223372036854775807);  view_349 = None
        unsqueeze_127: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_62, 2);  getitem_62 = None
        unsqueeze_128: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 4);  unsqueeze_127 = None
        expand_91: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_128, _shape_param_444);  _shape_param_444 = None
        clone_45: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_352: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_45, _shape_param_445);  clone_45 = _shape_param_445 = None
        unsqueeze_129: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_63, 2);  getitem_63 = None
        unsqueeze_130: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 4);  unsqueeze_129 = None
        expand_92: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_130, _shape_param_446);  _shape_param_446 = None
        clone_46: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_353: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_46, _shape_param_447);  clone_46 = _shape_param_447 = None
        mul_152: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_121, view_353)
        slice_125: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_121, 3, 0, 9223372036854775807, 2)
        slice_126: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_121, 3, 1, 9223372036854775807, 2);  slice_121 = None
        neg_30: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_126);  slice_126 = None
        unsqueeze_131: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_30, 4);  neg_30 = None
        unsqueeze_132: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_125, 4);  slice_125 = None
        cat_60: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_131, unsqueeze_132], -1);  unsqueeze_131 = unsqueeze_132 = None
        view_354: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_60, _shape_param_448);  cat_60 = _shape_param_448 = None
        mul_153: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_354, view_352);  view_354 = None
        add_137: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
        mul_154: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_123, view_353);  view_353 = None
        slice_127: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_123, 3, 0, 9223372036854775807, 2)
        slice_128: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_123, 3, 1, 9223372036854775807, 2);  slice_123 = None
        neg_31: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_128);  slice_128 = None
        unsqueeze_133: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_31, 4);  neg_31 = None
        unsqueeze_134: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_127, 4);  slice_127 = None
        cat_61: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_133, unsqueeze_134], -1);  unsqueeze_133 = unsqueeze_134 = None
        view_355: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_61, _shape_param_449);  cat_61 = _shape_param_449 = None
        mul_155: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_355, view_352);  view_355 = view_352 = None
        add_138: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None
        cat_62: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_137, slice_122], -1);  add_137 = slice_122 = None
        cat_63: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_138, slice_124], -1);  add_138 = slice_124 = None
        permute_169: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_62, [0, 2, 1, 3]);  cat_62 = None
        permute_170: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_63, [0, 2, 1, 3]);  cat_63 = None
        convert_element_type_245: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_169, torch.float32);  permute_169 = None
        permute_171: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_245, [0, 1, 3, 2]);  convert_element_type_245 = None
        convert_element_type_246: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_171, torch.bfloat16);  permute_171 = None
        convert_element_type_247: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_170, torch.bfloat16);  permute_170 = None
        expand_93: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_247, _shape_param_450);  convert_element_type_247 = _shape_param_450 = None
        view_356: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_93, _shape_param_451);  expand_93 = _shape_param_451 = None
        expand_94: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_246, _shape_param_452);  convert_element_type_246 = _shape_param_452 = None
        view_357: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_94, _shape_param_453);  expand_94 = _shape_param_453 = None
        bmm_30: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_356, view_357)
        view_358: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_30, _shape_param_454);  bmm_30 = _shape_param_454 = None
        div_30: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_358, 16.0);  view_358 = None
        add_139: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_30, where);  div_30 = None
        amax_15: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_139, [-1], True)
        sub_31: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_139, amax_15);  add_139 = amax_15 = None
        exp_15: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_16: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_31: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_248: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_31, torch.bfloat16)
        expand_95: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_248, _shape_param_455);  convert_element_type_248 = _shape_param_455 = None
        view_359: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_95, _shape_param_456);  expand_95 = _shape_param_456 = None
        expand_96: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_168, _shape_param_457);  permute_168 = _shape_param_457 = None
        view_360: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_96, _shape_param_458);  expand_96 = _shape_param_458 = None
        bmm_31: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_359, view_360)
        view_361: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_31, _shape_param_459);  bmm_31 = _shape_param_459 = None
        permute_172: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_361, [0, 2, 1, 3]);  view_361 = None
        clone_47: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_362: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_47, _shape_param_460);  clone_47 = _shape_param_460 = None
        convert_element_type_249: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg175_1, torch.bfloat16);  arg175_1 = None
        permute_173: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_249, [1, 0]);  convert_element_type_249 = None
        view_363: "bf16[128, 4096]" = torch.ops.aten.view.default(view_362, _shape_param_461);  view_362 = _shape_param_461 = None
        mm_63: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_363, permute_173)
        view_364: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_63, _shape_param_462);  mm_63 = _shape_param_462 = None
        convert_element_type_250: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg176_1, torch.bfloat16);  arg176_1 = None
        convert_element_type_251: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg177_1, torch.bfloat16);  arg177_1 = None
        permute_174: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_251, [1, 0]);  convert_element_type_251 = None
        addmm_30: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_250, view_345, permute_174);  convert_element_type_250 = None
        view_365: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_30, _shape_param_463);  _shape_param_463 = None
        mul_156: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_365, 0.5)
        convert_element_type_252: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_365, torch.float32)
        pow_16: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_252, 3.0);  convert_element_type_252 = None
        mul_157: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_140: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_365, mul_157);  view_365 = mul_157 = None
        mul_158: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_140, 0.7978845608028654);  add_140 = None
        tanh_15: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
        add_141: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_159: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_156, add_141);  mul_156 = add_141 = None
        convert_element_type_253: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg178_1, torch.bfloat16);  arg178_1 = None
        convert_element_type_254: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg179_1, torch.bfloat16);  arg179_1 = None
        convert_element_type_255: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_159, torch.bfloat16);  mul_159 = None
        view_366: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_255, _shape_param_464);  convert_element_type_255 = _shape_param_464 = None
        permute_175: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_254, [1, 0]);  convert_element_type_254 = None
        addmm_31: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_253, view_366, permute_175);  convert_element_type_253 = None
        view_367: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_31, _shape_param_465);  addmm_31 = _shape_param_465 = None
        add_142: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_364, view_367);  view_364 = view_367 = None
        add_143: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_142, add_134);  add_142 = add_134 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
        getitem_64: "f32[1, 128, 1]" = var_mean_16[0]
        getitem_65: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None
        add_144: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        sub_32: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_143, getitem_65);  getitem_65 = None
        mul_160: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_16);  sub_32 = None
        mul_161: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_160, arg180_1);  arg180_1 = None
        add_145: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_161, arg181_1);  mul_161 = arg181_1 = None
        convert_element_type_256: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg182_1, torch.bfloat16);  arg182_1 = None
        convert_element_type_257: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_145, torch.bfloat16);  add_145 = None
        permute_176: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_256, [1, 0]);  convert_element_type_256 = None
        view_368: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_257, _shape_param_466);  convert_element_type_257 = _shape_param_466 = None
        mm_64: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_368, permute_176)
        view_369: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_64, _shape_param_467);  mm_64 = _shape_param_467 = None
        convert_element_type_258: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg183_1, torch.bfloat16);  arg183_1 = None
        permute_177: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_258, [1, 0]);  convert_element_type_258 = None
        mm_65: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_368, permute_177)
        view_370: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_65, _shape_param_468);  mm_65 = _shape_param_468 = None
        convert_element_type_259: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg184_1, torch.bfloat16);  arg184_1 = None
        permute_178: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_259, [1, 0]);  convert_element_type_259 = None
        mm_66: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_368, permute_178)
        view_371: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_66, _shape_param_469);  mm_66 = _shape_param_469 = None
        view_372: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_369, _shape_param_470);  view_369 = _shape_param_470 = None
        view_373: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_370, _shape_param_471);  view_370 = _shape_param_471 = None
        view_374: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_371, _shape_param_472);  view_371 = _shape_param_472 = None
        permute_179: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
        repeat_17: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg185_1, [1, 1, 1]);  arg185_1 = None
        gather_16: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_17, 1, repeat_1);  repeat_17 = None
        convert_element_type_260: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_16, torch.bfloat16);  gather_16 = None
        split_16 = torch.ops.aten.split.Tensor(convert_element_type_260, 32, -1);  convert_element_type_260 = None
        getitem_66: "bf16[1, 128, 32]" = split_16[0]
        getitem_67: "bf16[1, 128, 32]" = split_16[1];  split_16 = None
        slice_129: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_373, 3, 0, 64)
        slice_130: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_373, 3, 64, 9223372036854775807);  view_373 = None
        slice_131: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_372, 3, 0, 64)
        slice_132: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_372, 3, 64, 9223372036854775807);  view_372 = None
        unsqueeze_135: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_66, 2);  getitem_66 = None
        unsqueeze_136: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 4);  unsqueeze_135 = None
        expand_97: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_136, _shape_param_473);  _shape_param_473 = None
        clone_48: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_375: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_48, _shape_param_474);  clone_48 = _shape_param_474 = None
        unsqueeze_137: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_67, 2);  getitem_67 = None
        unsqueeze_138: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 4);  unsqueeze_137 = None
        expand_98: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_138, _shape_param_475);  _shape_param_475 = None
        clone_49: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
        view_376: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_49, _shape_param_476);  clone_49 = _shape_param_476 = None
        mul_162: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_129, view_376)
        slice_133: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_129, 3, 0, 9223372036854775807, 2)
        slice_134: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_129, 3, 1, 9223372036854775807, 2);  slice_129 = None
        neg_32: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_134);  slice_134 = None
        unsqueeze_139: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_32, 4);  neg_32 = None
        unsqueeze_140: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_133, 4);  slice_133 = None
        cat_64: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_139, unsqueeze_140], -1);  unsqueeze_139 = unsqueeze_140 = None
        view_377: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_64, _shape_param_477);  cat_64 = _shape_param_477 = None
        mul_163: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_377, view_375);  view_377 = None
        add_146: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        mul_164: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_131, view_376);  view_376 = None
        slice_135: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_131, 3, 0, 9223372036854775807, 2)
        slice_136: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_131, 3, 1, 9223372036854775807, 2);  slice_131 = None
        neg_33: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_136);  slice_136 = None
        unsqueeze_141: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_33, 4);  neg_33 = None
        unsqueeze_142: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_135, 4);  slice_135 = None
        cat_65: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_141, unsqueeze_142], -1);  unsqueeze_141 = unsqueeze_142 = None
        view_378: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_65, _shape_param_478);  cat_65 = _shape_param_478 = None
        mul_165: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_378, view_375);  view_378 = view_375 = None
        add_147: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
        cat_66: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_146, slice_130], -1);  add_146 = slice_130 = None
        cat_67: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_147, slice_132], -1);  add_147 = slice_132 = None
        permute_180: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_66, [0, 2, 1, 3]);  cat_66 = None
        permute_181: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_67, [0, 2, 1, 3]);  cat_67 = None
        convert_element_type_261: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_180, torch.float32);  permute_180 = None
        permute_182: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_261, [0, 1, 3, 2]);  convert_element_type_261 = None
        convert_element_type_262: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_182, torch.bfloat16);  permute_182 = None
        convert_element_type_263: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_181, torch.bfloat16);  permute_181 = None
        expand_99: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_263, _shape_param_479);  convert_element_type_263 = _shape_param_479 = None
        view_379: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_99, _shape_param_480);  expand_99 = _shape_param_480 = None
        expand_100: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_262, _shape_param_481);  convert_element_type_262 = _shape_param_481 = None
        view_380: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_100, _shape_param_482);  expand_100 = _shape_param_482 = None
        bmm_32: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_379, view_380)
        view_381: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_32, _shape_param_483);  bmm_32 = _shape_param_483 = None
        div_32: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_381, 16.0);  view_381 = None
        add_148: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_32, where);  div_32 = None
        amax_16: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_148, [-1], True)
        sub_33: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_148, amax_16);  add_148 = amax_16 = None
        exp_16: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
        sum_17: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_33: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_264: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_33, torch.bfloat16)
        expand_101: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_264, _shape_param_484);  convert_element_type_264 = _shape_param_484 = None
        view_382: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_101, _shape_param_485);  expand_101 = _shape_param_485 = None
        expand_102: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_179, _shape_param_486);  permute_179 = _shape_param_486 = None
        view_383: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_102, _shape_param_487);  expand_102 = _shape_param_487 = None
        bmm_33: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_382, view_383)
        view_384: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_33, _shape_param_488);  bmm_33 = _shape_param_488 = None
        permute_183: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
        clone_50: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        view_385: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_50, _shape_param_489);  clone_50 = _shape_param_489 = None
        convert_element_type_265: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg186_1, torch.bfloat16);  arg186_1 = None
        permute_184: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_265, [1, 0]);  convert_element_type_265 = None
        view_386: "bf16[128, 4096]" = torch.ops.aten.view.default(view_385, _shape_param_490);  view_385 = _shape_param_490 = None
        mm_67: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_386, permute_184)
        view_387: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_67, _shape_param_491);  mm_67 = _shape_param_491 = None
        convert_element_type_266: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg187_1, torch.bfloat16);  arg187_1 = None
        convert_element_type_267: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg188_1, torch.bfloat16);  arg188_1 = None
        permute_185: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_267, [1, 0]);  convert_element_type_267 = None
        addmm_32: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_266, view_368, permute_185);  convert_element_type_266 = None
        view_388: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_32, _shape_param_492);  _shape_param_492 = None
        mul_166: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_388, 0.5)
        convert_element_type_268: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_388, torch.float32)
        pow_17: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_268, 3.0);  convert_element_type_268 = None
        mul_167: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_149: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_388, mul_167);  view_388 = mul_167 = None
        mul_168: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_149, 0.7978845608028654);  add_149 = None
        tanh_16: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_168);  mul_168 = None
        add_150: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_16, 1.0);  tanh_16 = None
        mul_169: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_166, add_150);  mul_166 = add_150 = None
        convert_element_type_269: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg189_1, torch.bfloat16);  arg189_1 = None
        convert_element_type_270: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg190_1, torch.bfloat16);  arg190_1 = None
        convert_element_type_271: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_169, torch.bfloat16);  mul_169 = None
        view_389: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_271, _shape_param_493);  convert_element_type_271 = _shape_param_493 = None
        permute_186: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_270, [1, 0]);  convert_element_type_270 = None
        addmm_33: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_269, view_389, permute_186);  convert_element_type_269 = None
        view_390: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_33, _shape_param_494);  addmm_33 = _shape_param_494 = None
        add_151: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_387, view_390);  view_387 = view_390 = None
        add_152: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_151, add_143);  add_151 = add_143 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_152, [2], correction = 0, keepdim = True)
        getitem_68: "f32[1, 128, 1]" = var_mean_17[0]
        getitem_69: "f32[1, 128, 1]" = var_mean_17[1];  var_mean_17 = None
        add_153: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_17: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        sub_34: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_152, getitem_69);  getitem_69 = None
        mul_170: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_17);  sub_34 = None
        mul_171: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_170, arg191_1);  arg191_1 = None
        add_154: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_171, arg192_1);  mul_171 = arg192_1 = None
        convert_element_type_272: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg193_1, torch.bfloat16);  arg193_1 = None
        convert_element_type_273: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_154, torch.bfloat16);  add_154 = None
        permute_187: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_272, [1, 0]);  convert_element_type_272 = None
        view_391: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_273, _shape_param_495);  convert_element_type_273 = _shape_param_495 = None
        mm_68: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_391, permute_187)
        view_392: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_68, _shape_param_496);  mm_68 = _shape_param_496 = None
        convert_element_type_274: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg194_1, torch.bfloat16);  arg194_1 = None
        permute_188: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_274, [1, 0]);  convert_element_type_274 = None
        mm_69: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_391, permute_188)
        view_393: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_69, _shape_param_497);  mm_69 = _shape_param_497 = None
        convert_element_type_275: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg195_1, torch.bfloat16);  arg195_1 = None
        permute_189: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_275, [1, 0]);  convert_element_type_275 = None
        mm_70: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_391, permute_189)
        view_394: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_70, _shape_param_498);  mm_70 = _shape_param_498 = None
        view_395: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_392, _shape_param_499);  view_392 = _shape_param_499 = None
        view_396: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_393, _shape_param_500);  view_393 = _shape_param_500 = None
        view_397: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_394, _shape_param_501);  view_394 = _shape_param_501 = None
        permute_190: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_397, [0, 2, 1, 3]);  view_397 = None
        repeat_18: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg196_1, [1, 1, 1]);  arg196_1 = None
        gather_17: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_18, 1, repeat_1);  repeat_18 = None
        convert_element_type_276: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_17, torch.bfloat16);  gather_17 = None
        split_17 = torch.ops.aten.split.Tensor(convert_element_type_276, 32, -1);  convert_element_type_276 = None
        getitem_70: "bf16[1, 128, 32]" = split_17[0]
        getitem_71: "bf16[1, 128, 32]" = split_17[1];  split_17 = None
        slice_137: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_396, 3, 0, 64)
        slice_138: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_396, 3, 64, 9223372036854775807);  view_396 = None
        slice_139: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_395, 3, 0, 64)
        slice_140: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_395, 3, 64, 9223372036854775807);  view_395 = None
        unsqueeze_143: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_70, 2);  getitem_70 = None
        unsqueeze_144: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 4);  unsqueeze_143 = None
        expand_103: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_144, _shape_param_502);  _shape_param_502 = None
        clone_51: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_103, memory_format = torch.contiguous_format);  expand_103 = None
        view_398: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_51, _shape_param_503);  clone_51 = _shape_param_503 = None
        unsqueeze_145: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_71, 2);  getitem_71 = None
        unsqueeze_146: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 4);  unsqueeze_145 = None
        expand_104: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_146, _shape_param_504);  _shape_param_504 = None
        clone_52: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
        view_399: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_52, _shape_param_505);  clone_52 = _shape_param_505 = None
        mul_172: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_137, view_399)
        slice_141: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_137, 3, 0, 9223372036854775807, 2)
        slice_142: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_137, 3, 1, 9223372036854775807, 2);  slice_137 = None
        neg_34: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_142);  slice_142 = None
        unsqueeze_147: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_34, 4);  neg_34 = None
        unsqueeze_148: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_141, 4);  slice_141 = None
        cat_68: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_147, unsqueeze_148], -1);  unsqueeze_147 = unsqueeze_148 = None
        view_400: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_68, _shape_param_506);  cat_68 = _shape_param_506 = None
        mul_173: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_400, view_398);  view_400 = None
        add_155: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
        mul_174: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_139, view_399);  view_399 = None
        slice_143: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_139, 3, 0, 9223372036854775807, 2)
        slice_144: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_139, 3, 1, 9223372036854775807, 2);  slice_139 = None
        neg_35: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_144);  slice_144 = None
        unsqueeze_149: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_35, 4);  neg_35 = None
        unsqueeze_150: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_143, 4);  slice_143 = None
        cat_69: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_149, unsqueeze_150], -1);  unsqueeze_149 = unsqueeze_150 = None
        view_401: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_69, _shape_param_507);  cat_69 = _shape_param_507 = None
        mul_175: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_401, view_398);  view_401 = view_398 = None
        add_156: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_174, mul_175);  mul_174 = mul_175 = None
        cat_70: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_155, slice_138], -1);  add_155 = slice_138 = None
        cat_71: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_156, slice_140], -1);  add_156 = slice_140 = None
        permute_191: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_70, [0, 2, 1, 3]);  cat_70 = None
        permute_192: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_71, [0, 2, 1, 3]);  cat_71 = None
        convert_element_type_277: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_191, torch.float32);  permute_191 = None
        permute_193: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_277, [0, 1, 3, 2]);  convert_element_type_277 = None
        convert_element_type_278: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_193, torch.bfloat16);  permute_193 = None
        convert_element_type_279: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_192, torch.bfloat16);  permute_192 = None
        expand_105: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_279, _shape_param_508);  convert_element_type_279 = _shape_param_508 = None
        view_402: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_105, _shape_param_509);  expand_105 = _shape_param_509 = None
        expand_106: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_278, _shape_param_510);  convert_element_type_278 = _shape_param_510 = None
        view_403: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_106, _shape_param_511);  expand_106 = _shape_param_511 = None
        bmm_34: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_402, view_403)
        view_404: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_34, _shape_param_512);  bmm_34 = _shape_param_512 = None
        div_34: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_404, 16.0);  view_404 = None
        add_157: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_34, where);  div_34 = None
        amax_17: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_157, [-1], True)
        sub_35: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_157, amax_17);  add_157 = amax_17 = None
        exp_17: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
        sum_18: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_35: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_280: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_35, torch.bfloat16)
        expand_107: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_280, _shape_param_513);  convert_element_type_280 = _shape_param_513 = None
        view_405: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_107, _shape_param_514);  expand_107 = _shape_param_514 = None
        expand_108: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_190, _shape_param_515);  permute_190 = _shape_param_515 = None
        view_406: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_108, _shape_param_516);  expand_108 = _shape_param_516 = None
        bmm_35: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_405, view_406)
        view_407: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_35, _shape_param_517);  bmm_35 = _shape_param_517 = None
        permute_194: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_407, [0, 2, 1, 3]);  view_407 = None
        clone_53: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_408: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_53, _shape_param_518);  clone_53 = _shape_param_518 = None
        convert_element_type_281: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg197_1, torch.bfloat16);  arg197_1 = None
        permute_195: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_281, [1, 0]);  convert_element_type_281 = None
        view_409: "bf16[128, 4096]" = torch.ops.aten.view.default(view_408, _shape_param_519);  view_408 = _shape_param_519 = None
        mm_71: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_409, permute_195)
        view_410: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_71, _shape_param_520);  mm_71 = _shape_param_520 = None
        convert_element_type_282: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg198_1, torch.bfloat16);  arg198_1 = None
        convert_element_type_283: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg199_1, torch.bfloat16);  arg199_1 = None
        permute_196: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_283, [1, 0]);  convert_element_type_283 = None
        addmm_34: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_282, view_391, permute_196);  convert_element_type_282 = None
        view_411: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_34, _shape_param_521);  _shape_param_521 = None
        mul_176: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_411, 0.5)
        convert_element_type_284: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_411, torch.float32)
        pow_18: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_284, 3.0);  convert_element_type_284 = None
        mul_177: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_158: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_411, mul_177);  view_411 = mul_177 = None
        mul_178: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_158, 0.7978845608028654);  add_158 = None
        tanh_17: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_178);  mul_178 = None
        add_159: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_17, 1.0);  tanh_17 = None
        mul_179: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_176, add_159);  mul_176 = add_159 = None
        convert_element_type_285: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg200_1, torch.bfloat16);  arg200_1 = None
        convert_element_type_286: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg201_1, torch.bfloat16);  arg201_1 = None
        convert_element_type_287: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_179, torch.bfloat16);  mul_179 = None
        view_412: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_287, _shape_param_522);  convert_element_type_287 = _shape_param_522 = None
        permute_197: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_286, [1, 0]);  convert_element_type_286 = None
        addmm_35: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_285, view_412, permute_197);  convert_element_type_285 = None
        view_413: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_35, _shape_param_523);  addmm_35 = _shape_param_523 = None
        add_160: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_410, view_413);  view_410 = view_413 = None
        add_161: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_160, add_152);  add_160 = add_152 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
        getitem_72: "f32[1, 128, 1]" = var_mean_18[0]
        getitem_73: "f32[1, 128, 1]" = var_mean_18[1];  var_mean_18 = None
        add_162: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_18: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_36: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_161, getitem_73);  getitem_73 = None
        mul_180: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_18);  sub_36 = None
        mul_181: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_180, arg202_1);  arg202_1 = None
        add_163: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_181, arg203_1);  mul_181 = arg203_1 = None
        convert_element_type_288: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg204_1, torch.bfloat16);  arg204_1 = None
        convert_element_type_289: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_163, torch.bfloat16);  add_163 = None
        permute_198: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_288, [1, 0]);  convert_element_type_288 = None
        view_414: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_289, _shape_param_524);  convert_element_type_289 = _shape_param_524 = None
        mm_72: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_414, permute_198)
        view_415: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_72, _shape_param_525);  mm_72 = _shape_param_525 = None
        convert_element_type_290: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg205_1, torch.bfloat16);  arg205_1 = None
        permute_199: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_290, [1, 0]);  convert_element_type_290 = None
        mm_73: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_414, permute_199)
        view_416: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_73, _shape_param_526);  mm_73 = _shape_param_526 = None
        convert_element_type_291: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg206_1, torch.bfloat16);  arg206_1 = None
        permute_200: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_291, [1, 0]);  convert_element_type_291 = None
        mm_74: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_414, permute_200)
        view_417: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_74, _shape_param_527);  mm_74 = _shape_param_527 = None
        view_418: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_415, _shape_param_528);  view_415 = _shape_param_528 = None
        view_419: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_416, _shape_param_529);  view_416 = _shape_param_529 = None
        view_420: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_417, _shape_param_530);  view_417 = _shape_param_530 = None
        permute_201: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
        repeat_19: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg207_1, [1, 1, 1]);  arg207_1 = None
        gather_18: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_19, 1, repeat_1);  repeat_19 = None
        convert_element_type_292: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_18, torch.bfloat16);  gather_18 = None
        split_18 = torch.ops.aten.split.Tensor(convert_element_type_292, 32, -1);  convert_element_type_292 = None
        getitem_74: "bf16[1, 128, 32]" = split_18[0]
        getitem_75: "bf16[1, 128, 32]" = split_18[1];  split_18 = None
        slice_145: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_419, 3, 0, 64)
        slice_146: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_419, 3, 64, 9223372036854775807);  view_419 = None
        slice_147: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_418, 3, 0, 64)
        slice_148: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_418, 3, 64, 9223372036854775807);  view_418 = None
        unsqueeze_151: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_74, 2);  getitem_74 = None
        unsqueeze_152: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 4);  unsqueeze_151 = None
        expand_109: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_152, _shape_param_531);  _shape_param_531 = None
        clone_54: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_109, memory_format = torch.contiguous_format);  expand_109 = None
        view_421: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_54, _shape_param_532);  clone_54 = _shape_param_532 = None
        unsqueeze_153: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_75, 2);  getitem_75 = None
        unsqueeze_154: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 4);  unsqueeze_153 = None
        expand_110: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_154, _shape_param_533);  _shape_param_533 = None
        clone_55: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_110, memory_format = torch.contiguous_format);  expand_110 = None
        view_422: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_55, _shape_param_534);  clone_55 = _shape_param_534 = None
        mul_182: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_145, view_422)
        slice_149: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_145, 3, 0, 9223372036854775807, 2)
        slice_150: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_145, 3, 1, 9223372036854775807, 2);  slice_145 = None
        neg_36: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_150);  slice_150 = None
        unsqueeze_155: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_36, 4);  neg_36 = None
        unsqueeze_156: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_149, 4);  slice_149 = None
        cat_72: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_155, unsqueeze_156], -1);  unsqueeze_155 = unsqueeze_156 = None
        view_423: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_72, _shape_param_535);  cat_72 = _shape_param_535 = None
        mul_183: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_423, view_421);  view_423 = None
        add_164: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None
        mul_184: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_147, view_422);  view_422 = None
        slice_151: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_147, 3, 0, 9223372036854775807, 2)
        slice_152: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_147, 3, 1, 9223372036854775807, 2);  slice_147 = None
        neg_37: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_152);  slice_152 = None
        unsqueeze_157: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_37, 4);  neg_37 = None
        unsqueeze_158: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_151, 4);  slice_151 = None
        cat_73: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_157, unsqueeze_158], -1);  unsqueeze_157 = unsqueeze_158 = None
        view_424: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_73, _shape_param_536);  cat_73 = _shape_param_536 = None
        mul_185: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_424, view_421);  view_424 = view_421 = None
        add_165: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_184, mul_185);  mul_184 = mul_185 = None
        cat_74: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_164, slice_146], -1);  add_164 = slice_146 = None
        cat_75: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_165, slice_148], -1);  add_165 = slice_148 = None
        permute_202: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_74, [0, 2, 1, 3]);  cat_74 = None
        permute_203: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_75, [0, 2, 1, 3]);  cat_75 = None
        convert_element_type_293: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_202, torch.float32);  permute_202 = None
        permute_204: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_293, [0, 1, 3, 2]);  convert_element_type_293 = None
        convert_element_type_294: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_204, torch.bfloat16);  permute_204 = None
        convert_element_type_295: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_203, torch.bfloat16);  permute_203 = None
        expand_111: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_295, _shape_param_537);  convert_element_type_295 = _shape_param_537 = None
        view_425: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_111, _shape_param_538);  expand_111 = _shape_param_538 = None
        expand_112: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_294, _shape_param_539);  convert_element_type_294 = _shape_param_539 = None
        view_426: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_112, _shape_param_540);  expand_112 = _shape_param_540 = None
        bmm_36: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_425, view_426)
        view_427: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_36, _shape_param_541);  bmm_36 = _shape_param_541 = None
        div_36: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_427, 16.0);  view_427 = None
        add_166: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_36, where);  div_36 = None
        amax_18: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_166, [-1], True)
        sub_37: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_166, amax_18);  add_166 = amax_18 = None
        exp_18: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_19: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_37: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_296: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_37, torch.bfloat16)
        expand_113: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_296, _shape_param_542);  convert_element_type_296 = _shape_param_542 = None
        view_428: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_113, _shape_param_543);  expand_113 = _shape_param_543 = None
        expand_114: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_201, _shape_param_544);  permute_201 = _shape_param_544 = None
        view_429: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_114, _shape_param_545);  expand_114 = _shape_param_545 = None
        bmm_37: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_428, view_429)
        view_430: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_37, _shape_param_546);  bmm_37 = _shape_param_546 = None
        permute_205: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
        clone_56: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_431: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_56, _shape_param_547);  clone_56 = _shape_param_547 = None
        convert_element_type_297: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg208_1, torch.bfloat16);  arg208_1 = None
        permute_206: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_297, [1, 0]);  convert_element_type_297 = None
        view_432: "bf16[128, 4096]" = torch.ops.aten.view.default(view_431, _shape_param_548);  view_431 = _shape_param_548 = None
        mm_75: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_432, permute_206)
        view_433: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_75, _shape_param_549);  mm_75 = _shape_param_549 = None
        convert_element_type_298: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg209_1, torch.bfloat16);  arg209_1 = None
        convert_element_type_299: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg210_1, torch.bfloat16);  arg210_1 = None
        permute_207: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_299, [1, 0]);  convert_element_type_299 = None
        addmm_36: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_298, view_414, permute_207);  convert_element_type_298 = None
        view_434: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_36, _shape_param_550);  _shape_param_550 = None
        mul_186: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_434, 0.5)
        convert_element_type_300: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_434, torch.float32)
        pow_19: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_300, 3.0);  convert_element_type_300 = None
        mul_187: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_167: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_434, mul_187);  view_434 = mul_187 = None
        mul_188: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_167, 0.7978845608028654);  add_167 = None
        tanh_18: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_188);  mul_188 = None
        add_168: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_18, 1.0);  tanh_18 = None
        mul_189: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_186, add_168);  mul_186 = add_168 = None
        convert_element_type_301: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg211_1, torch.bfloat16);  arg211_1 = None
        convert_element_type_302: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg212_1, torch.bfloat16);  arg212_1 = None
        convert_element_type_303: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_189, torch.bfloat16);  mul_189 = None
        view_435: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_303, _shape_param_551);  convert_element_type_303 = _shape_param_551 = None
        permute_208: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_302, [1, 0]);  convert_element_type_302 = None
        addmm_37: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_301, view_435, permute_208);  convert_element_type_301 = None
        view_436: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_37, _shape_param_552);  addmm_37 = _shape_param_552 = None
        add_169: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_433, view_436);  view_433 = view_436 = None
        add_170: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_169, add_161);  add_169 = add_161 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_170, [2], correction = 0, keepdim = True)
        getitem_76: "f32[1, 128, 1]" = var_mean_19[0]
        getitem_77: "f32[1, 128, 1]" = var_mean_19[1];  var_mean_19 = None
        add_171: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_19: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        sub_38: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_170, getitem_77);  getitem_77 = None
        mul_190: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_19);  sub_38 = None
        mul_191: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_190, arg213_1);  arg213_1 = None
        add_172: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_191, arg214_1);  mul_191 = arg214_1 = None
        convert_element_type_304: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg215_1, torch.bfloat16);  arg215_1 = None
        convert_element_type_305: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_172, torch.bfloat16);  add_172 = None
        permute_209: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_304, [1, 0]);  convert_element_type_304 = None
        view_437: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_305, _shape_param_553);  convert_element_type_305 = _shape_param_553 = None
        mm_76: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_437, permute_209)
        view_438: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_76, _shape_param_554);  mm_76 = _shape_param_554 = None
        convert_element_type_306: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg216_1, torch.bfloat16);  arg216_1 = None
        permute_210: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_306, [1, 0]);  convert_element_type_306 = None
        mm_77: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_437, permute_210)
        view_439: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_77, _shape_param_555);  mm_77 = _shape_param_555 = None
        convert_element_type_307: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg217_1, torch.bfloat16);  arg217_1 = None
        permute_211: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_307, [1, 0]);  convert_element_type_307 = None
        mm_78: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_437, permute_211)
        view_440: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_78, _shape_param_556);  mm_78 = _shape_param_556 = None
        view_441: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_438, _shape_param_557);  view_438 = _shape_param_557 = None
        view_442: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_439, _shape_param_558);  view_439 = _shape_param_558 = None
        view_443: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_440, _shape_param_559);  view_440 = _shape_param_559 = None
        permute_212: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None
        repeat_20: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg218_1, [1, 1, 1]);  arg218_1 = None
        gather_19: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_20, 1, repeat_1);  repeat_20 = None
        convert_element_type_308: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_19, torch.bfloat16);  gather_19 = None
        split_19 = torch.ops.aten.split.Tensor(convert_element_type_308, 32, -1);  convert_element_type_308 = None
        getitem_78: "bf16[1, 128, 32]" = split_19[0]
        getitem_79: "bf16[1, 128, 32]" = split_19[1];  split_19 = None
        slice_153: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_442, 3, 0, 64)
        slice_154: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_442, 3, 64, 9223372036854775807);  view_442 = None
        slice_155: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_441, 3, 0, 64)
        slice_156: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_441, 3, 64, 9223372036854775807);  view_441 = None
        unsqueeze_159: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_78, 2);  getitem_78 = None
        unsqueeze_160: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 4);  unsqueeze_159 = None
        expand_115: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_160, _shape_param_560);  _shape_param_560 = None
        clone_57: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_115, memory_format = torch.contiguous_format);  expand_115 = None
        view_444: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_57, _shape_param_561);  clone_57 = _shape_param_561 = None
        unsqueeze_161: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_79, 2);  getitem_79 = None
        unsqueeze_162: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 4);  unsqueeze_161 = None
        expand_116: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_162, _shape_param_562);  _shape_param_562 = None
        clone_58: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_116, memory_format = torch.contiguous_format);  expand_116 = None
        view_445: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_58, _shape_param_563);  clone_58 = _shape_param_563 = None
        mul_192: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_153, view_445)
        slice_157: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_153, 3, 0, 9223372036854775807, 2)
        slice_158: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_153, 3, 1, 9223372036854775807, 2);  slice_153 = None
        neg_38: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_158);  slice_158 = None
        unsqueeze_163: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_38, 4);  neg_38 = None
        unsqueeze_164: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_157, 4);  slice_157 = None
        cat_76: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_163, unsqueeze_164], -1);  unsqueeze_163 = unsqueeze_164 = None
        view_446: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_76, _shape_param_564);  cat_76 = _shape_param_564 = None
        mul_193: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_446, view_444);  view_446 = None
        add_173: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None
        mul_194: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_155, view_445);  view_445 = None
        slice_159: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_155, 3, 0, 9223372036854775807, 2)
        slice_160: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_155, 3, 1, 9223372036854775807, 2);  slice_155 = None
        neg_39: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_160);  slice_160 = None
        unsqueeze_165: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_39, 4);  neg_39 = None
        unsqueeze_166: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_159, 4);  slice_159 = None
        cat_77: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_165, unsqueeze_166], -1);  unsqueeze_165 = unsqueeze_166 = None
        view_447: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_77, _shape_param_565);  cat_77 = _shape_param_565 = None
        mul_195: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_447, view_444);  view_447 = view_444 = None
        add_174: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
        cat_78: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_173, slice_154], -1);  add_173 = slice_154 = None
        cat_79: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_174, slice_156], -1);  add_174 = slice_156 = None
        permute_213: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_78, [0, 2, 1, 3]);  cat_78 = None
        permute_214: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_79, [0, 2, 1, 3]);  cat_79 = None
        convert_element_type_309: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_213, torch.float32);  permute_213 = None
        permute_215: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_309, [0, 1, 3, 2]);  convert_element_type_309 = None
        convert_element_type_310: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_215, torch.bfloat16);  permute_215 = None
        convert_element_type_311: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_214, torch.bfloat16);  permute_214 = None
        expand_117: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_311, _shape_param_566);  convert_element_type_311 = _shape_param_566 = None
        view_448: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_117, _shape_param_567);  expand_117 = _shape_param_567 = None
        expand_118: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_310, _shape_param_568);  convert_element_type_310 = _shape_param_568 = None
        view_449: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_118, _shape_param_569);  expand_118 = _shape_param_569 = None
        bmm_38: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_448, view_449)
        view_450: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_38, _shape_param_570);  bmm_38 = _shape_param_570 = None
        div_38: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_450, 16.0);  view_450 = None
        add_175: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_38, where);  div_38 = None
        amax_19: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_175, [-1], True)
        sub_39: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_175, amax_19);  add_175 = amax_19 = None
        exp_19: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        sum_20: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_39: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        convert_element_type_312: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_39, torch.bfloat16)
        expand_119: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_312, _shape_param_571);  convert_element_type_312 = _shape_param_571 = None
        view_451: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_119, _shape_param_572);  expand_119 = _shape_param_572 = None
        expand_120: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_212, _shape_param_573);  permute_212 = _shape_param_573 = None
        view_452: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_120, _shape_param_574);  expand_120 = _shape_param_574 = None
        bmm_39: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_451, view_452)
        view_453: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_39, _shape_param_575);  bmm_39 = _shape_param_575 = None
        permute_216: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
        clone_59: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
        view_454: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_59, _shape_param_576);  clone_59 = _shape_param_576 = None
        convert_element_type_313: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg219_1, torch.bfloat16);  arg219_1 = None
        permute_217: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_313, [1, 0]);  convert_element_type_313 = None
        view_455: "bf16[128, 4096]" = torch.ops.aten.view.default(view_454, _shape_param_577);  view_454 = _shape_param_577 = None
        mm_79: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_455, permute_217)
        view_456: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_79, _shape_param_578);  mm_79 = _shape_param_578 = None
        convert_element_type_314: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg220_1, torch.bfloat16);  arg220_1 = None
        convert_element_type_315: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg221_1, torch.bfloat16);  arg221_1 = None
        permute_218: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_315, [1, 0]);  convert_element_type_315 = None
        addmm_38: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_314, view_437, permute_218);  convert_element_type_314 = None
        view_457: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_38, _shape_param_579);  _shape_param_579 = None
        mul_196: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_457, 0.5)
        convert_element_type_316: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_457, torch.float32)
        pow_20: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_316, 3.0);  convert_element_type_316 = None
        mul_197: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_176: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_457, mul_197);  view_457 = mul_197 = None
        mul_198: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_176, 0.7978845608028654);  add_176 = None
        tanh_19: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_198);  mul_198 = None
        add_177: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_19, 1.0);  tanh_19 = None
        mul_199: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_196, add_177);  mul_196 = add_177 = None
        convert_element_type_317: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg222_1, torch.bfloat16);  arg222_1 = None
        convert_element_type_318: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg223_1, torch.bfloat16);  arg223_1 = None
        convert_element_type_319: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_199, torch.bfloat16);  mul_199 = None
        view_458: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_319, _shape_param_580);  convert_element_type_319 = _shape_param_580 = None
        permute_219: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_318, [1, 0]);  convert_element_type_318 = None
        addmm_39: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_317, view_458, permute_219);  convert_element_type_317 = None
        view_459: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_39, _shape_param_581);  addmm_39 = _shape_param_581 = None
        add_178: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_456, view_459);  view_456 = view_459 = None
        add_179: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_178, add_170);  add_178 = add_170 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_179, [2], correction = 0, keepdim = True)
        getitem_80: "f32[1, 128, 1]" = var_mean_20[0]
        getitem_81: "f32[1, 128, 1]" = var_mean_20[1];  var_mean_20 = None
        add_180: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_20: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
        sub_40: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_179, getitem_81);  getitem_81 = None
        mul_200: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_20);  sub_40 = None
        mul_201: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_200, arg224_1);  arg224_1 = None
        add_181: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_201, arg225_1);  mul_201 = arg225_1 = None
        convert_element_type_320: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg226_1, torch.bfloat16);  arg226_1 = None
        convert_element_type_321: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_181, torch.bfloat16);  add_181 = None
        permute_220: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_320, [1, 0]);  convert_element_type_320 = None
        view_460: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_321, _shape_param_582);  convert_element_type_321 = _shape_param_582 = None
        mm_80: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_460, permute_220)
        view_461: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_80, _shape_param_583);  mm_80 = _shape_param_583 = None
        convert_element_type_322: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg227_1, torch.bfloat16);  arg227_1 = None
        permute_221: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_322, [1, 0]);  convert_element_type_322 = None
        mm_81: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_460, permute_221)
        view_462: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_81, _shape_param_584);  mm_81 = _shape_param_584 = None
        convert_element_type_323: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg228_1, torch.bfloat16);  arg228_1 = None
        permute_222: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_323, [1, 0]);  convert_element_type_323 = None
        mm_82: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_460, permute_222)
        view_463: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_82, _shape_param_585);  mm_82 = _shape_param_585 = None
        view_464: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_461, _shape_param_586);  view_461 = _shape_param_586 = None
        view_465: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_462, _shape_param_587);  view_462 = _shape_param_587 = None
        view_466: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_463, _shape_param_588);  view_463 = _shape_param_588 = None
        permute_223: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None
        repeat_21: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg229_1, [1, 1, 1]);  arg229_1 = None
        gather_20: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_21, 1, repeat_1);  repeat_21 = None
        convert_element_type_324: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_20, torch.bfloat16);  gather_20 = None
        split_20 = torch.ops.aten.split.Tensor(convert_element_type_324, 32, -1);  convert_element_type_324 = None
        getitem_82: "bf16[1, 128, 32]" = split_20[0]
        getitem_83: "bf16[1, 128, 32]" = split_20[1];  split_20 = None
        slice_161: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_465, 3, 0, 64)
        slice_162: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_465, 3, 64, 9223372036854775807);  view_465 = None
        slice_163: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_464, 3, 0, 64)
        slice_164: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_464, 3, 64, 9223372036854775807);  view_464 = None
        unsqueeze_167: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_82, 2);  getitem_82 = None
        unsqueeze_168: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 4);  unsqueeze_167 = None
        expand_121: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_168, _shape_param_589);  _shape_param_589 = None
        clone_60: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
        view_467: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_60, _shape_param_590);  clone_60 = _shape_param_590 = None
        unsqueeze_169: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_83, 2);  getitem_83 = None
        unsqueeze_170: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 4);  unsqueeze_169 = None
        expand_122: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_170, _shape_param_591);  _shape_param_591 = None
        clone_61: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
        view_468: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_61, _shape_param_592);  clone_61 = _shape_param_592 = None
        mul_202: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_161, view_468)
        slice_165: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_161, 3, 0, 9223372036854775807, 2)
        slice_166: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_161, 3, 1, 9223372036854775807, 2);  slice_161 = None
        neg_40: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_166);  slice_166 = None
        unsqueeze_171: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_40, 4);  neg_40 = None
        unsqueeze_172: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_165, 4);  slice_165 = None
        cat_80: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_171, unsqueeze_172], -1);  unsqueeze_171 = unsqueeze_172 = None
        view_469: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_80, _shape_param_593);  cat_80 = _shape_param_593 = None
        mul_203: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_469, view_467);  view_469 = None
        add_182: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_202, mul_203);  mul_202 = mul_203 = None
        mul_204: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_163, view_468);  view_468 = None
        slice_167: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_163, 3, 0, 9223372036854775807, 2)
        slice_168: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_163, 3, 1, 9223372036854775807, 2);  slice_163 = None
        neg_41: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_168);  slice_168 = None
        unsqueeze_173: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_41, 4);  neg_41 = None
        unsqueeze_174: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_167, 4);  slice_167 = None
        cat_81: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_173, unsqueeze_174], -1);  unsqueeze_173 = unsqueeze_174 = None
        view_470: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_81, _shape_param_594);  cat_81 = _shape_param_594 = None
        mul_205: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_470, view_467);  view_470 = view_467 = None
        add_183: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
        cat_82: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_182, slice_162], -1);  add_182 = slice_162 = None
        cat_83: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_183, slice_164], -1);  add_183 = slice_164 = None
        permute_224: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_82, [0, 2, 1, 3]);  cat_82 = None
        permute_225: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_83, [0, 2, 1, 3]);  cat_83 = None
        convert_element_type_325: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_224, torch.float32);  permute_224 = None
        permute_226: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_325, [0, 1, 3, 2]);  convert_element_type_325 = None
        convert_element_type_326: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_226, torch.bfloat16);  permute_226 = None
        convert_element_type_327: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_225, torch.bfloat16);  permute_225 = None
        expand_123: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_327, _shape_param_595);  convert_element_type_327 = _shape_param_595 = None
        view_471: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_123, _shape_param_596);  expand_123 = _shape_param_596 = None
        expand_124: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_326, _shape_param_597);  convert_element_type_326 = _shape_param_597 = None
        view_472: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_124, _shape_param_598);  expand_124 = _shape_param_598 = None
        bmm_40: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_471, view_472)
        view_473: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_40, _shape_param_599);  bmm_40 = _shape_param_599 = None
        div_40: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_473, 16.0);  view_473 = None
        add_184: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_40, where);  div_40 = None
        amax_20: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_184, [-1], True)
        sub_41: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_184, amax_20);  add_184 = amax_20 = None
        exp_20: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_41);  sub_41 = None
        sum_21: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_41: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_328: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_41, torch.bfloat16)
        expand_125: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_328, _shape_param_600);  convert_element_type_328 = _shape_param_600 = None
        view_474: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_125, _shape_param_601);  expand_125 = _shape_param_601 = None
        expand_126: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_223, _shape_param_602);  permute_223 = _shape_param_602 = None
        view_475: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_126, _shape_param_603);  expand_126 = _shape_param_603 = None
        bmm_41: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_474, view_475)
        view_476: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_41, _shape_param_604);  bmm_41 = _shape_param_604 = None
        permute_227: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
        clone_62: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_477: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_62, _shape_param_605);  clone_62 = _shape_param_605 = None
        convert_element_type_329: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg230_1, torch.bfloat16);  arg230_1 = None
        permute_228: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_329, [1, 0]);  convert_element_type_329 = None
        view_478: "bf16[128, 4096]" = torch.ops.aten.view.default(view_477, _shape_param_606);  view_477 = _shape_param_606 = None
        mm_83: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_478, permute_228)
        view_479: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_83, _shape_param_607);  mm_83 = _shape_param_607 = None
        convert_element_type_330: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg231_1, torch.bfloat16);  arg231_1 = None
        convert_element_type_331: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg232_1, torch.bfloat16);  arg232_1 = None
        permute_229: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_331, [1, 0]);  convert_element_type_331 = None
        addmm_40: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_330, view_460, permute_229);  convert_element_type_330 = None
        view_480: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_40, _shape_param_608);  _shape_param_608 = None
        mul_206: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_480, 0.5)
        convert_element_type_332: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_480, torch.float32)
        pow_21: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_332, 3.0);  convert_element_type_332 = None
        mul_207: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_185: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_480, mul_207);  view_480 = mul_207 = None
        mul_208: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_185, 0.7978845608028654);  add_185 = None
        tanh_20: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_208);  mul_208 = None
        add_186: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_20, 1.0);  tanh_20 = None
        mul_209: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_206, add_186);  mul_206 = add_186 = None
        convert_element_type_333: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg233_1, torch.bfloat16);  arg233_1 = None
        convert_element_type_334: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg234_1, torch.bfloat16);  arg234_1 = None
        convert_element_type_335: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_209, torch.bfloat16);  mul_209 = None
        view_481: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_335, _shape_param_609);  convert_element_type_335 = _shape_param_609 = None
        permute_230: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_334, [1, 0]);  convert_element_type_334 = None
        addmm_41: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_333, view_481, permute_230);  convert_element_type_333 = None
        view_482: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_41, _shape_param_610);  addmm_41 = _shape_param_610 = None
        add_187: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_479, view_482);  view_479 = view_482 = None
        add_188: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_187, add_179);  add_187 = add_179 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_188, [2], correction = 0, keepdim = True)
        getitem_84: "f32[1, 128, 1]" = var_mean_21[0]
        getitem_85: "f32[1, 128, 1]" = var_mean_21[1];  var_mean_21 = None
        add_189: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
        rsqrt_21: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
        sub_42: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_188, getitem_85);  getitem_85 = None
        mul_210: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_21);  sub_42 = None
        mul_211: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_210, arg235_1);  arg235_1 = None
        add_190: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_211, arg236_1);  mul_211 = arg236_1 = None
        convert_element_type_336: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg237_1, torch.bfloat16);  arg237_1 = None
        convert_element_type_337: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_190, torch.bfloat16);  add_190 = None
        permute_231: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_336, [1, 0]);  convert_element_type_336 = None
        view_483: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_337, _shape_param_611);  convert_element_type_337 = _shape_param_611 = None
        mm_84: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_483, permute_231)
        view_484: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_84, _shape_param_612);  mm_84 = _shape_param_612 = None
        convert_element_type_338: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg238_1, torch.bfloat16);  arg238_1 = None
        permute_232: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_338, [1, 0]);  convert_element_type_338 = None
        mm_85: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_483, permute_232)
        view_485: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_85, _shape_param_613);  mm_85 = _shape_param_613 = None
        convert_element_type_339: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg239_1, torch.bfloat16);  arg239_1 = None
        permute_233: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_339, [1, 0]);  convert_element_type_339 = None
        mm_86: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_483, permute_233)
        view_486: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_86, _shape_param_614);  mm_86 = _shape_param_614 = None
        view_487: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_484, _shape_param_615);  view_484 = _shape_param_615 = None
        view_488: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_485, _shape_param_616);  view_485 = _shape_param_616 = None
        view_489: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_486, _shape_param_617);  view_486 = _shape_param_617 = None
        permute_234: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None
        repeat_22: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg240_1, [1, 1, 1]);  arg240_1 = None
        gather_21: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_22, 1, repeat_1);  repeat_22 = None
        convert_element_type_340: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_21, torch.bfloat16);  gather_21 = None
        split_21 = torch.ops.aten.split.Tensor(convert_element_type_340, 32, -1);  convert_element_type_340 = None
        getitem_86: "bf16[1, 128, 32]" = split_21[0]
        getitem_87: "bf16[1, 128, 32]" = split_21[1];  split_21 = None
        slice_169: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_488, 3, 0, 64)
        slice_170: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_488, 3, 64, 9223372036854775807);  view_488 = None
        slice_171: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_487, 3, 0, 64)
        slice_172: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_487, 3, 64, 9223372036854775807);  view_487 = None
        unsqueeze_175: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_86, 2);  getitem_86 = None
        unsqueeze_176: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 4);  unsqueeze_175 = None
        expand_127: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_176, _shape_param_618);  _shape_param_618 = None
        clone_63: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_127, memory_format = torch.contiguous_format);  expand_127 = None
        view_490: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_63, _shape_param_619);  clone_63 = _shape_param_619 = None
        unsqueeze_177: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_87, 2);  getitem_87 = None
        unsqueeze_178: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 4);  unsqueeze_177 = None
        expand_128: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_178, _shape_param_620);  _shape_param_620 = None
        clone_64: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_128, memory_format = torch.contiguous_format);  expand_128 = None
        view_491: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_64, _shape_param_621);  clone_64 = _shape_param_621 = None
        mul_212: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_169, view_491)
        slice_173: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_169, 3, 0, 9223372036854775807, 2)
        slice_174: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_169, 3, 1, 9223372036854775807, 2);  slice_169 = None
        neg_42: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_174);  slice_174 = None
        unsqueeze_179: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_42, 4);  neg_42 = None
        unsqueeze_180: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_173, 4);  slice_173 = None
        cat_84: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_179, unsqueeze_180], -1);  unsqueeze_179 = unsqueeze_180 = None
        view_492: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_84, _shape_param_622);  cat_84 = _shape_param_622 = None
        mul_213: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_492, view_490);  view_492 = None
        add_191: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None
        mul_214: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_171, view_491);  view_491 = None
        slice_175: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_171, 3, 0, 9223372036854775807, 2)
        slice_176: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_171, 3, 1, 9223372036854775807, 2);  slice_171 = None
        neg_43: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_176);  slice_176 = None
        unsqueeze_181: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_43, 4);  neg_43 = None
        unsqueeze_182: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_175, 4);  slice_175 = None
        cat_85: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_181, unsqueeze_182], -1);  unsqueeze_181 = unsqueeze_182 = None
        view_493: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_85, _shape_param_623);  cat_85 = _shape_param_623 = None
        mul_215: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_493, view_490);  view_493 = view_490 = None
        add_192: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        cat_86: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_191, slice_170], -1);  add_191 = slice_170 = None
        cat_87: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_192, slice_172], -1);  add_192 = slice_172 = None
        permute_235: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_86, [0, 2, 1, 3]);  cat_86 = None
        permute_236: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_87, [0, 2, 1, 3]);  cat_87 = None
        convert_element_type_341: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_235, torch.float32);  permute_235 = None
        permute_237: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_341, [0, 1, 3, 2]);  convert_element_type_341 = None
        convert_element_type_342: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_237, torch.bfloat16);  permute_237 = None
        convert_element_type_343: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_236, torch.bfloat16);  permute_236 = None
        expand_129: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_343, _shape_param_624);  convert_element_type_343 = _shape_param_624 = None
        view_494: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_129, _shape_param_625);  expand_129 = _shape_param_625 = None
        expand_130: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_342, _shape_param_626);  convert_element_type_342 = _shape_param_626 = None
        view_495: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_130, _shape_param_627);  expand_130 = _shape_param_627 = None
        bmm_42: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_494, view_495)
        view_496: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_42, _shape_param_628);  bmm_42 = _shape_param_628 = None
        div_42: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_496, 16.0);  view_496 = None
        add_193: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_42, where);  div_42 = None
        amax_21: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_193, [-1], True)
        sub_43: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_193, amax_21);  add_193 = amax_21 = None
        exp_21: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_22: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_43: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        convert_element_type_344: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_43, torch.bfloat16)
        expand_131: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_344, _shape_param_629);  convert_element_type_344 = _shape_param_629 = None
        view_497: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_131, _shape_param_630);  expand_131 = _shape_param_630 = None
        expand_132: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_234, _shape_param_631);  permute_234 = _shape_param_631 = None
        view_498: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_132, _shape_param_632);  expand_132 = _shape_param_632 = None
        bmm_43: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_497, view_498)
        view_499: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_43, _shape_param_633);  bmm_43 = _shape_param_633 = None
        permute_238: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_499, [0, 2, 1, 3]);  view_499 = None
        clone_65: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
        view_500: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_65, _shape_param_634);  clone_65 = _shape_param_634 = None
        convert_element_type_345: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg241_1, torch.bfloat16);  arg241_1 = None
        permute_239: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_345, [1, 0]);  convert_element_type_345 = None
        view_501: "bf16[128, 4096]" = torch.ops.aten.view.default(view_500, _shape_param_635);  view_500 = _shape_param_635 = None
        mm_87: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_501, permute_239)
        view_502: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_87, _shape_param_636);  mm_87 = _shape_param_636 = None
        convert_element_type_346: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg242_1, torch.bfloat16);  arg242_1 = None
        convert_element_type_347: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg243_1, torch.bfloat16);  arg243_1 = None
        permute_240: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_347, [1, 0]);  convert_element_type_347 = None
        addmm_42: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_346, view_483, permute_240);  convert_element_type_346 = None
        view_503: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_42, _shape_param_637);  _shape_param_637 = None
        mul_216: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_503, 0.5)
        convert_element_type_348: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_503, torch.float32)
        pow_22: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_348, 3.0);  convert_element_type_348 = None
        mul_217: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_194: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_503, mul_217);  view_503 = mul_217 = None
        mul_218: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_194, 0.7978845608028654);  add_194 = None
        tanh_21: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_218);  mul_218 = None
        add_195: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_21, 1.0);  tanh_21 = None
        mul_219: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_216, add_195);  mul_216 = add_195 = None
        convert_element_type_349: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg244_1, torch.bfloat16);  arg244_1 = None
        convert_element_type_350: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg245_1, torch.bfloat16);  arg245_1 = None
        convert_element_type_351: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_219, torch.bfloat16);  mul_219 = None
        view_504: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_351, _shape_param_638);  convert_element_type_351 = _shape_param_638 = None
        permute_241: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_350, [1, 0]);  convert_element_type_350 = None
        addmm_43: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_349, view_504, permute_241);  convert_element_type_349 = None
        view_505: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_43, _shape_param_639);  addmm_43 = _shape_param_639 = None
        add_196: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_502, view_505);  view_502 = view_505 = None
        add_197: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_196, add_188);  add_196 = add_188 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_197, [2], correction = 0, keepdim = True)
        getitem_88: "f32[1, 128, 1]" = var_mean_22[0]
        getitem_89: "f32[1, 128, 1]" = var_mean_22[1];  var_mean_22 = None
        add_198: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_22: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
        sub_44: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_197, getitem_89);  getitem_89 = None
        mul_220: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_22);  sub_44 = None
        mul_221: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_220, arg246_1);  arg246_1 = None
        add_199: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_221, arg247_1);  mul_221 = arg247_1 = None
        convert_element_type_352: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg248_1, torch.bfloat16);  arg248_1 = None
        convert_element_type_353: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_199, torch.bfloat16);  add_199 = None
        permute_242: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_352, [1, 0]);  convert_element_type_352 = None
        view_506: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_353, _shape_param_640);  convert_element_type_353 = _shape_param_640 = None
        mm_88: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_506, permute_242)
        view_507: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_88, _shape_param_641);  mm_88 = _shape_param_641 = None
        convert_element_type_354: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg249_1, torch.bfloat16);  arg249_1 = None
        permute_243: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_354, [1, 0]);  convert_element_type_354 = None
        mm_89: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_506, permute_243)
        view_508: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_89, _shape_param_642);  mm_89 = _shape_param_642 = None
        convert_element_type_355: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg250_1, torch.bfloat16);  arg250_1 = None
        permute_244: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_355, [1, 0]);  convert_element_type_355 = None
        mm_90: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_506, permute_244)
        view_509: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_90, _shape_param_643);  mm_90 = _shape_param_643 = None
        view_510: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_507, _shape_param_644);  view_507 = _shape_param_644 = None
        view_511: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_508, _shape_param_645);  view_508 = _shape_param_645 = None
        view_512: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_509, _shape_param_646);  view_509 = _shape_param_646 = None
        permute_245: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
        repeat_23: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg251_1, [1, 1, 1]);  arg251_1 = None
        gather_22: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_23, 1, repeat_1);  repeat_23 = None
        convert_element_type_356: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_22, torch.bfloat16);  gather_22 = None
        split_22 = torch.ops.aten.split.Tensor(convert_element_type_356, 32, -1);  convert_element_type_356 = None
        getitem_90: "bf16[1, 128, 32]" = split_22[0]
        getitem_91: "bf16[1, 128, 32]" = split_22[1];  split_22 = None
        slice_177: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_511, 3, 0, 64)
        slice_178: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_511, 3, 64, 9223372036854775807);  view_511 = None
        slice_179: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_510, 3, 0, 64)
        slice_180: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_510, 3, 64, 9223372036854775807);  view_510 = None
        unsqueeze_183: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_90, 2);  getitem_90 = None
        unsqueeze_184: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 4);  unsqueeze_183 = None
        expand_133: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_184, _shape_param_647);  _shape_param_647 = None
        clone_66: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_133, memory_format = torch.contiguous_format);  expand_133 = None
        view_513: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_66, _shape_param_648);  clone_66 = _shape_param_648 = None
        unsqueeze_185: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_91, 2);  getitem_91 = None
        unsqueeze_186: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 4);  unsqueeze_185 = None
        expand_134: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_186, _shape_param_649);  _shape_param_649 = None
        clone_67: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_134, memory_format = torch.contiguous_format);  expand_134 = None
        view_514: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_67, _shape_param_650);  clone_67 = _shape_param_650 = None
        mul_222: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_177, view_514)
        slice_181: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_177, 3, 0, 9223372036854775807, 2)
        slice_182: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_177, 3, 1, 9223372036854775807, 2);  slice_177 = None
        neg_44: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_182);  slice_182 = None
        unsqueeze_187: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_44, 4);  neg_44 = None
        unsqueeze_188: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_181, 4);  slice_181 = None
        cat_88: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_187, unsqueeze_188], -1);  unsqueeze_187 = unsqueeze_188 = None
        view_515: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_88, _shape_param_651);  cat_88 = _shape_param_651 = None
        mul_223: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_515, view_513);  view_515 = None
        add_200: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_222, mul_223);  mul_222 = mul_223 = None
        mul_224: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_179, view_514);  view_514 = None
        slice_183: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_179, 3, 0, 9223372036854775807, 2)
        slice_184: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_179, 3, 1, 9223372036854775807, 2);  slice_179 = None
        neg_45: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_184);  slice_184 = None
        unsqueeze_189: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_45, 4);  neg_45 = None
        unsqueeze_190: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_183, 4);  slice_183 = None
        cat_89: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_189, unsqueeze_190], -1);  unsqueeze_189 = unsqueeze_190 = None
        view_516: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_89, _shape_param_652);  cat_89 = _shape_param_652 = None
        mul_225: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_516, view_513);  view_516 = view_513 = None
        add_201: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
        cat_90: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_200, slice_178], -1);  add_200 = slice_178 = None
        cat_91: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_201, slice_180], -1);  add_201 = slice_180 = None
        permute_246: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_90, [0, 2, 1, 3]);  cat_90 = None
        permute_247: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_91, [0, 2, 1, 3]);  cat_91 = None
        convert_element_type_357: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_246, torch.float32);  permute_246 = None
        permute_248: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_357, [0, 1, 3, 2]);  convert_element_type_357 = None
        convert_element_type_358: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_248, torch.bfloat16);  permute_248 = None
        convert_element_type_359: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_247, torch.bfloat16);  permute_247 = None
        expand_135: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_359, _shape_param_653);  convert_element_type_359 = _shape_param_653 = None
        view_517: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_135, _shape_param_654);  expand_135 = _shape_param_654 = None
        expand_136: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_358, _shape_param_655);  convert_element_type_358 = _shape_param_655 = None
        view_518: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_136, _shape_param_656);  expand_136 = _shape_param_656 = None
        bmm_44: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_517, view_518)
        view_519: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_44, _shape_param_657);  bmm_44 = _shape_param_657 = None
        div_44: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_519, 16.0);  view_519 = None
        add_202: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_44, where);  div_44 = None
        amax_22: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_202, [-1], True)
        sub_45: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_202, amax_22);  add_202 = amax_22 = None
        exp_22: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
        sum_23: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_45: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_360: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_45, torch.bfloat16)
        expand_137: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_360, _shape_param_658);  convert_element_type_360 = _shape_param_658 = None
        view_520: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_137, _shape_param_659);  expand_137 = _shape_param_659 = None
        expand_138: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_245, _shape_param_660);  permute_245 = _shape_param_660 = None
        view_521: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_138, _shape_param_661);  expand_138 = _shape_param_661 = None
        bmm_45: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_520, view_521)
        view_522: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_45, _shape_param_662);  bmm_45 = _shape_param_662 = None
        permute_249: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None
        clone_68: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_523: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_68, _shape_param_663);  clone_68 = _shape_param_663 = None
        convert_element_type_361: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg252_1, torch.bfloat16);  arg252_1 = None
        permute_250: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_361, [1, 0]);  convert_element_type_361 = None
        view_524: "bf16[128, 4096]" = torch.ops.aten.view.default(view_523, _shape_param_664);  view_523 = _shape_param_664 = None
        mm_91: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_524, permute_250)
        view_525: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_91, _shape_param_665);  mm_91 = _shape_param_665 = None
        convert_element_type_362: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg253_1, torch.bfloat16);  arg253_1 = None
        convert_element_type_363: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg254_1, torch.bfloat16);  arg254_1 = None
        permute_251: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_363, [1, 0]);  convert_element_type_363 = None
        addmm_44: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_362, view_506, permute_251);  convert_element_type_362 = None
        view_526: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_44, _shape_param_666);  _shape_param_666 = None
        mul_226: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_526, 0.5)
        convert_element_type_364: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_526, torch.float32)
        pow_23: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_364, 3.0);  convert_element_type_364 = None
        mul_227: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_203: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_526, mul_227);  view_526 = mul_227 = None
        mul_228: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_203, 0.7978845608028654);  add_203 = None
        tanh_22: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_228);  mul_228 = None
        add_204: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_22, 1.0);  tanh_22 = None
        mul_229: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_226, add_204);  mul_226 = add_204 = None
        convert_element_type_365: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg255_1, torch.bfloat16);  arg255_1 = None
        convert_element_type_366: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg256_1, torch.bfloat16);  arg256_1 = None
        convert_element_type_367: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_229, torch.bfloat16);  mul_229 = None
        view_527: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_367, _shape_param_667);  convert_element_type_367 = _shape_param_667 = None
        permute_252: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_366, [1, 0]);  convert_element_type_366 = None
        addmm_45: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_365, view_527, permute_252);  convert_element_type_365 = None
        view_528: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_45, _shape_param_668);  addmm_45 = _shape_param_668 = None
        add_205: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_525, view_528);  view_525 = view_528 = None
        add_206: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_205, add_197);  add_205 = add_197 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_206, [2], correction = 0, keepdim = True)
        getitem_92: "f32[1, 128, 1]" = var_mean_23[0]
        getitem_93: "f32[1, 128, 1]" = var_mean_23[1];  var_mean_23 = None
        add_207: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_23: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        sub_46: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_206, getitem_93);  getitem_93 = None
        mul_230: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_23);  sub_46 = None
        mul_231: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_230, arg257_1);  arg257_1 = None
        add_208: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_231, arg258_1);  mul_231 = arg258_1 = None
        convert_element_type_368: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg259_1, torch.bfloat16);  arg259_1 = None
        convert_element_type_369: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_208, torch.bfloat16);  add_208 = None
        permute_253: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_368, [1, 0]);  convert_element_type_368 = None
        view_529: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_369, _shape_param_669);  convert_element_type_369 = _shape_param_669 = None
        mm_92: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_529, permute_253)
        view_530: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_92, _shape_param_670);  mm_92 = _shape_param_670 = None
        convert_element_type_370: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg260_1, torch.bfloat16);  arg260_1 = None
        permute_254: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_370, [1, 0]);  convert_element_type_370 = None
        mm_93: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_529, permute_254)
        view_531: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_93, _shape_param_671);  mm_93 = _shape_param_671 = None
        convert_element_type_371: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg261_1, torch.bfloat16);  arg261_1 = None
        permute_255: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_371, [1, 0]);  convert_element_type_371 = None
        mm_94: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_529, permute_255)
        view_532: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_94, _shape_param_672);  mm_94 = _shape_param_672 = None
        view_533: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_530, _shape_param_673);  view_530 = _shape_param_673 = None
        view_534: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_531, _shape_param_674);  view_531 = _shape_param_674 = None
        view_535: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_532, _shape_param_675);  view_532 = _shape_param_675 = None
        permute_256: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None
        repeat_24: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg262_1, [1, 1, 1]);  arg262_1 = None
        gather_23: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_24, 1, repeat_1);  repeat_24 = None
        convert_element_type_372: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_23, torch.bfloat16);  gather_23 = None
        split_23 = torch.ops.aten.split.Tensor(convert_element_type_372, 32, -1);  convert_element_type_372 = None
        getitem_94: "bf16[1, 128, 32]" = split_23[0]
        getitem_95: "bf16[1, 128, 32]" = split_23[1];  split_23 = None
        slice_185: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_534, 3, 0, 64)
        slice_186: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_534, 3, 64, 9223372036854775807);  view_534 = None
        slice_187: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_533, 3, 0, 64)
        slice_188: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_533, 3, 64, 9223372036854775807);  view_533 = None
        unsqueeze_191: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_94, 2);  getitem_94 = None
        unsqueeze_192: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 4);  unsqueeze_191 = None
        expand_139: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_192, _shape_param_676);  _shape_param_676 = None
        clone_69: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_139, memory_format = torch.contiguous_format);  expand_139 = None
        view_536: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_69, _shape_param_677);  clone_69 = _shape_param_677 = None
        unsqueeze_193: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_95, 2);  getitem_95 = None
        unsqueeze_194: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 4);  unsqueeze_193 = None
        expand_140: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_194, _shape_param_678);  _shape_param_678 = None
        clone_70: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_140, memory_format = torch.contiguous_format);  expand_140 = None
        view_537: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_70, _shape_param_679);  clone_70 = _shape_param_679 = None
        mul_232: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_185, view_537)
        slice_189: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_185, 3, 0, 9223372036854775807, 2)
        slice_190: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_185, 3, 1, 9223372036854775807, 2);  slice_185 = None
        neg_46: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_190);  slice_190 = None
        unsqueeze_195: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_46, 4);  neg_46 = None
        unsqueeze_196: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_189, 4);  slice_189 = None
        cat_92: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_195, unsqueeze_196], -1);  unsqueeze_195 = unsqueeze_196 = None
        view_538: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_92, _shape_param_680);  cat_92 = _shape_param_680 = None
        mul_233: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_538, view_536);  view_538 = None
        add_209: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        mul_234: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_187, view_537);  view_537 = None
        slice_191: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_187, 3, 0, 9223372036854775807, 2)
        slice_192: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_187, 3, 1, 9223372036854775807, 2);  slice_187 = None
        neg_47: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_192);  slice_192 = None
        unsqueeze_197: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_47, 4);  neg_47 = None
        unsqueeze_198: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_191, 4);  slice_191 = None
        cat_93: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_197, unsqueeze_198], -1);  unsqueeze_197 = unsqueeze_198 = None
        view_539: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_93, _shape_param_681);  cat_93 = _shape_param_681 = None
        mul_235: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_539, view_536);  view_539 = view_536 = None
        add_210: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
        cat_94: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_209, slice_186], -1);  add_209 = slice_186 = None
        cat_95: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_210, slice_188], -1);  add_210 = slice_188 = None
        permute_257: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_94, [0, 2, 1, 3]);  cat_94 = None
        permute_258: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_95, [0, 2, 1, 3]);  cat_95 = None
        convert_element_type_373: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_257, torch.float32);  permute_257 = None
        permute_259: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_373, [0, 1, 3, 2]);  convert_element_type_373 = None
        convert_element_type_374: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_259, torch.bfloat16);  permute_259 = None
        convert_element_type_375: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_258, torch.bfloat16);  permute_258 = None
        expand_141: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_375, _shape_param_682);  convert_element_type_375 = _shape_param_682 = None
        view_540: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_141, _shape_param_683);  expand_141 = _shape_param_683 = None
        expand_142: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_374, _shape_param_684);  convert_element_type_374 = _shape_param_684 = None
        view_541: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_142, _shape_param_685);  expand_142 = _shape_param_685 = None
        bmm_46: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_540, view_541)
        view_542: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_46, _shape_param_686);  bmm_46 = _shape_param_686 = None
        div_46: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_542, 16.0);  view_542 = None
        add_211: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_46, where);  div_46 = None
        amax_23: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_211, [-1], True)
        sub_47: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_211, amax_23);  add_211 = amax_23 = None
        exp_23: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_47);  sub_47 = None
        sum_24: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_47: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        convert_element_type_376: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_47, torch.bfloat16)
        expand_143: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_376, _shape_param_687);  convert_element_type_376 = _shape_param_687 = None
        view_543: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_143, _shape_param_688);  expand_143 = _shape_param_688 = None
        expand_144: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_256, _shape_param_689);  permute_256 = _shape_param_689 = None
        view_544: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_144, _shape_param_690);  expand_144 = _shape_param_690 = None
        bmm_47: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_543, view_544)
        view_545: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_47, _shape_param_691);  bmm_47 = _shape_param_691 = None
        permute_260: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_545, [0, 2, 1, 3]);  view_545 = None
        clone_71: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_546: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_71, _shape_param_692);  clone_71 = _shape_param_692 = None
        convert_element_type_377: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg263_1, torch.bfloat16);  arg263_1 = None
        permute_261: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_377, [1, 0]);  convert_element_type_377 = None
        view_547: "bf16[128, 4096]" = torch.ops.aten.view.default(view_546, _shape_param_693);  view_546 = _shape_param_693 = None
        mm_95: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_547, permute_261)
        view_548: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_95, _shape_param_694);  mm_95 = _shape_param_694 = None
        convert_element_type_378: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg264_1, torch.bfloat16);  arg264_1 = None
        convert_element_type_379: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg265_1, torch.bfloat16);  arg265_1 = None
        permute_262: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_379, [1, 0]);  convert_element_type_379 = None
        addmm_46: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_378, view_529, permute_262);  convert_element_type_378 = None
        view_549: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_46, _shape_param_695);  _shape_param_695 = None
        mul_236: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_549, 0.5)
        convert_element_type_380: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_549, torch.float32)
        pow_24: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_380, 3.0);  convert_element_type_380 = None
        mul_237: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_212: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_549, mul_237);  view_549 = mul_237 = None
        mul_238: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_212, 0.7978845608028654);  add_212 = None
        tanh_23: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_238);  mul_238 = None
        add_213: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_23, 1.0);  tanh_23 = None
        mul_239: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_236, add_213);  mul_236 = add_213 = None
        convert_element_type_381: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg266_1, torch.bfloat16);  arg266_1 = None
        convert_element_type_382: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg267_1, torch.bfloat16);  arg267_1 = None
        convert_element_type_383: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_239, torch.bfloat16);  mul_239 = None
        view_550: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_383, _shape_param_696);  convert_element_type_383 = _shape_param_696 = None
        permute_263: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_382, [1, 0]);  convert_element_type_382 = None
        addmm_47: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_381, view_550, permute_263);  convert_element_type_381 = None
        view_551: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_47, _shape_param_697);  addmm_47 = _shape_param_697 = None
        add_214: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_548, view_551);  view_548 = view_551 = None
        add_215: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_214, add_206);  add_214 = add_206 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_215, [2], correction = 0, keepdim = True)
        getitem_96: "f32[1, 128, 1]" = var_mean_24[0]
        getitem_97: "f32[1, 128, 1]" = var_mean_24[1];  var_mean_24 = None
        add_216: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_24: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
        sub_48: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_215, getitem_97);  getitem_97 = None
        mul_240: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_24);  sub_48 = None
        mul_241: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_240, arg268_1);  arg268_1 = None
        add_217: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_241, arg269_1);  mul_241 = arg269_1 = None
        convert_element_type_384: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg270_1, torch.bfloat16);  arg270_1 = None
        convert_element_type_385: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_217, torch.bfloat16);  add_217 = None
        permute_264: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_384, [1, 0]);  convert_element_type_384 = None
        view_552: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_385, _shape_param_698);  convert_element_type_385 = _shape_param_698 = None
        mm_96: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_552, permute_264)
        view_553: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_96, _shape_param_699);  mm_96 = _shape_param_699 = None
        convert_element_type_386: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg271_1, torch.bfloat16);  arg271_1 = None
        permute_265: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_386, [1, 0]);  convert_element_type_386 = None
        mm_97: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_552, permute_265)
        view_554: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_97, _shape_param_700);  mm_97 = _shape_param_700 = None
        convert_element_type_387: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg272_1, torch.bfloat16);  arg272_1 = None
        permute_266: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_387, [1, 0]);  convert_element_type_387 = None
        mm_98: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_552, permute_266)
        view_555: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_98, _shape_param_701);  mm_98 = _shape_param_701 = None
        view_556: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_553, _shape_param_702);  view_553 = _shape_param_702 = None
        view_557: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_554, _shape_param_703);  view_554 = _shape_param_703 = None
        view_558: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_555, _shape_param_704);  view_555 = _shape_param_704 = None
        permute_267: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_558, [0, 2, 1, 3]);  view_558 = None
        repeat_25: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg273_1, [1, 1, 1]);  arg273_1 = None
        gather_24: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_25, 1, repeat_1);  repeat_25 = None
        convert_element_type_388: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_24, torch.bfloat16);  gather_24 = None
        split_24 = torch.ops.aten.split.Tensor(convert_element_type_388, 32, -1);  convert_element_type_388 = None
        getitem_98: "bf16[1, 128, 32]" = split_24[0]
        getitem_99: "bf16[1, 128, 32]" = split_24[1];  split_24 = None
        slice_193: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_557, 3, 0, 64)
        slice_194: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_557, 3, 64, 9223372036854775807);  view_557 = None
        slice_195: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_556, 3, 0, 64)
        slice_196: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_556, 3, 64, 9223372036854775807);  view_556 = None
        unsqueeze_199: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_98, 2);  getitem_98 = None
        unsqueeze_200: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 4);  unsqueeze_199 = None
        expand_145: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_200, _shape_param_705);  _shape_param_705 = None
        clone_72: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
        view_559: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_72, _shape_param_706);  clone_72 = _shape_param_706 = None
        unsqueeze_201: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_99, 2);  getitem_99 = None
        unsqueeze_202: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 4);  unsqueeze_201 = None
        expand_146: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_202, _shape_param_707);  _shape_param_707 = None
        clone_73: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_146, memory_format = torch.contiguous_format);  expand_146 = None
        view_560: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_73, _shape_param_708);  clone_73 = _shape_param_708 = None
        mul_242: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_193, view_560)
        slice_197: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_193, 3, 0, 9223372036854775807, 2)
        slice_198: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_193, 3, 1, 9223372036854775807, 2);  slice_193 = None
        neg_48: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_198);  slice_198 = None
        unsqueeze_203: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_48, 4);  neg_48 = None
        unsqueeze_204: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_197, 4);  slice_197 = None
        cat_96: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_203, unsqueeze_204], -1);  unsqueeze_203 = unsqueeze_204 = None
        view_561: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_96, _shape_param_709);  cat_96 = _shape_param_709 = None
        mul_243: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_561, view_559);  view_561 = None
        add_218: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        mul_244: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_195, view_560);  view_560 = None
        slice_199: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_195, 3, 0, 9223372036854775807, 2)
        slice_200: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_195, 3, 1, 9223372036854775807, 2);  slice_195 = None
        neg_49: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_200);  slice_200 = None
        unsqueeze_205: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_49, 4);  neg_49 = None
        unsqueeze_206: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_199, 4);  slice_199 = None
        cat_97: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_205, unsqueeze_206], -1);  unsqueeze_205 = unsqueeze_206 = None
        view_562: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_97, _shape_param_710);  cat_97 = _shape_param_710 = None
        mul_245: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_562, view_559);  view_562 = view_559 = None
        add_219: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
        cat_98: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_218, slice_194], -1);  add_218 = slice_194 = None
        cat_99: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_219, slice_196], -1);  add_219 = slice_196 = None
        permute_268: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_98, [0, 2, 1, 3]);  cat_98 = None
        permute_269: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_99, [0, 2, 1, 3]);  cat_99 = None
        convert_element_type_389: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_268, torch.float32);  permute_268 = None
        permute_270: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_389, [0, 1, 3, 2]);  convert_element_type_389 = None
        convert_element_type_390: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_270, torch.bfloat16);  permute_270 = None
        convert_element_type_391: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_269, torch.bfloat16);  permute_269 = None
        expand_147: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_391, _shape_param_711);  convert_element_type_391 = _shape_param_711 = None
        view_563: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_147, _shape_param_712);  expand_147 = _shape_param_712 = None
        expand_148: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_390, _shape_param_713);  convert_element_type_390 = _shape_param_713 = None
        view_564: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_148, _shape_param_714);  expand_148 = _shape_param_714 = None
        bmm_48: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_563, view_564)
        view_565: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_48, _shape_param_715);  bmm_48 = _shape_param_715 = None
        div_48: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_565, 16.0);  view_565 = None
        add_220: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_48, where);  div_48 = None
        amax_24: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_220, [-1], True)
        sub_49: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_220, amax_24);  add_220 = amax_24 = None
        exp_24: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_25: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_49: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
        convert_element_type_392: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_49, torch.bfloat16)
        expand_149: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_392, _shape_param_716);  convert_element_type_392 = _shape_param_716 = None
        view_566: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_149, _shape_param_717);  expand_149 = _shape_param_717 = None
        expand_150: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_267, _shape_param_718);  permute_267 = _shape_param_718 = None
        view_567: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_150, _shape_param_719);  expand_150 = _shape_param_719 = None
        bmm_49: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_566, view_567)
        view_568: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_49, _shape_param_720);  bmm_49 = _shape_param_720 = None
        permute_271: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
        clone_74: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_271, memory_format = torch.contiguous_format);  permute_271 = None
        view_569: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_74, _shape_param_721);  clone_74 = _shape_param_721 = None
        convert_element_type_393: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg274_1, torch.bfloat16);  arg274_1 = None
        permute_272: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_393, [1, 0]);  convert_element_type_393 = None
        view_570: "bf16[128, 4096]" = torch.ops.aten.view.default(view_569, _shape_param_722);  view_569 = _shape_param_722 = None
        mm_99: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_570, permute_272)
        view_571: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_99, _shape_param_723);  mm_99 = _shape_param_723 = None
        convert_element_type_394: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg275_1, torch.bfloat16);  arg275_1 = None
        convert_element_type_395: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg276_1, torch.bfloat16);  arg276_1 = None
        permute_273: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_395, [1, 0]);  convert_element_type_395 = None
        addmm_48: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_394, view_552, permute_273);  convert_element_type_394 = None
        view_572: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_48, _shape_param_724);  _shape_param_724 = None
        mul_246: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_572, 0.5)
        convert_element_type_396: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_572, torch.float32)
        pow_25: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_396, 3.0);  convert_element_type_396 = None
        mul_247: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_25, 0.044715);  pow_25 = None
        add_221: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_572, mul_247);  view_572 = mul_247 = None
        mul_248: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_221, 0.7978845608028654);  add_221 = None
        tanh_24: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_248);  mul_248 = None
        add_222: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_24, 1.0);  tanh_24 = None
        mul_249: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_246, add_222);  mul_246 = add_222 = None
        convert_element_type_397: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg277_1, torch.bfloat16);  arg277_1 = None
        convert_element_type_398: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg278_1, torch.bfloat16);  arg278_1 = None
        convert_element_type_399: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_249, torch.bfloat16);  mul_249 = None
        view_573: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_399, _shape_param_725);  convert_element_type_399 = _shape_param_725 = None
        permute_274: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_398, [1, 0]);  convert_element_type_398 = None
        addmm_49: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_397, view_573, permute_274);  convert_element_type_397 = None
        view_574: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_49, _shape_param_726);  addmm_49 = _shape_param_726 = None
        add_223: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_571, view_574);  view_571 = view_574 = None
        add_224: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_223, add_215);  add_223 = add_215 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_224, [2], correction = 0, keepdim = True)
        getitem_100: "f32[1, 128, 1]" = var_mean_25[0]
        getitem_101: "f32[1, 128, 1]" = var_mean_25[1];  var_mean_25 = None
        add_225: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
        rsqrt_25: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        sub_50: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_224, getitem_101);  getitem_101 = None
        mul_250: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_25);  sub_50 = None
        mul_251: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_250, arg279_1);  arg279_1 = None
        add_226: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_251, arg280_1);  mul_251 = arg280_1 = None
        convert_element_type_400: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg281_1, torch.bfloat16);  arg281_1 = None
        convert_element_type_401: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_226, torch.bfloat16);  add_226 = None
        permute_275: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_400, [1, 0]);  convert_element_type_400 = None
        view_575: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_401, _shape_param_727);  convert_element_type_401 = _shape_param_727 = None
        mm_100: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_575, permute_275)
        view_576: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_100, _shape_param_728);  mm_100 = _shape_param_728 = None
        convert_element_type_402: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg282_1, torch.bfloat16);  arg282_1 = None
        permute_276: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_402, [1, 0]);  convert_element_type_402 = None
        mm_101: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_575, permute_276)
        view_577: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_101, _shape_param_729);  mm_101 = _shape_param_729 = None
        convert_element_type_403: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg283_1, torch.bfloat16);  arg283_1 = None
        permute_277: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_403, [1, 0]);  convert_element_type_403 = None
        mm_102: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_575, permute_277)
        view_578: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_102, _shape_param_730);  mm_102 = _shape_param_730 = None
        view_579: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_576, _shape_param_731);  view_576 = _shape_param_731 = None
        view_580: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_577, _shape_param_732);  view_577 = _shape_param_732 = None
        view_581: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_578, _shape_param_733);  view_578 = _shape_param_733 = None
        permute_278: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_581, [0, 2, 1, 3]);  view_581 = None
        repeat_26: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg284_1, [1, 1, 1]);  arg284_1 = None
        gather_25: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_26, 1, repeat_1);  repeat_26 = None
        convert_element_type_404: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_25, torch.bfloat16);  gather_25 = None
        split_25 = torch.ops.aten.split.Tensor(convert_element_type_404, 32, -1);  convert_element_type_404 = None
        getitem_102: "bf16[1, 128, 32]" = split_25[0]
        getitem_103: "bf16[1, 128, 32]" = split_25[1];  split_25 = None
        slice_201: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_580, 3, 0, 64)
        slice_202: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_580, 3, 64, 9223372036854775807);  view_580 = None
        slice_203: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_579, 3, 0, 64)
        slice_204: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_579, 3, 64, 9223372036854775807);  view_579 = None
        unsqueeze_207: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_102, 2);  getitem_102 = None
        unsqueeze_208: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 4);  unsqueeze_207 = None
        expand_151: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_208, _shape_param_734);  _shape_param_734 = None
        clone_75: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_151, memory_format = torch.contiguous_format);  expand_151 = None
        view_582: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_75, _shape_param_735);  clone_75 = _shape_param_735 = None
        unsqueeze_209: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_103, 2);  getitem_103 = None
        unsqueeze_210: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 4);  unsqueeze_209 = None
        expand_152: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_210, _shape_param_736);  _shape_param_736 = None
        clone_76: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_152, memory_format = torch.contiguous_format);  expand_152 = None
        view_583: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_76, _shape_param_737);  clone_76 = _shape_param_737 = None
        mul_252: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_201, view_583)
        slice_205: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_201, 3, 0, 9223372036854775807, 2)
        slice_206: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_201, 3, 1, 9223372036854775807, 2);  slice_201 = None
        neg_50: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_206);  slice_206 = None
        unsqueeze_211: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_50, 4);  neg_50 = None
        unsqueeze_212: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_205, 4);  slice_205 = None
        cat_100: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_211, unsqueeze_212], -1);  unsqueeze_211 = unsqueeze_212 = None
        view_584: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_100, _shape_param_738);  cat_100 = _shape_param_738 = None
        mul_253: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_584, view_582);  view_584 = None
        add_227: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
        mul_254: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_203, view_583);  view_583 = None
        slice_207: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_203, 3, 0, 9223372036854775807, 2)
        slice_208: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_203, 3, 1, 9223372036854775807, 2);  slice_203 = None
        neg_51: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_208);  slice_208 = None
        unsqueeze_213: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_51, 4);  neg_51 = None
        unsqueeze_214: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_207, 4);  slice_207 = None
        cat_101: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_213, unsqueeze_214], -1);  unsqueeze_213 = unsqueeze_214 = None
        view_585: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_101, _shape_param_739);  cat_101 = _shape_param_739 = None
        mul_255: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_585, view_582);  view_585 = view_582 = None
        add_228: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None
        cat_102: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_227, slice_202], -1);  add_227 = slice_202 = None
        cat_103: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_228, slice_204], -1);  add_228 = slice_204 = None
        permute_279: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_102, [0, 2, 1, 3]);  cat_102 = None
        permute_280: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_103, [0, 2, 1, 3]);  cat_103 = None
        convert_element_type_405: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_279, torch.float32);  permute_279 = None
        permute_281: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_405, [0, 1, 3, 2]);  convert_element_type_405 = None
        convert_element_type_406: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_281, torch.bfloat16);  permute_281 = None
        convert_element_type_407: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_280, torch.bfloat16);  permute_280 = None
        expand_153: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_407, _shape_param_740);  convert_element_type_407 = _shape_param_740 = None
        view_586: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_153, _shape_param_741);  expand_153 = _shape_param_741 = None
        expand_154: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_406, _shape_param_742);  convert_element_type_406 = _shape_param_742 = None
        view_587: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_154, _shape_param_743);  expand_154 = _shape_param_743 = None
        bmm_50: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_586, view_587)
        view_588: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_50, _shape_param_744);  bmm_50 = _shape_param_744 = None
        div_50: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_588, 16.0);  view_588 = None
        add_229: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_50, where);  div_50 = None
        amax_25: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_229, [-1], True)
        sub_51: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_229, amax_25);  add_229 = amax_25 = None
        exp_25: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
        sum_26: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_51: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
        convert_element_type_408: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_51, torch.bfloat16)
        expand_155: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_408, _shape_param_745);  convert_element_type_408 = _shape_param_745 = None
        view_589: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_155, _shape_param_746);  expand_155 = _shape_param_746 = None
        expand_156: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_278, _shape_param_747);  permute_278 = _shape_param_747 = None
        view_590: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_156, _shape_param_748);  expand_156 = _shape_param_748 = None
        bmm_51: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_589, view_590)
        view_591: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_51, _shape_param_749);  bmm_51 = _shape_param_749 = None
        permute_282: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_591, [0, 2, 1, 3]);  view_591 = None
        clone_77: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_282, memory_format = torch.contiguous_format);  permute_282 = None
        view_592: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_77, _shape_param_750);  clone_77 = _shape_param_750 = None
        convert_element_type_409: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg285_1, torch.bfloat16);  arg285_1 = None
        permute_283: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_409, [1, 0]);  convert_element_type_409 = None
        view_593: "bf16[128, 4096]" = torch.ops.aten.view.default(view_592, _shape_param_751);  view_592 = _shape_param_751 = None
        mm_103: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_593, permute_283)
        view_594: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_103, _shape_param_752);  mm_103 = _shape_param_752 = None
        convert_element_type_410: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg286_1, torch.bfloat16);  arg286_1 = None
        convert_element_type_411: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg287_1, torch.bfloat16);  arg287_1 = None
        permute_284: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_411, [1, 0]);  convert_element_type_411 = None
        addmm_50: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_410, view_575, permute_284);  convert_element_type_410 = None
        view_595: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_50, _shape_param_753);  _shape_param_753 = None
        mul_256: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_595, 0.5)
        convert_element_type_412: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_595, torch.float32)
        pow_26: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_412, 3.0);  convert_element_type_412 = None
        mul_257: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_26, 0.044715);  pow_26 = None
        add_230: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_595, mul_257);  view_595 = mul_257 = None
        mul_258: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_230, 0.7978845608028654);  add_230 = None
        tanh_25: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_258);  mul_258 = None
        add_231: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_25, 1.0);  tanh_25 = None
        mul_259: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_256, add_231);  mul_256 = add_231 = None
        convert_element_type_413: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg288_1, torch.bfloat16);  arg288_1 = None
        convert_element_type_414: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg289_1, torch.bfloat16);  arg289_1 = None
        convert_element_type_415: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_259, torch.bfloat16);  mul_259 = None
        view_596: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_415, _shape_param_754);  convert_element_type_415 = _shape_param_754 = None
        permute_285: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_414, [1, 0]);  convert_element_type_414 = None
        addmm_51: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_413, view_596, permute_285);  convert_element_type_413 = None
        view_597: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_51, _shape_param_755);  addmm_51 = _shape_param_755 = None
        add_232: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_594, view_597);  view_594 = view_597 = None
        add_233: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_232, add_224);  add_232 = add_224 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(add_233, [2], correction = 0, keepdim = True)
        getitem_104: "f32[1, 128, 1]" = var_mean_26[0]
        getitem_105: "f32[1, 128, 1]" = var_mean_26[1];  var_mean_26 = None
        add_234: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_26: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
        sub_52: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_233, getitem_105);  getitem_105 = None
        mul_260: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_26);  sub_52 = None
        mul_261: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_260, arg290_1);  arg290_1 = None
        add_235: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_261, arg291_1);  mul_261 = arg291_1 = None
        convert_element_type_416: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg292_1, torch.bfloat16);  arg292_1 = None
        convert_element_type_417: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_235, torch.bfloat16);  add_235 = None
        permute_286: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_416, [1, 0]);  convert_element_type_416 = None
        view_598: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_417, _shape_param_756);  convert_element_type_417 = _shape_param_756 = None
        mm_104: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_598, permute_286)
        view_599: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_104, _shape_param_757);  mm_104 = _shape_param_757 = None
        convert_element_type_418: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg293_1, torch.bfloat16);  arg293_1 = None
        permute_287: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_418, [1, 0]);  convert_element_type_418 = None
        mm_105: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_598, permute_287)
        view_600: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_105, _shape_param_758);  mm_105 = _shape_param_758 = None
        convert_element_type_419: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg294_1, torch.bfloat16);  arg294_1 = None
        permute_288: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_419, [1, 0]);  convert_element_type_419 = None
        mm_106: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_598, permute_288)
        view_601: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_106, _shape_param_759);  mm_106 = _shape_param_759 = None
        view_602: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_599, _shape_param_760);  view_599 = _shape_param_760 = None
        view_603: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_600, _shape_param_761);  view_600 = _shape_param_761 = None
        view_604: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_601, _shape_param_762);  view_601 = _shape_param_762 = None
        permute_289: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_604, [0, 2, 1, 3]);  view_604 = None
        repeat_27: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg295_1, [1, 1, 1]);  arg295_1 = None
        gather_26: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_27, 1, repeat_1);  repeat_27 = None
        convert_element_type_420: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_26, torch.bfloat16);  gather_26 = None
        split_26 = torch.ops.aten.split.Tensor(convert_element_type_420, 32, -1);  convert_element_type_420 = None
        getitem_106: "bf16[1, 128, 32]" = split_26[0]
        getitem_107: "bf16[1, 128, 32]" = split_26[1];  split_26 = None
        slice_209: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_603, 3, 0, 64)
        slice_210: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_603, 3, 64, 9223372036854775807);  view_603 = None
        slice_211: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_602, 3, 0, 64)
        slice_212: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_602, 3, 64, 9223372036854775807);  view_602 = None
        unsqueeze_215: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_106, 2);  getitem_106 = None
        unsqueeze_216: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 4);  unsqueeze_215 = None
        expand_157: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_216, _shape_param_763);  _shape_param_763 = None
        clone_78: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_157, memory_format = torch.contiguous_format);  expand_157 = None
        view_605: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_78, _shape_param_764);  clone_78 = _shape_param_764 = None
        unsqueeze_217: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_107, 2);  getitem_107 = None
        unsqueeze_218: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 4);  unsqueeze_217 = None
        expand_158: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_218, _shape_param_765);  _shape_param_765 = None
        clone_79: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_158, memory_format = torch.contiguous_format);  expand_158 = None
        view_606: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_79, _shape_param_766);  clone_79 = _shape_param_766 = None
        mul_262: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_209, view_606)
        slice_213: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_209, 3, 0, 9223372036854775807, 2)
        slice_214: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_209, 3, 1, 9223372036854775807, 2);  slice_209 = None
        neg_52: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_214);  slice_214 = None
        unsqueeze_219: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_52, 4);  neg_52 = None
        unsqueeze_220: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_213, 4);  slice_213 = None
        cat_104: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_219, unsqueeze_220], -1);  unsqueeze_219 = unsqueeze_220 = None
        view_607: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_104, _shape_param_767);  cat_104 = _shape_param_767 = None
        mul_263: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_607, view_605);  view_607 = None
        add_236: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_262, mul_263);  mul_262 = mul_263 = None
        mul_264: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_211, view_606);  view_606 = None
        slice_215: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_211, 3, 0, 9223372036854775807, 2)
        slice_216: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_211, 3, 1, 9223372036854775807, 2);  slice_211 = None
        neg_53: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_216);  slice_216 = None
        unsqueeze_221: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_53, 4);  neg_53 = None
        unsqueeze_222: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_215, 4);  slice_215 = None
        cat_105: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_221, unsqueeze_222], -1);  unsqueeze_221 = unsqueeze_222 = None
        view_608: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_105, _shape_param_768);  cat_105 = _shape_param_768 = None
        mul_265: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_608, view_605);  view_608 = view_605 = None
        add_237: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None
        cat_106: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_236, slice_210], -1);  add_236 = slice_210 = None
        cat_107: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_237, slice_212], -1);  add_237 = slice_212 = None
        permute_290: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_106, [0, 2, 1, 3]);  cat_106 = None
        permute_291: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_107, [0, 2, 1, 3]);  cat_107 = None
        convert_element_type_421: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_290, torch.float32);  permute_290 = None
        permute_292: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_421, [0, 1, 3, 2]);  convert_element_type_421 = None
        convert_element_type_422: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_292, torch.bfloat16);  permute_292 = None
        convert_element_type_423: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_291, torch.bfloat16);  permute_291 = None
        expand_159: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_423, _shape_param_769);  convert_element_type_423 = _shape_param_769 = None
        view_609: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_159, _shape_param_770);  expand_159 = _shape_param_770 = None
        expand_160: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_422, _shape_param_771);  convert_element_type_422 = _shape_param_771 = None
        view_610: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_160, _shape_param_772);  expand_160 = _shape_param_772 = None
        bmm_52: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_609, view_610)
        view_611: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_52, _shape_param_773);  bmm_52 = _shape_param_773 = None
        div_52: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_611, 16.0);  view_611 = None
        add_238: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_52, where);  div_52 = None
        amax_26: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_238, [-1], True)
        sub_53: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_238, amax_26);  add_238 = amax_26 = None
        exp_26: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_53);  sub_53 = None
        sum_27: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_53: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
        convert_element_type_424: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_53, torch.bfloat16)
        expand_161: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_424, _shape_param_774);  convert_element_type_424 = _shape_param_774 = None
        view_612: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_161, _shape_param_775);  expand_161 = _shape_param_775 = None
        expand_162: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_289, _shape_param_776);  permute_289 = _shape_param_776 = None
        view_613: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_162, _shape_param_777);  expand_162 = _shape_param_777 = None
        bmm_53: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_612, view_613)
        view_614: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_53, _shape_param_778);  bmm_53 = _shape_param_778 = None
        permute_293: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None
        clone_80: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
        view_615: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_80, _shape_param_779);  clone_80 = _shape_param_779 = None
        convert_element_type_425: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg296_1, torch.bfloat16);  arg296_1 = None
        permute_294: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_425, [1, 0]);  convert_element_type_425 = None
        view_616: "bf16[128, 4096]" = torch.ops.aten.view.default(view_615, _shape_param_780);  view_615 = _shape_param_780 = None
        mm_107: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_616, permute_294)
        view_617: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_107, _shape_param_781);  mm_107 = _shape_param_781 = None
        convert_element_type_426: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg297_1, torch.bfloat16);  arg297_1 = None
        convert_element_type_427: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg298_1, torch.bfloat16);  arg298_1 = None
        permute_295: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_427, [1, 0]);  convert_element_type_427 = None
        addmm_52: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_426, view_598, permute_295);  convert_element_type_426 = None
        view_618: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_52, _shape_param_782);  _shape_param_782 = None
        mul_266: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_618, 0.5)
        convert_element_type_428: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_618, torch.float32)
        pow_27: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_428, 3.0);  convert_element_type_428 = None
        mul_267: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_27, 0.044715);  pow_27 = None
        add_239: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_618, mul_267);  view_618 = mul_267 = None
        mul_268: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_239, 0.7978845608028654);  add_239 = None
        tanh_26: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_268);  mul_268 = None
        add_240: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_26, 1.0);  tanh_26 = None
        mul_269: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_266, add_240);  mul_266 = add_240 = None
        convert_element_type_429: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg299_1, torch.bfloat16);  arg299_1 = None
        convert_element_type_430: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg300_1, torch.bfloat16);  arg300_1 = None
        convert_element_type_431: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_269, torch.bfloat16);  mul_269 = None
        view_619: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_431, _shape_param_783);  convert_element_type_431 = _shape_param_783 = None
        permute_296: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_430, [1, 0]);  convert_element_type_430 = None
        addmm_53: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_429, view_619, permute_296);  convert_element_type_429 = None
        view_620: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_53, _shape_param_784);  addmm_53 = _shape_param_784 = None
        add_241: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_617, view_620);  view_617 = view_620 = None
        add_242: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_241, add_233);  add_241 = add_233 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(add_242, [2], correction = 0, keepdim = True)
        getitem_108: "f32[1, 128, 1]" = var_mean_27[0]
        getitem_109: "f32[1, 128, 1]" = var_mean_27[1];  var_mean_27 = None
        add_243: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_27: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
        sub_54: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_242, getitem_109);  getitem_109 = None
        mul_270: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_27);  sub_54 = None
        mul_271: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_270, arg301_1);  arg301_1 = None
        add_244: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_271, arg302_1);  mul_271 = arg302_1 = None
        convert_element_type_432: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg303_1, torch.bfloat16);  arg303_1 = None
        convert_element_type_433: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_244, torch.bfloat16);  add_244 = None
        permute_297: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_432, [1, 0]);  convert_element_type_432 = None
        view_621: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_433, _shape_param_785);  convert_element_type_433 = _shape_param_785 = None
        mm_108: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_621, permute_297)
        view_622: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_108, _shape_param_786);  mm_108 = _shape_param_786 = None
        convert_element_type_434: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg304_1, torch.bfloat16);  arg304_1 = None
        permute_298: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_434, [1, 0]);  convert_element_type_434 = None
        mm_109: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_621, permute_298)
        view_623: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_109, _shape_param_787);  mm_109 = _shape_param_787 = None
        convert_element_type_435: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg305_1, torch.bfloat16);  arg305_1 = None
        permute_299: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_435, [1, 0]);  convert_element_type_435 = None
        mm_110: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_621, permute_299)
        view_624: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_110, _shape_param_788);  mm_110 = _shape_param_788 = None
        view_625: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_622, _shape_param_789);  view_622 = _shape_param_789 = None
        view_626: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_623, _shape_param_790);  view_623 = _shape_param_790 = None
        view_627: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_624, _shape_param_791);  view_624 = _shape_param_791 = None
        permute_300: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_627, [0, 2, 1, 3]);  view_627 = None
        repeat_28: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg306_1, [1, 1, 1]);  arg306_1 = None
        gather_27: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_28, 1, repeat_1);  repeat_28 = repeat_1 = None
        convert_element_type_436: "bf16[1, 128, 64]" = torch.ops.prims.convert_element_type.default(gather_27, torch.bfloat16);  gather_27 = None
        split_27 = torch.ops.aten.split.Tensor(convert_element_type_436, 32, -1);  convert_element_type_436 = None
        getitem_110: "bf16[1, 128, 32]" = split_27[0]
        getitem_111: "bf16[1, 128, 32]" = split_27[1];  split_27 = None
        slice_217: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_626, 3, 0, 64)
        slice_218: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_626, 3, 64, 9223372036854775807);  view_626 = None
        slice_219: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_625, 3, 0, 64)
        slice_220: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_625, 3, 64, 9223372036854775807);  view_625 = None
        unsqueeze_223: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_110, 2);  getitem_110 = None
        unsqueeze_224: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 4);  unsqueeze_223 = None
        expand_163: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_224, _shape_param_792);  _shape_param_792 = None
        clone_81: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_163, memory_format = torch.contiguous_format);  expand_163 = None
        view_628: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_81, _shape_param_793);  clone_81 = _shape_param_793 = None
        unsqueeze_225: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_111, 2);  getitem_111 = None
        unsqueeze_226: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 4);  unsqueeze_225 = None
        expand_164: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_226, _shape_param_794);  _shape_param_794 = None
        clone_82: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_164, memory_format = torch.contiguous_format);  expand_164 = None
        view_629: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_82, _shape_param_795);  clone_82 = _shape_param_795 = None
        mul_272: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_217, view_629)
        slice_221: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_217, 3, 0, 9223372036854775807, 2)
        slice_222: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_217, 3, 1, 9223372036854775807, 2);  slice_217 = None
        neg_54: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_222);  slice_222 = None
        unsqueeze_227: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_54, 4);  neg_54 = None
        unsqueeze_228: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_221, 4);  slice_221 = None
        cat_108: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_227, unsqueeze_228], -1);  unsqueeze_227 = unsqueeze_228 = None
        view_630: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_108, _shape_param_796);  cat_108 = _shape_param_796 = None
        mul_273: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_630, view_628);  view_630 = None
        add_245: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
        mul_274: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_219, view_629);  view_629 = None
        slice_223: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_219, 3, 0, 9223372036854775807, 2)
        slice_224: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_219, 3, 1, 9223372036854775807, 2);  slice_219 = None
        neg_55: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_224);  slice_224 = None
        unsqueeze_229: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_55, 4);  neg_55 = None
        unsqueeze_230: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_223, 4);  slice_223 = None
        cat_109: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_229, unsqueeze_230], -1);  unsqueeze_229 = unsqueeze_230 = None
        view_631: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_109, _shape_param_797);  cat_109 = _shape_param_797 = None
        mul_275: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_631, view_628);  view_631 = view_628 = None
        add_246: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
        cat_110: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_245, slice_218], -1);  add_245 = slice_218 = None
        cat_111: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_246, slice_220], -1);  add_246 = slice_220 = None
        permute_301: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_110, [0, 2, 1, 3]);  cat_110 = None
        permute_302: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_111, [0, 2, 1, 3]);  cat_111 = None
        convert_element_type_437: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_301, torch.float32);  permute_301 = None
        permute_303: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_437, [0, 1, 3, 2]);  convert_element_type_437 = None
        convert_element_type_438: "bf16[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(permute_303, torch.bfloat16);  permute_303 = None
        convert_element_type_439: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_302, torch.bfloat16);  permute_302 = None
        expand_165: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_439, _shape_param_798);  convert_element_type_439 = _shape_param_798 = None
        view_632: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_165, _shape_param_799);  expand_165 = _shape_param_799 = None
        expand_166: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(convert_element_type_438, _shape_param_800);  convert_element_type_438 = _shape_param_800 = None
        view_633: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_166, _shape_param_801);  expand_166 = _shape_param_801 = None
        bmm_54: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_632, view_633)
        view_634: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_54, _shape_param_802);  bmm_54 = _shape_param_802 = None
        div_54: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_634, 16.0);  view_634 = None
        add_247: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_54, where);  div_54 = where = None
        amax_27: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_247, [-1], True)
        sub_55: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_247, amax_27);  add_247 = amax_27 = None
        exp_27: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_28: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
        div_55: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None
        convert_element_type_440: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_55, torch.bfloat16)
        expand_167: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_440, _shape_param_803);  convert_element_type_440 = _shape_param_803 = None
        view_635: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_167, _shape_param_804);  expand_167 = _shape_param_804 = None
        expand_168: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_300, _shape_param_805);  permute_300 = _shape_param_805 = None
        view_636: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_168, _shape_param_806);  expand_168 = _shape_param_806 = None
        bmm_55: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_635, view_636)
        view_637: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_55, _shape_param_807);  bmm_55 = _shape_param_807 = None
        permute_304: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_637, [0, 2, 1, 3]);  view_637 = None
        clone_83: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_304, memory_format = torch.contiguous_format);  permute_304 = None
        view_638: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_83, _shape_param_808);  clone_83 = _shape_param_808 = None
        convert_element_type_441: "bf16[4096, 4096]" = torch.ops.prims.convert_element_type.default(arg307_1, torch.bfloat16);  arg307_1 = None
        permute_305: "bf16[4096, 4096]" = torch.ops.aten.permute.default(convert_element_type_441, [1, 0]);  convert_element_type_441 = None
        view_639: "bf16[128, 4096]" = torch.ops.aten.view.default(view_638, _shape_param_809);  view_638 = _shape_param_809 = None
        mm_111: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_639, permute_305)
        view_640: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_111, _shape_param_810);  mm_111 = _shape_param_810 = None
        convert_element_type_442: "bf16[16384]" = torch.ops.prims.convert_element_type.default(arg308_1, torch.bfloat16);  arg308_1 = None
        convert_element_type_443: "bf16[16384, 4096]" = torch.ops.prims.convert_element_type.default(arg309_1, torch.bfloat16);  arg309_1 = None
        permute_306: "bf16[4096, 16384]" = torch.ops.aten.permute.default(convert_element_type_443, [1, 0]);  convert_element_type_443 = None
        addmm_54: "bf16[128, 16384]" = torch.ops.aten.addmm.default(convert_element_type_442, view_621, permute_306);  convert_element_type_442 = None
        view_641: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_54, _shape_param_811);  _shape_param_811 = None
        mul_276: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_641, 0.5)
        convert_element_type_444: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_641, torch.float32)
        pow_28: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_444, 3.0);  convert_element_type_444 = None
        mul_277: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_28, 0.044715);  pow_28 = None
        add_248: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_641, mul_277);  view_641 = mul_277 = None
        mul_278: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_248, 0.7978845608028654);  add_248 = None
        tanh_27: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_278);  mul_278 = None
        add_249: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_27, 1.0);  tanh_27 = None
        mul_279: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_276, add_249);  mul_276 = add_249 = None
        convert_element_type_445: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg310_1, torch.bfloat16);  arg310_1 = None
        convert_element_type_446: "bf16[4096, 16384]" = torch.ops.prims.convert_element_type.default(arg311_1, torch.bfloat16);  arg311_1 = None
        convert_element_type_447: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_279, torch.bfloat16);  mul_279 = None
        view_642: "bf16[128, 16384]" = torch.ops.aten.view.default(convert_element_type_447, _shape_param_812);  convert_element_type_447 = _shape_param_812 = None
        permute_307: "bf16[16384, 4096]" = torch.ops.aten.permute.default(convert_element_type_446, [1, 0]);  convert_element_type_446 = None
        addmm_55: "bf16[128, 4096]" = torch.ops.aten.addmm.default(convert_element_type_445, view_642, permute_307);  convert_element_type_445 = None
        view_643: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_55, _shape_param_813);  addmm_55 = _shape_param_813 = None
        add_250: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_640, view_643);  view_640 = view_643 = None
        add_251: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_250, add_242);  add_250 = add_242 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add_251, [2], correction = 0, keepdim = True)
        getitem_112: "f32[1, 128, 1]" = var_mean_28[0]
        getitem_113: "f32[1, 128, 1]" = var_mean_28[1];  var_mean_28 = None
        add_252: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_28: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
        sub_56: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_251, getitem_113);  add_251 = getitem_113 = None
        mul_280: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_28);  sub_56 = None
        mul_281: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_280, arg312_1);  arg312_1 = None
        add_253: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_281, arg313_1);  mul_281 = arg313_1 = None
        convert_element_type_448: "bf16[50400]" = torch.ops.prims.convert_element_type.default(arg314_1, torch.bfloat16);  arg314_1 = None
        convert_element_type_449: "bf16[50400, 4096]" = torch.ops.prims.convert_element_type.default(arg315_1, torch.bfloat16);  arg315_1 = None
        convert_element_type_450: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_253, torch.bfloat16);  add_253 = None
        view_644: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_450, _shape_param_814);  convert_element_type_450 = _shape_param_814 = None
        permute_308: "bf16[4096, 50400]" = torch.ops.aten.permute.default(convert_element_type_449, [1, 0]);  convert_element_type_449 = None
        addmm_56: "bf16[128, 50400]" = torch.ops.aten.addmm.default(convert_element_type_448, view_644, permute_308);  convert_element_type_448 = None
        view_645: "bf16[1, 128, 50400]" = torch.ops.aten.view.default(addmm_56, _shape_param_815);  addmm_56 = _shape_param_815 = None
        convert_element_type_451: "f32[1, 128, 50400]" = torch.ops.prims.convert_element_type.default(view_645, torch.float32)
        constant_pad_nd: "i64[1, 129]" = torch.ops.aten.constant_pad_nd.default(arg316_1, [0, 1], -100.0);  arg316_1 = None
        slice_225: "i64[1, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807)
        view_646: "f32[128, 50400]" = torch.ops.aten.view.default(convert_element_type_451, _shape_param_816);  convert_element_type_451 = _shape_param_816 = None
        view_647: "i64[128]" = torch.ops.aten.view.default(slice_225, [-1]);  slice_225 = None
        amax_28: "f32[128, 1]" = torch.ops.aten.amax.default(view_646, [1], True)
        sub_57: "f32[128, 50400]" = torch.ops.aten.sub.Tensor(view_646, amax_28);  view_646 = None
        exp_28: "f32[128, 50400]" = torch.ops.aten.exp.default(sub_57)
        sum_29: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [1], True);  exp_28 = None
        log: "f32[128, 1]" = torch.ops.aten.log.default(sum_29);  sum_29 = None
        sub_58: "f32[128, 50400]" = torch.ops.aten.sub.Tensor(sub_57, log);  sub_57 = None
        ne: "b8[128]" = torch.ops.aten.ne.Scalar(view_647, -100)
        full_3: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[128]" = torch.ops.aten.where.self(ne, view_647, full_3);  view_647 = full_3 = None
        unsqueeze_231: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather_28: "f32[128, 1]" = torch.ops.aten.gather.default(sub_58, 1, unsqueeze_231);  sub_58 = unsqueeze_231 = None
        squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(gather_28, 1);  gather_28 = None
        neg_56: "f32[128]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_2: "f32[128]" = torch.ops.aten.where.self(ne, neg_56, full_1);  neg_56 = full_1 = None
        sum_30: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_452: "f32[]" = torch.ops.prims.convert_element_type.default(sum_30, torch.float32);  sum_30 = None
        sum_31: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
        div_56: "f32[]" = torch.ops.aten.div.Tensor(sum_31, convert_element_type_452);  sum_31 = None
        permute_309: "bf16[50400, 4096]" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
        div_57: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 4096);  rsqrt_28 = None
        permute_310: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_307, [1, 0]);  permute_307 = None
        permute_311: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
        permute_312: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
        permute_313: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_635, [0, 2, 1]);  view_635 = None
        permute_314: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_636, [0, 2, 1]);  view_636 = None
        permute_315: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_632, [0, 2, 1]);  view_632 = None
        permute_316: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_633, [0, 2, 1]);  view_633 = None
        permute_317: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_299, [1, 0]);  permute_299 = None
        permute_318: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
        permute_319: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
        div_58: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 4096);  rsqrt_27 = None
        permute_320: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
        permute_321: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_295, [1, 0]);  permute_295 = None
        permute_322: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_294, [1, 0]);  permute_294 = None
        permute_323: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_612, [0, 2, 1]);  view_612 = None
        permute_324: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_613, [0, 2, 1]);  view_613 = None
        permute_325: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_609, [0, 2, 1]);  view_609 = None
        permute_326: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_610, [0, 2, 1]);  view_610 = None
        permute_327: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_288, [1, 0]);  permute_288 = None
        permute_328: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_287, [1, 0]);  permute_287 = None
        permute_329: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_286, [1, 0]);  permute_286 = None
        div_59: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 4096);  rsqrt_26 = None
        permute_330: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
        permute_331: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
        permute_332: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_283, [1, 0]);  permute_283 = None
        permute_333: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_589, [0, 2, 1]);  view_589 = None
        permute_334: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_590, [0, 2, 1]);  view_590 = None
        permute_335: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_586, [0, 2, 1]);  view_586 = None
        permute_336: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_587, [0, 2, 1]);  view_587 = None
        permute_337: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
        permute_338: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
        permute_339: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
        div_60: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 4096);  rsqrt_25 = None
        permute_340: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
        permute_341: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
        permute_342: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
        permute_343: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_566, [0, 2, 1]);  view_566 = None
        permute_344: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_567, [0, 2, 1]);  view_567 = None
        permute_345: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_563, [0, 2, 1]);  view_563 = None
        permute_346: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_564, [0, 2, 1]);  view_564 = None
        permute_347: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
        permute_348: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
        permute_349: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
        div_61: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 4096);  rsqrt_24 = None
        permute_350: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        permute_351: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
        permute_352: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
        permute_353: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_543, [0, 2, 1]);  view_543 = None
        permute_354: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_544, [0, 2, 1]);  view_544 = None
        permute_355: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_540, [0, 2, 1]);  view_540 = None
        permute_356: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_541, [0, 2, 1]);  view_541 = None
        permute_357: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
        permute_358: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
        permute_359: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
        div_62: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 4096);  rsqrt_23 = None
        permute_360: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
        permute_361: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
        permute_362: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
        permute_363: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_520, [0, 2, 1]);  view_520 = None
        permute_364: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_521, [0, 2, 1]);  view_521 = None
        permute_365: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_517, [0, 2, 1]);  view_517 = None
        permute_366: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_518, [0, 2, 1]);  view_518 = None
        permute_367: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
        permute_368: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
        permute_369: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        div_63: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 4096);  rsqrt_22 = None
        permute_370: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        permute_371: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
        permute_372: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
        permute_373: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_497, [0, 2, 1]);  view_497 = None
        permute_374: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_498, [0, 2, 1]);  view_498 = None
        permute_375: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_494, [0, 2, 1]);  view_494 = None
        permute_376: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_495, [0, 2, 1]);  view_495 = None
        permute_377: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        permute_378: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
        permute_379: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
        div_64: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 4096);  rsqrt_21 = None
        permute_380: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
        permute_381: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
        permute_382: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
        permute_383: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_474, [0, 2, 1]);  view_474 = None
        permute_384: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_475, [0, 2, 1]);  view_475 = None
        permute_385: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_471, [0, 2, 1]);  view_471 = None
        permute_386: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_472, [0, 2, 1]);  view_472 = None
        permute_387: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
        permute_388: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
        permute_389: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
        div_65: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 4096);  rsqrt_20 = None
        permute_390: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
        permute_391: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
        permute_392: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
        permute_393: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_451, [0, 2, 1]);  view_451 = None
        permute_394: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_452, [0, 2, 1]);  view_452 = None
        permute_395: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_448, [0, 2, 1]);  view_448 = None
        permute_396: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_449, [0, 2, 1]);  view_449 = None
        permute_397: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        permute_398: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
        permute_399: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
        div_66: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 4096);  rsqrt_19 = None
        permute_400: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
        permute_401: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
        permute_402: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
        permute_403: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_428, [0, 2, 1]);  view_428 = None
        permute_404: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_429, [0, 2, 1]);  view_429 = None
        permute_405: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_425, [0, 2, 1]);  view_425 = None
        permute_406: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_426, [0, 2, 1]);  view_426 = None
        permute_407: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
        permute_408: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
        permute_409: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
        div_67: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 4096);  rsqrt_18 = None
        permute_410: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
        permute_411: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
        permute_412: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        permute_413: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_405, [0, 2, 1]);  view_405 = None
        permute_414: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_406, [0, 2, 1]);  view_406 = None
        permute_415: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_402, [0, 2, 1]);  view_402 = None
        permute_416: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_403, [0, 2, 1]);  view_403 = None
        permute_417: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
        permute_418: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
        permute_419: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
        div_68: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 4096);  rsqrt_17 = None
        permute_420: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        permute_421: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        permute_422: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
        permute_423: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_382, [0, 2, 1]);  view_382 = None
        permute_424: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_383, [0, 2, 1]);  view_383 = None
        permute_425: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_379, [0, 2, 1]);  view_379 = None
        permute_426: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_380, [0, 2, 1]);  view_380 = None
        permute_427: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        permute_428: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
        permute_429: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        div_69: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 4096);  rsqrt_16 = None
        permute_430: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
        permute_431: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        permute_432: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        permute_433: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_359, [0, 2, 1]);  view_359 = None
        permute_434: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_360, [0, 2, 1]);  view_360 = None
        permute_435: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_356, [0, 2, 1]);  view_356 = None
        permute_436: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_357, [0, 2, 1]);  view_357 = None
        permute_437: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        permute_438: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
        permute_439: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        div_70: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 4096);  rsqrt_15 = None
        permute_440: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
        permute_441: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        permute_442: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
        permute_443: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_336, [0, 2, 1]);  view_336 = None
        permute_444: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_337, [0, 2, 1]);  view_337 = None
        permute_445: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
        permute_446: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_334, [0, 2, 1]);  view_334 = None
        permute_447: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        permute_448: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
        permute_449: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
        div_71: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 4096);  rsqrt_14 = None
        permute_450: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        permute_451: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        permute_452: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        permute_453: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_313, [0, 2, 1]);  view_313 = None
        permute_454: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_314, [0, 2, 1]);  view_314 = None
        permute_455: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_310, [0, 2, 1]);  view_310 = None
        permute_456: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None
        permute_457: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        permute_458: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        permute_459: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        div_72: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 4096);  rsqrt_13 = None
        permute_460: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        permute_461: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        permute_462: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        permute_463: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_290, [0, 2, 1]);  view_290 = None
        permute_464: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_291, [0, 2, 1]);  view_291 = None
        permute_465: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None
        permute_466: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_288, [0, 2, 1]);  view_288 = None
        permute_467: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        permute_468: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        permute_469: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        div_73: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 4096);  rsqrt_12 = None
        permute_470: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        permute_471: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        permute_472: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        permute_473: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
        permute_474: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_268, [0, 2, 1]);  view_268 = None
        permute_475: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_264, [0, 2, 1]);  view_264 = None
        permute_476: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_265, [0, 2, 1]);  view_265 = None
        permute_477: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        permute_478: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        permute_479: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        div_74: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 4096);  rsqrt_11 = None
        permute_480: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        permute_481: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        permute_482: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        permute_483: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_244, [0, 2, 1]);  view_244 = None
        permute_484: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None
        permute_485: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_241, [0, 2, 1]);  view_241 = None
        permute_486: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_242, [0, 2, 1]);  view_242 = None
        permute_487: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        permute_488: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        permute_489: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        div_75: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 4096);  rsqrt_10 = None
        permute_490: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        permute_491: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        permute_492: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        permute_493: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_221, [0, 2, 1]);  view_221 = None
        permute_494: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_222, [0, 2, 1]);  view_222 = None
        permute_495: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_218, [0, 2, 1]);  view_218 = None
        permute_496: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_219, [0, 2, 1]);  view_219 = None
        permute_497: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        permute_498: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
        permute_499: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        div_76: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 4096);  rsqrt_9 = None
        permute_500: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        permute_501: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        permute_502: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        permute_503: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_198, [0, 2, 1]);  view_198 = None
        permute_504: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_199, [0, 2, 1]);  view_199 = None
        permute_505: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_195, [0, 2, 1]);  view_195 = None
        permute_506: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_196, [0, 2, 1]);  view_196 = None
        permute_507: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        permute_508: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        permute_509: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        div_77: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 4096);  rsqrt_8 = None
        permute_510: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        permute_511: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        permute_512: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        permute_513: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_175, [0, 2, 1]);  view_175 = None
        permute_514: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None
        permute_515: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
        permute_516: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
        permute_517: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        permute_518: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        permute_519: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        div_78: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 4096);  rsqrt_7 = None
        permute_520: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        permute_521: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        permute_522: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        permute_523: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
        permute_524: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
        permute_525: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_149, [0, 2, 1]);  view_149 = None
        permute_526: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_150, [0, 2, 1]);  view_150 = None
        permute_527: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        permute_528: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        permute_529: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        div_79: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 4096);  rsqrt_6 = None
        permute_530: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        permute_531: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        permute_532: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        permute_533: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_129, [0, 2, 1]);  view_129 = None
        permute_534: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
        permute_535: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_126, [0, 2, 1]);  view_126 = None
        permute_536: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_127, [0, 2, 1]);  view_127 = None
        permute_537: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        permute_538: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        permute_539: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        div_80: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 4096);  rsqrt_5 = None
        permute_540: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        permute_541: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        permute_542: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        permute_543: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
        permute_544: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
        permute_545: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
        permute_546: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_104, [0, 2, 1]);  view_104 = None
        permute_547: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        permute_548: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        permute_549: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        div_81: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 4096);  rsqrt_4 = None
        permute_550: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        permute_551: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        permute_552: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        permute_553: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
        permute_554: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_84, [0, 2, 1]);  view_84 = None
        permute_555: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
        permute_556: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
        permute_557: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        permute_558: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        permute_559: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        div_82: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 4096);  rsqrt_3 = None
        permute_560: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        permute_561: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        permute_562: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        permute_563: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
        permute_564: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
        permute_565: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
        permute_566: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
        permute_567: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        permute_568: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        permute_569: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        div_83: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 4096);  rsqrt_2 = None
        permute_570: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        permute_571: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        permute_572: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        permute_573: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
        permute_574: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_38, [0, 2, 1]);  view_38 = None
        permute_575: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        permute_576: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
        permute_577: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        permute_578: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_579: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        permute_580: "bf16[4096, 16384]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_581: "bf16[16384, 4096]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        permute_582: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_583: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
        permute_584: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        permute_585: "bf16[16, 256, 128]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        permute_586: "bf16[16, 128, 256]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_587: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_588: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        permute_589: "bf16[4096, 4096]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (embedding, getitem_1, rsqrt, view, unsqueeze_8, unsqueeze_10, div_1, view_18, addmm, view_21, add_7, getitem_5, rsqrt_1, view_23, unsqueeze_16, unsqueeze_18, div_3, view_41, addmm_2, view_44, mul_20, view_46, unsqueeze_24, unsqueeze_26, div_5, view_64, addmm_4, view_67, mul_30, view_69, unsqueeze_32, unsqueeze_34, div_7, view_87, addmm_6, view_90, mul_40, view_92, unsqueeze_40, unsqueeze_42, div_9, view_110, addmm_8, view_113, mul_50, view_115, unsqueeze_48, unsqueeze_50, div_11, view_133, addmm_10, view_136, mul_60, view_138, unsqueeze_56, unsqueeze_58, div_13, view_156, addmm_12, view_159, mul_70, view_161, unsqueeze_64, unsqueeze_66, div_15, view_179, addmm_14, view_182, mul_80, view_184, unsqueeze_72, unsqueeze_74, div_17, view_202, addmm_16, view_205, mul_90, view_207, unsqueeze_80, unsqueeze_82, div_19, view_225, addmm_18, view_228, mul_100, view_230, unsqueeze_88, unsqueeze_90, div_21, view_248, addmm_20, view_251, mul_110, view_253, unsqueeze_96, unsqueeze_98, div_23, view_271, addmm_22, view_274, mul_120, view_276, unsqueeze_104, unsqueeze_106, div_25, view_294, addmm_24, view_297, mul_130, view_299, unsqueeze_112, unsqueeze_114, div_27, view_317, addmm_26, view_320, mul_140, view_322, unsqueeze_120, unsqueeze_122, div_29, view_340, addmm_28, view_343, mul_150, view_345, unsqueeze_128, unsqueeze_130, div_31, view_363, addmm_30, view_366, mul_160, view_368, unsqueeze_136, unsqueeze_138, div_33, view_386, addmm_32, view_389, mul_170, view_391, unsqueeze_144, unsqueeze_146, div_35, view_409, addmm_34, view_412, mul_180, view_414, unsqueeze_152, unsqueeze_154, div_37, view_432, addmm_36, view_435, mul_190, view_437, unsqueeze_160, unsqueeze_162, div_39, view_455, addmm_38, view_458, mul_200, view_460, unsqueeze_168, unsqueeze_170, div_41, view_478, addmm_40, view_481, mul_210, view_483, unsqueeze_176, unsqueeze_178, div_43, view_501, addmm_42, view_504, mul_220, view_506, unsqueeze_184, unsqueeze_186, div_45, view_524, addmm_44, view_527, mul_230, view_529, unsqueeze_192, unsqueeze_194, div_47, view_547, addmm_46, view_550, mul_240, view_552, unsqueeze_200, unsqueeze_202, div_49, view_570, addmm_48, view_573, mul_250, view_575, unsqueeze_208, unsqueeze_210, div_51, view_593, addmm_50, view_596, mul_260, view_598, unsqueeze_216, unsqueeze_218, div_53, view_616, addmm_52, view_619, mul_270, view_621, unsqueeze_224, unsqueeze_226, div_55, view_639, addmm_54, view_642, mul_280, view_644, view_645, constant_pad_nd, amax_28, log, convert_element_type_452, div_56, permute_309, div_57, permute_310, permute_311, permute_312, permute_313, permute_314, permute_315, permute_316, permute_317, permute_318, permute_319, div_58, permute_320, permute_321, permute_322, permute_323, permute_324, permute_325, permute_326, permute_327, permute_328, permute_329, div_59, permute_330, permute_331, permute_332, permute_333, permute_334, permute_335, permute_336, permute_337, permute_338, permute_339, div_60, permute_340, permute_341, permute_342, permute_343, permute_344, permute_345, permute_346, permute_347, permute_348, permute_349, div_61, permute_350, permute_351, permute_352, permute_353, permute_354, permute_355, permute_356, permute_357, permute_358, permute_359, div_62, permute_360, permute_361, permute_362, permute_363, permute_364, permute_365, permute_366, permute_367, permute_368, permute_369, div_63, permute_370, permute_371, permute_372, permute_373, permute_374, permute_375, permute_376, permute_377, permute_378, permute_379, div_64, permute_380, permute_381, permute_382, permute_383, permute_384, permute_385, permute_386, permute_387, permute_388, permute_389, div_65, permute_390, permute_391, permute_392, permute_393, permute_394, permute_395, permute_396, permute_397, permute_398, permute_399, div_66, permute_400, permute_401, permute_402, permute_403, permute_404, permute_405, permute_406, permute_407, permute_408, permute_409, div_67, permute_410, permute_411, permute_412, permute_413, permute_414, permute_415, permute_416, permute_417, permute_418, permute_419, div_68, permute_420, permute_421, permute_422, permute_423, permute_424, permute_425, permute_426, permute_427, permute_428, permute_429, div_69, permute_430, permute_431, permute_432, permute_433, permute_434, permute_435, permute_436, permute_437, permute_438, permute_439, div_70, permute_440, permute_441, permute_442, permute_443, permute_444, permute_445, permute_446, permute_447, permute_448, permute_449, div_71, permute_450, permute_451, permute_452, permute_453, permute_454, permute_455, permute_456, permute_457, permute_458, permute_459, div_72, permute_460, permute_461, permute_462, permute_463, permute_464, permute_465, permute_466, permute_467, permute_468, permute_469, div_73, permute_470, permute_471, permute_472, permute_473, permute_474, permute_475, permute_476, permute_477, permute_478, permute_479, div_74, permute_480, permute_481, permute_482, permute_483, permute_484, permute_485, permute_486, permute_487, permute_488, permute_489, div_75, permute_490, permute_491, permute_492, permute_493, permute_494, permute_495, permute_496, permute_497, permute_498, permute_499, div_76, permute_500, permute_501, permute_502, permute_503, permute_504, permute_505, permute_506, permute_507, permute_508, permute_509, div_77, permute_510, permute_511, permute_512, permute_513, permute_514, permute_515, permute_516, permute_517, permute_518, permute_519, div_78, permute_520, permute_521, permute_522, permute_523, permute_524, permute_525, permute_526, permute_527, permute_528, permute_529, div_79, permute_530, permute_531, permute_532, permute_533, permute_534, permute_535, permute_536, permute_537, permute_538, permute_539, div_80, permute_540, permute_541, permute_542, permute_543, permute_544, permute_545, permute_546, permute_547, permute_548, permute_549, div_81, permute_550, permute_551, permute_552, permute_553, permute_554, permute_555, permute_556, permute_557, permute_558, permute_559, div_82, permute_560, permute_561, permute_562, permute_563, permute_564, permute_565, permute_566, permute_567, permute_568, permute_569, div_83, permute_570, permute_571, permute_572, permute_573, permute_574, permute_575, permute_576, permute_577, permute_578, permute_579, permute_580, permute_581, permute_582, permute_583, permute_584, permute_585, permute_586, permute_587, permute_588, permute_589)



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
