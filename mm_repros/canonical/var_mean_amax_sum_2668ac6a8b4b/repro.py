"""
Standalone repro captured via capture_hook.
Label: GPTJForQuestionAnswering
Pattern hash: 2668ac6a8b4b
Shape hash: 88cfdd18
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
    def forward(self, arg0_1: "bf16[50400, 4096]", arg1_1: "i64[1, 128]", arg2_1: "bf16[2048, 64]", arg3_1: "i64[1, 128]", arg4_1: "bf16[4096]", arg5_1: "bf16[4096]", arg6_1: "bf16[4096, 4096]", arg7_1: "bf16[4096, 4096]", arg8_1: "i64[1, 128]", arg9_1: "bf16[4096, 4096]", arg10_1: "bf16[4096, 4096]", arg11_1: "bf16[16384, 4096]", arg12_1: "bf16[16384]", arg13_1: "bf16[4096, 16384]", arg14_1: "bf16[4096]", arg15_1: "bf16[2048, 64]", arg16_1: "bf16[4096]", arg17_1: "bf16[4096]", arg18_1: "bf16[4096, 4096]", arg19_1: "bf16[4096, 4096]", arg20_1: "bf16[4096, 4096]", arg21_1: "bf16[4096, 4096]", arg22_1: "bf16[16384, 4096]", arg23_1: "bf16[16384]", arg24_1: "bf16[4096, 16384]", arg25_1: "bf16[4096]", arg26_1: "bf16[2048, 64]", arg27_1: "bf16[4096]", arg28_1: "bf16[4096]", arg29_1: "bf16[4096, 4096]", arg30_1: "bf16[4096, 4096]", arg31_1: "bf16[4096, 4096]", arg32_1: "bf16[4096, 4096]", arg33_1: "bf16[16384, 4096]", arg34_1: "bf16[16384]", arg35_1: "bf16[4096, 16384]", arg36_1: "bf16[4096]", arg37_1: "bf16[2048, 64]", arg38_1: "bf16[4096]", arg39_1: "bf16[4096]", arg40_1: "bf16[4096, 4096]", arg41_1: "bf16[4096, 4096]", arg42_1: "bf16[4096, 4096]", arg43_1: "bf16[4096, 4096]", arg44_1: "bf16[16384, 4096]", arg45_1: "bf16[16384]", arg46_1: "bf16[4096, 16384]", arg47_1: "bf16[4096]", arg48_1: "bf16[2048, 64]", arg49_1: "bf16[4096]", arg50_1: "bf16[4096]", arg51_1: "bf16[4096, 4096]", arg52_1: "bf16[4096, 4096]", arg53_1: "bf16[4096, 4096]", arg54_1: "bf16[4096, 4096]", arg55_1: "bf16[16384, 4096]", arg56_1: "bf16[16384]", arg57_1: "bf16[4096, 16384]", arg58_1: "bf16[4096]", arg59_1: "bf16[2048, 64]", arg60_1: "bf16[4096]", arg61_1: "bf16[4096]", arg62_1: "bf16[4096, 4096]", arg63_1: "bf16[4096, 4096]", arg64_1: "bf16[4096, 4096]", arg65_1: "bf16[4096, 4096]", arg66_1: "bf16[16384, 4096]", arg67_1: "bf16[16384]", arg68_1: "bf16[4096, 16384]", arg69_1: "bf16[4096]", arg70_1: "bf16[2048, 64]", arg71_1: "bf16[4096]", arg72_1: "bf16[4096]", arg73_1: "bf16[4096, 4096]", arg74_1: "bf16[4096, 4096]", arg75_1: "bf16[4096, 4096]", arg76_1: "bf16[4096, 4096]", arg77_1: "bf16[16384, 4096]", arg78_1: "bf16[16384]", arg79_1: "bf16[4096, 16384]", arg80_1: "bf16[4096]", arg81_1: "bf16[2048, 64]", arg82_1: "bf16[4096]", arg83_1: "bf16[4096]", arg84_1: "bf16[4096, 4096]", arg85_1: "bf16[4096, 4096]", arg86_1: "bf16[4096, 4096]", arg87_1: "bf16[4096, 4096]", arg88_1: "bf16[16384, 4096]", arg89_1: "bf16[16384]", arg90_1: "bf16[4096, 16384]", arg91_1: "bf16[4096]", arg92_1: "bf16[2048, 64]", arg93_1: "bf16[4096]", arg94_1: "bf16[4096]", arg95_1: "bf16[4096, 4096]", arg96_1: "bf16[4096, 4096]", arg97_1: "bf16[4096, 4096]", arg98_1: "bf16[4096, 4096]", arg99_1: "bf16[16384, 4096]", arg100_1: "bf16[16384]", arg101_1: "bf16[4096, 16384]", arg102_1: "bf16[4096]", arg103_1: "bf16[2048, 64]", arg104_1: "bf16[4096]", arg105_1: "bf16[4096]", arg106_1: "bf16[4096, 4096]", arg107_1: "bf16[4096, 4096]", arg108_1: "bf16[4096, 4096]", arg109_1: "bf16[4096, 4096]", arg110_1: "bf16[16384, 4096]", arg111_1: "bf16[16384]", arg112_1: "bf16[4096, 16384]", arg113_1: "bf16[4096]", arg114_1: "bf16[2048, 64]", arg115_1: "bf16[4096]", arg116_1: "bf16[4096]", arg117_1: "bf16[4096, 4096]", arg118_1: "bf16[4096, 4096]", arg119_1: "bf16[4096, 4096]", arg120_1: "bf16[4096, 4096]", arg121_1: "bf16[16384, 4096]", arg122_1: "bf16[16384]", arg123_1: "bf16[4096, 16384]", arg124_1: "bf16[4096]", arg125_1: "bf16[2048, 64]", arg126_1: "bf16[4096]", arg127_1: "bf16[4096]", arg128_1: "bf16[4096, 4096]", arg129_1: "bf16[4096, 4096]", arg130_1: "bf16[4096, 4096]", arg131_1: "bf16[4096, 4096]", arg132_1: "bf16[16384, 4096]", arg133_1: "bf16[16384]", arg134_1: "bf16[4096, 16384]", arg135_1: "bf16[4096]", arg136_1: "bf16[2048, 64]", arg137_1: "bf16[4096]", arg138_1: "bf16[4096]", arg139_1: "bf16[4096, 4096]", arg140_1: "bf16[4096, 4096]", arg141_1: "bf16[4096, 4096]", arg142_1: "bf16[4096, 4096]", arg143_1: "bf16[16384, 4096]", arg144_1: "bf16[16384]", arg145_1: "bf16[4096, 16384]", arg146_1: "bf16[4096]", arg147_1: "bf16[2048, 64]", arg148_1: "bf16[4096]", arg149_1: "bf16[4096]", arg150_1: "bf16[4096, 4096]", arg151_1: "bf16[4096, 4096]", arg152_1: "bf16[4096, 4096]", arg153_1: "bf16[4096, 4096]", arg154_1: "bf16[16384, 4096]", arg155_1: "bf16[16384]", arg156_1: "bf16[4096, 16384]", arg157_1: "bf16[4096]", arg158_1: "bf16[2048, 64]", arg159_1: "bf16[4096]", arg160_1: "bf16[4096]", arg161_1: "bf16[4096, 4096]", arg162_1: "bf16[4096, 4096]", arg163_1: "bf16[4096, 4096]", arg164_1: "bf16[4096, 4096]", arg165_1: "bf16[16384, 4096]", arg166_1: "bf16[16384]", arg167_1: "bf16[4096, 16384]", arg168_1: "bf16[4096]", arg169_1: "bf16[2048, 64]", arg170_1: "bf16[4096]", arg171_1: "bf16[4096]", arg172_1: "bf16[4096, 4096]", arg173_1: "bf16[4096, 4096]", arg174_1: "bf16[4096, 4096]", arg175_1: "bf16[4096, 4096]", arg176_1: "bf16[16384, 4096]", arg177_1: "bf16[16384]", arg178_1: "bf16[4096, 16384]", arg179_1: "bf16[4096]", arg180_1: "bf16[2048, 64]", arg181_1: "bf16[4096]", arg182_1: "bf16[4096]", arg183_1: "bf16[4096, 4096]", arg184_1: "bf16[4096, 4096]", arg185_1: "bf16[4096, 4096]", arg186_1: "bf16[4096, 4096]", arg187_1: "bf16[16384, 4096]", arg188_1: "bf16[16384]", arg189_1: "bf16[4096, 16384]", arg190_1: "bf16[4096]", arg191_1: "bf16[2048, 64]", arg192_1: "bf16[4096]", arg193_1: "bf16[4096]", arg194_1: "bf16[4096, 4096]", arg195_1: "bf16[4096, 4096]", arg196_1: "bf16[4096, 4096]", arg197_1: "bf16[4096, 4096]", arg198_1: "bf16[16384, 4096]", arg199_1: "bf16[16384]", arg200_1: "bf16[4096, 16384]", arg201_1: "bf16[4096]", arg202_1: "bf16[2048, 64]", arg203_1: "bf16[4096]", arg204_1: "bf16[4096]", arg205_1: "bf16[4096, 4096]", arg206_1: "bf16[4096, 4096]", arg207_1: "bf16[4096, 4096]", arg208_1: "bf16[4096, 4096]", arg209_1: "bf16[16384, 4096]", arg210_1: "bf16[16384]", arg211_1: "bf16[4096, 16384]", arg212_1: "bf16[4096]", arg213_1: "bf16[2048, 64]", arg214_1: "bf16[4096]", arg215_1: "bf16[4096]", arg216_1: "bf16[4096, 4096]", arg217_1: "bf16[4096, 4096]", arg218_1: "bf16[4096, 4096]", arg219_1: "bf16[4096, 4096]", arg220_1: "bf16[16384, 4096]", arg221_1: "bf16[16384]", arg222_1: "bf16[4096, 16384]", arg223_1: "bf16[4096]", arg224_1: "bf16[2048, 64]", arg225_1: "bf16[4096]", arg226_1: "bf16[4096]", arg227_1: "bf16[4096, 4096]", arg228_1: "bf16[4096, 4096]", arg229_1: "bf16[4096, 4096]", arg230_1: "bf16[4096, 4096]", arg231_1: "bf16[16384, 4096]", arg232_1: "bf16[16384]", arg233_1: "bf16[4096, 16384]", arg234_1: "bf16[4096]", arg235_1: "bf16[2048, 64]", arg236_1: "bf16[4096]", arg237_1: "bf16[4096]", arg238_1: "bf16[4096, 4096]", arg239_1: "bf16[4096, 4096]", arg240_1: "bf16[4096, 4096]", arg241_1: "bf16[4096, 4096]", arg242_1: "bf16[16384, 4096]", arg243_1: "bf16[16384]", arg244_1: "bf16[4096, 16384]", arg245_1: "bf16[4096]", arg246_1: "bf16[2048, 64]", arg247_1: "bf16[4096]", arg248_1: "bf16[4096]", arg249_1: "bf16[4096, 4096]", arg250_1: "bf16[4096, 4096]", arg251_1: "bf16[4096, 4096]", arg252_1: "bf16[4096, 4096]", arg253_1: "bf16[16384, 4096]", arg254_1: "bf16[16384]", arg255_1: "bf16[4096, 16384]", arg256_1: "bf16[4096]", arg257_1: "bf16[2048, 64]", arg258_1: "bf16[4096]", arg259_1: "bf16[4096]", arg260_1: "bf16[4096, 4096]", arg261_1: "bf16[4096, 4096]", arg262_1: "bf16[4096, 4096]", arg263_1: "bf16[4096, 4096]", arg264_1: "bf16[16384, 4096]", arg265_1: "bf16[16384]", arg266_1: "bf16[4096, 16384]", arg267_1: "bf16[4096]", arg268_1: "bf16[2048, 64]", arg269_1: "bf16[4096]", arg270_1: "bf16[4096]", arg271_1: "bf16[4096, 4096]", arg272_1: "bf16[4096, 4096]", arg273_1: "bf16[4096, 4096]", arg274_1: "bf16[4096, 4096]", arg275_1: "bf16[16384, 4096]", arg276_1: "bf16[16384]", arg277_1: "bf16[4096, 16384]", arg278_1: "bf16[4096]", arg279_1: "bf16[2048, 64]", arg280_1: "bf16[4096]", arg281_1: "bf16[4096]", arg282_1: "bf16[4096, 4096]", arg283_1: "bf16[4096, 4096]", arg284_1: "bf16[4096, 4096]", arg285_1: "bf16[4096, 4096]", arg286_1: "bf16[16384, 4096]", arg287_1: "bf16[16384]", arg288_1: "bf16[4096, 16384]", arg289_1: "bf16[4096]", arg290_1: "bf16[2048, 64]", arg291_1: "bf16[4096]", arg292_1: "bf16[4096]", arg293_1: "bf16[4096, 4096]", arg294_1: "bf16[4096, 4096]", arg295_1: "bf16[4096, 4096]", arg296_1: "bf16[4096, 4096]", arg297_1: "bf16[16384, 4096]", arg298_1: "bf16[16384]", arg299_1: "bf16[4096, 16384]", arg300_1: "bf16[4096]", arg301_1: "bf16[2048, 64]", arg302_1: "bf16[4096]", arg303_1: "bf16[4096]", arg304_1: "bf16[4096, 4096]", arg305_1: "bf16[4096, 4096]", arg306_1: "bf16[4096, 4096]", arg307_1: "bf16[4096, 4096]", arg308_1: "bf16[16384, 4096]", arg309_1: "bf16[16384]", arg310_1: "bf16[4096, 16384]", arg311_1: "bf16[4096]", arg312_1: "bf16[4096]", arg313_1: "bf16[4096]", arg314_1: "bf16[2, 4096]", arg315_1: "bf16[2]", arg316_1: "i64[1]", arg317_1: "i64[1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631, _shape_param_632, _shape_param_633, _shape_param_634, _shape_param_635, _shape_param_636, _shape_param_637, _shape_param_638, _shape_param_639, _shape_param_640, _shape_param_641, _shape_param_642, _shape_param_643, _shape_param_644, _shape_param_645, _shape_param_646, _shape_param_647, _shape_param_648, _shape_param_649, _shape_param_650, _shape_param_651, _shape_param_652, _shape_param_653, _shape_param_654, _shape_param_655, _shape_param_656, _shape_param_657, _shape_param_658, _shape_param_659, _shape_param_660, _shape_param_661, _shape_param_662, _shape_param_663, _shape_param_664, _shape_param_665, _shape_param_666, _shape_param_667, _shape_param_668, _shape_param_669, _shape_param_670, _shape_param_671, _shape_param_672, _shape_param_673, _shape_param_674, _shape_param_675, _shape_param_676, _shape_param_677, _shape_param_678, _shape_param_679, _shape_param_680, _shape_param_681, _shape_param_682, _shape_param_683, _shape_param_684, _shape_param_685, _shape_param_686, _shape_param_687, _shape_param_688, _shape_param_689, _shape_param_690, _shape_param_691, _shape_param_692, _shape_param_693, _shape_param_694, _shape_param_695, _shape_param_696, _shape_param_697, _shape_param_698, _shape_param_699, _shape_param_700, _shape_param_701, _shape_param_702, _shape_param_703, _shape_param_704, _shape_param_705, _shape_param_706, _shape_param_707, _shape_param_708, _shape_param_709, _shape_param_710, _shape_param_711, _shape_param_712, _shape_param_713, _shape_param_714, _shape_param_715, _shape_param_716, _shape_param_717, _shape_param_718, _shape_param_719, _shape_param_720, _shape_param_721, _shape_param_722, _shape_param_723, _shape_param_724, _shape_param_725, _shape_param_726, _shape_param_727, _shape_param_728, _shape_param_729, _shape_param_730, _shape_param_731, _shape_param_732, _shape_param_733, _shape_param_734, _shape_param_735, _shape_param_736, _shape_param_737, _shape_param_738, _shape_param_739, _shape_param_740, _shape_param_741, _shape_param_742, _shape_param_743, _shape_param_744, _shape_param_745, _shape_param_746, _shape_param_747, _shape_param_748, _shape_param_749, _shape_param_750, _shape_param_751, _shape_param_752, _shape_param_753, _shape_param_754, _shape_param_755, _shape_param_756, _shape_param_757, _shape_param_758, _shape_param_759, _shape_param_760, _shape_param_761, _shape_param_762, _shape_param_763, _shape_param_764, _shape_param_765, _shape_param_766, _shape_param_767, _shape_param_768, _shape_param_769, _shape_param_770, _shape_param_771, _shape_param_772, _shape_param_773, _shape_param_774, _shape_param_775, _shape_param_776, _shape_param_777, _shape_param_778, _shape_param_779, _shape_param_780, _shape_param_781, _shape_param_782, _shape_param_783, _shape_param_784, _shape_param_785, _shape_param_786, _shape_param_787, _shape_param_788, _shape_param_789, _shape_param_790, _shape_param_791, _shape_param_792, _shape_param_793, _shape_param_794, _shape_param_795, _shape_param_796, _shape_param_797, _shape_param_798, _shape_param_799, _shape_param_800, _shape_param_801, _shape_param_802, _shape_param_803, _shape_param_804, _shape_param_805, _shape_param_806, _shape_param_807, _shape_param_808, _shape_param_809, _shape_param_810, _shape_param_811, _shape_param_812, _shape_param_813, _shape_param_814, _shape_param_815, _shape_param_816, _shape_param_817, _shape_param_818, _shape_param_819, _shape_param_820, _shape_param_821, _shape_param_822, _shape_param_823, _shape_param_824, _shape_param_825, _shape_param_826, _shape_param_827, _shape_param_828, _shape_param_829, _shape_param_830, _shape_param_831, _shape_param_832, _shape_param_833, _shape_param_834, _shape_param_835, _shape_param_836, _shape_param_837, _shape_param_838, _shape_param_839, _shape_param_840, _shape_param_841, _shape_param_842, _shape_param_843, _shape_param_844, _shape_param_845, _shape_param_846, _shape_param_847, _shape_param_848, _shape_param_849, _shape_param_850, _shape_param_851, _shape_param_852, _shape_param_853, _shape_param_854, _shape_param_855, _shape_param_856, _shape_param_857, _shape_param_858, _shape_param_859, _shape_param_860, _shape_param_861, _shape_param_862, _shape_param_863, _shape_param_864, _shape_param_865, _shape_param_866, _shape_param_867, _shape_param_868, _shape_param_869, _shape_param_870, _shape_param_871, _shape_param_872, _shape_param_873, _shape_param_874, _shape_param_875, _shape_param_876, _shape_param_877, _shape_param_878, _shape_param_879, _shape_param_880, _shape_param_881, _shape_param_882, _shape_param_883, _shape_param_884, _shape_param_885, _shape_param_886, _shape_param_887, _shape_param_888, _shape_param_889, _shape_param_890, _shape_param_891, _shape_param_892, _shape_param_893, _shape_param_894, _shape_param_895, _shape_param_896, _shape_param_897, _shape_param_898, _shape_param_899, _shape_param_900, _shape_param_901, _shape_param_902, _shape_param_903, _shape_param_904, _shape_param_905, _shape_param_906, _shape_param_907, _shape_param_908, _shape_param_909, _shape_param_910, _shape_param_911, _shape_param_912, _shape_param_913, _shape_param_914, _shape_param_915, _shape_param_916, _shape_param_917, _shape_param_918, _shape_param_919, _shape_param_920, _shape_param_921, _shape_param_922, _shape_param_923, _shape_param_924, _shape_param_925, _shape_param_926, _shape_param_927, _shape_param_928, _shape_param_929, _shape_param_930, _shape_param_931, _shape_param_932, _shape_param_933, _shape_param_934, _shape_param_935, _shape_param_936, _shape_param_937, _shape_param_938, _shape_param_939, _shape_param_940, _shape_param_941, _shape_param_942, _shape_param_943, _shape_param_944, _shape_param_945, _shape_param_946, _shape_param_947, _shape_param_948, _shape_param_949, _shape_param_950, _shape_param_951, _shape_param_952, _shape_param_953, _shape_param_954, _shape_param_955, _shape_param_956, _shape_param_957, _shape_param_958, _shape_param_959, _shape_param_960, _shape_param_961, _shape_param_962, _shape_param_963, _shape_param_964, _shape_param_965, _shape_param_966, _shape_param_967, _shape_param_968, _shape_param_969, _shape_param_970, _shape_param_971, _shape_param_972, _shape_param_973, _shape_param_974, _shape_param_975, _shape_param_976, _shape_param_977, _shape_param_978, _shape_param_979, _shape_param_980, _shape_param_981, _shape_param_982, _shape_param_983, _shape_param_984, _shape_param_985, _shape_param_986, _shape_param_987, _shape_param_988, _shape_param_989, _shape_param_990, _shape_param_991, _shape_param_992, _shape_param_993, _shape_param_994, _shape_param_995, _shape_param_996, _shape_param_997, _shape_param_998, _shape_param_999, _shape_param_1000, _shape_param_1001, _shape_param_1002, _shape_param_1003, _shape_param_1004, _shape_param_1005, _shape_param_1006, _shape_param_1007, _shape_param_1008, _shape_param_1009, _shape_param_1010, _shape_param_1011, _shape_param_1012, _shape_param_1013, _shape_param_1014, _shape_param_1015, _shape_param_1016, _shape_param_1017, _shape_param_1018, _shape_param_1019, _shape_param_1020, _shape_param_1021, _shape_param_1022, _shape_param_1023, _shape_param_1024, _shape_param_1025, _shape_param_1026, _shape_param_1027, _shape_param_1028, _shape_param_1029, _shape_param_1030, _shape_param_1031, _shape_param_1032, _shape_param_1033, _shape_param_1034, _shape_param_1035, _shape_param_1036, _shape_param_1037, _shape_param_1038):
        # No stacktrace found for following nodes
        embedding: "bf16[1, 128, 4096]" = torch.ops.aten.embedding.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 128, 1]" = var_mean[0]
        getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
        repeat: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg2_1, [1, 1, 1]);  arg2_1 = None
        unsqueeze: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_1: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze, _shape_param_0);  unsqueeze = _shape_param_0 = None
        gather: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat, 1, repeat_1);  repeat = repeat_1 = None
        split = torch.ops.aten.split.Tensor(gather, 32, -1);  gather = None
        getitem_2: "bf16[1, 128, 32]" = split[0]
        getitem_3: "bf16[1, 128, 32]" = split[1];  split = None
        sub: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        convert_element_type_1: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  _shape_param_1 = None
        permute: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        view_2: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_1, _shape_param_3);  view_1 = _shape_param_3 = None
        slice_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_2, 3, 0, 64)
        unsqueeze_1: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_3, 2)
        unsqueeze_2: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 4);  unsqueeze_1 = None
        expand: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_2, _shape_param_4);  unsqueeze_2 = _shape_param_4 = None
        clone: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_3: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        mul_2: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1, view_3);  view_3 = None
        slice_2: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1, 3, 1, 9223372036854775807, 2)
        neg: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_2);  slice_2 = None
        unsqueeze_3: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg, 4);  neg = None
        slice_3: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 9223372036854775807, 2);  slice_1 = None
        unsqueeze_4: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_3, 4);  slice_3 = None
        cat: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_3, unsqueeze_4], -1);  unsqueeze_3 = unsqueeze_4 = None
        view_4: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat, _shape_param_6);  cat = _shape_param_6 = None
        unsqueeze_5: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_2, 2)
        unsqueeze_6: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 4);  unsqueeze_5 = None
        expand_1: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_6, _shape_param_7);  unsqueeze_6 = _shape_param_7 = None
        clone_1: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_5: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_8);  clone_1 = _shape_param_8 = None
        mul_3: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_4, view_5);  view_4 = view_5 = None
        add_2: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        slice_4: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_2, 3, 64, 9223372036854775807);  view_2 = None
        cat_1: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_2, slice_4], -1);  add_2 = slice_4 = None
        permute_1: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_1, [0, 2, 1, 3]);  cat_1 = None
        convert_element_type_2: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        expand_2: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_2, _shape_param_9);  convert_element_type_2 = _shape_param_9 = None
        view_6: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_2, _shape_param_10);  expand_2 = _shape_param_10 = None
        view_7: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_11);  _shape_param_11 = None
        permute_2: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm_1: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_7, permute_2);  view_7 = permute_2 = None
        view_8: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_1, _shape_param_12);  mm_1 = _shape_param_12 = None
        view_9: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_8, _shape_param_13);  view_8 = _shape_param_13 = None
        slice_5: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_9, 3, 0, 64)
        unsqueeze_7: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_3, 2);  getitem_3 = None
        unsqueeze_8: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 4);  unsqueeze_7 = None
        expand_3: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_8, _shape_param_14);  unsqueeze_8 = _shape_param_14 = None
        clone_2: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_10: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_15);  clone_2 = _shape_param_15 = None
        mul_4: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_5, view_10);  view_10 = None
        slice_6: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_5, 3, 1, 9223372036854775807, 2)
        neg_1: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_6);  slice_6 = None
        unsqueeze_9: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_1, 4);  neg_1 = None
        slice_7: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_5, 3, 0, 9223372036854775807, 2);  slice_5 = None
        unsqueeze_10: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_7, 4);  slice_7 = None
        cat_2: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_9, unsqueeze_10], -1);  unsqueeze_9 = unsqueeze_10 = None
        view_11: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_2, _shape_param_16);  cat_2 = _shape_param_16 = None
        unsqueeze_11: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_2, 2);  getitem_2 = None
        unsqueeze_12: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 4);  unsqueeze_11 = None
        expand_4: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_12, _shape_param_17);  unsqueeze_12 = _shape_param_17 = None
        clone_3: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_12: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_3, _shape_param_18);  clone_3 = _shape_param_18 = None
        mul_5: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_11, view_12);  view_11 = view_12 = None
        add_3: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        slice_8: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_9, 3, 64, 9223372036854775807);  view_9 = None
        cat_3: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_3, slice_8], -1);  add_3 = slice_8 = None
        permute_3: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_3, [0, 2, 1, 3]);  cat_3 = None
        convert_element_type_3: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_3, torch.float32);  permute_3 = None
        permute_4: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_3, [0, 1, 3, 2]);  convert_element_type_3 = None
        expand_5: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_4, _shape_param_19);  permute_4 = _shape_param_19 = None
        view_13: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_5, _shape_param_20);  expand_5 = _shape_param_20 = None
        bmm: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_6, view_13);  view_6 = view_13 = None
        view_14: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_21);  bmm = _shape_param_21 = None
        div: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_14, 16.0);  view_14 = None
        full: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_4: "i64[128]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze_13: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_4, 0);  add_4 = None
        unsqueeze_14: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 1);  unsqueeze_13 = None
        unsqueeze_15: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 2);  unsqueeze_14 = None
        iota_1: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_5: "i64[128]" = torch.ops.aten.add.Tensor(iota_1, 0);  iota_1 = None
        unsqueeze_16: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_5, 0);  add_5 = None
        unsqueeze_17: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 1);  unsqueeze_16 = None
        unsqueeze_18: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 3);  unsqueeze_17 = None
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_15, unsqueeze_18)
        bitwise_and: "b8[1, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(full, le);  full = le = None
        iota_2: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_19: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_2, 1);  iota_2 = None
        unsqueeze_20: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_19, 2);  unsqueeze_19 = None
        unsqueeze_21: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, 3);  unsqueeze_20 = None
        index: "i64[1, 1, 128, 1]" = torch.ops.aten.index.Tensor(arg8_1, [unsqueeze_21, unsqueeze_18]);  unsqueeze_18 = None
        index_1: "i64[1, 1, 1, 128]" = torch.ops.aten.index.Tensor(arg8_1, [unsqueeze_21, unsqueeze_15]);  arg8_1 = unsqueeze_21 = unsqueeze_15 = None
        eq: "b8[1, 1, 128, 128]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None
        bitwise_and_1: "b8[1, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None
        expand_6: "b8[1, 1, 128, 128]" = torch.ops.aten.expand.default(bitwise_and_1, _shape_param_22);  bitwise_and_1 = _shape_param_22 = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_2: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[1, 1, 128, 128]" = torch.ops.aten.where.self(expand_6, full_1, full_2);  expand_6 = full_1 = full_2 = None
        add_6: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div, where);  div = None
        amax: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_6, [-1], True)
        sub_1: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_6, amax);  add_6 = amax = None
        exp: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_4: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        expand_7: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_4, _shape_param_23);  convert_element_type_4 = _shape_param_23 = None
        view_15: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_7, _shape_param_24);  expand_7 = _shape_param_24 = None
        view_16: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_25);  _shape_param_25 = None
        permute_5: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_2: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_16, permute_5);  view_16 = permute_5 = None
        view_17: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_2, _shape_param_26);  mm_2 = _shape_param_26 = None
        view_18: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_17, _shape_param_27);  view_17 = _shape_param_27 = None
        permute_6: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        expand_8: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_6, _shape_param_28);  permute_6 = _shape_param_28 = None
        view_19: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_8, _shape_param_29);  expand_8 = _shape_param_29 = None
        bmm_1: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_15, view_19);  view_15 = view_19 = None
        view_20: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_1, _shape_param_30);  bmm_1 = _shape_param_30 = None
        permute_7: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
        clone_4: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_21: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_4, _shape_param_31);  clone_4 = _shape_param_31 = None
        view_22: "bf16[128, 4096]" = torch.ops.aten.view.default(view_21, _shape_param_32);  view_21 = _shape_param_32 = None
        permute_8: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm_3: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_22, permute_8);  view_22 = permute_8 = None
        view_23: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_3, _shape_param_33);  mm_3 = _shape_param_33 = None
        view_24: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_34);  convert_element_type_1 = _shape_param_34 = None
        permute_9: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg12_1, view_24, permute_9);  arg12_1 = view_24 = permute_9 = None
        view_25: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm, _shape_param_35);  addmm = _shape_param_35 = None
        mul_6: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_25, 0.5)
        pow_1: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_25, 3.0)
        mul_7: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_7: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_25, mul_7);  view_25 = mul_7 = None
        mul_8: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
        add_8: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_9: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_6, add_8);  mul_6 = add_8 = None
        view_26: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_9, _shape_param_36);  mul_9 = _shape_param_36 = None
        permute_10: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_1: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg14_1, view_26, permute_10);  arg14_1 = view_26 = permute_10 = None
        view_27: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_1, _shape_param_37);  addmm_1 = _shape_param_37 = None
        add_9: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_23, view_27);  view_23 = view_27 = None
        add_10: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_9, embedding);  add_9 = embedding = None
        convert_element_type_5: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_10, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_5, [2], correction = 0, keepdim = True)
        getitem_4: "f32[1, 128, 1]" = var_mean_1[0]
        getitem_5: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        repeat_2: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg15_1, [1, 1, 1]);  arg15_1 = None
        unsqueeze_22: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_3: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_22, _shape_param_38);  unsqueeze_22 = _shape_param_38 = None
        gather_1: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_2, 1, repeat_3);  repeat_2 = repeat_3 = None
        split_1 = torch.ops.aten.split.Tensor(gather_1, 32, -1);  gather_1 = None
        getitem_6: "bf16[1, 128, 32]" = split_1[0]
        getitem_7: "bf16[1, 128, 32]" = split_1[1];  split_1 = None
        sub_2: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_5, getitem_5);  convert_element_type_5 = getitem_5 = None
        add_11: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_10: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_11: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_10, arg16_1);  mul_10 = arg16_1 = None
        add_12: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_11, arg17_1);  mul_11 = arg17_1 = None
        convert_element_type_6: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16);  add_12 = None
        view_28: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_6, _shape_param_39);  _shape_param_39 = None
        permute_11: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        mm_4: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_28, permute_11);  view_28 = permute_11 = None
        view_29: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_4, _shape_param_40);  mm_4 = _shape_param_40 = None
        view_30: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_29, _shape_param_41);  view_29 = _shape_param_41 = None
        slice_9: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_30, 3, 0, 64)
        unsqueeze_23: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_7, 2)
        unsqueeze_24: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, 4);  unsqueeze_23 = None
        expand_9: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_24, _shape_param_42);  unsqueeze_24 = _shape_param_42 = None
        clone_5: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_31: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_5, _shape_param_43);  clone_5 = _shape_param_43 = None
        mul_12: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_9, view_31);  view_31 = None
        slice_10: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_9, 3, 1, 9223372036854775807, 2)
        neg_2: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_10);  slice_10 = None
        unsqueeze_25: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_2, 4);  neg_2 = None
        slice_11: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_9, 3, 0, 9223372036854775807, 2);  slice_9 = None
        unsqueeze_26: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_11, 4);  slice_11 = None
        cat_4: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_25, unsqueeze_26], -1);  unsqueeze_25 = unsqueeze_26 = None
        view_32: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_4, _shape_param_44);  cat_4 = _shape_param_44 = None
        unsqueeze_27: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_6, 2)
        unsqueeze_28: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_27, 4);  unsqueeze_27 = None
        expand_10: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_28, _shape_param_45);  unsqueeze_28 = _shape_param_45 = None
        clone_6: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_33: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_6, _shape_param_46);  clone_6 = _shape_param_46 = None
        mul_13: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_32, view_33);  view_32 = view_33 = None
        add_13: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
        slice_12: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_30, 3, 64, 9223372036854775807);  view_30 = None
        cat_5: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_13, slice_12], -1);  add_13 = slice_12 = None
        permute_12: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_5, [0, 2, 1, 3]);  cat_5 = None
        convert_element_type_7: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_12, torch.float32);  permute_12 = None
        expand_11: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_7, _shape_param_47);  convert_element_type_7 = _shape_param_47 = None
        view_34: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_11, _shape_param_48);  expand_11 = _shape_param_48 = None
        view_35: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_6, _shape_param_49);  _shape_param_49 = None
        permute_13: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        mm_5: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_35, permute_13);  view_35 = permute_13 = None
        view_36: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_5, _shape_param_50);  mm_5 = _shape_param_50 = None
        view_37: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_36, _shape_param_51);  view_36 = _shape_param_51 = None
        slice_13: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_37, 3, 0, 64)
        unsqueeze_29: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_7, 2);  getitem_7 = None
        unsqueeze_30: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_29, 4);  unsqueeze_29 = None
        expand_12: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_30, _shape_param_52);  unsqueeze_30 = _shape_param_52 = None
        clone_7: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_38: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_7, _shape_param_53);  clone_7 = _shape_param_53 = None
        mul_14: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_13, view_38);  view_38 = None
        slice_14: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_13, 3, 1, 9223372036854775807, 2)
        neg_3: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_14);  slice_14 = None
        unsqueeze_31: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_3, 4);  neg_3 = None
        slice_15: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_13, 3, 0, 9223372036854775807, 2);  slice_13 = None
        unsqueeze_32: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_15, 4);  slice_15 = None
        cat_6: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_31, unsqueeze_32], -1);  unsqueeze_31 = unsqueeze_32 = None
        view_39: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_6, _shape_param_54);  cat_6 = _shape_param_54 = None
        unsqueeze_33: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_6, 2);  getitem_6 = None
        unsqueeze_34: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_33, 4);  unsqueeze_33 = None
        expand_13: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_34, _shape_param_55);  unsqueeze_34 = _shape_param_55 = None
        clone_8: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_40: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_56);  clone_8 = _shape_param_56 = None
        mul_15: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_39, view_40);  view_39 = view_40 = None
        add_14: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None
        slice_16: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_37, 3, 64, 9223372036854775807);  view_37 = None
        cat_7: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_14, slice_16], -1);  add_14 = slice_16 = None
        permute_14: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_7, [0, 2, 1, 3]);  cat_7 = None
        convert_element_type_8: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_14, torch.float32);  permute_14 = None
        permute_15: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_8, [0, 1, 3, 2]);  convert_element_type_8 = None
        expand_14: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_15, _shape_param_57);  permute_15 = _shape_param_57 = None
        view_41: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_14, _shape_param_58);  expand_14 = _shape_param_58 = None
        bmm_2: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_34, view_41);  view_34 = view_41 = None
        view_42: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_59);  bmm_2 = _shape_param_59 = None
        div_2: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_42, 16.0);  view_42 = None
        add_15: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_2, where);  div_2 = None
        amax_1: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_15, [-1], True)
        sub_3: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_15, amax_1);  add_15 = amax_1 = None
        exp_1: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_2: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_9: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        expand_15: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_9, _shape_param_60);  convert_element_type_9 = _shape_param_60 = None
        view_43: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_15, _shape_param_61);  expand_15 = _shape_param_61 = None
        view_44: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_6, _shape_param_62);  _shape_param_62 = None
        permute_16: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
        mm_6: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_44, permute_16);  view_44 = permute_16 = None
        view_45: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_6, _shape_param_63);  mm_6 = _shape_param_63 = None
        view_46: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_45, _shape_param_64);  view_45 = _shape_param_64 = None
        permute_17: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        expand_16: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_17, _shape_param_65);  permute_17 = _shape_param_65 = None
        view_47: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_16, _shape_param_66);  expand_16 = _shape_param_66 = None
        bmm_3: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_43, view_47);  view_43 = view_47 = None
        view_48: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_3, _shape_param_67);  bmm_3 = _shape_param_67 = None
        permute_18: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        clone_9: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_49: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_9, _shape_param_68);  clone_9 = _shape_param_68 = None
        view_50: "bf16[128, 4096]" = torch.ops.aten.view.default(view_49, _shape_param_69);  view_49 = _shape_param_69 = None
        permute_19: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        mm_7: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_50, permute_19);  view_50 = permute_19 = None
        view_51: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_7, _shape_param_70);  mm_7 = _shape_param_70 = None
        view_52: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_6, _shape_param_71);  convert_element_type_6 = _shape_param_71 = None
        permute_20: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_2: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg23_1, view_52, permute_20);  arg23_1 = view_52 = permute_20 = None
        view_53: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_2, _shape_param_72);  addmm_2 = _shape_param_72 = None
        mul_16: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_53, 0.5)
        pow_2: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_53, 3.0)
        mul_17: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_16: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_53, mul_17);  view_53 = mul_17 = None
        mul_18: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_16, 0.7978845608028654);  add_16 = None
        tanh_1: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
        add_17: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_19: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_16, add_17);  mul_16 = add_17 = None
        view_54: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_19, _shape_param_73);  mul_19 = _shape_param_73 = None
        permute_21: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_3: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg25_1, view_54, permute_21);  arg25_1 = view_54 = permute_21 = None
        view_55: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_3, _shape_param_74);  addmm_3 = _shape_param_74 = None
        add_18: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_51, view_55);  view_51 = view_55 = None
        add_19: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_18, add_10);  add_18 = add_10 = None
        convert_element_type_10: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_19, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_10, [2], correction = 0, keepdim = True)
        getitem_8: "f32[1, 128, 1]" = var_mean_2[0]
        getitem_9: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        repeat_4: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg26_1, [1, 1, 1]);  arg26_1 = None
        unsqueeze_35: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_5: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_35, _shape_param_75);  unsqueeze_35 = _shape_param_75 = None
        gather_2: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_4, 1, repeat_5);  repeat_4 = repeat_5 = None
        split_2 = torch.ops.aten.split.Tensor(gather_2, 32, -1);  gather_2 = None
        getitem_10: "bf16[1, 128, 32]" = split_2[0]
        getitem_11: "bf16[1, 128, 32]" = split_2[1];  split_2 = None
        sub_4: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_10, getitem_9);  convert_element_type_10 = getitem_9 = None
        add_20: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_20: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
        mul_21: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_20, arg27_1);  mul_20 = arg27_1 = None
        add_21: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_21, arg28_1);  mul_21 = arg28_1 = None
        convert_element_type_11: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16);  add_21 = None
        view_56: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_11, _shape_param_76);  _shape_param_76 = None
        permute_22: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        mm_8: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_56, permute_22);  view_56 = permute_22 = None
        view_57: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_8, _shape_param_77);  mm_8 = _shape_param_77 = None
        view_58: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_57, _shape_param_78);  view_57 = _shape_param_78 = None
        slice_17: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_58, 3, 0, 64)
        unsqueeze_36: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_11, 2)
        unsqueeze_37: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 4);  unsqueeze_36 = None
        expand_17: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_37, _shape_param_79);  unsqueeze_37 = _shape_param_79 = None
        clone_10: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_59: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_10, _shape_param_80);  clone_10 = _shape_param_80 = None
        mul_22: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_17, view_59);  view_59 = None
        slice_18: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_17, 3, 1, 9223372036854775807, 2)
        neg_4: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_18);  slice_18 = None
        unsqueeze_38: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_4, 4);  neg_4 = None
        slice_19: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_17, 3, 0, 9223372036854775807, 2);  slice_17 = None
        unsqueeze_39: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_19, 4);  slice_19 = None
        cat_8: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_38, unsqueeze_39], -1);  unsqueeze_38 = unsqueeze_39 = None
        view_60: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_8, _shape_param_81);  cat_8 = _shape_param_81 = None
        unsqueeze_40: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_10, 2)
        unsqueeze_41: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 4);  unsqueeze_40 = None
        expand_18: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_41, _shape_param_82);  unsqueeze_41 = _shape_param_82 = None
        clone_11: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_61: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_11, _shape_param_83);  clone_11 = _shape_param_83 = None
        mul_23: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_60, view_61);  view_60 = view_61 = None
        add_22: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        slice_20: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_58, 3, 64, 9223372036854775807);  view_58 = None
        cat_9: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_22, slice_20], -1);  add_22 = slice_20 = None
        permute_23: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_9, [0, 2, 1, 3]);  cat_9 = None
        convert_element_type_12: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_23, torch.float32);  permute_23 = None
        expand_19: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_12, _shape_param_84);  convert_element_type_12 = _shape_param_84 = None
        view_62: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_19, _shape_param_85);  expand_19 = _shape_param_85 = None
        view_63: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_11, _shape_param_86);  _shape_param_86 = None
        permute_24: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        mm_9: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_63, permute_24);  view_63 = permute_24 = None
        view_64: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_9, _shape_param_87);  mm_9 = _shape_param_87 = None
        view_65: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_64, _shape_param_88);  view_64 = _shape_param_88 = None
        slice_21: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_65, 3, 0, 64)
        unsqueeze_42: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_11, 2);  getitem_11 = None
        unsqueeze_43: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 4);  unsqueeze_42 = None
        expand_20: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_43, _shape_param_89);  unsqueeze_43 = _shape_param_89 = None
        clone_12: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_66: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_90);  clone_12 = _shape_param_90 = None
        mul_24: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_21, view_66);  view_66 = None
        slice_22: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_21, 3, 1, 9223372036854775807, 2)
        neg_5: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_22);  slice_22 = None
        unsqueeze_44: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_5, 4);  neg_5 = None
        slice_23: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_21, 3, 0, 9223372036854775807, 2);  slice_21 = None
        unsqueeze_45: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_23, 4);  slice_23 = None
        cat_10: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_44, unsqueeze_45], -1);  unsqueeze_44 = unsqueeze_45 = None
        view_67: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_10, _shape_param_91);  cat_10 = _shape_param_91 = None
        unsqueeze_46: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_10, 2);  getitem_10 = None
        unsqueeze_47: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 4);  unsqueeze_46 = None
        expand_21: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_47, _shape_param_92);  unsqueeze_47 = _shape_param_92 = None
        clone_13: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_68: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_13, _shape_param_93);  clone_13 = _shape_param_93 = None
        mul_25: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_67, view_68);  view_67 = view_68 = None
        add_23: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_24, mul_25);  mul_24 = mul_25 = None
        slice_24: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_65, 3, 64, 9223372036854775807);  view_65 = None
        cat_11: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_23, slice_24], -1);  add_23 = slice_24 = None
        permute_25: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_11, [0, 2, 1, 3]);  cat_11 = None
        convert_element_type_13: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_25, torch.float32);  permute_25 = None
        permute_26: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_13, [0, 1, 3, 2]);  convert_element_type_13 = None
        expand_22: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_26, _shape_param_94);  permute_26 = _shape_param_94 = None
        view_69: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_22, _shape_param_95);  expand_22 = _shape_param_95 = None
        bmm_4: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_62, view_69);  view_62 = view_69 = None
        view_70: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_4, _shape_param_96);  bmm_4 = _shape_param_96 = None
        div_4: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_70, 16.0);  view_70 = None
        add_24: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_4, where);  div_4 = None
        amax_2: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_24, [-1], True)
        sub_5: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_24, amax_2);  add_24 = amax_2 = None
        exp_2: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_3: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_5: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_14: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        expand_23: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_14, _shape_param_97);  convert_element_type_14 = _shape_param_97 = None
        view_71: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_23, _shape_param_98);  expand_23 = _shape_param_98 = None
        view_72: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_11, _shape_param_99);  _shape_param_99 = None
        permute_27: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        mm_10: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_72, permute_27);  view_72 = permute_27 = None
        view_73: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_10, _shape_param_100);  mm_10 = _shape_param_100 = None
        view_74: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_73, _shape_param_101);  view_73 = _shape_param_101 = None
        permute_28: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        expand_24: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_28, _shape_param_102);  permute_28 = _shape_param_102 = None
        view_75: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_24, _shape_param_103);  expand_24 = _shape_param_103 = None
        bmm_5: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_71, view_75);  view_71 = view_75 = None
        view_76: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_5, _shape_param_104);  bmm_5 = _shape_param_104 = None
        permute_29: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        clone_14: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_77: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_14, _shape_param_105);  clone_14 = _shape_param_105 = None
        view_78: "bf16[128, 4096]" = torch.ops.aten.view.default(view_77, _shape_param_106);  view_77 = _shape_param_106 = None
        permute_30: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        mm_11: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_78, permute_30);  view_78 = permute_30 = None
        view_79: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_11, _shape_param_107);  mm_11 = _shape_param_107 = None
        view_80: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_11, _shape_param_108);  convert_element_type_11 = _shape_param_108 = None
        permute_31: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_4: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg34_1, view_80, permute_31);  arg34_1 = view_80 = permute_31 = None
        view_81: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_4, _shape_param_109);  addmm_4 = _shape_param_109 = None
        mul_26: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_81, 0.5)
        pow_3: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_81, 3.0)
        mul_27: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_25: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_81, mul_27);  view_81 = mul_27 = None
        mul_28: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_2: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_28);  mul_28 = None
        add_26: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_29: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_26, add_26);  mul_26 = add_26 = None
        view_82: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_29, _shape_param_110);  mul_29 = _shape_param_110 = None
        permute_32: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_5: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg36_1, view_82, permute_32);  arg36_1 = view_82 = permute_32 = None
        view_83: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_5, _shape_param_111);  addmm_5 = _shape_param_111 = None
        add_27: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_79, view_83);  view_79 = view_83 = None
        add_28: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_27, add_19);  add_27 = add_19 = None
        convert_element_type_15: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_28, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_15, [2], correction = 0, keepdim = True)
        getitem_12: "f32[1, 128, 1]" = var_mean_3[0]
        getitem_13: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        repeat_6: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg37_1, [1, 1, 1]);  arg37_1 = None
        unsqueeze_48: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_7: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_48, _shape_param_112);  unsqueeze_48 = _shape_param_112 = None
        gather_3: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_6, 1, repeat_7);  repeat_6 = repeat_7 = None
        split_3 = torch.ops.aten.split.Tensor(gather_3, 32, -1);  gather_3 = None
        getitem_14: "bf16[1, 128, 32]" = split_3[0]
        getitem_15: "bf16[1, 128, 32]" = split_3[1];  split_3 = None
        sub_6: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_15, getitem_13);  convert_element_type_15 = getitem_13 = None
        add_29: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        mul_30: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = rsqrt_3 = None
        mul_31: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_30, arg38_1);  mul_30 = arg38_1 = None
        add_30: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_31, arg39_1);  mul_31 = arg39_1 = None
        convert_element_type_16: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_30, torch.bfloat16);  add_30 = None
        view_84: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_16, _shape_param_113);  _shape_param_113 = None
        permute_33: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
        mm_12: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_84, permute_33);  view_84 = permute_33 = None
        view_85: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_12, _shape_param_114);  mm_12 = _shape_param_114 = None
        view_86: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_85, _shape_param_115);  view_85 = _shape_param_115 = None
        slice_25: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_86, 3, 0, 64)
        unsqueeze_49: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_15, 2)
        unsqueeze_50: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_49, 4);  unsqueeze_49 = None
        expand_25: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_50, _shape_param_116);  unsqueeze_50 = _shape_param_116 = None
        clone_15: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_87: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_15, _shape_param_117);  clone_15 = _shape_param_117 = None
        mul_32: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_25, view_87);  view_87 = None
        slice_26: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_25, 3, 1, 9223372036854775807, 2)
        neg_6: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_26);  slice_26 = None
        unsqueeze_51: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_6, 4);  neg_6 = None
        slice_27: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_25, 3, 0, 9223372036854775807, 2);  slice_25 = None
        unsqueeze_52: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_27, 4);  slice_27 = None
        cat_12: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_51, unsqueeze_52], -1);  unsqueeze_51 = unsqueeze_52 = None
        view_88: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_12, _shape_param_118);  cat_12 = _shape_param_118 = None
        unsqueeze_53: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_14, 2)
        unsqueeze_54: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_53, 4);  unsqueeze_53 = None
        expand_26: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_54, _shape_param_119);  unsqueeze_54 = _shape_param_119 = None
        clone_16: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_89: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_120);  clone_16 = _shape_param_120 = None
        mul_33: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_88, view_89);  view_88 = view_89 = None
        add_31: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        slice_28: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_86, 3, 64, 9223372036854775807);  view_86 = None
        cat_13: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_31, slice_28], -1);  add_31 = slice_28 = None
        permute_34: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_13, [0, 2, 1, 3]);  cat_13 = None
        convert_element_type_17: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_34, torch.float32);  permute_34 = None
        expand_27: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_17, _shape_param_121);  convert_element_type_17 = _shape_param_121 = None
        view_90: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_27, _shape_param_122);  expand_27 = _shape_param_122 = None
        view_91: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_16, _shape_param_123);  _shape_param_123 = None
        permute_35: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        mm_13: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_91, permute_35);  view_91 = permute_35 = None
        view_92: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_13, _shape_param_124);  mm_13 = _shape_param_124 = None
        view_93: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_92, _shape_param_125);  view_92 = _shape_param_125 = None
        slice_29: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_93, 3, 0, 64)
        unsqueeze_55: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_15, 2);  getitem_15 = None
        unsqueeze_56: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_55, 4);  unsqueeze_55 = None
        expand_28: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_56, _shape_param_126);  unsqueeze_56 = _shape_param_126 = None
        clone_17: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_94: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_17, _shape_param_127);  clone_17 = _shape_param_127 = None
        mul_34: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_29, view_94);  view_94 = None
        slice_30: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_29, 3, 1, 9223372036854775807, 2)
        neg_7: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_30);  slice_30 = None
        unsqueeze_57: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_7, 4);  neg_7 = None
        slice_31: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_29, 3, 0, 9223372036854775807, 2);  slice_29 = None
        unsqueeze_58: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_31, 4);  slice_31 = None
        cat_14: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_57, unsqueeze_58], -1);  unsqueeze_57 = unsqueeze_58 = None
        view_95: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_14, _shape_param_128);  cat_14 = _shape_param_128 = None
        unsqueeze_59: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_14, 2);  getitem_14 = None
        unsqueeze_60: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_59, 4);  unsqueeze_59 = None
        expand_29: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_60, _shape_param_129);  unsqueeze_60 = _shape_param_129 = None
        clone_18: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_96: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_18, _shape_param_130);  clone_18 = _shape_param_130 = None
        mul_35: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_95, view_96);  view_95 = view_96 = None
        add_32: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None
        slice_32: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_93, 3, 64, 9223372036854775807);  view_93 = None
        cat_15: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_32, slice_32], -1);  add_32 = slice_32 = None
        permute_36: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_15, [0, 2, 1, 3]);  cat_15 = None
        convert_element_type_18: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_36, torch.float32);  permute_36 = None
        permute_37: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_18, [0, 1, 3, 2]);  convert_element_type_18 = None
        expand_30: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_37, _shape_param_131);  permute_37 = _shape_param_131 = None
        view_97: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_30, _shape_param_132);  expand_30 = _shape_param_132 = None
        bmm_6: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_90, view_97);  view_90 = view_97 = None
        view_98: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_133);  bmm_6 = _shape_param_133 = None
        div_6: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_98, 16.0);  view_98 = None
        add_33: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_6, where);  div_6 = None
        amax_3: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_33, [-1], True)
        sub_7: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_33, amax_3);  add_33 = amax_3 = None
        exp_3: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_4: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_19: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        expand_31: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_19, _shape_param_134);  convert_element_type_19 = _shape_param_134 = None
        view_99: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_31, _shape_param_135);  expand_31 = _shape_param_135 = None
        view_100: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_16, _shape_param_136);  _shape_param_136 = None
        permute_38: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
        mm_14: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_100, permute_38);  view_100 = permute_38 = None
        view_101: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_14, _shape_param_137);  mm_14 = _shape_param_137 = None
        view_102: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_101, _shape_param_138);  view_101 = _shape_param_138 = None
        permute_39: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        expand_32: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_39, _shape_param_139);  permute_39 = _shape_param_139 = None
        view_103: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_32, _shape_param_140);  expand_32 = _shape_param_140 = None
        bmm_7: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_99, view_103);  view_99 = view_103 = None
        view_104: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_7, _shape_param_141);  bmm_7 = _shape_param_141 = None
        permute_40: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        clone_19: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_105: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_19, _shape_param_142);  clone_19 = _shape_param_142 = None
        view_106: "bf16[128, 4096]" = torch.ops.aten.view.default(view_105, _shape_param_143);  view_105 = _shape_param_143 = None
        permute_41: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        mm_15: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_106, permute_41);  view_106 = permute_41 = None
        view_107: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_15, _shape_param_144);  mm_15 = _shape_param_144 = None
        view_108: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_16, _shape_param_145);  convert_element_type_16 = _shape_param_145 = None
        permute_42: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
        addmm_6: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg45_1, view_108, permute_42);  arg45_1 = view_108 = permute_42 = None
        view_109: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_6, _shape_param_146);  addmm_6 = _shape_param_146 = None
        mul_36: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        pow_4: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 3.0)
        mul_37: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_34: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_109, mul_37);  view_109 = mul_37 = None
        mul_38: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_34, 0.7978845608028654);  add_34 = None
        tanh_3: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_35: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_39: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_36, add_35);  mul_36 = add_35 = None
        view_110: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_39, _shape_param_147);  mul_39 = _shape_param_147 = None
        permute_43: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
        addmm_7: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg47_1, view_110, permute_43);  arg47_1 = view_110 = permute_43 = None
        view_111: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_7, _shape_param_148);  addmm_7 = _shape_param_148 = None
        add_36: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_107, view_111);  view_107 = view_111 = None
        add_37: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_36, add_28);  add_36 = add_28 = None
        convert_element_type_20: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_37, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_20, [2], correction = 0, keepdim = True)
        getitem_16: "f32[1, 128, 1]" = var_mean_4[0]
        getitem_17: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        repeat_8: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg48_1, [1, 1, 1]);  arg48_1 = None
        unsqueeze_61: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_9: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_61, _shape_param_149);  unsqueeze_61 = _shape_param_149 = None
        gather_4: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_8, 1, repeat_9);  repeat_8 = repeat_9 = None
        split_4 = torch.ops.aten.split.Tensor(gather_4, 32, -1);  gather_4 = None
        getitem_18: "bf16[1, 128, 32]" = split_4[0]
        getitem_19: "bf16[1, 128, 32]" = split_4[1];  split_4 = None
        sub_8: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_20, getitem_17);  convert_element_type_20 = getitem_17 = None
        add_38: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_40: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_4);  sub_8 = rsqrt_4 = None
        mul_41: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_40, arg49_1);  mul_40 = arg49_1 = None
        add_39: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_41, arg50_1);  mul_41 = arg50_1 = None
        convert_element_type_21: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None
        view_112: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_21, _shape_param_150);  _shape_param_150 = None
        permute_44: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        mm_16: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_112, permute_44);  view_112 = permute_44 = None
        view_113: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_16, _shape_param_151);  mm_16 = _shape_param_151 = None
        view_114: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_113, _shape_param_152);  view_113 = _shape_param_152 = None
        slice_33: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_114, 3, 0, 64)
        unsqueeze_62: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_19, 2)
        unsqueeze_63: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, 4);  unsqueeze_62 = None
        expand_33: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_63, _shape_param_153);  unsqueeze_63 = _shape_param_153 = None
        clone_20: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_115: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_154);  clone_20 = _shape_param_154 = None
        mul_42: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_33, view_115);  view_115 = None
        slice_34: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_33, 3, 1, 9223372036854775807, 2)
        neg_8: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_34);  slice_34 = None
        unsqueeze_64: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_8, 4);  neg_8 = None
        slice_35: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_33, 3, 0, 9223372036854775807, 2);  slice_33 = None
        unsqueeze_65: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_35, 4);  slice_35 = None
        cat_16: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_64, unsqueeze_65], -1);  unsqueeze_64 = unsqueeze_65 = None
        view_116: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_16, _shape_param_155);  cat_16 = _shape_param_155 = None
        unsqueeze_66: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_18, 2)
        unsqueeze_67: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, 4);  unsqueeze_66 = None
        expand_34: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_67, _shape_param_156);  unsqueeze_67 = _shape_param_156 = None
        clone_21: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_117: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_21, _shape_param_157);  clone_21 = _shape_param_157 = None
        mul_43: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_116, view_117);  view_116 = view_117 = None
        add_40: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_42, mul_43);  mul_42 = mul_43 = None
        slice_36: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_114, 3, 64, 9223372036854775807);  view_114 = None
        cat_17: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_40, slice_36], -1);  add_40 = slice_36 = None
        permute_45: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_17, [0, 2, 1, 3]);  cat_17 = None
        convert_element_type_22: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_45, torch.float32);  permute_45 = None
        expand_35: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_22, _shape_param_158);  convert_element_type_22 = _shape_param_158 = None
        view_118: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_35, _shape_param_159);  expand_35 = _shape_param_159 = None
        view_119: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_21, _shape_param_160);  _shape_param_160 = None
        permute_46: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        mm_17: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_119, permute_46);  view_119 = permute_46 = None
        view_120: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_17, _shape_param_161);  mm_17 = _shape_param_161 = None
        view_121: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_120, _shape_param_162);  view_120 = _shape_param_162 = None
        slice_37: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_121, 3, 0, 64)
        unsqueeze_68: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_19, 2);  getitem_19 = None
        unsqueeze_69: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, 4);  unsqueeze_68 = None
        expand_36: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_69, _shape_param_163);  unsqueeze_69 = _shape_param_163 = None
        clone_22: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_122: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_22, _shape_param_164);  clone_22 = _shape_param_164 = None
        mul_44: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_37, view_122);  view_122 = None
        slice_38: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_37, 3, 1, 9223372036854775807, 2)
        neg_9: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_38);  slice_38 = None
        unsqueeze_70: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_9, 4);  neg_9 = None
        slice_39: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_37, 3, 0, 9223372036854775807, 2);  slice_37 = None
        unsqueeze_71: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_39, 4);  slice_39 = None
        cat_18: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_70, unsqueeze_71], -1);  unsqueeze_70 = unsqueeze_71 = None
        view_123: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_18, _shape_param_165);  cat_18 = _shape_param_165 = None
        unsqueeze_72: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_18, 2);  getitem_18 = None
        unsqueeze_73: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, 4);  unsqueeze_72 = None
        expand_37: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_73, _shape_param_166);  unsqueeze_73 = _shape_param_166 = None
        clone_23: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_124: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_23, _shape_param_167);  clone_23 = _shape_param_167 = None
        mul_45: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_123, view_124);  view_123 = view_124 = None
        add_41: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
        slice_40: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_121, 3, 64, 9223372036854775807);  view_121 = None
        cat_19: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_41, slice_40], -1);  add_41 = slice_40 = None
        permute_47: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_19, [0, 2, 1, 3]);  cat_19 = None
        convert_element_type_23: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_47, torch.float32);  permute_47 = None
        permute_48: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_23, [0, 1, 3, 2]);  convert_element_type_23 = None
        expand_38: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_48, _shape_param_168);  permute_48 = _shape_param_168 = None
        view_125: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_38, _shape_param_169);  expand_38 = _shape_param_169 = None
        bmm_8: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_118, view_125);  view_118 = view_125 = None
        view_126: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_8, _shape_param_170);  bmm_8 = _shape_param_170 = None
        div_8: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_126, 16.0);  view_126 = None
        add_42: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_8, where);  div_8 = None
        amax_4: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_42, [-1], True)
        sub_9: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_42, amax_4);  add_42 = amax_4 = None
        exp_4: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_5: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_9: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_24: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        expand_39: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_24, _shape_param_171);  convert_element_type_24 = _shape_param_171 = None
        view_127: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_39, _shape_param_172);  expand_39 = _shape_param_172 = None
        view_128: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_21, _shape_param_173);  _shape_param_173 = None
        permute_49: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        mm_18: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_128, permute_49);  view_128 = permute_49 = None
        view_129: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_18, _shape_param_174);  mm_18 = _shape_param_174 = None
        view_130: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_129, _shape_param_175);  view_129 = _shape_param_175 = None
        permute_50: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        expand_40: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_50, _shape_param_176);  permute_50 = _shape_param_176 = None
        view_131: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_40, _shape_param_177);  expand_40 = _shape_param_177 = None
        bmm_9: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_127, view_131);  view_127 = view_131 = None
        view_132: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_9, _shape_param_178);  bmm_9 = _shape_param_178 = None
        permute_51: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
        clone_24: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_133: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_24, _shape_param_179);  clone_24 = _shape_param_179 = None
        view_134: "bf16[128, 4096]" = torch.ops.aten.view.default(view_133, _shape_param_180);  view_133 = _shape_param_180 = None
        permute_52: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        mm_19: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_134, permute_52);  view_134 = permute_52 = None
        view_135: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_19, _shape_param_181);  mm_19 = _shape_param_181 = None
        view_136: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_21, _shape_param_182);  convert_element_type_21 = _shape_param_182 = None
        permute_53: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_8: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg56_1, view_136, permute_53);  arg56_1 = view_136 = permute_53 = None
        view_137: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_8, _shape_param_183);  addmm_8 = _shape_param_183 = None
        mul_46: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_137, 0.5)
        pow_5: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_137, 3.0)
        mul_47: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_43: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_137, mul_47);  view_137 = mul_47 = None
        mul_48: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_43, 0.7978845608028654);  add_43 = None
        tanh_4: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
        add_44: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_49: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_46, add_44);  mul_46 = add_44 = None
        view_138: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_49, _shape_param_184);  mul_49 = _shape_param_184 = None
        permute_54: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_9: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg58_1, view_138, permute_54);  arg58_1 = view_138 = permute_54 = None
        view_139: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_9, _shape_param_185);  addmm_9 = _shape_param_185 = None
        add_45: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_135, view_139);  view_135 = view_139 = None
        add_46: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_45, add_37);  add_45 = add_37 = None
        convert_element_type_25: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_46, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_25, [2], correction = 0, keepdim = True)
        getitem_20: "f32[1, 128, 1]" = var_mean_5[0]
        getitem_21: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        repeat_10: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg59_1, [1, 1, 1]);  arg59_1 = None
        unsqueeze_74: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_11: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_74, _shape_param_186);  unsqueeze_74 = _shape_param_186 = None
        gather_5: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_10, 1, repeat_11);  repeat_10 = repeat_11 = None
        split_5 = torch.ops.aten.split.Tensor(gather_5, 32, -1);  gather_5 = None
        getitem_22: "bf16[1, 128, 32]" = split_5[0]
        getitem_23: "bf16[1, 128, 32]" = split_5[1];  split_5 = None
        sub_10: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_25, getitem_21);  convert_element_type_25 = getitem_21 = None
        add_47: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        mul_50: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_5);  sub_10 = rsqrt_5 = None
        mul_51: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_50, arg60_1);  mul_50 = arg60_1 = None
        add_48: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_51, arg61_1);  mul_51 = arg61_1 = None
        convert_element_type_26: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_48, torch.bfloat16);  add_48 = None
        view_140: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_26, _shape_param_187);  _shape_param_187 = None
        permute_55: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        mm_20: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_140, permute_55);  view_140 = permute_55 = None
        view_141: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_20, _shape_param_188);  mm_20 = _shape_param_188 = None
        view_142: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_141, _shape_param_189);  view_141 = _shape_param_189 = None
        slice_41: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_142, 3, 0, 64)
        unsqueeze_75: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_23, 2)
        unsqueeze_76: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_75, 4);  unsqueeze_75 = None
        expand_41: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_76, _shape_param_190);  unsqueeze_76 = _shape_param_190 = None
        clone_25: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_143: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_25, _shape_param_191);  clone_25 = _shape_param_191 = None
        mul_52: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_41, view_143);  view_143 = None
        slice_42: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_41, 3, 1, 9223372036854775807, 2)
        neg_10: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_42);  slice_42 = None
        unsqueeze_77: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_10, 4);  neg_10 = None
        slice_43: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_41, 3, 0, 9223372036854775807, 2);  slice_41 = None
        unsqueeze_78: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_43, 4);  slice_43 = None
        cat_20: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_77, unsqueeze_78], -1);  unsqueeze_77 = unsqueeze_78 = None
        view_144: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_20, _shape_param_192);  cat_20 = _shape_param_192 = None
        unsqueeze_79: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_22, 2)
        unsqueeze_80: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_79, 4);  unsqueeze_79 = None
        expand_42: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_80, _shape_param_193);  unsqueeze_80 = _shape_param_193 = None
        clone_26: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_145: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_26, _shape_param_194);  clone_26 = _shape_param_194 = None
        mul_53: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_144, view_145);  view_144 = view_145 = None
        add_49: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
        slice_44: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_142, 3, 64, 9223372036854775807);  view_142 = None
        cat_21: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_49, slice_44], -1);  add_49 = slice_44 = None
        permute_56: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_21, [0, 2, 1, 3]);  cat_21 = None
        convert_element_type_27: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_56, torch.float32);  permute_56 = None
        expand_43: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_27, _shape_param_195);  convert_element_type_27 = _shape_param_195 = None
        view_146: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_43, _shape_param_196);  expand_43 = _shape_param_196 = None
        view_147: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_26, _shape_param_197);  _shape_param_197 = None
        permute_57: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        mm_21: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_147, permute_57);  view_147 = permute_57 = None
        view_148: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_21, _shape_param_198);  mm_21 = _shape_param_198 = None
        view_149: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_148, _shape_param_199);  view_148 = _shape_param_199 = None
        slice_45: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_149, 3, 0, 64)
        unsqueeze_81: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_23, 2);  getitem_23 = None
        unsqueeze_82: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_81, 4);  unsqueeze_81 = None
        expand_44: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_82, _shape_param_200);  unsqueeze_82 = _shape_param_200 = None
        clone_27: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_150: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_27, _shape_param_201);  clone_27 = _shape_param_201 = None
        mul_54: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_45, view_150);  view_150 = None
        slice_46: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_45, 3, 1, 9223372036854775807, 2)
        neg_11: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_46);  slice_46 = None
        unsqueeze_83: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_11, 4);  neg_11 = None
        slice_47: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_45, 3, 0, 9223372036854775807, 2);  slice_45 = None
        unsqueeze_84: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_47, 4);  slice_47 = None
        cat_22: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_83, unsqueeze_84], -1);  unsqueeze_83 = unsqueeze_84 = None
        view_151: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_22, _shape_param_202);  cat_22 = _shape_param_202 = None
        unsqueeze_85: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_22, 2);  getitem_22 = None
        unsqueeze_86: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 4);  unsqueeze_85 = None
        expand_45: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_86, _shape_param_203);  unsqueeze_86 = _shape_param_203 = None
        clone_28: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_152: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_204);  clone_28 = _shape_param_204 = None
        mul_55: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_151, view_152);  view_151 = view_152 = None
        add_50: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_54, mul_55);  mul_54 = mul_55 = None
        slice_48: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_149, 3, 64, 9223372036854775807);  view_149 = None
        cat_23: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_50, slice_48], -1);  add_50 = slice_48 = None
        permute_58: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_23, [0, 2, 1, 3]);  cat_23 = None
        convert_element_type_28: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_58, torch.float32);  permute_58 = None
        permute_59: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_28, [0, 1, 3, 2]);  convert_element_type_28 = None
        expand_46: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_59, _shape_param_205);  permute_59 = _shape_param_205 = None
        view_153: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_46, _shape_param_206);  expand_46 = _shape_param_206 = None
        bmm_10: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_146, view_153);  view_146 = view_153 = None
        view_154: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_207);  bmm_10 = _shape_param_207 = None
        div_10: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_154, 16.0);  view_154 = None
        add_51: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_10, where);  div_10 = None
        amax_5: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_51, [-1], True)
        sub_11: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_51, amax_5);  add_51 = amax_5 = None
        exp_5: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_6: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_11: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_29: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        expand_47: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_29, _shape_param_208);  convert_element_type_29 = _shape_param_208 = None
        view_155: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_47, _shape_param_209);  expand_47 = _shape_param_209 = None
        view_156: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_26, _shape_param_210);  _shape_param_210 = None
        permute_60: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        mm_22: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_156, permute_60);  view_156 = permute_60 = None
        view_157: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_22, _shape_param_211);  mm_22 = _shape_param_211 = None
        view_158: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_157, _shape_param_212);  view_157 = _shape_param_212 = None
        permute_61: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
        expand_48: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_61, _shape_param_213);  permute_61 = _shape_param_213 = None
        view_159: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_48, _shape_param_214);  expand_48 = _shape_param_214 = None
        bmm_11: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_155, view_159);  view_155 = view_159 = None
        view_160: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_11, _shape_param_215);  bmm_11 = _shape_param_215 = None
        permute_62: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
        clone_29: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_161: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_29, _shape_param_216);  clone_29 = _shape_param_216 = None
        view_162: "bf16[128, 4096]" = torch.ops.aten.view.default(view_161, _shape_param_217);  view_161 = _shape_param_217 = None
        permute_63: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        mm_23: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_162, permute_63);  view_162 = permute_63 = None
        view_163: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_23, _shape_param_218);  mm_23 = _shape_param_218 = None
        view_164: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_26, _shape_param_219);  convert_element_type_26 = _shape_param_219 = None
        permute_64: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        addmm_10: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg67_1, view_164, permute_64);  arg67_1 = view_164 = permute_64 = None
        view_165: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_10, _shape_param_220);  addmm_10 = _shape_param_220 = None
        mul_56: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_165, 0.5)
        pow_6: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_165, 3.0)
        mul_57: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_52: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_165, mul_57);  view_165 = mul_57 = None
        mul_58: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_52, 0.7978845608028654);  add_52 = None
        tanh_5: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_53: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_59: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_56, add_53);  mul_56 = add_53 = None
        view_166: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_59, _shape_param_221);  mul_59 = _shape_param_221 = None
        permute_65: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
        addmm_11: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg69_1, view_166, permute_65);  arg69_1 = view_166 = permute_65 = None
        view_167: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_11, _shape_param_222);  addmm_11 = _shape_param_222 = None
        add_54: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_163, view_167);  view_163 = view_167 = None
        add_55: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_54, add_46);  add_54 = add_46 = None
        convert_element_type_30: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_55, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_30, [2], correction = 0, keepdim = True)
        getitem_24: "f32[1, 128, 1]" = var_mean_6[0]
        getitem_25: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        repeat_12: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg70_1, [1, 1, 1]);  arg70_1 = None
        unsqueeze_87: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_13: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_87, _shape_param_223);  unsqueeze_87 = _shape_param_223 = None
        gather_6: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_12, 1, repeat_13);  repeat_12 = repeat_13 = None
        split_6 = torch.ops.aten.split.Tensor(gather_6, 32, -1);  gather_6 = None
        getitem_26: "bf16[1, 128, 32]" = split_6[0]
        getitem_27: "bf16[1, 128, 32]" = split_6[1];  split_6 = None
        sub_12: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_30, getitem_25);  convert_element_type_30 = getitem_25 = None
        add_56: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_60: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_6);  sub_12 = rsqrt_6 = None
        mul_61: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_60, arg71_1);  mul_60 = arg71_1 = None
        add_57: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_61, arg72_1);  mul_61 = arg72_1 = None
        convert_element_type_31: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16);  add_57 = None
        view_168: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_31, _shape_param_224);  _shape_param_224 = None
        permute_66: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        mm_24: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_168, permute_66);  view_168 = permute_66 = None
        view_169: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_24, _shape_param_225);  mm_24 = _shape_param_225 = None
        view_170: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_169, _shape_param_226);  view_169 = _shape_param_226 = None
        slice_49: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_170, 3, 0, 64)
        unsqueeze_88: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_27, 2)
        unsqueeze_89: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, 4);  unsqueeze_88 = None
        expand_49: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_89, _shape_param_227);  unsqueeze_89 = _shape_param_227 = None
        clone_30: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_171: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_30, _shape_param_228);  clone_30 = _shape_param_228 = None
        mul_62: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_49, view_171);  view_171 = None
        slice_50: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_49, 3, 1, 9223372036854775807, 2)
        neg_12: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_50);  slice_50 = None
        unsqueeze_90: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_12, 4);  neg_12 = None
        slice_51: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_49, 3, 0, 9223372036854775807, 2);  slice_49 = None
        unsqueeze_91: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_51, 4);  slice_51 = None
        cat_24: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_90, unsqueeze_91], -1);  unsqueeze_90 = unsqueeze_91 = None
        view_172: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_24, _shape_param_229);  cat_24 = _shape_param_229 = None
        unsqueeze_92: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_26, 2)
        unsqueeze_93: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, 4);  unsqueeze_92 = None
        expand_50: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_93, _shape_param_230);  unsqueeze_93 = _shape_param_230 = None
        clone_31: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_173: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_31, _shape_param_231);  clone_31 = _shape_param_231 = None
        mul_63: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_172, view_173);  view_172 = view_173 = None
        add_58: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
        slice_52: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_170, 3, 64, 9223372036854775807);  view_170 = None
        cat_25: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_58, slice_52], -1);  add_58 = slice_52 = None
        permute_67: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_25, [0, 2, 1, 3]);  cat_25 = None
        convert_element_type_32: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_67, torch.float32);  permute_67 = None
        expand_51: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_32, _shape_param_232);  convert_element_type_32 = _shape_param_232 = None
        view_174: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_51, _shape_param_233);  expand_51 = _shape_param_233 = None
        view_175: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_31, _shape_param_234);  _shape_param_234 = None
        permute_68: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
        mm_25: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_175, permute_68);  view_175 = permute_68 = None
        view_176: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_25, _shape_param_235);  mm_25 = _shape_param_235 = None
        view_177: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_176, _shape_param_236);  view_176 = _shape_param_236 = None
        slice_53: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_177, 3, 0, 64)
        unsqueeze_94: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_27, 2);  getitem_27 = None
        unsqueeze_95: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, 4);  unsqueeze_94 = None
        expand_52: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_95, _shape_param_237);  unsqueeze_95 = _shape_param_237 = None
        clone_32: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_178: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_32, _shape_param_238);  clone_32 = _shape_param_238 = None
        mul_64: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_53, view_178);  view_178 = None
        slice_54: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_53, 3, 1, 9223372036854775807, 2)
        neg_13: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_54);  slice_54 = None
        unsqueeze_96: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_13, 4);  neg_13 = None
        slice_55: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_53, 3, 0, 9223372036854775807, 2);  slice_53 = None
        unsqueeze_97: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_55, 4);  slice_55 = None
        cat_26: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_96, unsqueeze_97], -1);  unsqueeze_96 = unsqueeze_97 = None
        view_179: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_26, _shape_param_239);  cat_26 = _shape_param_239 = None
        unsqueeze_98: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_26, 2);  getitem_26 = None
        unsqueeze_99: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, 4);  unsqueeze_98 = None
        expand_53: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_99, _shape_param_240);  unsqueeze_99 = _shape_param_240 = None
        clone_33: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_180: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_33, _shape_param_241);  clone_33 = _shape_param_241 = None
        mul_65: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_179, view_180);  view_179 = view_180 = None
        add_59: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        slice_56: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_177, 3, 64, 9223372036854775807);  view_177 = None
        cat_27: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_59, slice_56], -1);  add_59 = slice_56 = None
        permute_69: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_27, [0, 2, 1, 3]);  cat_27 = None
        convert_element_type_33: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_69, torch.float32);  permute_69 = None
        permute_70: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_33, [0, 1, 3, 2]);  convert_element_type_33 = None
        expand_54: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_70, _shape_param_242);  permute_70 = _shape_param_242 = None
        view_181: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_54, _shape_param_243);  expand_54 = _shape_param_243 = None
        bmm_12: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_174, view_181);  view_174 = view_181 = None
        view_182: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_12, _shape_param_244);  bmm_12 = _shape_param_244 = None
        div_12: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_182, 16.0);  view_182 = None
        add_60: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_12, where);  div_12 = None
        amax_6: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_60, [-1], True)
        sub_13: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_60, amax_6);  add_60 = amax_6 = None
        exp_6: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_7: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_13: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_34: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None
        expand_55: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_34, _shape_param_245);  convert_element_type_34 = _shape_param_245 = None
        view_183: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_55, _shape_param_246);  expand_55 = _shape_param_246 = None
        view_184: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_31, _shape_param_247);  _shape_param_247 = None
        permute_71: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        mm_26: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_184, permute_71);  view_184 = permute_71 = None
        view_185: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_26, _shape_param_248);  mm_26 = _shape_param_248 = None
        view_186: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_185, _shape_param_249);  view_185 = _shape_param_249 = None
        permute_72: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        expand_56: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_72, _shape_param_250);  permute_72 = _shape_param_250 = None
        view_187: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_56, _shape_param_251);  expand_56 = _shape_param_251 = None
        bmm_13: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_183, view_187);  view_183 = view_187 = None
        view_188: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_13, _shape_param_252);  bmm_13 = _shape_param_252 = None
        permute_73: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
        clone_34: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_189: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_34, _shape_param_253);  clone_34 = _shape_param_253 = None
        view_190: "bf16[128, 4096]" = torch.ops.aten.view.default(view_189, _shape_param_254);  view_189 = _shape_param_254 = None
        permute_74: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
        mm_27: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_190, permute_74);  view_190 = permute_74 = None
        view_191: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_27, _shape_param_255);  mm_27 = _shape_param_255 = None
        view_192: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_31, _shape_param_256);  convert_element_type_31 = _shape_param_256 = None
        permute_75: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_12: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg78_1, view_192, permute_75);  arg78_1 = view_192 = permute_75 = None
        view_193: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_12, _shape_param_257);  addmm_12 = _shape_param_257 = None
        mul_66: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_193, 0.5)
        pow_7: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_193, 3.0)
        mul_67: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_61: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_193, mul_67);  view_193 = mul_67 = None
        mul_68: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_61, 0.7978845608028654);  add_61 = None
        tanh_6: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_68);  mul_68 = None
        add_62: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_69: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_66, add_62);  mul_66 = add_62 = None
        view_194: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_69, _shape_param_258);  mul_69 = _shape_param_258 = None
        permute_76: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_13: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg80_1, view_194, permute_76);  arg80_1 = view_194 = permute_76 = None
        view_195: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_13, _shape_param_259);  addmm_13 = _shape_param_259 = None
        add_63: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_191, view_195);  view_191 = view_195 = None
        add_64: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_63, add_55);  add_63 = add_55 = None
        convert_element_type_35: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_64, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_35, [2], correction = 0, keepdim = True)
        getitem_28: "f32[1, 128, 1]" = var_mean_7[0]
        getitem_29: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        repeat_14: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg81_1, [1, 1, 1]);  arg81_1 = None
        unsqueeze_100: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_15: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_100, _shape_param_260);  unsqueeze_100 = _shape_param_260 = None
        gather_7: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_14, 1, repeat_15);  repeat_14 = repeat_15 = None
        split_7 = torch.ops.aten.split.Tensor(gather_7, 32, -1);  gather_7 = None
        getitem_30: "bf16[1, 128, 32]" = split_7[0]
        getitem_31: "bf16[1, 128, 32]" = split_7[1];  split_7 = None
        sub_14: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_35, getitem_29);  convert_element_type_35 = getitem_29 = None
        add_65: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_70: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_7);  sub_14 = rsqrt_7 = None
        mul_71: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_70, arg82_1);  mul_70 = arg82_1 = None
        add_66: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_71, arg83_1);  mul_71 = arg83_1 = None
        convert_element_type_36: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_66, torch.bfloat16);  add_66 = None
        view_196: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_36, _shape_param_261);  _shape_param_261 = None
        permute_77: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
        mm_28: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_196, permute_77);  view_196 = permute_77 = None
        view_197: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_28, _shape_param_262);  mm_28 = _shape_param_262 = None
        view_198: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_197, _shape_param_263);  view_197 = _shape_param_263 = None
        slice_57: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_198, 3, 0, 64)
        unsqueeze_101: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_31, 2)
        unsqueeze_102: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 4);  unsqueeze_101 = None
        expand_57: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_102, _shape_param_264);  unsqueeze_102 = _shape_param_264 = None
        clone_35: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_199: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_35, _shape_param_265);  clone_35 = _shape_param_265 = None
        mul_72: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_57, view_199);  view_199 = None
        slice_58: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_57, 3, 1, 9223372036854775807, 2)
        neg_14: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_58);  slice_58 = None
        unsqueeze_103: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_14, 4);  neg_14 = None
        slice_59: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_57, 3, 0, 9223372036854775807, 2);  slice_57 = None
        unsqueeze_104: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_59, 4);  slice_59 = None
        cat_28: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_103, unsqueeze_104], -1);  unsqueeze_103 = unsqueeze_104 = None
        view_200: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_28, _shape_param_266);  cat_28 = _shape_param_266 = None
        unsqueeze_105: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_30, 2)
        unsqueeze_106: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 4);  unsqueeze_105 = None
        expand_58: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_106, _shape_param_267);  unsqueeze_106 = _shape_param_267 = None
        clone_36: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_201: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_268);  clone_36 = _shape_param_268 = None
        mul_73: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_200, view_201);  view_200 = view_201 = None
        add_67: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
        slice_60: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_198, 3, 64, 9223372036854775807);  view_198 = None
        cat_29: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_67, slice_60], -1);  add_67 = slice_60 = None
        permute_78: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_29, [0, 2, 1, 3]);  cat_29 = None
        convert_element_type_37: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_78, torch.float32);  permute_78 = None
        expand_59: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_37, _shape_param_269);  convert_element_type_37 = _shape_param_269 = None
        view_202: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_59, _shape_param_270);  expand_59 = _shape_param_270 = None
        view_203: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_36, _shape_param_271);  _shape_param_271 = None
        permute_79: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        mm_29: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_203, permute_79);  view_203 = permute_79 = None
        view_204: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_29, _shape_param_272);  mm_29 = _shape_param_272 = None
        view_205: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_204, _shape_param_273);  view_204 = _shape_param_273 = None
        slice_61: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_205, 3, 0, 64)
        unsqueeze_107: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_31, 2);  getitem_31 = None
        unsqueeze_108: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_107, 4);  unsqueeze_107 = None
        expand_60: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_108, _shape_param_274);  unsqueeze_108 = _shape_param_274 = None
        clone_37: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_206: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_37, _shape_param_275);  clone_37 = _shape_param_275 = None
        mul_74: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_61, view_206);  view_206 = None
        slice_62: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_61, 3, 1, 9223372036854775807, 2)
        neg_15: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_62);  slice_62 = None
        unsqueeze_109: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_15, 4);  neg_15 = None
        slice_63: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_61, 3, 0, 9223372036854775807, 2);  slice_61 = None
        unsqueeze_110: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_63, 4);  slice_63 = None
        cat_30: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_109, unsqueeze_110], -1);  unsqueeze_109 = unsqueeze_110 = None
        view_207: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_30, _shape_param_276);  cat_30 = _shape_param_276 = None
        unsqueeze_111: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_30, 2);  getitem_30 = None
        unsqueeze_112: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 4);  unsqueeze_111 = None
        expand_61: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_112, _shape_param_277);  unsqueeze_112 = _shape_param_277 = None
        clone_38: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_208: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_38, _shape_param_278);  clone_38 = _shape_param_278 = None
        mul_75: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_207, view_208);  view_207 = view_208 = None
        add_68: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        slice_64: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_205, 3, 64, 9223372036854775807);  view_205 = None
        cat_31: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_68, slice_64], -1);  add_68 = slice_64 = None
        permute_80: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_31, [0, 2, 1, 3]);  cat_31 = None
        convert_element_type_38: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_80, torch.float32);  permute_80 = None
        permute_81: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_38, [0, 1, 3, 2]);  convert_element_type_38 = None
        expand_62: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_81, _shape_param_279);  permute_81 = _shape_param_279 = None
        view_209: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_62, _shape_param_280);  expand_62 = _shape_param_280 = None
        bmm_14: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_202, view_209);  view_202 = view_209 = None
        view_210: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_14, _shape_param_281);  bmm_14 = _shape_param_281 = None
        div_14: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_210, 16.0);  view_210 = None
        add_69: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_14, where);  div_14 = None
        amax_7: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_69, [-1], True)
        sub_15: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_69, amax_7);  add_69 = amax_7 = None
        exp_7: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_8: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_15: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_39: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None
        expand_63: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_39, _shape_param_282);  convert_element_type_39 = _shape_param_282 = None
        view_211: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_63, _shape_param_283);  expand_63 = _shape_param_283 = None
        view_212: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_36, _shape_param_284);  _shape_param_284 = None
        permute_82: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        mm_30: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_212, permute_82);  view_212 = permute_82 = None
        view_213: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_30, _shape_param_285);  mm_30 = _shape_param_285 = None
        view_214: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_213, _shape_param_286);  view_213 = _shape_param_286 = None
        permute_83: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
        expand_64: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_83, _shape_param_287);  permute_83 = _shape_param_287 = None
        view_215: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_64, _shape_param_288);  expand_64 = _shape_param_288 = None
        bmm_15: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_211, view_215);  view_211 = view_215 = None
        view_216: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_15, _shape_param_289);  bmm_15 = _shape_param_289 = None
        permute_84: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_216, [0, 2, 1, 3]);  view_216 = None
        clone_39: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_217: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_39, _shape_param_290);  clone_39 = _shape_param_290 = None
        view_218: "bf16[128, 4096]" = torch.ops.aten.view.default(view_217, _shape_param_291);  view_217 = _shape_param_291 = None
        permute_85: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        mm_31: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_218, permute_85);  view_218 = permute_85 = None
        view_219: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_31, _shape_param_292);  mm_31 = _shape_param_292 = None
        view_220: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_36, _shape_param_293);  convert_element_type_36 = _shape_param_293 = None
        permute_86: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_14: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg89_1, view_220, permute_86);  arg89_1 = view_220 = permute_86 = None
        view_221: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_14, _shape_param_294);  addmm_14 = _shape_param_294 = None
        mul_76: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_221, 0.5)
        pow_8: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_221, 3.0)
        mul_77: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_70: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_221, mul_77);  view_221 = mul_77 = None
        mul_78: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_70, 0.7978845608028654);  add_70 = None
        tanh_7: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_71: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_79: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_76, add_71);  mul_76 = add_71 = None
        view_222: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_79, _shape_param_295);  mul_79 = _shape_param_295 = None
        permute_87: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
        addmm_15: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg91_1, view_222, permute_87);  arg91_1 = view_222 = permute_87 = None
        view_223: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_15, _shape_param_296);  addmm_15 = _shape_param_296 = None
        add_72: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_219, view_223);  view_219 = view_223 = None
        add_73: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_72, add_64);  add_72 = add_64 = None
        convert_element_type_40: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_73, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_40, [2], correction = 0, keepdim = True)
        getitem_32: "f32[1, 128, 1]" = var_mean_8[0]
        getitem_33: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        repeat_16: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg92_1, [1, 1, 1]);  arg92_1 = None
        unsqueeze_113: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_17: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_113, _shape_param_297);  unsqueeze_113 = _shape_param_297 = None
        gather_8: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_16, 1, repeat_17);  repeat_16 = repeat_17 = None
        split_8 = torch.ops.aten.split.Tensor(gather_8, 32, -1);  gather_8 = None
        getitem_34: "bf16[1, 128, 32]" = split_8[0]
        getitem_35: "bf16[1, 128, 32]" = split_8[1];  split_8 = None
        sub_16: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_40, getitem_33);  convert_element_type_40 = getitem_33 = None
        add_74: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_80: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_8);  sub_16 = rsqrt_8 = None
        mul_81: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_80, arg93_1);  mul_80 = arg93_1 = None
        add_75: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_81, arg94_1);  mul_81 = arg94_1 = None
        convert_element_type_41: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_75, torch.bfloat16);  add_75 = None
        view_224: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_41, _shape_param_298);  _shape_param_298 = None
        permute_88: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        mm_32: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_224, permute_88);  view_224 = permute_88 = None
        view_225: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_32, _shape_param_299);  mm_32 = _shape_param_299 = None
        view_226: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_225, _shape_param_300);  view_225 = _shape_param_300 = None
        slice_65: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_226, 3, 0, 64)
        unsqueeze_114: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_35, 2)
        unsqueeze_115: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 4);  unsqueeze_114 = None
        expand_65: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_115, _shape_param_301);  unsqueeze_115 = _shape_param_301 = None
        clone_40: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_227: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_302);  clone_40 = _shape_param_302 = None
        mul_82: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_65, view_227);  view_227 = None
        slice_66: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_65, 3, 1, 9223372036854775807, 2)
        neg_16: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_66);  slice_66 = None
        unsqueeze_116: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_16, 4);  neg_16 = None
        slice_67: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_65, 3, 0, 9223372036854775807, 2);  slice_65 = None
        unsqueeze_117: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_67, 4);  slice_67 = None
        cat_32: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_116, unsqueeze_117], -1);  unsqueeze_116 = unsqueeze_117 = None
        view_228: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_32, _shape_param_303);  cat_32 = _shape_param_303 = None
        unsqueeze_118: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_34, 2)
        unsqueeze_119: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, 4);  unsqueeze_118 = None
        expand_66: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_119, _shape_param_304);  unsqueeze_119 = _shape_param_304 = None
        clone_41: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_229: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_41, _shape_param_305);  clone_41 = _shape_param_305 = None
        mul_83: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_228, view_229);  view_228 = view_229 = None
        add_76: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None
        slice_68: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_226, 3, 64, 9223372036854775807);  view_226 = None
        cat_33: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_76, slice_68], -1);  add_76 = slice_68 = None
        permute_89: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_33, [0, 2, 1, 3]);  cat_33 = None
        convert_element_type_42: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_89, torch.float32);  permute_89 = None
        expand_67: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_42, _shape_param_306);  convert_element_type_42 = _shape_param_306 = None
        view_230: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_67, _shape_param_307);  expand_67 = _shape_param_307 = None
        view_231: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_41, _shape_param_308);  _shape_param_308 = None
        permute_90: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
        mm_33: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_231, permute_90);  view_231 = permute_90 = None
        view_232: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_33, _shape_param_309);  mm_33 = _shape_param_309 = None
        view_233: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_232, _shape_param_310);  view_232 = _shape_param_310 = None
        slice_69: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_233, 3, 0, 64)
        unsqueeze_120: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_35, 2);  getitem_35 = None
        unsqueeze_121: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 4);  unsqueeze_120 = None
        expand_68: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_121, _shape_param_311);  unsqueeze_121 = _shape_param_311 = None
        clone_42: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_234: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_42, _shape_param_312);  clone_42 = _shape_param_312 = None
        mul_84: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_69, view_234);  view_234 = None
        slice_70: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_69, 3, 1, 9223372036854775807, 2)
        neg_17: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_70);  slice_70 = None
        unsqueeze_122: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_17, 4);  neg_17 = None
        slice_71: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_69, 3, 0, 9223372036854775807, 2);  slice_69 = None
        unsqueeze_123: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_71, 4);  slice_71 = None
        cat_34: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_122, unsqueeze_123], -1);  unsqueeze_122 = unsqueeze_123 = None
        view_235: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_34, _shape_param_313);  cat_34 = _shape_param_313 = None
        unsqueeze_124: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_34, 2);  getitem_34 = None
        unsqueeze_125: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, 4);  unsqueeze_124 = None
        expand_69: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_125, _shape_param_314);  unsqueeze_125 = _shape_param_314 = None
        clone_43: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_236: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_43, _shape_param_315);  clone_43 = _shape_param_315 = None
        mul_85: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_235, view_236);  view_235 = view_236 = None
        add_77: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None
        slice_72: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_233, 3, 64, 9223372036854775807);  view_233 = None
        cat_35: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_77, slice_72], -1);  add_77 = slice_72 = None
        permute_91: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_35, [0, 2, 1, 3]);  cat_35 = None
        convert_element_type_43: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_91, torch.float32);  permute_91 = None
        permute_92: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_43, [0, 1, 3, 2]);  convert_element_type_43 = None
        expand_70: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_92, _shape_param_316);  permute_92 = _shape_param_316 = None
        view_237: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_70, _shape_param_317);  expand_70 = _shape_param_317 = None
        bmm_16: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_230, view_237);  view_230 = view_237 = None
        view_238: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_16, _shape_param_318);  bmm_16 = _shape_param_318 = None
        div_16: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_238, 16.0);  view_238 = None
        add_78: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_16, where);  div_16 = None
        amax_8: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_78, [-1], True)
        sub_17: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_78, amax_8);  add_78 = amax_8 = None
        exp_8: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_9: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_17: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_44: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None
        expand_71: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_44, _shape_param_319);  convert_element_type_44 = _shape_param_319 = None
        view_239: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_71, _shape_param_320);  expand_71 = _shape_param_320 = None
        view_240: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_41, _shape_param_321);  _shape_param_321 = None
        permute_93: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        mm_34: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_240, permute_93);  view_240 = permute_93 = None
        view_241: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_34, _shape_param_322);  mm_34 = _shape_param_322 = None
        view_242: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_241, _shape_param_323);  view_241 = _shape_param_323 = None
        permute_94: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_242, [0, 2, 1, 3]);  view_242 = None
        expand_72: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_94, _shape_param_324);  permute_94 = _shape_param_324 = None
        view_243: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_72, _shape_param_325);  expand_72 = _shape_param_325 = None
        bmm_17: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_239, view_243);  view_239 = view_243 = None
        view_244: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_17, _shape_param_326);  bmm_17 = _shape_param_326 = None
        permute_95: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
        clone_44: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_245: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_44, _shape_param_327);  clone_44 = _shape_param_327 = None
        view_246: "bf16[128, 4096]" = torch.ops.aten.view.default(view_245, _shape_param_328);  view_245 = _shape_param_328 = None
        permute_96: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        mm_35: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_246, permute_96);  view_246 = permute_96 = None
        view_247: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_35, _shape_param_329);  mm_35 = _shape_param_329 = None
        view_248: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_41, _shape_param_330);  convert_element_type_41 = _shape_param_330 = None
        permute_97: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_16: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg100_1, view_248, permute_97);  arg100_1 = view_248 = permute_97 = None
        view_249: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_16, _shape_param_331);  addmm_16 = _shape_param_331 = None
        mul_86: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_249, 0.5)
        pow_9: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_249, 3.0)
        mul_87: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_79: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_249, mul_87);  view_249 = mul_87 = None
        mul_88: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
        tanh_8: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_88);  mul_88 = None
        add_80: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_89: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_86, add_80);  mul_86 = add_80 = None
        view_250: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_89, _shape_param_332);  mul_89 = _shape_param_332 = None
        permute_98: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_17: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg102_1, view_250, permute_98);  arg102_1 = view_250 = permute_98 = None
        view_251: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_17, _shape_param_333);  addmm_17 = _shape_param_333 = None
        add_81: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_247, view_251);  view_247 = view_251 = None
        add_82: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_81, add_73);  add_81 = add_73 = None
        convert_element_type_45: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_82, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_45, [2], correction = 0, keepdim = True)
        getitem_36: "f32[1, 128, 1]" = var_mean_9[0]
        getitem_37: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        repeat_18: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg103_1, [1, 1, 1]);  arg103_1 = None
        unsqueeze_126: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_19: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_126, _shape_param_334);  unsqueeze_126 = _shape_param_334 = None
        gather_9: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_18, 1, repeat_19);  repeat_18 = repeat_19 = None
        split_9 = torch.ops.aten.split.Tensor(gather_9, 32, -1);  gather_9 = None
        getitem_38: "bf16[1, 128, 32]" = split_9[0]
        getitem_39: "bf16[1, 128, 32]" = split_9[1];  split_9 = None
        sub_18: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_45, getitem_37);  convert_element_type_45 = getitem_37 = None
        add_83: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
        mul_90: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_9);  sub_18 = rsqrt_9 = None
        mul_91: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_90, arg104_1);  mul_90 = arg104_1 = None
        add_84: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_91, arg105_1);  mul_91 = arg105_1 = None
        convert_element_type_46: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_84, torch.bfloat16);  add_84 = None
        view_252: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_46, _shape_param_335);  _shape_param_335 = None
        permute_99: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        mm_36: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_252, permute_99);  view_252 = permute_99 = None
        view_253: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_36, _shape_param_336);  mm_36 = _shape_param_336 = None
        view_254: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_253, _shape_param_337);  view_253 = _shape_param_337 = None
        slice_73: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_254, 3, 0, 64)
        unsqueeze_127: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_39, 2)
        unsqueeze_128: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 4);  unsqueeze_127 = None
        expand_73: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_128, _shape_param_338);  unsqueeze_128 = _shape_param_338 = None
        clone_45: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_255: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_45, _shape_param_339);  clone_45 = _shape_param_339 = None
        mul_92: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_73, view_255);  view_255 = None
        slice_74: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_73, 3, 1, 9223372036854775807, 2)
        neg_18: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_74);  slice_74 = None
        unsqueeze_129: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_18, 4);  neg_18 = None
        slice_75: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_73, 3, 0, 9223372036854775807, 2);  slice_73 = None
        unsqueeze_130: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_75, 4);  slice_75 = None
        cat_36: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_129, unsqueeze_130], -1);  unsqueeze_129 = unsqueeze_130 = None
        view_256: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_36, _shape_param_340);  cat_36 = _shape_param_340 = None
        unsqueeze_131: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_38, 2)
        unsqueeze_132: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 4);  unsqueeze_131 = None
        expand_74: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_132, _shape_param_341);  unsqueeze_132 = _shape_param_341 = None
        clone_46: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_257: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_46, _shape_param_342);  clone_46 = _shape_param_342 = None
        mul_93: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_256, view_257);  view_256 = view_257 = None
        add_85: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        slice_76: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_254, 3, 64, 9223372036854775807);  view_254 = None
        cat_37: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_85, slice_76], -1);  add_85 = slice_76 = None
        permute_100: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_37, [0, 2, 1, 3]);  cat_37 = None
        convert_element_type_47: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_100, torch.float32);  permute_100 = None
        expand_75: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_47, _shape_param_343);  convert_element_type_47 = _shape_param_343 = None
        view_258: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_75, _shape_param_344);  expand_75 = _shape_param_344 = None
        view_259: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_46, _shape_param_345);  _shape_param_345 = None
        permute_101: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        mm_37: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_259, permute_101);  view_259 = permute_101 = None
        view_260: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_37, _shape_param_346);  mm_37 = _shape_param_346 = None
        view_261: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_260, _shape_param_347);  view_260 = _shape_param_347 = None
        slice_77: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_261, 3, 0, 64)
        unsqueeze_133: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_39, 2);  getitem_39 = None
        unsqueeze_134: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 4);  unsqueeze_133 = None
        expand_76: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_134, _shape_param_348);  unsqueeze_134 = _shape_param_348 = None
        clone_47: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_262: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_47, _shape_param_349);  clone_47 = _shape_param_349 = None
        mul_94: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_77, view_262);  view_262 = None
        slice_78: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_77, 3, 1, 9223372036854775807, 2)
        neg_19: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_78);  slice_78 = None
        unsqueeze_135: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_19, 4);  neg_19 = None
        slice_79: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_77, 3, 0, 9223372036854775807, 2);  slice_77 = None
        unsqueeze_136: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_79, 4);  slice_79 = None
        cat_38: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_135, unsqueeze_136], -1);  unsqueeze_135 = unsqueeze_136 = None
        view_263: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_38, _shape_param_350);  cat_38 = _shape_param_350 = None
        unsqueeze_137: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_38, 2);  getitem_38 = None
        unsqueeze_138: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 4);  unsqueeze_137 = None
        expand_77: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_138, _shape_param_351);  unsqueeze_138 = _shape_param_351 = None
        clone_48: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_264: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_48, _shape_param_352);  clone_48 = _shape_param_352 = None
        mul_95: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_263, view_264);  view_263 = view_264 = None
        add_86: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
        slice_80: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_261, 3, 64, 9223372036854775807);  view_261 = None
        cat_39: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_86, slice_80], -1);  add_86 = slice_80 = None
        permute_102: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_39, [0, 2, 1, 3]);  cat_39 = None
        convert_element_type_48: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_102, torch.float32);  permute_102 = None
        permute_103: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_48, [0, 1, 3, 2]);  convert_element_type_48 = None
        expand_78: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_103, _shape_param_353);  permute_103 = _shape_param_353 = None
        view_265: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_78, _shape_param_354);  expand_78 = _shape_param_354 = None
        bmm_18: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_258, view_265);  view_258 = view_265 = None
        view_266: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_18, _shape_param_355);  bmm_18 = _shape_param_355 = None
        div_18: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_266, 16.0);  view_266 = None
        add_87: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_18, where);  div_18 = None
        amax_9: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_87, [-1], True)
        sub_19: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_87, amax_9);  add_87 = amax_9 = None
        exp_9: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_10: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_19: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_49: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None
        expand_79: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_49, _shape_param_356);  convert_element_type_49 = _shape_param_356 = None
        view_267: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_79, _shape_param_357);  expand_79 = _shape_param_357 = None
        view_268: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_46, _shape_param_358);  _shape_param_358 = None
        permute_104: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        mm_38: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_268, permute_104);  view_268 = permute_104 = None
        view_269: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_38, _shape_param_359);  mm_38 = _shape_param_359 = None
        view_270: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_269, _shape_param_360);  view_269 = _shape_param_360 = None
        permute_105: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
        expand_80: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_105, _shape_param_361);  permute_105 = _shape_param_361 = None
        view_271: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_80, _shape_param_362);  expand_80 = _shape_param_362 = None
        bmm_19: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_267, view_271);  view_267 = view_271 = None
        view_272: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_19, _shape_param_363);  bmm_19 = _shape_param_363 = None
        permute_106: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
        clone_49: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_273: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_49, _shape_param_364);  clone_49 = _shape_param_364 = None
        view_274: "bf16[128, 4096]" = torch.ops.aten.view.default(view_273, _shape_param_365);  view_273 = _shape_param_365 = None
        permute_107: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        mm_39: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_274, permute_107);  view_274 = permute_107 = None
        view_275: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_39, _shape_param_366);  mm_39 = _shape_param_366 = None
        view_276: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_46, _shape_param_367);  convert_element_type_46 = _shape_param_367 = None
        permute_108: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
        addmm_18: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg111_1, view_276, permute_108);  arg111_1 = view_276 = permute_108 = None
        view_277: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_18, _shape_param_368);  addmm_18 = _shape_param_368 = None
        mul_96: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_277, 0.5)
        pow_10: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_277, 3.0)
        mul_97: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_88: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_277, mul_97);  view_277 = mul_97 = None
        mul_98: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_88, 0.7978845608028654);  add_88 = None
        tanh_9: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_98);  mul_98 = None
        add_89: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_99: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_96, add_89);  mul_96 = add_89 = None
        view_278: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_99, _shape_param_369);  mul_99 = _shape_param_369 = None
        permute_109: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        addmm_19: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg113_1, view_278, permute_109);  arg113_1 = view_278 = permute_109 = None
        view_279: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_19, _shape_param_370);  addmm_19 = _shape_param_370 = None
        add_90: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_275, view_279);  view_275 = view_279 = None
        add_91: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_90, add_82);  add_90 = add_82 = None
        convert_element_type_50: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_91, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_50, [2], correction = 0, keepdim = True)
        getitem_40: "f32[1, 128, 1]" = var_mean_10[0]
        getitem_41: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        repeat_20: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg114_1, [1, 1, 1]);  arg114_1 = None
        unsqueeze_139: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_21: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_139, _shape_param_371);  unsqueeze_139 = _shape_param_371 = None
        gather_10: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_20, 1, repeat_21);  repeat_20 = repeat_21 = None
        split_10 = torch.ops.aten.split.Tensor(gather_10, 32, -1);  gather_10 = None
        getitem_42: "bf16[1, 128, 32]" = split_10[0]
        getitem_43: "bf16[1, 128, 32]" = split_10[1];  split_10 = None
        sub_20: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_50, getitem_41);  convert_element_type_50 = getitem_41 = None
        add_92: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_100: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_10);  sub_20 = rsqrt_10 = None
        mul_101: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_100, arg115_1);  mul_100 = arg115_1 = None
        add_93: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_101, arg116_1);  mul_101 = arg116_1 = None
        convert_element_type_51: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_93, torch.bfloat16);  add_93 = None
        view_280: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_51, _shape_param_372);  _shape_param_372 = None
        permute_110: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        mm_40: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_280, permute_110);  view_280 = permute_110 = None
        view_281: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_40, _shape_param_373);  mm_40 = _shape_param_373 = None
        view_282: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_281, _shape_param_374);  view_281 = _shape_param_374 = None
        slice_81: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_282, 3, 0, 64)
        unsqueeze_140: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_43, 2)
        unsqueeze_141: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 4);  unsqueeze_140 = None
        expand_81: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_141, _shape_param_375);  unsqueeze_141 = _shape_param_375 = None
        clone_50: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_283: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_50, _shape_param_376);  clone_50 = _shape_param_376 = None
        mul_102: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_81, view_283);  view_283 = None
        slice_82: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_81, 3, 1, 9223372036854775807, 2)
        neg_20: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_82);  slice_82 = None
        unsqueeze_142: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_20, 4);  neg_20 = None
        slice_83: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_81, 3, 0, 9223372036854775807, 2);  slice_81 = None
        unsqueeze_143: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_83, 4);  slice_83 = None
        cat_40: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_142, unsqueeze_143], -1);  unsqueeze_142 = unsqueeze_143 = None
        view_284: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_40, _shape_param_377);  cat_40 = _shape_param_377 = None
        unsqueeze_144: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_42, 2)
        unsqueeze_145: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 4);  unsqueeze_144 = None
        expand_82: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_145, _shape_param_378);  unsqueeze_145 = _shape_param_378 = None
        clone_51: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_285: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_51, _shape_param_379);  clone_51 = _shape_param_379 = None
        mul_103: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_284, view_285);  view_284 = view_285 = None
        add_94: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        slice_84: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_282, 3, 64, 9223372036854775807);  view_282 = None
        cat_41: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_94, slice_84], -1);  add_94 = slice_84 = None
        permute_111: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_41, [0, 2, 1, 3]);  cat_41 = None
        convert_element_type_52: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_111, torch.float32);  permute_111 = None
        expand_83: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_52, _shape_param_380);  convert_element_type_52 = _shape_param_380 = None
        view_286: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_83, _shape_param_381);  expand_83 = _shape_param_381 = None
        view_287: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_51, _shape_param_382);  _shape_param_382 = None
        permute_112: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        mm_41: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_287, permute_112);  view_287 = permute_112 = None
        view_288: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_41, _shape_param_383);  mm_41 = _shape_param_383 = None
        view_289: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_288, _shape_param_384);  view_288 = _shape_param_384 = None
        slice_85: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_289, 3, 0, 64)
        unsqueeze_146: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_43, 2);  getitem_43 = None
        unsqueeze_147: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 4);  unsqueeze_146 = None
        expand_84: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_147, _shape_param_385);  unsqueeze_147 = _shape_param_385 = None
        clone_52: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_290: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_52, _shape_param_386);  clone_52 = _shape_param_386 = None
        mul_104: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_85, view_290);  view_290 = None
        slice_86: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_85, 3, 1, 9223372036854775807, 2)
        neg_21: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_86);  slice_86 = None
        unsqueeze_148: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_21, 4);  neg_21 = None
        slice_87: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_85, 3, 0, 9223372036854775807, 2);  slice_85 = None
        unsqueeze_149: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_87, 4);  slice_87 = None
        cat_42: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_148, unsqueeze_149], -1);  unsqueeze_148 = unsqueeze_149 = None
        view_291: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_42, _shape_param_387);  cat_42 = _shape_param_387 = None
        unsqueeze_150: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_42, 2);  getitem_42 = None
        unsqueeze_151: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 4);  unsqueeze_150 = None
        expand_85: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_151, _shape_param_388);  unsqueeze_151 = _shape_param_388 = None
        clone_53: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_292: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_53, _shape_param_389);  clone_53 = _shape_param_389 = None
        mul_105: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_291, view_292);  view_291 = view_292 = None
        add_95: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
        slice_88: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_289, 3, 64, 9223372036854775807);  view_289 = None
        cat_43: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_95, slice_88], -1);  add_95 = slice_88 = None
        permute_113: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_43, [0, 2, 1, 3]);  cat_43 = None
        convert_element_type_53: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_113, torch.float32);  permute_113 = None
        permute_114: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_53, [0, 1, 3, 2]);  convert_element_type_53 = None
        expand_86: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_114, _shape_param_390);  permute_114 = _shape_param_390 = None
        view_293: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_86, _shape_param_391);  expand_86 = _shape_param_391 = None
        bmm_20: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_286, view_293);  view_286 = view_293 = None
        view_294: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_20, _shape_param_392);  bmm_20 = _shape_param_392 = None
        div_20: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_294, 16.0);  view_294 = None
        add_96: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_20, where);  div_20 = None
        amax_10: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_96, [-1], True)
        sub_21: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_96, amax_10);  add_96 = amax_10 = None
        exp_10: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_11: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_21: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_54: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None
        expand_87: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_54, _shape_param_393);  convert_element_type_54 = _shape_param_393 = None
        view_295: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_87, _shape_param_394);  expand_87 = _shape_param_394 = None
        view_296: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_51, _shape_param_395);  _shape_param_395 = None
        permute_115: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        mm_42: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_296, permute_115);  view_296 = permute_115 = None
        view_297: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_42, _shape_param_396);  mm_42 = _shape_param_396 = None
        view_298: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_297, _shape_param_397);  view_297 = _shape_param_397 = None
        permute_116: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_298, [0, 2, 1, 3]);  view_298 = None
        expand_88: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_116, _shape_param_398);  permute_116 = _shape_param_398 = None
        view_299: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_88, _shape_param_399);  expand_88 = _shape_param_399 = None
        bmm_21: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_295, view_299);  view_295 = view_299 = None
        view_300: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_21, _shape_param_400);  bmm_21 = _shape_param_400 = None
        permute_117: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
        clone_54: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_301: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_54, _shape_param_401);  clone_54 = _shape_param_401 = None
        view_302: "bf16[128, 4096]" = torch.ops.aten.view.default(view_301, _shape_param_402);  view_301 = _shape_param_402 = None
        permute_118: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        mm_43: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_302, permute_118);  view_302 = permute_118 = None
        view_303: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_43, _shape_param_403);  mm_43 = _shape_param_403 = None
        view_304: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_51, _shape_param_404);  convert_element_type_51 = _shape_param_404 = None
        permute_119: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_20: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg122_1, view_304, permute_119);  arg122_1 = view_304 = permute_119 = None
        view_305: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_20, _shape_param_405);  addmm_20 = _shape_param_405 = None
        mul_106: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_305, 0.5)
        pow_11: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_305, 3.0)
        mul_107: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_97: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_305, mul_107);  view_305 = mul_107 = None
        mul_108: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_97, 0.7978845608028654);  add_97 = None
        tanh_10: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_108);  mul_108 = None
        add_98: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_109: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_106, add_98);  mul_106 = add_98 = None
        view_306: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_109, _shape_param_406);  mul_109 = _shape_param_406 = None
        permute_120: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_21: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg124_1, view_306, permute_120);  arg124_1 = view_306 = permute_120 = None
        view_307: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_21, _shape_param_407);  addmm_21 = _shape_param_407 = None
        add_99: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_303, view_307);  view_303 = view_307 = None
        add_100: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_99, add_91);  add_99 = add_91 = None
        convert_element_type_55: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_100, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_55, [2], correction = 0, keepdim = True)
        getitem_44: "f32[1, 128, 1]" = var_mean_11[0]
        getitem_45: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        repeat_22: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg125_1, [1, 1, 1]);  arg125_1 = None
        unsqueeze_152: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_23: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_152, _shape_param_408);  unsqueeze_152 = _shape_param_408 = None
        gather_11: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_22, 1, repeat_23);  repeat_22 = repeat_23 = None
        split_11 = torch.ops.aten.split.Tensor(gather_11, 32, -1);  gather_11 = None
        getitem_46: "bf16[1, 128, 32]" = split_11[0]
        getitem_47: "bf16[1, 128, 32]" = split_11[1];  split_11 = None
        sub_22: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_55, getitem_45);  convert_element_type_55 = getitem_45 = None
        add_101: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        mul_110: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_11);  sub_22 = rsqrt_11 = None
        mul_111: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_110, arg126_1);  mul_110 = arg126_1 = None
        add_102: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_111, arg127_1);  mul_111 = arg127_1 = None
        convert_element_type_56: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_102, torch.bfloat16);  add_102 = None
        view_308: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_56, _shape_param_409);  _shape_param_409 = None
        permute_121: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
        mm_44: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_308, permute_121);  view_308 = permute_121 = None
        view_309: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_44, _shape_param_410);  mm_44 = _shape_param_410 = None
        view_310: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_309, _shape_param_411);  view_309 = _shape_param_411 = None
        slice_89: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_310, 3, 0, 64)
        unsqueeze_153: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_47, 2)
        unsqueeze_154: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 4);  unsqueeze_153 = None
        expand_89: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_154, _shape_param_412);  unsqueeze_154 = _shape_param_412 = None
        clone_55: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_311: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_55, _shape_param_413);  clone_55 = _shape_param_413 = None
        mul_112: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_89, view_311);  view_311 = None
        slice_90: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_89, 3, 1, 9223372036854775807, 2)
        neg_22: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_90);  slice_90 = None
        unsqueeze_155: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_22, 4);  neg_22 = None
        slice_91: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_89, 3, 0, 9223372036854775807, 2);  slice_89 = None
        unsqueeze_156: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_91, 4);  slice_91 = None
        cat_44: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_155, unsqueeze_156], -1);  unsqueeze_155 = unsqueeze_156 = None
        view_312: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_44, _shape_param_414);  cat_44 = _shape_param_414 = None
        unsqueeze_157: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_46, 2)
        unsqueeze_158: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 4);  unsqueeze_157 = None
        expand_90: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_158, _shape_param_415);  unsqueeze_158 = _shape_param_415 = None
        clone_56: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_313: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_56, _shape_param_416);  clone_56 = _shape_param_416 = None
        mul_113: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_312, view_313);  view_312 = view_313 = None
        add_103: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
        slice_92: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_310, 3, 64, 9223372036854775807);  view_310 = None
        cat_45: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_103, slice_92], -1);  add_103 = slice_92 = None
        permute_122: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_45, [0, 2, 1, 3]);  cat_45 = None
        convert_element_type_57: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_122, torch.float32);  permute_122 = None
        expand_91: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_57, _shape_param_417);  convert_element_type_57 = _shape_param_417 = None
        view_314: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_91, _shape_param_418);  expand_91 = _shape_param_418 = None
        view_315: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_56, _shape_param_419);  _shape_param_419 = None
        permute_123: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        mm_45: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_315, permute_123);  view_315 = permute_123 = None
        view_316: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_45, _shape_param_420);  mm_45 = _shape_param_420 = None
        view_317: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_316, _shape_param_421);  view_316 = _shape_param_421 = None
        slice_93: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_317, 3, 0, 64)
        unsqueeze_159: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_47, 2);  getitem_47 = None
        unsqueeze_160: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 4);  unsqueeze_159 = None
        expand_92: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_160, _shape_param_422);  unsqueeze_160 = _shape_param_422 = None
        clone_57: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_318: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_57, _shape_param_423);  clone_57 = _shape_param_423 = None
        mul_114: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_93, view_318);  view_318 = None
        slice_94: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_93, 3, 1, 9223372036854775807, 2)
        neg_23: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_94);  slice_94 = None
        unsqueeze_161: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_23, 4);  neg_23 = None
        slice_95: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_93, 3, 0, 9223372036854775807, 2);  slice_93 = None
        unsqueeze_162: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_95, 4);  slice_95 = None
        cat_46: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_161, unsqueeze_162], -1);  unsqueeze_161 = unsqueeze_162 = None
        view_319: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_46, _shape_param_424);  cat_46 = _shape_param_424 = None
        unsqueeze_163: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_46, 2);  getitem_46 = None
        unsqueeze_164: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 4);  unsqueeze_163 = None
        expand_93: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_164, _shape_param_425);  unsqueeze_164 = _shape_param_425 = None
        clone_58: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_320: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_58, _shape_param_426);  clone_58 = _shape_param_426 = None
        mul_115: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_319, view_320);  view_319 = view_320 = None
        add_104: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None
        slice_96: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_317, 3, 64, 9223372036854775807);  view_317 = None
        cat_47: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_104, slice_96], -1);  add_104 = slice_96 = None
        permute_124: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_47, [0, 2, 1, 3]);  cat_47 = None
        convert_element_type_58: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_124, torch.float32);  permute_124 = None
        permute_125: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_58, [0, 1, 3, 2]);  convert_element_type_58 = None
        expand_94: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_125, _shape_param_427);  permute_125 = _shape_param_427 = None
        view_321: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_94, _shape_param_428);  expand_94 = _shape_param_428 = None
        bmm_22: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_314, view_321);  view_314 = view_321 = None
        view_322: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_429);  bmm_22 = _shape_param_429 = None
        div_22: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_322, 16.0);  view_322 = None
        add_105: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_22, where);  div_22 = None
        amax_11: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_105, [-1], True)
        sub_23: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_105, amax_11);  add_105 = amax_11 = None
        exp_11: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_12: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_23: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_59: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None
        expand_95: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_59, _shape_param_430);  convert_element_type_59 = _shape_param_430 = None
        view_323: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_95, _shape_param_431);  expand_95 = _shape_param_431 = None
        view_324: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_56, _shape_param_432);  _shape_param_432 = None
        permute_126: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
        mm_46: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_324, permute_126);  view_324 = permute_126 = None
        view_325: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_46, _shape_param_433);  mm_46 = _shape_param_433 = None
        view_326: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_325, _shape_param_434);  view_325 = _shape_param_434 = None
        permute_127: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None
        expand_96: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_127, _shape_param_435);  permute_127 = _shape_param_435 = None
        view_327: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_96, _shape_param_436);  expand_96 = _shape_param_436 = None
        bmm_23: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_323, view_327);  view_323 = view_327 = None
        view_328: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_23, _shape_param_437);  bmm_23 = _shape_param_437 = None
        permute_128: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None
        clone_59: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_329: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_59, _shape_param_438);  clone_59 = _shape_param_438 = None
        view_330: "bf16[128, 4096]" = torch.ops.aten.view.default(view_329, _shape_param_439);  view_329 = _shape_param_439 = None
        permute_129: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        mm_47: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_330, permute_129);  view_330 = permute_129 = None
        view_331: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_47, _shape_param_440);  mm_47 = _shape_param_440 = None
        view_332: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_56, _shape_param_441);  convert_element_type_56 = _shape_param_441 = None
        permute_130: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
        addmm_22: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg133_1, view_332, permute_130);  arg133_1 = view_332 = permute_130 = None
        view_333: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_22, _shape_param_442);  addmm_22 = _shape_param_442 = None
        mul_116: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_333, 0.5)
        pow_12: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_333, 3.0)
        mul_117: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_106: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_333, mul_117);  view_333 = mul_117 = None
        mul_118: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_106, 0.7978845608028654);  add_106 = None
        tanh_11: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_107: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_119: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_116, add_107);  mul_116 = add_107 = None
        view_334: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_119, _shape_param_443);  mul_119 = _shape_param_443 = None
        permute_131: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        addmm_23: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg135_1, view_334, permute_131);  arg135_1 = view_334 = permute_131 = None
        view_335: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_23, _shape_param_444);  addmm_23 = _shape_param_444 = None
        add_108: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_331, view_335);  view_331 = view_335 = None
        add_109: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_108, add_100);  add_108 = add_100 = None
        convert_element_type_60: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_109, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_60, [2], correction = 0, keepdim = True)
        getitem_48: "f32[1, 128, 1]" = var_mean_12[0]
        getitem_49: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        repeat_24: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg136_1, [1, 1, 1]);  arg136_1 = None
        unsqueeze_165: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_25: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_165, _shape_param_445);  unsqueeze_165 = _shape_param_445 = None
        gather_12: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_24, 1, repeat_25);  repeat_24 = repeat_25 = None
        split_12 = torch.ops.aten.split.Tensor(gather_12, 32, -1);  gather_12 = None
        getitem_50: "bf16[1, 128, 32]" = split_12[0]
        getitem_51: "bf16[1, 128, 32]" = split_12[1];  split_12 = None
        sub_24: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_60, getitem_49);  convert_element_type_60 = getitem_49 = None
        add_110: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        mul_120: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_12);  sub_24 = rsqrt_12 = None
        mul_121: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_120, arg137_1);  mul_120 = arg137_1 = None
        add_111: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_121, arg138_1);  mul_121 = arg138_1 = None
        convert_element_type_61: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_111, torch.bfloat16);  add_111 = None
        view_336: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_61, _shape_param_446);  _shape_param_446 = None
        permute_132: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        mm_48: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_336, permute_132);  view_336 = permute_132 = None
        view_337: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_48, _shape_param_447);  mm_48 = _shape_param_447 = None
        view_338: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_337, _shape_param_448);  view_337 = _shape_param_448 = None
        slice_97: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_338, 3, 0, 64)
        unsqueeze_166: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_51, 2)
        unsqueeze_167: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 4);  unsqueeze_166 = None
        expand_97: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_167, _shape_param_449);  unsqueeze_167 = _shape_param_449 = None
        clone_60: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_339: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_60, _shape_param_450);  clone_60 = _shape_param_450 = None
        mul_122: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_97, view_339);  view_339 = None
        slice_98: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_97, 3, 1, 9223372036854775807, 2)
        neg_24: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_98);  slice_98 = None
        unsqueeze_168: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_24, 4);  neg_24 = None
        slice_99: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_97, 3, 0, 9223372036854775807, 2);  slice_97 = None
        unsqueeze_169: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_99, 4);  slice_99 = None
        cat_48: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_168, unsqueeze_169], -1);  unsqueeze_168 = unsqueeze_169 = None
        view_340: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_48, _shape_param_451);  cat_48 = _shape_param_451 = None
        unsqueeze_170: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_50, 2)
        unsqueeze_171: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 4);  unsqueeze_170 = None
        expand_98: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_171, _shape_param_452);  unsqueeze_171 = _shape_param_452 = None
        clone_61: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
        view_341: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_61, _shape_param_453);  clone_61 = _shape_param_453 = None
        mul_123: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_340, view_341);  view_340 = view_341 = None
        add_112: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None
        slice_100: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_338, 3, 64, 9223372036854775807);  view_338 = None
        cat_49: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_112, slice_100], -1);  add_112 = slice_100 = None
        permute_133: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_49, [0, 2, 1, 3]);  cat_49 = None
        convert_element_type_62: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_133, torch.float32);  permute_133 = None
        expand_99: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_62, _shape_param_454);  convert_element_type_62 = _shape_param_454 = None
        view_342: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_99, _shape_param_455);  expand_99 = _shape_param_455 = None
        view_343: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_61, _shape_param_456);  _shape_param_456 = None
        permute_134: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
        mm_49: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_343, permute_134);  view_343 = permute_134 = None
        view_344: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_49, _shape_param_457);  mm_49 = _shape_param_457 = None
        view_345: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_344, _shape_param_458);  view_344 = _shape_param_458 = None
        slice_101: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_345, 3, 0, 64)
        unsqueeze_172: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_51, 2);  getitem_51 = None
        unsqueeze_173: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 4);  unsqueeze_172 = None
        expand_100: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_173, _shape_param_459);  unsqueeze_173 = _shape_param_459 = None
        clone_62: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_100, memory_format = torch.contiguous_format);  expand_100 = None
        view_346: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_62, _shape_param_460);  clone_62 = _shape_param_460 = None
        mul_124: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_101, view_346);  view_346 = None
        slice_102: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_101, 3, 1, 9223372036854775807, 2)
        neg_25: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_102);  slice_102 = None
        unsqueeze_174: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_25, 4);  neg_25 = None
        slice_103: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_101, 3, 0, 9223372036854775807, 2);  slice_101 = None
        unsqueeze_175: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_103, 4);  slice_103 = None
        cat_50: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_174, unsqueeze_175], -1);  unsqueeze_174 = unsqueeze_175 = None
        view_347: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_50, _shape_param_461);  cat_50 = _shape_param_461 = None
        unsqueeze_176: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_50, 2);  getitem_50 = None
        unsqueeze_177: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 4);  unsqueeze_176 = None
        expand_101: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_177, _shape_param_462);  unsqueeze_177 = _shape_param_462 = None
        clone_63: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_101, memory_format = torch.contiguous_format);  expand_101 = None
        view_348: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_63, _shape_param_463);  clone_63 = _shape_param_463 = None
        mul_125: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_347, view_348);  view_347 = view_348 = None
        add_113: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
        slice_104: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_345, 3, 64, 9223372036854775807);  view_345 = None
        cat_51: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_113, slice_104], -1);  add_113 = slice_104 = None
        permute_135: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_51, [0, 2, 1, 3]);  cat_51 = None
        convert_element_type_63: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_135, torch.float32);  permute_135 = None
        permute_136: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_63, [0, 1, 3, 2]);  convert_element_type_63 = None
        expand_102: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_136, _shape_param_464);  permute_136 = _shape_param_464 = None
        view_349: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_102, _shape_param_465);  expand_102 = _shape_param_465 = None
        bmm_24: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_342, view_349);  view_342 = view_349 = None
        view_350: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_24, _shape_param_466);  bmm_24 = _shape_param_466 = None
        div_24: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_350, 16.0);  view_350 = None
        add_114: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_24, where);  div_24 = None
        amax_12: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_114, [-1], True)
        sub_25: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_114, amax_12);  add_114 = amax_12 = None
        exp_12: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_13: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_25: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_64: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None
        expand_103: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_64, _shape_param_467);  convert_element_type_64 = _shape_param_467 = None
        view_351: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_103, _shape_param_468);  expand_103 = _shape_param_468 = None
        view_352: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_61, _shape_param_469);  _shape_param_469 = None
        permute_137: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        mm_50: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_352, permute_137);  view_352 = permute_137 = None
        view_353: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_50, _shape_param_470);  mm_50 = _shape_param_470 = None
        view_354: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_353, _shape_param_471);  view_353 = _shape_param_471 = None
        permute_138: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
        expand_104: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_138, _shape_param_472);  permute_138 = _shape_param_472 = None
        view_355: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_104, _shape_param_473);  expand_104 = _shape_param_473 = None
        bmm_25: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_351, view_355);  view_351 = view_355 = None
        view_356: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_25, _shape_param_474);  bmm_25 = _shape_param_474 = None
        permute_139: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None
        clone_64: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        view_357: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_64, _shape_param_475);  clone_64 = _shape_param_475 = None
        view_358: "bf16[128, 4096]" = torch.ops.aten.view.default(view_357, _shape_param_476);  view_357 = _shape_param_476 = None
        permute_140: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
        mm_51: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_358, permute_140);  view_358 = permute_140 = None
        view_359: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_51, _shape_param_477);  mm_51 = _shape_param_477 = None
        view_360: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_61, _shape_param_478);  convert_element_type_61 = _shape_param_478 = None
        permute_141: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_24: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg144_1, view_360, permute_141);  arg144_1 = view_360 = permute_141 = None
        view_361: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_24, _shape_param_479);  addmm_24 = _shape_param_479 = None
        mul_126: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_361, 0.5)
        pow_13: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_361, 3.0)
        mul_127: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_115: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_361, mul_127);  view_361 = mul_127 = None
        mul_128: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_115, 0.7978845608028654);  add_115 = None
        tanh_12: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_128);  mul_128 = None
        add_116: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_129: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_126, add_116);  mul_126 = add_116 = None
        view_362: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_129, _shape_param_480);  mul_129 = _shape_param_480 = None
        permute_142: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_25: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg146_1, view_362, permute_142);  arg146_1 = view_362 = permute_142 = None
        view_363: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_25, _shape_param_481);  addmm_25 = _shape_param_481 = None
        add_117: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_359, view_363);  view_359 = view_363 = None
        add_118: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_117, add_109);  add_117 = add_109 = None
        convert_element_type_65: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_118, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_65, [2], correction = 0, keepdim = True)
        getitem_52: "f32[1, 128, 1]" = var_mean_13[0]
        getitem_53: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None
        repeat_26: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg147_1, [1, 1, 1]);  arg147_1 = None
        unsqueeze_178: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_27: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_178, _shape_param_482);  unsqueeze_178 = _shape_param_482 = None
        gather_13: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_26, 1, repeat_27);  repeat_26 = repeat_27 = None
        split_13 = torch.ops.aten.split.Tensor(gather_13, 32, -1);  gather_13 = None
        getitem_54: "bf16[1, 128, 32]" = split_13[0]
        getitem_55: "bf16[1, 128, 32]" = split_13[1];  split_13 = None
        sub_26: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_65, getitem_53);  convert_element_type_65 = getitem_53 = None
        add_119: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        mul_130: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_13);  sub_26 = rsqrt_13 = None
        mul_131: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_130, arg148_1);  mul_130 = arg148_1 = None
        add_120: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_131, arg149_1);  mul_131 = arg149_1 = None
        convert_element_type_66: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_120, torch.bfloat16);  add_120 = None
        view_364: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_66, _shape_param_483);  _shape_param_483 = None
        permute_143: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        mm_52: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_364, permute_143);  view_364 = permute_143 = None
        view_365: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_52, _shape_param_484);  mm_52 = _shape_param_484 = None
        view_366: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_365, _shape_param_485);  view_365 = _shape_param_485 = None
        slice_105: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_366, 3, 0, 64)
        unsqueeze_179: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_55, 2)
        unsqueeze_180: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 4);  unsqueeze_179 = None
        expand_105: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_180, _shape_param_486);  unsqueeze_180 = _shape_param_486 = None
        clone_65: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
        view_367: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_65, _shape_param_487);  clone_65 = _shape_param_487 = None
        mul_132: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_105, view_367);  view_367 = None
        slice_106: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_105, 3, 1, 9223372036854775807, 2)
        neg_26: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_106);  slice_106 = None
        unsqueeze_181: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_26, 4);  neg_26 = None
        slice_107: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_105, 3, 0, 9223372036854775807, 2);  slice_105 = None
        unsqueeze_182: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_107, 4);  slice_107 = None
        cat_52: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_181, unsqueeze_182], -1);  unsqueeze_181 = unsqueeze_182 = None
        view_368: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_52, _shape_param_488);  cat_52 = _shape_param_488 = None
        unsqueeze_183: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_54, 2)
        unsqueeze_184: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 4);  unsqueeze_183 = None
        expand_106: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_184, _shape_param_489);  unsqueeze_184 = _shape_param_489 = None
        clone_66: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_106, memory_format = torch.contiguous_format);  expand_106 = None
        view_369: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_66, _shape_param_490);  clone_66 = _shape_param_490 = None
        mul_133: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_368, view_369);  view_368 = view_369 = None
        add_121: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
        slice_108: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_366, 3, 64, 9223372036854775807);  view_366 = None
        cat_53: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_121, slice_108], -1);  add_121 = slice_108 = None
        permute_144: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_53, [0, 2, 1, 3]);  cat_53 = None
        convert_element_type_67: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_144, torch.float32);  permute_144 = None
        expand_107: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_67, _shape_param_491);  convert_element_type_67 = _shape_param_491 = None
        view_370: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_107, _shape_param_492);  expand_107 = _shape_param_492 = None
        view_371: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_66, _shape_param_493);  _shape_param_493 = None
        permute_145: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        mm_53: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_371, permute_145);  view_371 = permute_145 = None
        view_372: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_53, _shape_param_494);  mm_53 = _shape_param_494 = None
        view_373: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_372, _shape_param_495);  view_372 = _shape_param_495 = None
        slice_109: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_373, 3, 0, 64)
        unsqueeze_185: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_55, 2);  getitem_55 = None
        unsqueeze_186: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 4);  unsqueeze_185 = None
        expand_108: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_186, _shape_param_496);  unsqueeze_186 = _shape_param_496 = None
        clone_67: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_108, memory_format = torch.contiguous_format);  expand_108 = None
        view_374: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_67, _shape_param_497);  clone_67 = _shape_param_497 = None
        mul_134: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_109, view_374);  view_374 = None
        slice_110: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_109, 3, 1, 9223372036854775807, 2)
        neg_27: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_110);  slice_110 = None
        unsqueeze_187: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_27, 4);  neg_27 = None
        slice_111: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_109, 3, 0, 9223372036854775807, 2);  slice_109 = None
        unsqueeze_188: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_111, 4);  slice_111 = None
        cat_54: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_187, unsqueeze_188], -1);  unsqueeze_187 = unsqueeze_188 = None
        view_375: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_54, _shape_param_498);  cat_54 = _shape_param_498 = None
        unsqueeze_189: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_54, 2);  getitem_54 = None
        unsqueeze_190: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 4);  unsqueeze_189 = None
        expand_109: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_190, _shape_param_499);  unsqueeze_190 = _shape_param_499 = None
        clone_68: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_109, memory_format = torch.contiguous_format);  expand_109 = None
        view_376: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_68, _shape_param_500);  clone_68 = _shape_param_500 = None
        mul_135: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_375, view_376);  view_375 = view_376 = None
        add_122: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        slice_112: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_373, 3, 64, 9223372036854775807);  view_373 = None
        cat_55: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_122, slice_112], -1);  add_122 = slice_112 = None
        permute_146: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_55, [0, 2, 1, 3]);  cat_55 = None
        convert_element_type_68: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_146, torch.float32);  permute_146 = None
        permute_147: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_68, [0, 1, 3, 2]);  convert_element_type_68 = None
        expand_110: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_147, _shape_param_501);  permute_147 = _shape_param_501 = None
        view_377: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_110, _shape_param_502);  expand_110 = _shape_param_502 = None
        bmm_26: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_370, view_377);  view_370 = view_377 = None
        view_378: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_26, _shape_param_503);  bmm_26 = _shape_param_503 = None
        div_26: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_378, 16.0);  view_378 = None
        add_123: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_26, where);  div_26 = None
        amax_13: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_123, [-1], True)
        sub_27: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_123, amax_13);  add_123 = amax_13 = None
        exp_13: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_14: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_27: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_69: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16);  div_27 = None
        expand_111: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_69, _shape_param_504);  convert_element_type_69 = _shape_param_504 = None
        view_379: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_111, _shape_param_505);  expand_111 = _shape_param_505 = None
        view_380: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_66, _shape_param_506);  _shape_param_506 = None
        permute_148: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        mm_54: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_380, permute_148);  view_380 = permute_148 = None
        view_381: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_54, _shape_param_507);  mm_54 = _shape_param_507 = None
        view_382: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_381, _shape_param_508);  view_381 = _shape_param_508 = None
        permute_149: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
        expand_112: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_149, _shape_param_509);  permute_149 = _shape_param_509 = None
        view_383: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_112, _shape_param_510);  expand_112 = _shape_param_510 = None
        bmm_27: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_379, view_383);  view_379 = view_383 = None
        view_384: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_27, _shape_param_511);  bmm_27 = _shape_param_511 = None
        permute_150: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
        clone_69: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_385: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_69, _shape_param_512);  clone_69 = _shape_param_512 = None
        view_386: "bf16[128, 4096]" = torch.ops.aten.view.default(view_385, _shape_param_513);  view_385 = _shape_param_513 = None
        permute_151: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        mm_55: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_386, permute_151);  view_386 = permute_151 = None
        view_387: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_55, _shape_param_514);  mm_55 = _shape_param_514 = None
        view_388: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_66, _shape_param_515);  convert_element_type_66 = _shape_param_515 = None
        permute_152: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_26: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg155_1, view_388, permute_152);  arg155_1 = view_388 = permute_152 = None
        view_389: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_26, _shape_param_516);  addmm_26 = _shape_param_516 = None
        mul_136: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_389, 0.5)
        pow_14: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_389, 3.0)
        mul_137: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_124: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_389, mul_137);  view_389 = mul_137 = None
        mul_138: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_124, 0.7978845608028654);  add_124 = None
        tanh_13: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_138);  mul_138 = None
        add_125: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_139: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_136, add_125);  mul_136 = add_125 = None
        view_390: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_139, _shape_param_517);  mul_139 = _shape_param_517 = None
        permute_153: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_27: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg157_1, view_390, permute_153);  arg157_1 = view_390 = permute_153 = None
        view_391: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_27, _shape_param_518);  addmm_27 = _shape_param_518 = None
        add_126: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_387, view_391);  view_387 = view_391 = None
        add_127: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_126, add_118);  add_126 = add_118 = None
        convert_element_type_70: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_127, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_70, [2], correction = 0, keepdim = True)
        getitem_56: "f32[1, 128, 1]" = var_mean_14[0]
        getitem_57: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None
        repeat_28: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg158_1, [1, 1, 1]);  arg158_1 = None
        unsqueeze_191: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_29: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_191, _shape_param_519);  unsqueeze_191 = _shape_param_519 = None
        gather_14: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_28, 1, repeat_29);  repeat_28 = repeat_29 = None
        split_14 = torch.ops.aten.split.Tensor(gather_14, 32, -1);  gather_14 = None
        getitem_58: "bf16[1, 128, 32]" = split_14[0]
        getitem_59: "bf16[1, 128, 32]" = split_14[1];  split_14 = None
        sub_28: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_70, getitem_57);  convert_element_type_70 = getitem_57 = None
        add_128: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_140: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_14);  sub_28 = rsqrt_14 = None
        mul_141: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_140, arg159_1);  mul_140 = arg159_1 = None
        add_129: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_141, arg160_1);  mul_141 = arg160_1 = None
        convert_element_type_71: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16);  add_129 = None
        view_392: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_71, _shape_param_520);  _shape_param_520 = None
        permute_154: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        mm_56: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_392, permute_154);  view_392 = permute_154 = None
        view_393: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_56, _shape_param_521);  mm_56 = _shape_param_521 = None
        view_394: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_393, _shape_param_522);  view_393 = _shape_param_522 = None
        slice_113: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_394, 3, 0, 64)
        unsqueeze_192: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_59, 2)
        unsqueeze_193: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 4);  unsqueeze_192 = None
        expand_113: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_193, _shape_param_523);  unsqueeze_193 = _shape_param_523 = None
        clone_70: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
        view_395: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_70, _shape_param_524);  clone_70 = _shape_param_524 = None
        mul_142: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_113, view_395);  view_395 = None
        slice_114: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_113, 3, 1, 9223372036854775807, 2)
        neg_28: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_114);  slice_114 = None
        unsqueeze_194: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_28, 4);  neg_28 = None
        slice_115: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_113, 3, 0, 9223372036854775807, 2);  slice_113 = None
        unsqueeze_195: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_115, 4);  slice_115 = None
        cat_56: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_194, unsqueeze_195], -1);  unsqueeze_194 = unsqueeze_195 = None
        view_396: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_56, _shape_param_525);  cat_56 = _shape_param_525 = None
        unsqueeze_196: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_58, 2)
        unsqueeze_197: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 4);  unsqueeze_196 = None
        expand_114: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_197, _shape_param_526);  unsqueeze_197 = _shape_param_526 = None
        clone_71: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_114, memory_format = torch.contiguous_format);  expand_114 = None
        view_397: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_71, _shape_param_527);  clone_71 = _shape_param_527 = None
        mul_143: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_396, view_397);  view_396 = view_397 = None
        add_130: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
        slice_116: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_394, 3, 64, 9223372036854775807);  view_394 = None
        cat_57: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_130, slice_116], -1);  add_130 = slice_116 = None
        permute_155: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_57, [0, 2, 1, 3]);  cat_57 = None
        convert_element_type_72: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_155, torch.float32);  permute_155 = None
        expand_115: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_72, _shape_param_528);  convert_element_type_72 = _shape_param_528 = None
        view_398: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_115, _shape_param_529);  expand_115 = _shape_param_529 = None
        view_399: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_71, _shape_param_530);  _shape_param_530 = None
        permute_156: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        mm_57: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_399, permute_156);  view_399 = permute_156 = None
        view_400: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_57, _shape_param_531);  mm_57 = _shape_param_531 = None
        view_401: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_400, _shape_param_532);  view_400 = _shape_param_532 = None
        slice_117: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_401, 3, 0, 64)
        unsqueeze_198: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_59, 2);  getitem_59 = None
        unsqueeze_199: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 4);  unsqueeze_198 = None
        expand_116: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_199, _shape_param_533);  unsqueeze_199 = _shape_param_533 = None
        clone_72: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_116, memory_format = torch.contiguous_format);  expand_116 = None
        view_402: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_72, _shape_param_534);  clone_72 = _shape_param_534 = None
        mul_144: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_117, view_402);  view_402 = None
        slice_118: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_117, 3, 1, 9223372036854775807, 2)
        neg_29: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_118);  slice_118 = None
        unsqueeze_200: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_29, 4);  neg_29 = None
        slice_119: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_117, 3, 0, 9223372036854775807, 2);  slice_117 = None
        unsqueeze_201: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_119, 4);  slice_119 = None
        cat_58: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_200, unsqueeze_201], -1);  unsqueeze_200 = unsqueeze_201 = None
        view_403: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_58, _shape_param_535);  cat_58 = _shape_param_535 = None
        unsqueeze_202: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_58, 2);  getitem_58 = None
        unsqueeze_203: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 4);  unsqueeze_202 = None
        expand_117: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_203, _shape_param_536);  unsqueeze_203 = _shape_param_536 = None
        clone_73: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_117, memory_format = torch.contiguous_format);  expand_117 = None
        view_404: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_73, _shape_param_537);  clone_73 = _shape_param_537 = None
        mul_145: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_403, view_404);  view_403 = view_404 = None
        add_131: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        slice_120: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_401, 3, 64, 9223372036854775807);  view_401 = None
        cat_59: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_131, slice_120], -1);  add_131 = slice_120 = None
        permute_157: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_59, [0, 2, 1, 3]);  cat_59 = None
        convert_element_type_73: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_157, torch.float32);  permute_157 = None
        permute_158: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_73, [0, 1, 3, 2]);  convert_element_type_73 = None
        expand_118: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_158, _shape_param_538);  permute_158 = _shape_param_538 = None
        view_405: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_118, _shape_param_539);  expand_118 = _shape_param_539 = None
        bmm_28: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_398, view_405);  view_398 = view_405 = None
        view_406: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_28, _shape_param_540);  bmm_28 = _shape_param_540 = None
        div_28: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_406, 16.0);  view_406 = None
        add_132: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_28, where);  div_28 = None
        amax_14: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_132, [-1], True)
        sub_29: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_132, amax_14);  add_132 = amax_14 = None
        exp_14: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
        sum_15: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_29: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_74: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_29, torch.bfloat16);  div_29 = None
        expand_119: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_74, _shape_param_541);  convert_element_type_74 = _shape_param_541 = None
        view_407: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_119, _shape_param_542);  expand_119 = _shape_param_542 = None
        view_408: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_71, _shape_param_543);  _shape_param_543 = None
        permute_159: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        mm_58: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_408, permute_159);  view_408 = permute_159 = None
        view_409: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_58, _shape_param_544);  mm_58 = _shape_param_544 = None
        view_410: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_409, _shape_param_545);  view_409 = _shape_param_545 = None
        permute_160: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
        expand_120: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_160, _shape_param_546);  permute_160 = _shape_param_546 = None
        view_411: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_120, _shape_param_547);  expand_120 = _shape_param_547 = None
        bmm_29: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_407, view_411);  view_407 = view_411 = None
        view_412: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_29, _shape_param_548);  bmm_29 = _shape_param_548 = None
        permute_161: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
        clone_74: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_413: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_74, _shape_param_549);  clone_74 = _shape_param_549 = None
        view_414: "bf16[128, 4096]" = torch.ops.aten.view.default(view_413, _shape_param_550);  view_413 = _shape_param_550 = None
        permute_162: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        mm_59: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_414, permute_162);  view_414 = permute_162 = None
        view_415: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_59, _shape_param_551);  mm_59 = _shape_param_551 = None
        view_416: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_71, _shape_param_552);  convert_element_type_71 = _shape_param_552 = None
        permute_163: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_28: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg166_1, view_416, permute_163);  arg166_1 = view_416 = permute_163 = None
        view_417: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_28, _shape_param_553);  addmm_28 = _shape_param_553 = None
        mul_146: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_417, 0.5)
        pow_15: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_417, 3.0)
        mul_147: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_133: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_417, mul_147);  view_417 = mul_147 = None
        mul_148: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_133, 0.7978845608028654);  add_133 = None
        tanh_14: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_148);  mul_148 = None
        add_134: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_149: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_146, add_134);  mul_146 = add_134 = None
        view_418: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_149, _shape_param_554);  mul_149 = _shape_param_554 = None
        permute_164: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_29: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg168_1, view_418, permute_164);  arg168_1 = view_418 = permute_164 = None
        view_419: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_29, _shape_param_555);  addmm_29 = _shape_param_555 = None
        add_135: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_415, view_419);  view_415 = view_419 = None
        add_136: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_135, add_127);  add_135 = add_127 = None
        convert_element_type_75: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_136, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_75, [2], correction = 0, keepdim = True)
        getitem_60: "f32[1, 128, 1]" = var_mean_15[0]
        getitem_61: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None
        repeat_30: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg169_1, [1, 1, 1]);  arg169_1 = None
        unsqueeze_204: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_31: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_204, _shape_param_556);  unsqueeze_204 = _shape_param_556 = None
        gather_15: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_30, 1, repeat_31);  repeat_30 = repeat_31 = None
        split_15 = torch.ops.aten.split.Tensor(gather_15, 32, -1);  gather_15 = None
        getitem_62: "bf16[1, 128, 32]" = split_15[0]
        getitem_63: "bf16[1, 128, 32]" = split_15[1];  split_15 = None
        sub_30: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_75, getitem_61);  convert_element_type_75 = getitem_61 = None
        add_137: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        mul_150: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_15);  sub_30 = rsqrt_15 = None
        mul_151: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_150, arg170_1);  mul_150 = arg170_1 = None
        add_138: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_151, arg171_1);  mul_151 = arg171_1 = None
        convert_element_type_76: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_138, torch.bfloat16);  add_138 = None
        view_420: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_76, _shape_param_557);  _shape_param_557 = None
        permute_165: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        mm_60: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_420, permute_165);  view_420 = permute_165 = None
        view_421: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_60, _shape_param_558);  mm_60 = _shape_param_558 = None
        view_422: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_421, _shape_param_559);  view_421 = _shape_param_559 = None
        slice_121: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_422, 3, 0, 64)
        unsqueeze_205: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_63, 2)
        unsqueeze_206: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 4);  unsqueeze_205 = None
        expand_121: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_206, _shape_param_560);  unsqueeze_206 = _shape_param_560 = None
        clone_75: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
        view_423: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_75, _shape_param_561);  clone_75 = _shape_param_561 = None
        mul_152: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_121, view_423);  view_423 = None
        slice_122: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_121, 3, 1, 9223372036854775807, 2)
        neg_30: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_122);  slice_122 = None
        unsqueeze_207: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_30, 4);  neg_30 = None
        slice_123: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_121, 3, 0, 9223372036854775807, 2);  slice_121 = None
        unsqueeze_208: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_123, 4);  slice_123 = None
        cat_60: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_207, unsqueeze_208], -1);  unsqueeze_207 = unsqueeze_208 = None
        view_424: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_60, _shape_param_562);  cat_60 = _shape_param_562 = None
        unsqueeze_209: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_62, 2)
        unsqueeze_210: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 4);  unsqueeze_209 = None
        expand_122: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_210, _shape_param_563);  unsqueeze_210 = _shape_param_563 = None
        clone_76: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
        view_425: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_76, _shape_param_564);  clone_76 = _shape_param_564 = None
        mul_153: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_424, view_425);  view_424 = view_425 = None
        add_139: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
        slice_124: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_422, 3, 64, 9223372036854775807);  view_422 = None
        cat_61: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_139, slice_124], -1);  add_139 = slice_124 = None
        permute_166: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_61, [0, 2, 1, 3]);  cat_61 = None
        convert_element_type_77: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_166, torch.float32);  permute_166 = None
        expand_123: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_77, _shape_param_565);  convert_element_type_77 = _shape_param_565 = None
        view_426: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_123, _shape_param_566);  expand_123 = _shape_param_566 = None
        view_427: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_76, _shape_param_567);  _shape_param_567 = None
        permute_167: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        mm_61: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_427, permute_167);  view_427 = permute_167 = None
        view_428: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_61, _shape_param_568);  mm_61 = _shape_param_568 = None
        view_429: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_428, _shape_param_569);  view_428 = _shape_param_569 = None
        slice_125: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_429, 3, 0, 64)
        unsqueeze_211: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_63, 2);  getitem_63 = None
        unsqueeze_212: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 4);  unsqueeze_211 = None
        expand_124: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_212, _shape_param_570);  unsqueeze_212 = _shape_param_570 = None
        clone_77: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_124, memory_format = torch.contiguous_format);  expand_124 = None
        view_430: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_77, _shape_param_571);  clone_77 = _shape_param_571 = None
        mul_154: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_125, view_430);  view_430 = None
        slice_126: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_125, 3, 1, 9223372036854775807, 2)
        neg_31: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_126);  slice_126 = None
        unsqueeze_213: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_31, 4);  neg_31 = None
        slice_127: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_125, 3, 0, 9223372036854775807, 2);  slice_125 = None
        unsqueeze_214: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_127, 4);  slice_127 = None
        cat_62: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_213, unsqueeze_214], -1);  unsqueeze_213 = unsqueeze_214 = None
        view_431: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_62, _shape_param_572);  cat_62 = _shape_param_572 = None
        unsqueeze_215: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_62, 2);  getitem_62 = None
        unsqueeze_216: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 4);  unsqueeze_215 = None
        expand_125: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_216, _shape_param_573);  unsqueeze_216 = _shape_param_573 = None
        clone_78: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_125, memory_format = torch.contiguous_format);  expand_125 = None
        view_432: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_78, _shape_param_574);  clone_78 = _shape_param_574 = None
        mul_155: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_431, view_432);  view_431 = view_432 = None
        add_140: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None
        slice_128: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_429, 3, 64, 9223372036854775807);  view_429 = None
        cat_63: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_140, slice_128], -1);  add_140 = slice_128 = None
        permute_168: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_63, [0, 2, 1, 3]);  cat_63 = None
        convert_element_type_78: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_168, torch.float32);  permute_168 = None
        permute_169: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_78, [0, 1, 3, 2]);  convert_element_type_78 = None
        expand_126: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_169, _shape_param_575);  permute_169 = _shape_param_575 = None
        view_433: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_126, _shape_param_576);  expand_126 = _shape_param_576 = None
        bmm_30: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_426, view_433);  view_426 = view_433 = None
        view_434: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_30, _shape_param_577);  bmm_30 = _shape_param_577 = None
        div_30: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_434, 16.0);  view_434 = None
        add_141: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_30, where);  div_30 = None
        amax_15: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_141, [-1], True)
        sub_31: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_141, amax_15);  add_141 = amax_15 = None
        exp_15: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_16: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_31: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_79: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_31, torch.bfloat16);  div_31 = None
        expand_127: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_79, _shape_param_578);  convert_element_type_79 = _shape_param_578 = None
        view_435: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_127, _shape_param_579);  expand_127 = _shape_param_579 = None
        view_436: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_76, _shape_param_580);  _shape_param_580 = None
        permute_170: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
        mm_62: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_436, permute_170);  view_436 = permute_170 = None
        view_437: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_62, _shape_param_581);  mm_62 = _shape_param_581 = None
        view_438: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_437, _shape_param_582);  view_437 = _shape_param_582 = None
        permute_171: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_438, [0, 2, 1, 3]);  view_438 = None
        expand_128: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_171, _shape_param_583);  permute_171 = _shape_param_583 = None
        view_439: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_128, _shape_param_584);  expand_128 = _shape_param_584 = None
        bmm_31: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_435, view_439);  view_435 = view_439 = None
        view_440: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_31, _shape_param_585);  bmm_31 = _shape_param_585 = None
        permute_172: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_440, [0, 2, 1, 3]);  view_440 = None
        clone_79: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_441: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_79, _shape_param_586);  clone_79 = _shape_param_586 = None
        view_442: "bf16[128, 4096]" = torch.ops.aten.view.default(view_441, _shape_param_587);  view_441 = _shape_param_587 = None
        permute_173: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        mm_63: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_442, permute_173);  view_442 = permute_173 = None
        view_443: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_63, _shape_param_588);  mm_63 = _shape_param_588 = None
        view_444: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_76, _shape_param_589);  convert_element_type_76 = _shape_param_589 = None
        permute_174: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_30: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg177_1, view_444, permute_174);  arg177_1 = view_444 = permute_174 = None
        view_445: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_30, _shape_param_590);  addmm_30 = _shape_param_590 = None
        mul_156: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_445, 0.5)
        pow_16: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_445, 3.0)
        mul_157: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_142: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_445, mul_157);  view_445 = mul_157 = None
        mul_158: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_142, 0.7978845608028654);  add_142 = None
        tanh_15: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
        add_143: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_159: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_156, add_143);  mul_156 = add_143 = None
        view_446: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_159, _shape_param_591);  mul_159 = _shape_param_591 = None
        permute_175: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_31: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg179_1, view_446, permute_175);  arg179_1 = view_446 = permute_175 = None
        view_447: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_31, _shape_param_592);  addmm_31 = _shape_param_592 = None
        add_144: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_443, view_447);  view_443 = view_447 = None
        add_145: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_144, add_136);  add_144 = add_136 = None
        convert_element_type_80: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_145, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_80, [2], correction = 0, keepdim = True)
        getitem_64: "f32[1, 128, 1]" = var_mean_16[0]
        getitem_65: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None
        repeat_32: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg180_1, [1, 1, 1]);  arg180_1 = None
        unsqueeze_217: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_33: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_217, _shape_param_593);  unsqueeze_217 = _shape_param_593 = None
        gather_16: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_32, 1, repeat_33);  repeat_32 = repeat_33 = None
        split_16 = torch.ops.aten.split.Tensor(gather_16, 32, -1);  gather_16 = None
        getitem_66: "bf16[1, 128, 32]" = split_16[0]
        getitem_67: "bf16[1, 128, 32]" = split_16[1];  split_16 = None
        sub_32: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_80, getitem_65);  convert_element_type_80 = getitem_65 = None
        add_146: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        mul_160: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_16);  sub_32 = rsqrt_16 = None
        mul_161: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_160, arg181_1);  mul_160 = arg181_1 = None
        add_147: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_161, arg182_1);  mul_161 = arg182_1 = None
        convert_element_type_81: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_147, torch.bfloat16);  add_147 = None
        view_448: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_81, _shape_param_594);  _shape_param_594 = None
        permute_176: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        mm_64: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_448, permute_176);  view_448 = permute_176 = None
        view_449: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_64, _shape_param_595);  mm_64 = _shape_param_595 = None
        view_450: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_449, _shape_param_596);  view_449 = _shape_param_596 = None
        slice_129: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_450, 3, 0, 64)
        unsqueeze_218: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_67, 2)
        unsqueeze_219: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 4);  unsqueeze_218 = None
        expand_129: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_219, _shape_param_597);  unsqueeze_219 = _shape_param_597 = None
        clone_80: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
        view_451: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_80, _shape_param_598);  clone_80 = _shape_param_598 = None
        mul_162: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_129, view_451);  view_451 = None
        slice_130: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_129, 3, 1, 9223372036854775807, 2)
        neg_32: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_130);  slice_130 = None
        unsqueeze_220: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_32, 4);  neg_32 = None
        slice_131: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_129, 3, 0, 9223372036854775807, 2);  slice_129 = None
        unsqueeze_221: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_131, 4);  slice_131 = None
        cat_64: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_220, unsqueeze_221], -1);  unsqueeze_220 = unsqueeze_221 = None
        view_452: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_64, _shape_param_599);  cat_64 = _shape_param_599 = None
        unsqueeze_222: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_66, 2)
        unsqueeze_223: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 4);  unsqueeze_222 = None
        expand_130: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_223, _shape_param_600);  unsqueeze_223 = _shape_param_600 = None
        clone_81: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_130, memory_format = torch.contiguous_format);  expand_130 = None
        view_453: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_81, _shape_param_601);  clone_81 = _shape_param_601 = None
        mul_163: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_452, view_453);  view_452 = view_453 = None
        add_148: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        slice_132: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_450, 3, 64, 9223372036854775807);  view_450 = None
        cat_65: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_148, slice_132], -1);  add_148 = slice_132 = None
        permute_177: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_65, [0, 2, 1, 3]);  cat_65 = None
        convert_element_type_82: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_177, torch.float32);  permute_177 = None
        expand_131: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_82, _shape_param_602);  convert_element_type_82 = _shape_param_602 = None
        view_454: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_131, _shape_param_603);  expand_131 = _shape_param_603 = None
        view_455: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_81, _shape_param_604);  _shape_param_604 = None
        permute_178: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        mm_65: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_455, permute_178);  view_455 = permute_178 = None
        view_456: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_65, _shape_param_605);  mm_65 = _shape_param_605 = None
        view_457: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_456, _shape_param_606);  view_456 = _shape_param_606 = None
        slice_133: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_457, 3, 0, 64)
        unsqueeze_224: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_67, 2);  getitem_67 = None
        unsqueeze_225: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 4);  unsqueeze_224 = None
        expand_132: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_225, _shape_param_607);  unsqueeze_225 = _shape_param_607 = None
        clone_82: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_132, memory_format = torch.contiguous_format);  expand_132 = None
        view_458: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_82, _shape_param_608);  clone_82 = _shape_param_608 = None
        mul_164: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_133, view_458);  view_458 = None
        slice_134: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_133, 3, 1, 9223372036854775807, 2)
        neg_33: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_134);  slice_134 = None
        unsqueeze_226: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_33, 4);  neg_33 = None
        slice_135: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_133, 3, 0, 9223372036854775807, 2);  slice_133 = None
        unsqueeze_227: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_135, 4);  slice_135 = None
        cat_66: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_226, unsqueeze_227], -1);  unsqueeze_226 = unsqueeze_227 = None
        view_459: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_66, _shape_param_609);  cat_66 = _shape_param_609 = None
        unsqueeze_228: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_66, 2);  getitem_66 = None
        unsqueeze_229: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 4);  unsqueeze_228 = None
        expand_133: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_229, _shape_param_610);  unsqueeze_229 = _shape_param_610 = None
        clone_83: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_133, memory_format = torch.contiguous_format);  expand_133 = None
        view_460: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_83, _shape_param_611);  clone_83 = _shape_param_611 = None
        mul_165: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_459, view_460);  view_459 = view_460 = None
        add_149: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
        slice_136: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_457, 3, 64, 9223372036854775807);  view_457 = None
        cat_67: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_149, slice_136], -1);  add_149 = slice_136 = None
        permute_179: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_67, [0, 2, 1, 3]);  cat_67 = None
        convert_element_type_83: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_179, torch.float32);  permute_179 = None
        permute_180: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_83, [0, 1, 3, 2]);  convert_element_type_83 = None
        expand_134: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_180, _shape_param_612);  permute_180 = _shape_param_612 = None
        view_461: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_134, _shape_param_613);  expand_134 = _shape_param_613 = None
        bmm_32: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_454, view_461);  view_454 = view_461 = None
        view_462: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_32, _shape_param_614);  bmm_32 = _shape_param_614 = None
        div_32: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_462, 16.0);  view_462 = None
        add_150: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_32, where);  div_32 = None
        amax_16: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_150, [-1], True)
        sub_33: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_150, amax_16);  add_150 = amax_16 = None
        exp_16: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
        sum_17: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_33: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_84: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_33, torch.bfloat16);  div_33 = None
        expand_135: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_84, _shape_param_615);  convert_element_type_84 = _shape_param_615 = None
        view_463: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_135, _shape_param_616);  expand_135 = _shape_param_616 = None
        view_464: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_81, _shape_param_617);  _shape_param_617 = None
        permute_181: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        mm_66: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_464, permute_181);  view_464 = permute_181 = None
        view_465: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_66, _shape_param_618);  mm_66 = _shape_param_618 = None
        view_466: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_465, _shape_param_619);  view_465 = _shape_param_619 = None
        permute_182: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None
        expand_136: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_182, _shape_param_620);  permute_182 = _shape_param_620 = None
        view_467: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_136, _shape_param_621);  expand_136 = _shape_param_621 = None
        bmm_33: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_463, view_467);  view_463 = view_467 = None
        view_468: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_33, _shape_param_622);  bmm_33 = _shape_param_622 = None
        permute_183: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
        clone_84: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        view_469: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_84, _shape_param_623);  clone_84 = _shape_param_623 = None
        view_470: "bf16[128, 4096]" = torch.ops.aten.view.default(view_469, _shape_param_624);  view_469 = _shape_param_624 = None
        permute_184: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        mm_67: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_470, permute_184);  view_470 = permute_184 = None
        view_471: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_67, _shape_param_625);  mm_67 = _shape_param_625 = None
        view_472: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_81, _shape_param_626);  convert_element_type_81 = _shape_param_626 = None
        permute_185: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_32: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg188_1, view_472, permute_185);  arg188_1 = view_472 = permute_185 = None
        view_473: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_32, _shape_param_627);  addmm_32 = _shape_param_627 = None
        mul_166: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_473, 0.5)
        pow_17: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_473, 3.0)
        mul_167: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_151: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_473, mul_167);  view_473 = mul_167 = None
        mul_168: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_151, 0.7978845608028654);  add_151 = None
        tanh_16: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_168);  mul_168 = None
        add_152: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_16, 1.0);  tanh_16 = None
        mul_169: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_166, add_152);  mul_166 = add_152 = None
        view_474: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_169, _shape_param_628);  mul_169 = _shape_param_628 = None
        permute_186: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_33: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg190_1, view_474, permute_186);  arg190_1 = view_474 = permute_186 = None
        view_475: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_33, _shape_param_629);  addmm_33 = _shape_param_629 = None
        add_153: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_471, view_475);  view_471 = view_475 = None
        add_154: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_153, add_145);  add_153 = add_145 = None
        convert_element_type_85: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_154, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_85, [2], correction = 0, keepdim = True)
        getitem_68: "f32[1, 128, 1]" = var_mean_17[0]
        getitem_69: "f32[1, 128, 1]" = var_mean_17[1];  var_mean_17 = None
        repeat_34: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg191_1, [1, 1, 1]);  arg191_1 = None
        unsqueeze_230: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_35: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_230, _shape_param_630);  unsqueeze_230 = _shape_param_630 = None
        gather_17: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_34, 1, repeat_35);  repeat_34 = repeat_35 = None
        split_17 = torch.ops.aten.split.Tensor(gather_17, 32, -1);  gather_17 = None
        getitem_70: "bf16[1, 128, 32]" = split_17[0]
        getitem_71: "bf16[1, 128, 32]" = split_17[1];  split_17 = None
        sub_34: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_85, getitem_69);  convert_element_type_85 = getitem_69 = None
        add_155: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_17: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        mul_170: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_17);  sub_34 = rsqrt_17 = None
        mul_171: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_170, arg192_1);  mul_170 = arg192_1 = None
        add_156: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_171, arg193_1);  mul_171 = arg193_1 = None
        convert_element_type_86: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_156, torch.bfloat16);  add_156 = None
        view_476: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_86, _shape_param_631);  _shape_param_631 = None
        permute_187: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
        mm_68: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_476, permute_187);  view_476 = permute_187 = None
        view_477: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_68, _shape_param_632);  mm_68 = _shape_param_632 = None
        view_478: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_477, _shape_param_633);  view_477 = _shape_param_633 = None
        slice_137: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_478, 3, 0, 64)
        unsqueeze_231: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_71, 2)
        unsqueeze_232: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 4);  unsqueeze_231 = None
        expand_137: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_232, _shape_param_634);  unsqueeze_232 = _shape_param_634 = None
        clone_85: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
        view_479: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_85, _shape_param_635);  clone_85 = _shape_param_635 = None
        mul_172: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_137, view_479);  view_479 = None
        slice_138: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_137, 3, 1, 9223372036854775807, 2)
        neg_34: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_138);  slice_138 = None
        unsqueeze_233: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_34, 4);  neg_34 = None
        slice_139: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_137, 3, 0, 9223372036854775807, 2);  slice_137 = None
        unsqueeze_234: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_139, 4);  slice_139 = None
        cat_68: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_233, unsqueeze_234], -1);  unsqueeze_233 = unsqueeze_234 = None
        view_480: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_68, _shape_param_636);  cat_68 = _shape_param_636 = None
        unsqueeze_235: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_70, 2)
        unsqueeze_236: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 4);  unsqueeze_235 = None
        expand_138: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_236, _shape_param_637);  unsqueeze_236 = _shape_param_637 = None
        clone_86: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_138, memory_format = torch.contiguous_format);  expand_138 = None
        view_481: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_86, _shape_param_638);  clone_86 = _shape_param_638 = None
        mul_173: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_480, view_481);  view_480 = view_481 = None
        add_157: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
        slice_140: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_478, 3, 64, 9223372036854775807);  view_478 = None
        cat_69: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_157, slice_140], -1);  add_157 = slice_140 = None
        permute_188: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_69, [0, 2, 1, 3]);  cat_69 = None
        convert_element_type_87: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_188, torch.float32);  permute_188 = None
        expand_139: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_87, _shape_param_639);  convert_element_type_87 = _shape_param_639 = None
        view_482: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_139, _shape_param_640);  expand_139 = _shape_param_640 = None
        view_483: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_86, _shape_param_641);  _shape_param_641 = None
        permute_189: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        mm_69: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_483, permute_189);  view_483 = permute_189 = None
        view_484: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_69, _shape_param_642);  mm_69 = _shape_param_642 = None
        view_485: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_484, _shape_param_643);  view_484 = _shape_param_643 = None
        slice_141: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_485, 3, 0, 64)
        unsqueeze_237: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_71, 2);  getitem_71 = None
        unsqueeze_238: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 4);  unsqueeze_237 = None
        expand_140: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_238, _shape_param_644);  unsqueeze_238 = _shape_param_644 = None
        clone_87: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_140, memory_format = torch.contiguous_format);  expand_140 = None
        view_486: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_87, _shape_param_645);  clone_87 = _shape_param_645 = None
        mul_174: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_141, view_486);  view_486 = None
        slice_142: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_141, 3, 1, 9223372036854775807, 2)
        neg_35: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_142);  slice_142 = None
        unsqueeze_239: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_35, 4);  neg_35 = None
        slice_143: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_141, 3, 0, 9223372036854775807, 2);  slice_141 = None
        unsqueeze_240: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_143, 4);  slice_143 = None
        cat_70: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_239, unsqueeze_240], -1);  unsqueeze_239 = unsqueeze_240 = None
        view_487: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_70, _shape_param_646);  cat_70 = _shape_param_646 = None
        unsqueeze_241: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_70, 2);  getitem_70 = None
        unsqueeze_242: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 4);  unsqueeze_241 = None
        expand_141: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_242, _shape_param_647);  unsqueeze_242 = _shape_param_647 = None
        clone_88: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_141, memory_format = torch.contiguous_format);  expand_141 = None
        view_488: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_88, _shape_param_648);  clone_88 = _shape_param_648 = None
        mul_175: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_487, view_488);  view_487 = view_488 = None
        add_158: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_174, mul_175);  mul_174 = mul_175 = None
        slice_144: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_485, 3, 64, 9223372036854775807);  view_485 = None
        cat_71: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_158, slice_144], -1);  add_158 = slice_144 = None
        permute_190: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_71, [0, 2, 1, 3]);  cat_71 = None
        convert_element_type_88: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_190, torch.float32);  permute_190 = None
        permute_191: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_88, [0, 1, 3, 2]);  convert_element_type_88 = None
        expand_142: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_191, _shape_param_649);  permute_191 = _shape_param_649 = None
        view_489: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_142, _shape_param_650);  expand_142 = _shape_param_650 = None
        bmm_34: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_482, view_489);  view_482 = view_489 = None
        view_490: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_34, _shape_param_651);  bmm_34 = _shape_param_651 = None
        div_34: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_490, 16.0);  view_490 = None
        add_159: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_34, where);  div_34 = None
        amax_17: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_159, [-1], True)
        sub_35: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_159, amax_17);  add_159 = amax_17 = None
        exp_17: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
        sum_18: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_35: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_89: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_35, torch.bfloat16);  div_35 = None
        expand_143: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_89, _shape_param_652);  convert_element_type_89 = _shape_param_652 = None
        view_491: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_143, _shape_param_653);  expand_143 = _shape_param_653 = None
        view_492: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_86, _shape_param_654);  _shape_param_654 = None
        permute_192: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
        mm_70: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_492, permute_192);  view_492 = permute_192 = None
        view_493: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_70, _shape_param_655);  mm_70 = _shape_param_655 = None
        view_494: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_493, _shape_param_656);  view_493 = _shape_param_656 = None
        permute_193: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
        expand_144: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_193, _shape_param_657);  permute_193 = _shape_param_657 = None
        view_495: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_144, _shape_param_658);  expand_144 = _shape_param_658 = None
        bmm_35: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_491, view_495);  view_491 = view_495 = None
        view_496: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_35, _shape_param_659);  bmm_35 = _shape_param_659 = None
        permute_194: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_496, [0, 2, 1, 3]);  view_496 = None
        clone_89: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_497: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_89, _shape_param_660);  clone_89 = _shape_param_660 = None
        view_498: "bf16[128, 4096]" = torch.ops.aten.view.default(view_497, _shape_param_661);  view_497 = _shape_param_661 = None
        permute_195: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        mm_71: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_498, permute_195);  view_498 = permute_195 = None
        view_499: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_71, _shape_param_662);  mm_71 = _shape_param_662 = None
        view_500: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_86, _shape_param_663);  convert_element_type_86 = _shape_param_663 = None
        permute_196: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
        addmm_34: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg199_1, view_500, permute_196);  arg199_1 = view_500 = permute_196 = None
        view_501: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_34, _shape_param_664);  addmm_34 = _shape_param_664 = None
        mul_176: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_501, 0.5)
        pow_18: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_501, 3.0)
        mul_177: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_160: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_501, mul_177);  view_501 = mul_177 = None
        mul_178: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_160, 0.7978845608028654);  add_160 = None
        tanh_17: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_178);  mul_178 = None
        add_161: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_17, 1.0);  tanh_17 = None
        mul_179: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_176, add_161);  mul_176 = add_161 = None
        view_502: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_179, _shape_param_665);  mul_179 = _shape_param_665 = None
        permute_197: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
        addmm_35: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg201_1, view_502, permute_197);  arg201_1 = view_502 = permute_197 = None
        view_503: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_35, _shape_param_666);  addmm_35 = _shape_param_666 = None
        add_162: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_499, view_503);  view_499 = view_503 = None
        add_163: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_162, add_154);  add_162 = add_154 = None
        convert_element_type_90: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_163, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_90, [2], correction = 0, keepdim = True)
        getitem_72: "f32[1, 128, 1]" = var_mean_18[0]
        getitem_73: "f32[1, 128, 1]" = var_mean_18[1];  var_mean_18 = None
        repeat_36: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg202_1, [1, 1, 1]);  arg202_1 = None
        unsqueeze_243: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_37: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_243, _shape_param_667);  unsqueeze_243 = _shape_param_667 = None
        gather_18: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_36, 1, repeat_37);  repeat_36 = repeat_37 = None
        split_18 = torch.ops.aten.split.Tensor(gather_18, 32, -1);  gather_18 = None
        getitem_74: "bf16[1, 128, 32]" = split_18[0]
        getitem_75: "bf16[1, 128, 32]" = split_18[1];  split_18 = None
        sub_36: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_90, getitem_73);  convert_element_type_90 = getitem_73 = None
        add_164: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_18: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        mul_180: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_18);  sub_36 = rsqrt_18 = None
        mul_181: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_180, arg203_1);  mul_180 = arg203_1 = None
        add_165: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_181, arg204_1);  mul_181 = arg204_1 = None
        convert_element_type_91: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_165, torch.bfloat16);  add_165 = None
        view_504: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_91, _shape_param_668);  _shape_param_668 = None
        permute_198: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        mm_72: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_504, permute_198);  view_504 = permute_198 = None
        view_505: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_72, _shape_param_669);  mm_72 = _shape_param_669 = None
        view_506: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_505, _shape_param_670);  view_505 = _shape_param_670 = None
        slice_145: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_506, 3, 0, 64)
        unsqueeze_244: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_75, 2)
        unsqueeze_245: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 4);  unsqueeze_244 = None
        expand_145: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_245, _shape_param_671);  unsqueeze_245 = _shape_param_671 = None
        clone_90: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
        view_507: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_90, _shape_param_672);  clone_90 = _shape_param_672 = None
        mul_182: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_145, view_507);  view_507 = None
        slice_146: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_145, 3, 1, 9223372036854775807, 2)
        neg_36: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_146);  slice_146 = None
        unsqueeze_246: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_36, 4);  neg_36 = None
        slice_147: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_145, 3, 0, 9223372036854775807, 2);  slice_145 = None
        unsqueeze_247: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_147, 4);  slice_147 = None
        cat_72: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_246, unsqueeze_247], -1);  unsqueeze_246 = unsqueeze_247 = None
        view_508: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_72, _shape_param_673);  cat_72 = _shape_param_673 = None
        unsqueeze_248: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_74, 2)
        unsqueeze_249: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 4);  unsqueeze_248 = None
        expand_146: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_249, _shape_param_674);  unsqueeze_249 = _shape_param_674 = None
        clone_91: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_146, memory_format = torch.contiguous_format);  expand_146 = None
        view_509: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_91, _shape_param_675);  clone_91 = _shape_param_675 = None
        mul_183: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_508, view_509);  view_508 = view_509 = None
        add_166: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None
        slice_148: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_506, 3, 64, 9223372036854775807);  view_506 = None
        cat_73: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_166, slice_148], -1);  add_166 = slice_148 = None
        permute_199: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_73, [0, 2, 1, 3]);  cat_73 = None
        convert_element_type_92: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_199, torch.float32);  permute_199 = None
        expand_147: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_92, _shape_param_676);  convert_element_type_92 = _shape_param_676 = None
        view_510: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_147, _shape_param_677);  expand_147 = _shape_param_677 = None
        view_511: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_91, _shape_param_678);  _shape_param_678 = None
        permute_200: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        mm_73: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_511, permute_200);  view_511 = permute_200 = None
        view_512: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_73, _shape_param_679);  mm_73 = _shape_param_679 = None
        view_513: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_512, _shape_param_680);  view_512 = _shape_param_680 = None
        slice_149: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_513, 3, 0, 64)
        unsqueeze_250: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_75, 2);  getitem_75 = None
        unsqueeze_251: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 4);  unsqueeze_250 = None
        expand_148: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_251, _shape_param_681);  unsqueeze_251 = _shape_param_681 = None
        clone_92: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_148, memory_format = torch.contiguous_format);  expand_148 = None
        view_514: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_92, _shape_param_682);  clone_92 = _shape_param_682 = None
        mul_184: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_149, view_514);  view_514 = None
        slice_150: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_149, 3, 1, 9223372036854775807, 2)
        neg_37: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_150);  slice_150 = None
        unsqueeze_252: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_37, 4);  neg_37 = None
        slice_151: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_149, 3, 0, 9223372036854775807, 2);  slice_149 = None
        unsqueeze_253: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_151, 4);  slice_151 = None
        cat_74: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_252, unsqueeze_253], -1);  unsqueeze_252 = unsqueeze_253 = None
        view_515: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_74, _shape_param_683);  cat_74 = _shape_param_683 = None
        unsqueeze_254: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_74, 2);  getitem_74 = None
        unsqueeze_255: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 4);  unsqueeze_254 = None
        expand_149: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_255, _shape_param_684);  unsqueeze_255 = _shape_param_684 = None
        clone_93: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_149, memory_format = torch.contiguous_format);  expand_149 = None
        view_516: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_93, _shape_param_685);  clone_93 = _shape_param_685 = None
        mul_185: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_515, view_516);  view_515 = view_516 = None
        add_167: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_184, mul_185);  mul_184 = mul_185 = None
        slice_152: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_513, 3, 64, 9223372036854775807);  view_513 = None
        cat_75: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_167, slice_152], -1);  add_167 = slice_152 = None
        permute_201: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_75, [0, 2, 1, 3]);  cat_75 = None
        convert_element_type_93: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_201, torch.float32);  permute_201 = None
        permute_202: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_93, [0, 1, 3, 2]);  convert_element_type_93 = None
        expand_150: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_202, _shape_param_686);  permute_202 = _shape_param_686 = None
        view_517: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_150, _shape_param_687);  expand_150 = _shape_param_687 = None
        bmm_36: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_510, view_517);  view_510 = view_517 = None
        view_518: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_36, _shape_param_688);  bmm_36 = _shape_param_688 = None
        div_36: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_518, 16.0);  view_518 = None
        add_168: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_36, where);  div_36 = None
        amax_18: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_168, [-1], True)
        sub_37: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_168, amax_18);  add_168 = amax_18 = None
        exp_18: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_19: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_37: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_94: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_37, torch.bfloat16);  div_37 = None
        expand_151: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_94, _shape_param_689);  convert_element_type_94 = _shape_param_689 = None
        view_519: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_151, _shape_param_690);  expand_151 = _shape_param_690 = None
        view_520: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_91, _shape_param_691);  _shape_param_691 = None
        permute_203: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        mm_74: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_520, permute_203);  view_520 = permute_203 = None
        view_521: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_74, _shape_param_692);  mm_74 = _shape_param_692 = None
        view_522: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_521, _shape_param_693);  view_521 = _shape_param_693 = None
        permute_204: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None
        expand_152: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_204, _shape_param_694);  permute_204 = _shape_param_694 = None
        view_523: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_152, _shape_param_695);  expand_152 = _shape_param_695 = None
        bmm_37: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_519, view_523);  view_519 = view_523 = None
        view_524: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_37, _shape_param_696);  bmm_37 = _shape_param_696 = None
        permute_205: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_524, [0, 2, 1, 3]);  view_524 = None
        clone_94: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_525: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_94, _shape_param_697);  clone_94 = _shape_param_697 = None
        view_526: "bf16[128, 4096]" = torch.ops.aten.view.default(view_525, _shape_param_698);  view_525 = _shape_param_698 = None
        permute_206: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        mm_75: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_526, permute_206);  view_526 = permute_206 = None
        view_527: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_75, _shape_param_699);  mm_75 = _shape_param_699 = None
        view_528: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_91, _shape_param_700);  convert_element_type_91 = _shape_param_700 = None
        permute_207: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        addmm_36: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg210_1, view_528, permute_207);  arg210_1 = view_528 = permute_207 = None
        view_529: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_36, _shape_param_701);  addmm_36 = _shape_param_701 = None
        mul_186: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_529, 0.5)
        pow_19: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_529, 3.0)
        mul_187: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_169: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_529, mul_187);  view_529 = mul_187 = None
        mul_188: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_169, 0.7978845608028654);  add_169 = None
        tanh_18: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_188);  mul_188 = None
        add_170: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_18, 1.0);  tanh_18 = None
        mul_189: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_186, add_170);  mul_186 = add_170 = None
        view_530: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_189, _shape_param_702);  mul_189 = _shape_param_702 = None
        permute_208: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_37: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg212_1, view_530, permute_208);  arg212_1 = view_530 = permute_208 = None
        view_531: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_37, _shape_param_703);  addmm_37 = _shape_param_703 = None
        add_171: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_527, view_531);  view_527 = view_531 = None
        add_172: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_171, add_163);  add_171 = add_163 = None
        convert_element_type_95: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_172, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_95, [2], correction = 0, keepdim = True)
        getitem_76: "f32[1, 128, 1]" = var_mean_19[0]
        getitem_77: "f32[1, 128, 1]" = var_mean_19[1];  var_mean_19 = None
        repeat_38: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg213_1, [1, 1, 1]);  arg213_1 = None
        unsqueeze_256: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_39: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_256, _shape_param_704);  unsqueeze_256 = _shape_param_704 = None
        gather_19: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_38, 1, repeat_39);  repeat_38 = repeat_39 = None
        split_19 = torch.ops.aten.split.Tensor(gather_19, 32, -1);  gather_19 = None
        getitem_78: "bf16[1, 128, 32]" = split_19[0]
        getitem_79: "bf16[1, 128, 32]" = split_19[1];  split_19 = None
        sub_38: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_95, getitem_77);  convert_element_type_95 = getitem_77 = None
        add_173: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_19: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
        mul_190: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_19);  sub_38 = rsqrt_19 = None
        mul_191: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_190, arg214_1);  mul_190 = arg214_1 = None
        add_174: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_191, arg215_1);  mul_191 = arg215_1 = None
        convert_element_type_96: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_174, torch.bfloat16);  add_174 = None
        view_532: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_96, _shape_param_705);  _shape_param_705 = None
        permute_209: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
        mm_76: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_532, permute_209);  view_532 = permute_209 = None
        view_533: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_76, _shape_param_706);  mm_76 = _shape_param_706 = None
        view_534: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_533, _shape_param_707);  view_533 = _shape_param_707 = None
        slice_153: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_534, 3, 0, 64)
        unsqueeze_257: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_79, 2)
        unsqueeze_258: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 4);  unsqueeze_257 = None
        expand_153: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_258, _shape_param_708);  unsqueeze_258 = _shape_param_708 = None
        clone_95: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
        view_535: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_95, _shape_param_709);  clone_95 = _shape_param_709 = None
        mul_192: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_153, view_535);  view_535 = None
        slice_154: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_153, 3, 1, 9223372036854775807, 2)
        neg_38: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_154);  slice_154 = None
        unsqueeze_259: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_38, 4);  neg_38 = None
        slice_155: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_153, 3, 0, 9223372036854775807, 2);  slice_153 = None
        unsqueeze_260: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_155, 4);  slice_155 = None
        cat_76: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_259, unsqueeze_260], -1);  unsqueeze_259 = unsqueeze_260 = None
        view_536: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_76, _shape_param_710);  cat_76 = _shape_param_710 = None
        unsqueeze_261: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_78, 2)
        unsqueeze_262: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 4);  unsqueeze_261 = None
        expand_154: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_262, _shape_param_711);  unsqueeze_262 = _shape_param_711 = None
        clone_96: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_154, memory_format = torch.contiguous_format);  expand_154 = None
        view_537: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_96, _shape_param_712);  clone_96 = _shape_param_712 = None
        mul_193: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_536, view_537);  view_536 = view_537 = None
        add_175: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None
        slice_156: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_534, 3, 64, 9223372036854775807);  view_534 = None
        cat_77: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_175, slice_156], -1);  add_175 = slice_156 = None
        permute_210: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_77, [0, 2, 1, 3]);  cat_77 = None
        convert_element_type_97: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_210, torch.float32);  permute_210 = None
        expand_155: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_97, _shape_param_713);  convert_element_type_97 = _shape_param_713 = None
        view_538: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_155, _shape_param_714);  expand_155 = _shape_param_714 = None
        view_539: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_96, _shape_param_715);  _shape_param_715 = None
        permute_211: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        mm_77: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_539, permute_211);  view_539 = permute_211 = None
        view_540: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_77, _shape_param_716);  mm_77 = _shape_param_716 = None
        view_541: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_540, _shape_param_717);  view_540 = _shape_param_717 = None
        slice_157: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_541, 3, 0, 64)
        unsqueeze_263: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_79, 2);  getitem_79 = None
        unsqueeze_264: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 4);  unsqueeze_263 = None
        expand_156: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_264, _shape_param_718);  unsqueeze_264 = _shape_param_718 = None
        clone_97: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_156, memory_format = torch.contiguous_format);  expand_156 = None
        view_542: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_97, _shape_param_719);  clone_97 = _shape_param_719 = None
        mul_194: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_157, view_542);  view_542 = None
        slice_158: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_157, 3, 1, 9223372036854775807, 2)
        neg_39: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_158);  slice_158 = None
        unsqueeze_265: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_39, 4);  neg_39 = None
        slice_159: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_157, 3, 0, 9223372036854775807, 2);  slice_157 = None
        unsqueeze_266: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_159, 4);  slice_159 = None
        cat_78: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_265, unsqueeze_266], -1);  unsqueeze_265 = unsqueeze_266 = None
        view_543: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_78, _shape_param_720);  cat_78 = _shape_param_720 = None
        unsqueeze_267: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_78, 2);  getitem_78 = None
        unsqueeze_268: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 4);  unsqueeze_267 = None
        expand_157: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_268, _shape_param_721);  unsqueeze_268 = _shape_param_721 = None
        clone_98: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_157, memory_format = torch.contiguous_format);  expand_157 = None
        view_544: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_98, _shape_param_722);  clone_98 = _shape_param_722 = None
        mul_195: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_543, view_544);  view_543 = view_544 = None
        add_176: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
        slice_160: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_541, 3, 64, 9223372036854775807);  view_541 = None
        cat_79: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_176, slice_160], -1);  add_176 = slice_160 = None
        permute_212: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_79, [0, 2, 1, 3]);  cat_79 = None
        convert_element_type_98: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_212, torch.float32);  permute_212 = None
        permute_213: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_98, [0, 1, 3, 2]);  convert_element_type_98 = None
        expand_158: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_213, _shape_param_723);  permute_213 = _shape_param_723 = None
        view_545: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_158, _shape_param_724);  expand_158 = _shape_param_724 = None
        bmm_38: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_538, view_545);  view_538 = view_545 = None
        view_546: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_38, _shape_param_725);  bmm_38 = _shape_param_725 = None
        div_38: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_546, 16.0);  view_546 = None
        add_177: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_38, where);  div_38 = None
        amax_19: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_177, [-1], True)
        sub_39: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_177, amax_19);  add_177 = amax_19 = None
        exp_19: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        sum_20: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_39: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        convert_element_type_99: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_39, torch.bfloat16);  div_39 = None
        expand_159: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_99, _shape_param_726);  convert_element_type_99 = _shape_param_726 = None
        view_547: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_159, _shape_param_727);  expand_159 = _shape_param_727 = None
        view_548: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_96, _shape_param_728);  _shape_param_728 = None
        permute_214: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        mm_78: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_548, permute_214);  view_548 = permute_214 = None
        view_549: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_78, _shape_param_729);  mm_78 = _shape_param_729 = None
        view_550: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_549, _shape_param_730);  view_549 = _shape_param_730 = None
        permute_215: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_550, [0, 2, 1, 3]);  view_550 = None
        expand_160: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_215, _shape_param_731);  permute_215 = _shape_param_731 = None
        view_551: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_160, _shape_param_732);  expand_160 = _shape_param_732 = None
        bmm_39: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_547, view_551);  view_547 = view_551 = None
        view_552: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_39, _shape_param_733);  bmm_39 = _shape_param_733 = None
        permute_216: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_552, [0, 2, 1, 3]);  view_552 = None
        clone_99: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
        view_553: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_99, _shape_param_734);  clone_99 = _shape_param_734 = None
        view_554: "bf16[128, 4096]" = torch.ops.aten.view.default(view_553, _shape_param_735);  view_553 = _shape_param_735 = None
        permute_217: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        mm_79: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_554, permute_217);  view_554 = permute_217 = None
        view_555: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_79, _shape_param_736);  mm_79 = _shape_param_736 = None
        view_556: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_96, _shape_param_737);  convert_element_type_96 = _shape_param_737 = None
        permute_218: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        addmm_38: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg221_1, view_556, permute_218);  arg221_1 = view_556 = permute_218 = None
        view_557: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_38, _shape_param_738);  addmm_38 = _shape_param_738 = None
        mul_196: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_557, 0.5)
        pow_20: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_557, 3.0)
        mul_197: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_178: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_557, mul_197);  view_557 = mul_197 = None
        mul_198: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_178, 0.7978845608028654);  add_178 = None
        tanh_19: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_198);  mul_198 = None
        add_179: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_19, 1.0);  tanh_19 = None
        mul_199: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_196, add_179);  mul_196 = add_179 = None
        view_558: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_199, _shape_param_739);  mul_199 = _shape_param_739 = None
        permute_219: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        addmm_39: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg223_1, view_558, permute_219);  arg223_1 = view_558 = permute_219 = None
        view_559: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_39, _shape_param_740);  addmm_39 = _shape_param_740 = None
        add_180: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_555, view_559);  view_555 = view_559 = None
        add_181: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_180, add_172);  add_180 = add_172 = None
        convert_element_type_100: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_181, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_100, [2], correction = 0, keepdim = True)
        getitem_80: "f32[1, 128, 1]" = var_mean_20[0]
        getitem_81: "f32[1, 128, 1]" = var_mean_20[1];  var_mean_20 = None
        repeat_40: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg224_1, [1, 1, 1]);  arg224_1 = None
        unsqueeze_269: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_41: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_269, _shape_param_741);  unsqueeze_269 = _shape_param_741 = None
        gather_20: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_40, 1, repeat_41);  repeat_40 = repeat_41 = None
        split_20 = torch.ops.aten.split.Tensor(gather_20, 32, -1);  gather_20 = None
        getitem_82: "bf16[1, 128, 32]" = split_20[0]
        getitem_83: "bf16[1, 128, 32]" = split_20[1];  split_20 = None
        sub_40: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_100, getitem_81);  convert_element_type_100 = getitem_81 = None
        add_182: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_20: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        mul_200: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_20);  sub_40 = rsqrt_20 = None
        mul_201: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_200, arg225_1);  mul_200 = arg225_1 = None
        add_183: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_201, arg226_1);  mul_201 = arg226_1 = None
        convert_element_type_101: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_183, torch.bfloat16);  add_183 = None
        view_560: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_101, _shape_param_742);  _shape_param_742 = None
        permute_220: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        mm_80: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_560, permute_220);  view_560 = permute_220 = None
        view_561: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_80, _shape_param_743);  mm_80 = _shape_param_743 = None
        view_562: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_561, _shape_param_744);  view_561 = _shape_param_744 = None
        slice_161: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_562, 3, 0, 64)
        unsqueeze_270: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_83, 2)
        unsqueeze_271: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 4);  unsqueeze_270 = None
        expand_161: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_271, _shape_param_745);  unsqueeze_271 = _shape_param_745 = None
        clone_100: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_161, memory_format = torch.contiguous_format);  expand_161 = None
        view_563: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_100, _shape_param_746);  clone_100 = _shape_param_746 = None
        mul_202: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_161, view_563);  view_563 = None
        slice_162: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_161, 3, 1, 9223372036854775807, 2)
        neg_40: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_162);  slice_162 = None
        unsqueeze_272: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_40, 4);  neg_40 = None
        slice_163: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_161, 3, 0, 9223372036854775807, 2);  slice_161 = None
        unsqueeze_273: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_163, 4);  slice_163 = None
        cat_80: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_272, unsqueeze_273], -1);  unsqueeze_272 = unsqueeze_273 = None
        view_564: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_80, _shape_param_747);  cat_80 = _shape_param_747 = None
        unsqueeze_274: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_82, 2)
        unsqueeze_275: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 4);  unsqueeze_274 = None
        expand_162: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_275, _shape_param_748);  unsqueeze_275 = _shape_param_748 = None
        clone_101: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_162, memory_format = torch.contiguous_format);  expand_162 = None
        view_565: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_101, _shape_param_749);  clone_101 = _shape_param_749 = None
        mul_203: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_564, view_565);  view_564 = view_565 = None
        add_184: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_202, mul_203);  mul_202 = mul_203 = None
        slice_164: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_562, 3, 64, 9223372036854775807);  view_562 = None
        cat_81: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_184, slice_164], -1);  add_184 = slice_164 = None
        permute_221: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_81, [0, 2, 1, 3]);  cat_81 = None
        convert_element_type_102: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_221, torch.float32);  permute_221 = None
        expand_163: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_102, _shape_param_750);  convert_element_type_102 = _shape_param_750 = None
        view_566: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_163, _shape_param_751);  expand_163 = _shape_param_751 = None
        view_567: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_101, _shape_param_752);  _shape_param_752 = None
        permute_222: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
        mm_81: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_567, permute_222);  view_567 = permute_222 = None
        view_568: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_81, _shape_param_753);  mm_81 = _shape_param_753 = None
        view_569: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_568, _shape_param_754);  view_568 = _shape_param_754 = None
        slice_165: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_569, 3, 0, 64)
        unsqueeze_276: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_83, 2);  getitem_83 = None
        unsqueeze_277: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 4);  unsqueeze_276 = None
        expand_164: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_277, _shape_param_755);  unsqueeze_277 = _shape_param_755 = None
        clone_102: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_164, memory_format = torch.contiguous_format);  expand_164 = None
        view_570: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_102, _shape_param_756);  clone_102 = _shape_param_756 = None
        mul_204: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_165, view_570);  view_570 = None
        slice_166: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_165, 3, 1, 9223372036854775807, 2)
        neg_41: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_166);  slice_166 = None
        unsqueeze_278: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_41, 4);  neg_41 = None
        slice_167: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_165, 3, 0, 9223372036854775807, 2);  slice_165 = None
        unsqueeze_279: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_167, 4);  slice_167 = None
        cat_82: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_278, unsqueeze_279], -1);  unsqueeze_278 = unsqueeze_279 = None
        view_571: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_82, _shape_param_757);  cat_82 = _shape_param_757 = None
        unsqueeze_280: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_82, 2);  getitem_82 = None
        unsqueeze_281: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 4);  unsqueeze_280 = None
        expand_165: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_281, _shape_param_758);  unsqueeze_281 = _shape_param_758 = None
        clone_103: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_165, memory_format = torch.contiguous_format);  expand_165 = None
        view_572: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_103, _shape_param_759);  clone_103 = _shape_param_759 = None
        mul_205: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_571, view_572);  view_571 = view_572 = None
        add_185: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
        slice_168: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_569, 3, 64, 9223372036854775807);  view_569 = None
        cat_83: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_185, slice_168], -1);  add_185 = slice_168 = None
        permute_223: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_83, [0, 2, 1, 3]);  cat_83 = None
        convert_element_type_103: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_223, torch.float32);  permute_223 = None
        permute_224: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_103, [0, 1, 3, 2]);  convert_element_type_103 = None
        expand_166: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_224, _shape_param_760);  permute_224 = _shape_param_760 = None
        view_573: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_166, _shape_param_761);  expand_166 = _shape_param_761 = None
        bmm_40: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_566, view_573);  view_566 = view_573 = None
        view_574: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_40, _shape_param_762);  bmm_40 = _shape_param_762 = None
        div_40: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_574, 16.0);  view_574 = None
        add_186: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_40, where);  div_40 = None
        amax_20: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_186, [-1], True)
        sub_41: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_186, amax_20);  add_186 = amax_20 = None
        exp_20: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_41);  sub_41 = None
        sum_21: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_41: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_104: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_41, torch.bfloat16);  div_41 = None
        expand_167: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_104, _shape_param_763);  convert_element_type_104 = _shape_param_763 = None
        view_575: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_167, _shape_param_764);  expand_167 = _shape_param_764 = None
        view_576: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_101, _shape_param_765);  _shape_param_765 = None
        permute_225: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        mm_82: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_576, permute_225);  view_576 = permute_225 = None
        view_577: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_82, _shape_param_766);  mm_82 = _shape_param_766 = None
        view_578: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_577, _shape_param_767);  view_577 = _shape_param_767 = None
        permute_226: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None
        expand_168: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_226, _shape_param_768);  permute_226 = _shape_param_768 = None
        view_579: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_168, _shape_param_769);  expand_168 = _shape_param_769 = None
        bmm_41: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_575, view_579);  view_575 = view_579 = None
        view_580: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_41, _shape_param_770);  bmm_41 = _shape_param_770 = None
        permute_227: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_580, [0, 2, 1, 3]);  view_580 = None
        clone_104: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_581: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_104, _shape_param_771);  clone_104 = _shape_param_771 = None
        view_582: "bf16[128, 4096]" = torch.ops.aten.view.default(view_581, _shape_param_772);  view_581 = _shape_param_772 = None
        permute_228: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
        mm_83: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_582, permute_228);  view_582 = permute_228 = None
        view_583: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_83, _shape_param_773);  mm_83 = _shape_param_773 = None
        view_584: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_101, _shape_param_774);  convert_element_type_101 = _shape_param_774 = None
        permute_229: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_40: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg232_1, view_584, permute_229);  arg232_1 = view_584 = permute_229 = None
        view_585: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_40, _shape_param_775);  addmm_40 = _shape_param_775 = None
        mul_206: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_585, 0.5)
        pow_21: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_585, 3.0)
        mul_207: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_187: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_585, mul_207);  view_585 = mul_207 = None
        mul_208: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_187, 0.7978845608028654);  add_187 = None
        tanh_20: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_208);  mul_208 = None
        add_188: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_20, 1.0);  tanh_20 = None
        mul_209: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_206, add_188);  mul_206 = add_188 = None
        view_586: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_209, _shape_param_776);  mul_209 = _shape_param_776 = None
        permute_230: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_41: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg234_1, view_586, permute_230);  arg234_1 = view_586 = permute_230 = None
        view_587: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_41, _shape_param_777);  addmm_41 = _shape_param_777 = None
        add_189: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_583, view_587);  view_583 = view_587 = None
        add_190: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_189, add_181);  add_189 = add_181 = None
        convert_element_type_105: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_190, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_105, [2], correction = 0, keepdim = True)
        getitem_84: "f32[1, 128, 1]" = var_mean_21[0]
        getitem_85: "f32[1, 128, 1]" = var_mean_21[1];  var_mean_21 = None
        repeat_42: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg235_1, [1, 1, 1]);  arg235_1 = None
        unsqueeze_282: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_43: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_282, _shape_param_778);  unsqueeze_282 = _shape_param_778 = None
        gather_21: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_42, 1, repeat_43);  repeat_42 = repeat_43 = None
        split_21 = torch.ops.aten.split.Tensor(gather_21, 32, -1);  gather_21 = None
        getitem_86: "bf16[1, 128, 32]" = split_21[0]
        getitem_87: "bf16[1, 128, 32]" = split_21[1];  split_21 = None
        sub_42: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_105, getitem_85);  convert_element_type_105 = getitem_85 = None
        add_191: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
        rsqrt_21: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        mul_210: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_21);  sub_42 = rsqrt_21 = None
        mul_211: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_210, arg236_1);  mul_210 = arg236_1 = None
        add_192: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_211, arg237_1);  mul_211 = arg237_1 = None
        convert_element_type_106: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_192, torch.bfloat16);  add_192 = None
        view_588: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_106, _shape_param_779);  _shape_param_779 = None
        permute_231: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        mm_84: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_588, permute_231);  view_588 = permute_231 = None
        view_589: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_84, _shape_param_780);  mm_84 = _shape_param_780 = None
        view_590: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_589, _shape_param_781);  view_589 = _shape_param_781 = None
        slice_169: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_590, 3, 0, 64)
        unsqueeze_283: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_87, 2)
        unsqueeze_284: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 4);  unsqueeze_283 = None
        expand_169: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_284, _shape_param_782);  unsqueeze_284 = _shape_param_782 = None
        clone_105: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_169, memory_format = torch.contiguous_format);  expand_169 = None
        view_591: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_105, _shape_param_783);  clone_105 = _shape_param_783 = None
        mul_212: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_169, view_591);  view_591 = None
        slice_170: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_169, 3, 1, 9223372036854775807, 2)
        neg_42: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_170);  slice_170 = None
        unsqueeze_285: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_42, 4);  neg_42 = None
        slice_171: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_169, 3, 0, 9223372036854775807, 2);  slice_169 = None
        unsqueeze_286: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_171, 4);  slice_171 = None
        cat_84: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_285, unsqueeze_286], -1);  unsqueeze_285 = unsqueeze_286 = None
        view_592: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_84, _shape_param_784);  cat_84 = _shape_param_784 = None
        unsqueeze_287: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_86, 2)
        unsqueeze_288: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 4);  unsqueeze_287 = None
        expand_170: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_288, _shape_param_785);  unsqueeze_288 = _shape_param_785 = None
        clone_106: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_170, memory_format = torch.contiguous_format);  expand_170 = None
        view_593: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_106, _shape_param_786);  clone_106 = _shape_param_786 = None
        mul_213: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_592, view_593);  view_592 = view_593 = None
        add_193: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None
        slice_172: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_590, 3, 64, 9223372036854775807);  view_590 = None
        cat_85: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_193, slice_172], -1);  add_193 = slice_172 = None
        permute_232: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_85, [0, 2, 1, 3]);  cat_85 = None
        convert_element_type_107: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_232, torch.float32);  permute_232 = None
        expand_171: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_107, _shape_param_787);  convert_element_type_107 = _shape_param_787 = None
        view_594: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_171, _shape_param_788);  expand_171 = _shape_param_788 = None
        view_595: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_106, _shape_param_789);  _shape_param_789 = None
        permute_233: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        mm_85: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_595, permute_233);  view_595 = permute_233 = None
        view_596: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_85, _shape_param_790);  mm_85 = _shape_param_790 = None
        view_597: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_596, _shape_param_791);  view_596 = _shape_param_791 = None
        slice_173: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_597, 3, 0, 64)
        unsqueeze_289: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_87, 2);  getitem_87 = None
        unsqueeze_290: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 4);  unsqueeze_289 = None
        expand_172: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_290, _shape_param_792);  unsqueeze_290 = _shape_param_792 = None
        clone_107: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_172, memory_format = torch.contiguous_format);  expand_172 = None
        view_598: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_107, _shape_param_793);  clone_107 = _shape_param_793 = None
        mul_214: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_173, view_598);  view_598 = None
        slice_174: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_173, 3, 1, 9223372036854775807, 2)
        neg_43: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_174);  slice_174 = None
        unsqueeze_291: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_43, 4);  neg_43 = None
        slice_175: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_173, 3, 0, 9223372036854775807, 2);  slice_173 = None
        unsqueeze_292: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_175, 4);  slice_175 = None
        cat_86: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_291, unsqueeze_292], -1);  unsqueeze_291 = unsqueeze_292 = None
        view_599: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_86, _shape_param_794);  cat_86 = _shape_param_794 = None
        unsqueeze_293: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_86, 2);  getitem_86 = None
        unsqueeze_294: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 4);  unsqueeze_293 = None
        expand_173: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_294, _shape_param_795);  unsqueeze_294 = _shape_param_795 = None
        clone_108: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_173, memory_format = torch.contiguous_format);  expand_173 = None
        view_600: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_108, _shape_param_796);  clone_108 = _shape_param_796 = None
        mul_215: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_599, view_600);  view_599 = view_600 = None
        add_194: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        slice_176: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_597, 3, 64, 9223372036854775807);  view_597 = None
        cat_87: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_194, slice_176], -1);  add_194 = slice_176 = None
        permute_234: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_87, [0, 2, 1, 3]);  cat_87 = None
        convert_element_type_108: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_234, torch.float32);  permute_234 = None
        permute_235: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_108, [0, 1, 3, 2]);  convert_element_type_108 = None
        expand_174: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_235, _shape_param_797);  permute_235 = _shape_param_797 = None
        view_601: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_174, _shape_param_798);  expand_174 = _shape_param_798 = None
        bmm_42: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_594, view_601);  view_594 = view_601 = None
        view_602: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_42, _shape_param_799);  bmm_42 = _shape_param_799 = None
        div_42: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_602, 16.0);  view_602 = None
        add_195: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_42, where);  div_42 = None
        amax_21: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_195, [-1], True)
        sub_43: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_195, amax_21);  add_195 = amax_21 = None
        exp_21: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_22: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_43: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        convert_element_type_109: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_43, torch.bfloat16);  div_43 = None
        expand_175: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_109, _shape_param_800);  convert_element_type_109 = _shape_param_800 = None
        view_603: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_175, _shape_param_801);  expand_175 = _shape_param_801 = None
        view_604: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_106, _shape_param_802);  _shape_param_802 = None
        permute_236: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
        mm_86: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_604, permute_236);  view_604 = permute_236 = None
        view_605: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_86, _shape_param_803);  mm_86 = _shape_param_803 = None
        view_606: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_605, _shape_param_804);  view_605 = _shape_param_804 = None
        permute_237: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_606, [0, 2, 1, 3]);  view_606 = None
        expand_176: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_237, _shape_param_805);  permute_237 = _shape_param_805 = None
        view_607: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_176, _shape_param_806);  expand_176 = _shape_param_806 = None
        bmm_43: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_603, view_607);  view_603 = view_607 = None
        view_608: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_43, _shape_param_807);  bmm_43 = _shape_param_807 = None
        permute_238: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_608, [0, 2, 1, 3]);  view_608 = None
        clone_109: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
        view_609: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_109, _shape_param_808);  clone_109 = _shape_param_808 = None
        view_610: "bf16[128, 4096]" = torch.ops.aten.view.default(view_609, _shape_param_809);  view_609 = _shape_param_809 = None
        permute_239: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        mm_87: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_610, permute_239);  view_610 = permute_239 = None
        view_611: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_87, _shape_param_810);  mm_87 = _shape_param_810 = None
        view_612: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_106, _shape_param_811);  convert_element_type_106 = _shape_param_811 = None
        permute_240: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg242_1, [1, 0]);  arg242_1 = None
        addmm_42: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg243_1, view_612, permute_240);  arg243_1 = view_612 = permute_240 = None
        view_613: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_42, _shape_param_812);  addmm_42 = _shape_param_812 = None
        mul_216: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_613, 0.5)
        pow_22: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_613, 3.0)
        mul_217: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_196: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_613, mul_217);  view_613 = mul_217 = None
        mul_218: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_196, 0.7978845608028654);  add_196 = None
        tanh_21: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_218);  mul_218 = None
        add_197: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_21, 1.0);  tanh_21 = None
        mul_219: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_216, add_197);  mul_216 = add_197 = None
        view_614: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_219, _shape_param_813);  mul_219 = _shape_param_813 = None
        permute_241: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
        addmm_43: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg245_1, view_614, permute_241);  arg245_1 = view_614 = permute_241 = None
        view_615: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_43, _shape_param_814);  addmm_43 = _shape_param_814 = None
        add_198: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_611, view_615);  view_611 = view_615 = None
        add_199: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_198, add_190);  add_198 = add_190 = None
        convert_element_type_110: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_199, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_110, [2], correction = 0, keepdim = True)
        getitem_88: "f32[1, 128, 1]" = var_mean_22[0]
        getitem_89: "f32[1, 128, 1]" = var_mean_22[1];  var_mean_22 = None
        repeat_44: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg246_1, [1, 1, 1]);  arg246_1 = None
        unsqueeze_295: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_45: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_295, _shape_param_815);  unsqueeze_295 = _shape_param_815 = None
        gather_22: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_44, 1, repeat_45);  repeat_44 = repeat_45 = None
        split_22 = torch.ops.aten.split.Tensor(gather_22, 32, -1);  gather_22 = None
        getitem_90: "bf16[1, 128, 32]" = split_22[0]
        getitem_91: "bf16[1, 128, 32]" = split_22[1];  split_22 = None
        sub_44: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_110, getitem_89);  convert_element_type_110 = getitem_89 = None
        add_200: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_22: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
        mul_220: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_22);  sub_44 = rsqrt_22 = None
        mul_221: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_220, arg247_1);  mul_220 = arg247_1 = None
        add_201: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_221, arg248_1);  mul_221 = arg248_1 = None
        convert_element_type_111: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_201, torch.bfloat16);  add_201 = None
        view_616: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_111, _shape_param_816);  _shape_param_816 = None
        permute_242: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        mm_88: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_616, permute_242);  view_616 = permute_242 = None
        view_617: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_88, _shape_param_817);  mm_88 = _shape_param_817 = None
        view_618: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_617, _shape_param_818);  view_617 = _shape_param_818 = None
        slice_177: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_618, 3, 0, 64)
        unsqueeze_296: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_91, 2)
        unsqueeze_297: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 4);  unsqueeze_296 = None
        expand_177: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_297, _shape_param_819);  unsqueeze_297 = _shape_param_819 = None
        clone_110: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
        view_619: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_110, _shape_param_820);  clone_110 = _shape_param_820 = None
        mul_222: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_177, view_619);  view_619 = None
        slice_178: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_177, 3, 1, 9223372036854775807, 2)
        neg_44: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_178);  slice_178 = None
        unsqueeze_298: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_44, 4);  neg_44 = None
        slice_179: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_177, 3, 0, 9223372036854775807, 2);  slice_177 = None
        unsqueeze_299: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_179, 4);  slice_179 = None
        cat_88: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_298, unsqueeze_299], -1);  unsqueeze_298 = unsqueeze_299 = None
        view_620: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_88, _shape_param_821);  cat_88 = _shape_param_821 = None
        unsqueeze_300: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_90, 2)
        unsqueeze_301: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 4);  unsqueeze_300 = None
        expand_178: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_301, _shape_param_822);  unsqueeze_301 = _shape_param_822 = None
        clone_111: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_178, memory_format = torch.contiguous_format);  expand_178 = None
        view_621: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_111, _shape_param_823);  clone_111 = _shape_param_823 = None
        mul_223: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_620, view_621);  view_620 = view_621 = None
        add_202: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_222, mul_223);  mul_222 = mul_223 = None
        slice_180: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_618, 3, 64, 9223372036854775807);  view_618 = None
        cat_89: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_202, slice_180], -1);  add_202 = slice_180 = None
        permute_243: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_89, [0, 2, 1, 3]);  cat_89 = None
        convert_element_type_112: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_243, torch.float32);  permute_243 = None
        expand_179: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_112, _shape_param_824);  convert_element_type_112 = _shape_param_824 = None
        view_622: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_179, _shape_param_825);  expand_179 = _shape_param_825 = None
        view_623: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_111, _shape_param_826);  _shape_param_826 = None
        permute_244: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
        mm_89: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_623, permute_244);  view_623 = permute_244 = None
        view_624: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_89, _shape_param_827);  mm_89 = _shape_param_827 = None
        view_625: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_624, _shape_param_828);  view_624 = _shape_param_828 = None
        slice_181: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_625, 3, 0, 64)
        unsqueeze_302: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_91, 2);  getitem_91 = None
        unsqueeze_303: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 4);  unsqueeze_302 = None
        expand_180: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_303, _shape_param_829);  unsqueeze_303 = _shape_param_829 = None
        clone_112: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_180, memory_format = torch.contiguous_format);  expand_180 = None
        view_626: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_112, _shape_param_830);  clone_112 = _shape_param_830 = None
        mul_224: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_181, view_626);  view_626 = None
        slice_182: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_181, 3, 1, 9223372036854775807, 2)
        neg_45: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_182);  slice_182 = None
        unsqueeze_304: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_45, 4);  neg_45 = None
        slice_183: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_181, 3, 0, 9223372036854775807, 2);  slice_181 = None
        unsqueeze_305: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_183, 4);  slice_183 = None
        cat_90: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_304, unsqueeze_305], -1);  unsqueeze_304 = unsqueeze_305 = None
        view_627: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_90, _shape_param_831);  cat_90 = _shape_param_831 = None
        unsqueeze_306: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_90, 2);  getitem_90 = None
        unsqueeze_307: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 4);  unsqueeze_306 = None
        expand_181: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_307, _shape_param_832);  unsqueeze_307 = _shape_param_832 = None
        clone_113: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_181, memory_format = torch.contiguous_format);  expand_181 = None
        view_628: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_113, _shape_param_833);  clone_113 = _shape_param_833 = None
        mul_225: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_627, view_628);  view_627 = view_628 = None
        add_203: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
        slice_184: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_625, 3, 64, 9223372036854775807);  view_625 = None
        cat_91: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_203, slice_184], -1);  add_203 = slice_184 = None
        permute_245: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_91, [0, 2, 1, 3]);  cat_91 = None
        convert_element_type_113: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_245, torch.float32);  permute_245 = None
        permute_246: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_113, [0, 1, 3, 2]);  convert_element_type_113 = None
        expand_182: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_246, _shape_param_834);  permute_246 = _shape_param_834 = None
        view_629: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_182, _shape_param_835);  expand_182 = _shape_param_835 = None
        bmm_44: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_622, view_629);  view_622 = view_629 = None
        view_630: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_44, _shape_param_836);  bmm_44 = _shape_param_836 = None
        div_44: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_630, 16.0);  view_630 = None
        add_204: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_44, where);  div_44 = None
        amax_22: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_204, [-1], True)
        sub_45: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_204, amax_22);  add_204 = amax_22 = None
        exp_22: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
        sum_23: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_45: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_114: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_45, torch.bfloat16);  div_45 = None
        expand_183: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_114, _shape_param_837);  convert_element_type_114 = _shape_param_837 = None
        view_631: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_183, _shape_param_838);  expand_183 = _shape_param_838 = None
        view_632: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_111, _shape_param_839);  _shape_param_839 = None
        permute_247: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        mm_90: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_632, permute_247);  view_632 = permute_247 = None
        view_633: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_90, _shape_param_840);  mm_90 = _shape_param_840 = None
        view_634: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_633, _shape_param_841);  view_633 = _shape_param_841 = None
        permute_248: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_634, [0, 2, 1, 3]);  view_634 = None
        expand_184: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_248, _shape_param_842);  permute_248 = _shape_param_842 = None
        view_635: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_184, _shape_param_843);  expand_184 = _shape_param_843 = None
        bmm_45: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_631, view_635);  view_631 = view_635 = None
        view_636: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_45, _shape_param_844);  bmm_45 = _shape_param_844 = None
        permute_249: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_636, [0, 2, 1, 3]);  view_636 = None
        clone_114: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_637: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_114, _shape_param_845);  clone_114 = _shape_param_845 = None
        view_638: "bf16[128, 4096]" = torch.ops.aten.view.default(view_637, _shape_param_846);  view_637 = _shape_param_846 = None
        permute_250: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
        mm_91: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_638, permute_250);  view_638 = permute_250 = None
        view_639: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_91, _shape_param_847);  mm_91 = _shape_param_847 = None
        view_640: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_111, _shape_param_848);  convert_element_type_111 = _shape_param_848 = None
        permute_251: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_44: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg254_1, view_640, permute_251);  arg254_1 = view_640 = permute_251 = None
        view_641: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_44, _shape_param_849);  addmm_44 = _shape_param_849 = None
        mul_226: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_641, 0.5)
        pow_23: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_641, 3.0)
        mul_227: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_205: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_641, mul_227);  view_641 = mul_227 = None
        mul_228: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_205, 0.7978845608028654);  add_205 = None
        tanh_22: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_228);  mul_228 = None
        add_206: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_22, 1.0);  tanh_22 = None
        mul_229: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_226, add_206);  mul_226 = add_206 = None
        view_642: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_229, _shape_param_850);  mul_229 = _shape_param_850 = None
        permute_252: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_45: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg256_1, view_642, permute_252);  arg256_1 = view_642 = permute_252 = None
        view_643: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_45, _shape_param_851);  addmm_45 = _shape_param_851 = None
        add_207: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_639, view_643);  view_639 = view_643 = None
        add_208: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_207, add_199);  add_207 = add_199 = None
        convert_element_type_115: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_208, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_115, [2], correction = 0, keepdim = True)
        getitem_92: "f32[1, 128, 1]" = var_mean_23[0]
        getitem_93: "f32[1, 128, 1]" = var_mean_23[1];  var_mean_23 = None
        repeat_46: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg257_1, [1, 1, 1]);  arg257_1 = None
        unsqueeze_308: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_47: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_308, _shape_param_852);  unsqueeze_308 = _shape_param_852 = None
        gather_23: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_46, 1, repeat_47);  repeat_46 = repeat_47 = None
        split_23 = torch.ops.aten.split.Tensor(gather_23, 32, -1);  gather_23 = None
        getitem_94: "bf16[1, 128, 32]" = split_23[0]
        getitem_95: "bf16[1, 128, 32]" = split_23[1];  split_23 = None
        sub_46: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_115, getitem_93);  convert_element_type_115 = getitem_93 = None
        add_209: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_23: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
        mul_230: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_23);  sub_46 = rsqrt_23 = None
        mul_231: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_230, arg258_1);  mul_230 = arg258_1 = None
        add_210: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_231, arg259_1);  mul_231 = arg259_1 = None
        convert_element_type_116: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_210, torch.bfloat16);  add_210 = None
        view_644: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_116, _shape_param_853);  _shape_param_853 = None
        permute_253: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
        mm_92: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_644, permute_253);  view_644 = permute_253 = None
        view_645: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_92, _shape_param_854);  mm_92 = _shape_param_854 = None
        view_646: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_645, _shape_param_855);  view_645 = _shape_param_855 = None
        slice_185: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_646, 3, 0, 64)
        unsqueeze_309: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_95, 2)
        unsqueeze_310: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 4);  unsqueeze_309 = None
        expand_185: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_310, _shape_param_856);  unsqueeze_310 = _shape_param_856 = None
        clone_115: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_185, memory_format = torch.contiguous_format);  expand_185 = None
        view_647: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_115, _shape_param_857);  clone_115 = _shape_param_857 = None
        mul_232: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_185, view_647);  view_647 = None
        slice_186: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_185, 3, 1, 9223372036854775807, 2)
        neg_46: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_186);  slice_186 = None
        unsqueeze_311: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_46, 4);  neg_46 = None
        slice_187: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_185, 3, 0, 9223372036854775807, 2);  slice_185 = None
        unsqueeze_312: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_187, 4);  slice_187 = None
        cat_92: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_311, unsqueeze_312], -1);  unsqueeze_311 = unsqueeze_312 = None
        view_648: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_92, _shape_param_858);  cat_92 = _shape_param_858 = None
        unsqueeze_313: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_94, 2)
        unsqueeze_314: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 4);  unsqueeze_313 = None
        expand_186: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_314, _shape_param_859);  unsqueeze_314 = _shape_param_859 = None
        clone_116: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_186, memory_format = torch.contiguous_format);  expand_186 = None
        view_649: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_116, _shape_param_860);  clone_116 = _shape_param_860 = None
        mul_233: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_648, view_649);  view_648 = view_649 = None
        add_211: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        slice_188: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_646, 3, 64, 9223372036854775807);  view_646 = None
        cat_93: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_211, slice_188], -1);  add_211 = slice_188 = None
        permute_254: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_93, [0, 2, 1, 3]);  cat_93 = None
        convert_element_type_117: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_254, torch.float32);  permute_254 = None
        expand_187: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_117, _shape_param_861);  convert_element_type_117 = _shape_param_861 = None
        view_650: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_187, _shape_param_862);  expand_187 = _shape_param_862 = None
        view_651: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_116, _shape_param_863);  _shape_param_863 = None
        permute_255: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        mm_93: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_651, permute_255);  view_651 = permute_255 = None
        view_652: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_93, _shape_param_864);  mm_93 = _shape_param_864 = None
        view_653: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_652, _shape_param_865);  view_652 = _shape_param_865 = None
        slice_189: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_653, 3, 0, 64)
        unsqueeze_315: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_95, 2);  getitem_95 = None
        unsqueeze_316: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 4);  unsqueeze_315 = None
        expand_188: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_316, _shape_param_866);  unsqueeze_316 = _shape_param_866 = None
        clone_117: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_188, memory_format = torch.contiguous_format);  expand_188 = None
        view_654: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_117, _shape_param_867);  clone_117 = _shape_param_867 = None
        mul_234: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_189, view_654);  view_654 = None
        slice_190: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_189, 3, 1, 9223372036854775807, 2)
        neg_47: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_190);  slice_190 = None
        unsqueeze_317: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_47, 4);  neg_47 = None
        slice_191: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_189, 3, 0, 9223372036854775807, 2);  slice_189 = None
        unsqueeze_318: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_191, 4);  slice_191 = None
        cat_94: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_317, unsqueeze_318], -1);  unsqueeze_317 = unsqueeze_318 = None
        view_655: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_94, _shape_param_868);  cat_94 = _shape_param_868 = None
        unsqueeze_319: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_94, 2);  getitem_94 = None
        unsqueeze_320: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 4);  unsqueeze_319 = None
        expand_189: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_320, _shape_param_869);  unsqueeze_320 = _shape_param_869 = None
        clone_118: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_189, memory_format = torch.contiguous_format);  expand_189 = None
        view_656: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_118, _shape_param_870);  clone_118 = _shape_param_870 = None
        mul_235: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_655, view_656);  view_655 = view_656 = None
        add_212: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
        slice_192: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_653, 3, 64, 9223372036854775807);  view_653 = None
        cat_95: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_212, slice_192], -1);  add_212 = slice_192 = None
        permute_256: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_95, [0, 2, 1, 3]);  cat_95 = None
        convert_element_type_118: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_256, torch.float32);  permute_256 = None
        permute_257: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_118, [0, 1, 3, 2]);  convert_element_type_118 = None
        expand_190: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_257, _shape_param_871);  permute_257 = _shape_param_871 = None
        view_657: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_190, _shape_param_872);  expand_190 = _shape_param_872 = None
        bmm_46: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_650, view_657);  view_650 = view_657 = None
        view_658: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_46, _shape_param_873);  bmm_46 = _shape_param_873 = None
        div_46: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_658, 16.0);  view_658 = None
        add_213: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_46, where);  div_46 = None
        amax_23: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_213, [-1], True)
        sub_47: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_213, amax_23);  add_213 = amax_23 = None
        exp_23: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_47);  sub_47 = None
        sum_24: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_47: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        convert_element_type_119: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_47, torch.bfloat16);  div_47 = None
        expand_191: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_119, _shape_param_874);  convert_element_type_119 = _shape_param_874 = None
        view_659: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_191, _shape_param_875);  expand_191 = _shape_param_875 = None
        view_660: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_116, _shape_param_876);  _shape_param_876 = None
        permute_258: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        mm_94: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_660, permute_258);  view_660 = permute_258 = None
        view_661: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_94, _shape_param_877);  mm_94 = _shape_param_877 = None
        view_662: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_661, _shape_param_878);  view_661 = _shape_param_878 = None
        permute_259: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_662, [0, 2, 1, 3]);  view_662 = None
        expand_192: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_259, _shape_param_879);  permute_259 = _shape_param_879 = None
        view_663: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_192, _shape_param_880);  expand_192 = _shape_param_880 = None
        bmm_47: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_659, view_663);  view_659 = view_663 = None
        view_664: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_47, _shape_param_881);  bmm_47 = _shape_param_881 = None
        permute_260: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_664, [0, 2, 1, 3]);  view_664 = None
        clone_119: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_665: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_119, _shape_param_882);  clone_119 = _shape_param_882 = None
        view_666: "bf16[128, 4096]" = torch.ops.aten.view.default(view_665, _shape_param_883);  view_665 = _shape_param_883 = None
        permute_261: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        mm_95: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_666, permute_261);  view_666 = permute_261 = None
        view_667: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_95, _shape_param_884);  mm_95 = _shape_param_884 = None
        view_668: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_116, _shape_param_885);  convert_element_type_116 = _shape_param_885 = None
        permute_262: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
        addmm_46: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg265_1, view_668, permute_262);  arg265_1 = view_668 = permute_262 = None
        view_669: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_46, _shape_param_886);  addmm_46 = _shape_param_886 = None
        mul_236: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_669, 0.5)
        pow_24: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_669, 3.0)
        mul_237: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_214: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_669, mul_237);  view_669 = mul_237 = None
        mul_238: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_214, 0.7978845608028654);  add_214 = None
        tanh_23: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_238);  mul_238 = None
        add_215: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_23, 1.0);  tanh_23 = None
        mul_239: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_236, add_215);  mul_236 = add_215 = None
        view_670: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_239, _shape_param_887);  mul_239 = _shape_param_887 = None
        permute_263: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        addmm_47: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg267_1, view_670, permute_263);  arg267_1 = view_670 = permute_263 = None
        view_671: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_47, _shape_param_888);  addmm_47 = _shape_param_888 = None
        add_216: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_667, view_671);  view_667 = view_671 = None
        add_217: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_216, add_208);  add_216 = add_208 = None
        convert_element_type_120: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_217, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_120, [2], correction = 0, keepdim = True)
        getitem_96: "f32[1, 128, 1]" = var_mean_24[0]
        getitem_97: "f32[1, 128, 1]" = var_mean_24[1];  var_mean_24 = None
        repeat_48: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg268_1, [1, 1, 1]);  arg268_1 = None
        unsqueeze_321: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_49: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_321, _shape_param_889);  unsqueeze_321 = _shape_param_889 = None
        gather_24: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_48, 1, repeat_49);  repeat_48 = repeat_49 = None
        split_24 = torch.ops.aten.split.Tensor(gather_24, 32, -1);  gather_24 = None
        getitem_98: "bf16[1, 128, 32]" = split_24[0]
        getitem_99: "bf16[1, 128, 32]" = split_24[1];  split_24 = None
        sub_48: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_120, getitem_97);  convert_element_type_120 = getitem_97 = None
        add_218: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_24: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        mul_240: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_24);  sub_48 = rsqrt_24 = None
        mul_241: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_240, arg269_1);  mul_240 = arg269_1 = None
        add_219: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_241, arg270_1);  mul_241 = arg270_1 = None
        convert_element_type_121: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_219, torch.bfloat16);  add_219 = None
        view_672: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_121, _shape_param_890);  _shape_param_890 = None
        permute_264: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        mm_96: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_672, permute_264);  view_672 = permute_264 = None
        view_673: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_96, _shape_param_891);  mm_96 = _shape_param_891 = None
        view_674: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_673, _shape_param_892);  view_673 = _shape_param_892 = None
        slice_193: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_674, 3, 0, 64)
        unsqueeze_322: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_99, 2)
        unsqueeze_323: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 4);  unsqueeze_322 = None
        expand_193: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_323, _shape_param_893);  unsqueeze_323 = _shape_param_893 = None
        clone_120: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_193, memory_format = torch.contiguous_format);  expand_193 = None
        view_675: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_120, _shape_param_894);  clone_120 = _shape_param_894 = None
        mul_242: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_193, view_675);  view_675 = None
        slice_194: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_193, 3, 1, 9223372036854775807, 2)
        neg_48: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_194);  slice_194 = None
        unsqueeze_324: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_48, 4);  neg_48 = None
        slice_195: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_193, 3, 0, 9223372036854775807, 2);  slice_193 = None
        unsqueeze_325: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_195, 4);  slice_195 = None
        cat_96: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_324, unsqueeze_325], -1);  unsqueeze_324 = unsqueeze_325 = None
        view_676: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_96, _shape_param_895);  cat_96 = _shape_param_895 = None
        unsqueeze_326: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_98, 2)
        unsqueeze_327: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 4);  unsqueeze_326 = None
        expand_194: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_327, _shape_param_896);  unsqueeze_327 = _shape_param_896 = None
        clone_121: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_194, memory_format = torch.contiguous_format);  expand_194 = None
        view_677: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_121, _shape_param_897);  clone_121 = _shape_param_897 = None
        mul_243: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_676, view_677);  view_676 = view_677 = None
        add_220: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        slice_196: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_674, 3, 64, 9223372036854775807);  view_674 = None
        cat_97: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_220, slice_196], -1);  add_220 = slice_196 = None
        permute_265: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_97, [0, 2, 1, 3]);  cat_97 = None
        convert_element_type_122: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_265, torch.float32);  permute_265 = None
        expand_195: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_122, _shape_param_898);  convert_element_type_122 = _shape_param_898 = None
        view_678: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_195, _shape_param_899);  expand_195 = _shape_param_899 = None
        view_679: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_121, _shape_param_900);  _shape_param_900 = None
        permute_266: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        mm_97: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_679, permute_266);  view_679 = permute_266 = None
        view_680: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_97, _shape_param_901);  mm_97 = _shape_param_901 = None
        view_681: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_680, _shape_param_902);  view_680 = _shape_param_902 = None
        slice_197: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_681, 3, 0, 64)
        unsqueeze_328: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_99, 2);  getitem_99 = None
        unsqueeze_329: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 4);  unsqueeze_328 = None
        expand_196: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_329, _shape_param_903);  unsqueeze_329 = _shape_param_903 = None
        clone_122: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_196, memory_format = torch.contiguous_format);  expand_196 = None
        view_682: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_122, _shape_param_904);  clone_122 = _shape_param_904 = None
        mul_244: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_197, view_682);  view_682 = None
        slice_198: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_197, 3, 1, 9223372036854775807, 2)
        neg_49: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_198);  slice_198 = None
        unsqueeze_330: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_49, 4);  neg_49 = None
        slice_199: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_197, 3, 0, 9223372036854775807, 2);  slice_197 = None
        unsqueeze_331: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_199, 4);  slice_199 = None
        cat_98: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_330, unsqueeze_331], -1);  unsqueeze_330 = unsqueeze_331 = None
        view_683: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_98, _shape_param_905);  cat_98 = _shape_param_905 = None
        unsqueeze_332: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_98, 2);  getitem_98 = None
        unsqueeze_333: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 4);  unsqueeze_332 = None
        expand_197: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_333, _shape_param_906);  unsqueeze_333 = _shape_param_906 = None
        clone_123: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_197, memory_format = torch.contiguous_format);  expand_197 = None
        view_684: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_123, _shape_param_907);  clone_123 = _shape_param_907 = None
        mul_245: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_683, view_684);  view_683 = view_684 = None
        add_221: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
        slice_200: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_681, 3, 64, 9223372036854775807);  view_681 = None
        cat_99: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_221, slice_200], -1);  add_221 = slice_200 = None
        permute_267: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_99, [0, 2, 1, 3]);  cat_99 = None
        convert_element_type_123: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_267, torch.float32);  permute_267 = None
        permute_268: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_123, [0, 1, 3, 2]);  convert_element_type_123 = None
        expand_198: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_268, _shape_param_908);  permute_268 = _shape_param_908 = None
        view_685: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_198, _shape_param_909);  expand_198 = _shape_param_909 = None
        bmm_48: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_678, view_685);  view_678 = view_685 = None
        view_686: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_48, _shape_param_910);  bmm_48 = _shape_param_910 = None
        div_48: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_686, 16.0);  view_686 = None
        add_222: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_48, where);  div_48 = None
        amax_24: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_222, [-1], True)
        sub_49: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_222, amax_24);  add_222 = amax_24 = None
        exp_24: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_25: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_49: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
        convert_element_type_124: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_49, torch.bfloat16);  div_49 = None
        expand_199: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_124, _shape_param_911);  convert_element_type_124 = _shape_param_911 = None
        view_687: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_199, _shape_param_912);  expand_199 = _shape_param_912 = None
        view_688: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_121, _shape_param_913);  _shape_param_913 = None
        permute_269: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        mm_98: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_688, permute_269);  view_688 = permute_269 = None
        view_689: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_98, _shape_param_914);  mm_98 = _shape_param_914 = None
        view_690: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_689, _shape_param_915);  view_689 = _shape_param_915 = None
        permute_270: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_690, [0, 2, 1, 3]);  view_690 = None
        expand_200: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_270, _shape_param_916);  permute_270 = _shape_param_916 = None
        view_691: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_200, _shape_param_917);  expand_200 = _shape_param_917 = None
        bmm_49: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_687, view_691);  view_687 = view_691 = None
        view_692: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_49, _shape_param_918);  bmm_49 = _shape_param_918 = None
        permute_271: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_692, [0, 2, 1, 3]);  view_692 = None
        clone_124: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_271, memory_format = torch.contiguous_format);  permute_271 = None
        view_693: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_124, _shape_param_919);  clone_124 = _shape_param_919 = None
        view_694: "bf16[128, 4096]" = torch.ops.aten.view.default(view_693, _shape_param_920);  view_693 = _shape_param_920 = None
        permute_272: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        mm_99: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_694, permute_272);  view_694 = permute_272 = None
        view_695: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_99, _shape_param_921);  mm_99 = _shape_param_921 = None
        view_696: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_121, _shape_param_922);  convert_element_type_121 = _shape_param_922 = None
        permute_273: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        addmm_48: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg276_1, view_696, permute_273);  arg276_1 = view_696 = permute_273 = None
        view_697: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_48, _shape_param_923);  addmm_48 = _shape_param_923 = None
        mul_246: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_697, 0.5)
        pow_25: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_697, 3.0)
        mul_247: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_25, 0.044715);  pow_25 = None
        add_223: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_697, mul_247);  view_697 = mul_247 = None
        mul_248: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_223, 0.7978845608028654);  add_223 = None
        tanh_24: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_248);  mul_248 = None
        add_224: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_24, 1.0);  tanh_24 = None
        mul_249: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_246, add_224);  mul_246 = add_224 = None
        view_698: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_249, _shape_param_924);  mul_249 = _shape_param_924 = None
        permute_274: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        addmm_49: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg278_1, view_698, permute_274);  arg278_1 = view_698 = permute_274 = None
        view_699: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_49, _shape_param_925);  addmm_49 = _shape_param_925 = None
        add_225: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_695, view_699);  view_695 = view_699 = None
        add_226: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_225, add_217);  add_225 = add_217 = None
        convert_element_type_125: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_226, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_125, [2], correction = 0, keepdim = True)
        getitem_100: "f32[1, 128, 1]" = var_mean_25[0]
        getitem_101: "f32[1, 128, 1]" = var_mean_25[1];  var_mean_25 = None
        repeat_50: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg279_1, [1, 1, 1]);  arg279_1 = None
        unsqueeze_334: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_51: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_334, _shape_param_926);  unsqueeze_334 = _shape_param_926 = None
        gather_25: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_50, 1, repeat_51);  repeat_50 = repeat_51 = None
        split_25 = torch.ops.aten.split.Tensor(gather_25, 32, -1);  gather_25 = None
        getitem_102: "bf16[1, 128, 32]" = split_25[0]
        getitem_103: "bf16[1, 128, 32]" = split_25[1];  split_25 = None
        sub_50: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_125, getitem_101);  convert_element_type_125 = getitem_101 = None
        add_227: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
        rsqrt_25: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_227);  add_227 = None
        mul_250: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_25);  sub_50 = rsqrt_25 = None
        mul_251: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_250, arg280_1);  mul_250 = arg280_1 = None
        add_228: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_251, arg281_1);  mul_251 = arg281_1 = None
        convert_element_type_126: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_228, torch.bfloat16);  add_228 = None
        view_700: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_126, _shape_param_927);  _shape_param_927 = None
        permute_275: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
        mm_100: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_700, permute_275);  view_700 = permute_275 = None
        view_701: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_100, _shape_param_928);  mm_100 = _shape_param_928 = None
        view_702: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_701, _shape_param_929);  view_701 = _shape_param_929 = None
        slice_201: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_702, 3, 0, 64)
        unsqueeze_335: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_103, 2)
        unsqueeze_336: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 4);  unsqueeze_335 = None
        expand_201: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_336, _shape_param_930);  unsqueeze_336 = _shape_param_930 = None
        clone_125: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_201, memory_format = torch.contiguous_format);  expand_201 = None
        view_703: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_125, _shape_param_931);  clone_125 = _shape_param_931 = None
        mul_252: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_201, view_703);  view_703 = None
        slice_202: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_201, 3, 1, 9223372036854775807, 2)
        neg_50: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_202);  slice_202 = None
        unsqueeze_337: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_50, 4);  neg_50 = None
        slice_203: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_201, 3, 0, 9223372036854775807, 2);  slice_201 = None
        unsqueeze_338: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_203, 4);  slice_203 = None
        cat_100: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_337, unsqueeze_338], -1);  unsqueeze_337 = unsqueeze_338 = None
        view_704: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_100, _shape_param_932);  cat_100 = _shape_param_932 = None
        unsqueeze_339: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_102, 2)
        unsqueeze_340: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 4);  unsqueeze_339 = None
        expand_202: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_340, _shape_param_933);  unsqueeze_340 = _shape_param_933 = None
        clone_126: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_202, memory_format = torch.contiguous_format);  expand_202 = None
        view_705: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_126, _shape_param_934);  clone_126 = _shape_param_934 = None
        mul_253: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_704, view_705);  view_704 = view_705 = None
        add_229: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
        slice_204: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_702, 3, 64, 9223372036854775807);  view_702 = None
        cat_101: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_229, slice_204], -1);  add_229 = slice_204 = None
        permute_276: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_101, [0, 2, 1, 3]);  cat_101 = None
        convert_element_type_127: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_276, torch.float32);  permute_276 = None
        expand_203: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_127, _shape_param_935);  convert_element_type_127 = _shape_param_935 = None
        view_706: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_203, _shape_param_936);  expand_203 = _shape_param_936 = None
        view_707: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_126, _shape_param_937);  _shape_param_937 = None
        permute_277: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        mm_101: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_707, permute_277);  view_707 = permute_277 = None
        view_708: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_101, _shape_param_938);  mm_101 = _shape_param_938 = None
        view_709: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_708, _shape_param_939);  view_708 = _shape_param_939 = None
        slice_205: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_709, 3, 0, 64)
        unsqueeze_341: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_103, 2);  getitem_103 = None
        unsqueeze_342: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 4);  unsqueeze_341 = None
        expand_204: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_342, _shape_param_940);  unsqueeze_342 = _shape_param_940 = None
        clone_127: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_204, memory_format = torch.contiguous_format);  expand_204 = None
        view_710: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_127, _shape_param_941);  clone_127 = _shape_param_941 = None
        mul_254: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_205, view_710);  view_710 = None
        slice_206: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_205, 3, 1, 9223372036854775807, 2)
        neg_51: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_206);  slice_206 = None
        unsqueeze_343: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_51, 4);  neg_51 = None
        slice_207: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_205, 3, 0, 9223372036854775807, 2);  slice_205 = None
        unsqueeze_344: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_207, 4);  slice_207 = None
        cat_102: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_343, unsqueeze_344], -1);  unsqueeze_343 = unsqueeze_344 = None
        view_711: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_102, _shape_param_942);  cat_102 = _shape_param_942 = None
        unsqueeze_345: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_102, 2);  getitem_102 = None
        unsqueeze_346: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 4);  unsqueeze_345 = None
        expand_205: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_346, _shape_param_943);  unsqueeze_346 = _shape_param_943 = None
        clone_128: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_205, memory_format = torch.contiguous_format);  expand_205 = None
        view_712: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_128, _shape_param_944);  clone_128 = _shape_param_944 = None
        mul_255: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_711, view_712);  view_711 = view_712 = None
        add_230: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None
        slice_208: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_709, 3, 64, 9223372036854775807);  view_709 = None
        cat_103: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_230, slice_208], -1);  add_230 = slice_208 = None
        permute_278: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_103, [0, 2, 1, 3]);  cat_103 = None
        convert_element_type_128: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_278, torch.float32);  permute_278 = None
        permute_279: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_128, [0, 1, 3, 2]);  convert_element_type_128 = None
        expand_206: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_279, _shape_param_945);  permute_279 = _shape_param_945 = None
        view_713: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_206, _shape_param_946);  expand_206 = _shape_param_946 = None
        bmm_50: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_706, view_713);  view_706 = view_713 = None
        view_714: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_50, _shape_param_947);  bmm_50 = _shape_param_947 = None
        div_50: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_714, 16.0);  view_714 = None
        add_231: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_50, where);  div_50 = None
        amax_25: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_231, [-1], True)
        sub_51: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_231, amax_25);  add_231 = amax_25 = None
        exp_25: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
        sum_26: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_51: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
        convert_element_type_129: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_51, torch.bfloat16);  div_51 = None
        expand_207: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_129, _shape_param_948);  convert_element_type_129 = _shape_param_948 = None
        view_715: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_207, _shape_param_949);  expand_207 = _shape_param_949 = None
        view_716: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_126, _shape_param_950);  _shape_param_950 = None
        permute_280: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
        mm_102: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_716, permute_280);  view_716 = permute_280 = None
        view_717: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_102, _shape_param_951);  mm_102 = _shape_param_951 = None
        view_718: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_717, _shape_param_952);  view_717 = _shape_param_952 = None
        permute_281: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_718, [0, 2, 1, 3]);  view_718 = None
        expand_208: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_281, _shape_param_953);  permute_281 = _shape_param_953 = None
        view_719: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_208, _shape_param_954);  expand_208 = _shape_param_954 = None
        bmm_51: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_715, view_719);  view_715 = view_719 = None
        view_720: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_51, _shape_param_955);  bmm_51 = _shape_param_955 = None
        permute_282: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_720, [0, 2, 1, 3]);  view_720 = None
        clone_129: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_282, memory_format = torch.contiguous_format);  permute_282 = None
        view_721: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_129, _shape_param_956);  clone_129 = _shape_param_956 = None
        view_722: "bf16[128, 4096]" = torch.ops.aten.view.default(view_721, _shape_param_957);  view_721 = _shape_param_957 = None
        permute_283: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        mm_103: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_722, permute_283);  view_722 = permute_283 = None
        view_723: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_103, _shape_param_958);  mm_103 = _shape_param_958 = None
        view_724: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_126, _shape_param_959);  convert_element_type_126 = _shape_param_959 = None
        permute_284: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
        addmm_50: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg287_1, view_724, permute_284);  arg287_1 = view_724 = permute_284 = None
        view_725: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_50, _shape_param_960);  addmm_50 = _shape_param_960 = None
        mul_256: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_725, 0.5)
        pow_26: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_725, 3.0)
        mul_257: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_26, 0.044715);  pow_26 = None
        add_232: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_725, mul_257);  view_725 = mul_257 = None
        mul_258: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_232, 0.7978845608028654);  add_232 = None
        tanh_25: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_258);  mul_258 = None
        add_233: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_25, 1.0);  tanh_25 = None
        mul_259: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_256, add_233);  mul_256 = add_233 = None
        view_726: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_259, _shape_param_961);  mul_259 = _shape_param_961 = None
        permute_285: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
        addmm_51: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg289_1, view_726, permute_285);  arg289_1 = view_726 = permute_285 = None
        view_727: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_51, _shape_param_962);  addmm_51 = _shape_param_962 = None
        add_234: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_723, view_727);  view_723 = view_727 = None
        add_235: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_234, add_226);  add_234 = add_226 = None
        convert_element_type_130: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_235, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_130, [2], correction = 0, keepdim = True)
        getitem_104: "f32[1, 128, 1]" = var_mean_26[0]
        getitem_105: "f32[1, 128, 1]" = var_mean_26[1];  var_mean_26 = None
        repeat_52: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg290_1, [1, 1, 1]);  arg290_1 = None
        unsqueeze_347: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1)
        repeat_53: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_347, _shape_param_963);  unsqueeze_347 = _shape_param_963 = None
        gather_26: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_52, 1, repeat_53);  repeat_52 = repeat_53 = None
        split_26 = torch.ops.aten.split.Tensor(gather_26, 32, -1);  gather_26 = None
        getitem_106: "bf16[1, 128, 32]" = split_26[0]
        getitem_107: "bf16[1, 128, 32]" = split_26[1];  split_26 = None
        sub_52: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_130, getitem_105);  convert_element_type_130 = getitem_105 = None
        add_236: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_26: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        mul_260: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_26);  sub_52 = rsqrt_26 = None
        mul_261: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_260, arg291_1);  mul_260 = arg291_1 = None
        add_237: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_261, arg292_1);  mul_261 = arg292_1 = None
        convert_element_type_131: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_237, torch.bfloat16);  add_237 = None
        view_728: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_131, _shape_param_964);  _shape_param_964 = None
        permute_286: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        mm_104: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_728, permute_286);  view_728 = permute_286 = None
        view_729: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_104, _shape_param_965);  mm_104 = _shape_param_965 = None
        view_730: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_729, _shape_param_966);  view_729 = _shape_param_966 = None
        slice_209: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_730, 3, 0, 64)
        unsqueeze_348: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_107, 2)
        unsqueeze_349: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 4);  unsqueeze_348 = None
        expand_209: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_349, _shape_param_967);  unsqueeze_349 = _shape_param_967 = None
        clone_130: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_209, memory_format = torch.contiguous_format);  expand_209 = None
        view_731: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_130, _shape_param_968);  clone_130 = _shape_param_968 = None
        mul_262: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_209, view_731);  view_731 = None
        slice_210: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_209, 3, 1, 9223372036854775807, 2)
        neg_52: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_210);  slice_210 = None
        unsqueeze_350: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_52, 4);  neg_52 = None
        slice_211: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_209, 3, 0, 9223372036854775807, 2);  slice_209 = None
        unsqueeze_351: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_211, 4);  slice_211 = None
        cat_104: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_350, unsqueeze_351], -1);  unsqueeze_350 = unsqueeze_351 = None
        view_732: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_104, _shape_param_969);  cat_104 = _shape_param_969 = None
        unsqueeze_352: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_106, 2)
        unsqueeze_353: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 4);  unsqueeze_352 = None
        expand_210: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_353, _shape_param_970);  unsqueeze_353 = _shape_param_970 = None
        clone_131: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_210, memory_format = torch.contiguous_format);  expand_210 = None
        view_733: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_131, _shape_param_971);  clone_131 = _shape_param_971 = None
        mul_263: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_732, view_733);  view_732 = view_733 = None
        add_238: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_262, mul_263);  mul_262 = mul_263 = None
        slice_212: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_730, 3, 64, 9223372036854775807);  view_730 = None
        cat_105: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_238, slice_212], -1);  add_238 = slice_212 = None
        permute_287: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_105, [0, 2, 1, 3]);  cat_105 = None
        convert_element_type_132: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_287, torch.float32);  permute_287 = None
        expand_211: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_132, _shape_param_972);  convert_element_type_132 = _shape_param_972 = None
        view_734: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_211, _shape_param_973);  expand_211 = _shape_param_973 = None
        view_735: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_131, _shape_param_974);  _shape_param_974 = None
        permute_288: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
        mm_105: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_735, permute_288);  view_735 = permute_288 = None
        view_736: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_105, _shape_param_975);  mm_105 = _shape_param_975 = None
        view_737: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_736, _shape_param_976);  view_736 = _shape_param_976 = None
        slice_213: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_737, 3, 0, 64)
        unsqueeze_354: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_107, 2);  getitem_107 = None
        unsqueeze_355: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 4);  unsqueeze_354 = None
        expand_212: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_355, _shape_param_977);  unsqueeze_355 = _shape_param_977 = None
        clone_132: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_212, memory_format = torch.contiguous_format);  expand_212 = None
        view_738: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_132, _shape_param_978);  clone_132 = _shape_param_978 = None
        mul_264: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_213, view_738);  view_738 = None
        slice_214: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_213, 3, 1, 9223372036854775807, 2)
        neg_53: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_214);  slice_214 = None
        unsqueeze_356: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_53, 4);  neg_53 = None
        slice_215: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_213, 3, 0, 9223372036854775807, 2);  slice_213 = None
        unsqueeze_357: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_215, 4);  slice_215 = None
        cat_106: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_356, unsqueeze_357], -1);  unsqueeze_356 = unsqueeze_357 = None
        view_739: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_106, _shape_param_979);  cat_106 = _shape_param_979 = None
        unsqueeze_358: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_106, 2);  getitem_106 = None
        unsqueeze_359: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 4);  unsqueeze_358 = None
        expand_213: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_359, _shape_param_980);  unsqueeze_359 = _shape_param_980 = None
        clone_133: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_213, memory_format = torch.contiguous_format);  expand_213 = None
        view_740: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_133, _shape_param_981);  clone_133 = _shape_param_981 = None
        mul_265: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_739, view_740);  view_739 = view_740 = None
        add_239: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None
        slice_216: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_737, 3, 64, 9223372036854775807);  view_737 = None
        cat_107: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_239, slice_216], -1);  add_239 = slice_216 = None
        permute_289: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_107, [0, 2, 1, 3]);  cat_107 = None
        convert_element_type_133: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_289, torch.float32);  permute_289 = None
        permute_290: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_133, [0, 1, 3, 2]);  convert_element_type_133 = None
        expand_214: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_290, _shape_param_982);  permute_290 = _shape_param_982 = None
        view_741: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_214, _shape_param_983);  expand_214 = _shape_param_983 = None
        bmm_52: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_734, view_741);  view_734 = view_741 = None
        view_742: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_52, _shape_param_984);  bmm_52 = _shape_param_984 = None
        div_52: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_742, 16.0);  view_742 = None
        add_240: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_52, where);  div_52 = None
        amax_26: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_240, [-1], True)
        sub_53: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_240, amax_26);  add_240 = amax_26 = None
        exp_26: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_53);  sub_53 = None
        sum_27: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
        div_53: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
        convert_element_type_134: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_53, torch.bfloat16);  div_53 = None
        expand_215: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_134, _shape_param_985);  convert_element_type_134 = _shape_param_985 = None
        view_743: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_215, _shape_param_986);  expand_215 = _shape_param_986 = None
        view_744: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_131, _shape_param_987);  _shape_param_987 = None
        permute_291: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        mm_106: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_744, permute_291);  view_744 = permute_291 = None
        view_745: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_106, _shape_param_988);  mm_106 = _shape_param_988 = None
        view_746: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_745, _shape_param_989);  view_745 = _shape_param_989 = None
        permute_292: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_746, [0, 2, 1, 3]);  view_746 = None
        expand_216: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_292, _shape_param_990);  permute_292 = _shape_param_990 = None
        view_747: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_216, _shape_param_991);  expand_216 = _shape_param_991 = None
        bmm_53: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_743, view_747);  view_743 = view_747 = None
        view_748: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_53, _shape_param_992);  bmm_53 = _shape_param_992 = None
        permute_293: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_748, [0, 2, 1, 3]);  view_748 = None
        clone_134: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
        view_749: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_134, _shape_param_993);  clone_134 = _shape_param_993 = None
        view_750: "bf16[128, 4096]" = torch.ops.aten.view.default(view_749, _shape_param_994);  view_749 = _shape_param_994 = None
        permute_294: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
        mm_107: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_750, permute_294);  view_750 = permute_294 = None
        view_751: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_107, _shape_param_995);  mm_107 = _shape_param_995 = None
        view_752: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_131, _shape_param_996);  convert_element_type_131 = _shape_param_996 = None
        permute_295: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_52: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg298_1, view_752, permute_295);  arg298_1 = view_752 = permute_295 = None
        view_753: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_52, _shape_param_997);  addmm_52 = _shape_param_997 = None
        mul_266: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_753, 0.5)
        pow_27: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_753, 3.0)
        mul_267: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_27, 0.044715);  pow_27 = None
        add_241: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_753, mul_267);  view_753 = mul_267 = None
        mul_268: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_241, 0.7978845608028654);  add_241 = None
        tanh_26: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_268);  mul_268 = None
        add_242: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_26, 1.0);  tanh_26 = None
        mul_269: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_266, add_242);  mul_266 = add_242 = None
        view_754: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_269, _shape_param_998);  mul_269 = _shape_param_998 = None
        permute_296: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        addmm_53: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg300_1, view_754, permute_296);  arg300_1 = view_754 = permute_296 = None
        view_755: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_53, _shape_param_999);  addmm_53 = _shape_param_999 = None
        add_243: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_751, view_755);  view_751 = view_755 = None
        add_244: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_243, add_235);  add_243 = add_235 = None
        convert_element_type_135: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_244, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_135, [2], correction = 0, keepdim = True)
        getitem_108: "f32[1, 128, 1]" = var_mean_27[0]
        getitem_109: "f32[1, 128, 1]" = var_mean_27[1];  var_mean_27 = None
        repeat_54: "bf16[1, 2048, 64]" = torch.ops.aten.repeat.default(arg301_1, [1, 1, 1]);  arg301_1 = None
        unsqueeze_360: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg3_1, -1);  arg3_1 = None
        repeat_55: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_360, _shape_param_1000);  unsqueeze_360 = _shape_param_1000 = None
        gather_27: "bf16[1, 128, 64]" = torch.ops.aten.gather.default(repeat_54, 1, repeat_55);  repeat_54 = repeat_55 = None
        split_27 = torch.ops.aten.split.Tensor(gather_27, 32, -1);  gather_27 = None
        getitem_110: "bf16[1, 128, 32]" = split_27[0]
        getitem_111: "bf16[1, 128, 32]" = split_27[1];  split_27 = None
        sub_54: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_135, getitem_109);  convert_element_type_135 = getitem_109 = None
        add_245: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_27: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_245);  add_245 = None
        mul_270: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_27);  sub_54 = rsqrt_27 = None
        mul_271: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_270, arg302_1);  mul_270 = arg302_1 = None
        add_246: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_271, arg303_1);  mul_271 = arg303_1 = None
        convert_element_type_136: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_246, torch.bfloat16);  add_246 = None
        view_756: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_136, _shape_param_1001);  _shape_param_1001 = None
        permute_297: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg304_1, [1, 0]);  arg304_1 = None
        mm_108: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_756, permute_297);  view_756 = permute_297 = None
        view_757: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_108, _shape_param_1002);  mm_108 = _shape_param_1002 = None
        view_758: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_757, _shape_param_1003);  view_757 = _shape_param_1003 = None
        slice_217: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_758, 3, 0, 64)
        unsqueeze_361: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_111, 2)
        unsqueeze_362: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 4);  unsqueeze_361 = None
        expand_217: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_362, _shape_param_1004);  unsqueeze_362 = _shape_param_1004 = None
        clone_135: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_217, memory_format = torch.contiguous_format);  expand_217 = None
        view_759: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_135, _shape_param_1005);  clone_135 = _shape_param_1005 = None
        mul_272: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_217, view_759);  view_759 = None
        slice_218: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_217, 3, 1, 9223372036854775807, 2)
        neg_54: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_218);  slice_218 = None
        unsqueeze_363: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_54, 4);  neg_54 = None
        slice_219: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_217, 3, 0, 9223372036854775807, 2);  slice_217 = None
        unsqueeze_364: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_219, 4);  slice_219 = None
        cat_108: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_363, unsqueeze_364], -1);  unsqueeze_363 = unsqueeze_364 = None
        view_760: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_108, _shape_param_1006);  cat_108 = _shape_param_1006 = None
        unsqueeze_365: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_110, 2)
        unsqueeze_366: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 4);  unsqueeze_365 = None
        expand_218: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_366, _shape_param_1007);  unsqueeze_366 = _shape_param_1007 = None
        clone_136: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_218, memory_format = torch.contiguous_format);  expand_218 = None
        view_761: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_136, _shape_param_1008);  clone_136 = _shape_param_1008 = None
        mul_273: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_760, view_761);  view_760 = view_761 = None
        add_247: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
        slice_220: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_758, 3, 64, 9223372036854775807);  view_758 = None
        cat_109: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_247, slice_220], -1);  add_247 = slice_220 = None
        permute_298: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_109, [0, 2, 1, 3]);  cat_109 = None
        convert_element_type_137: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_298, torch.float32);  permute_298 = None
        expand_219: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(convert_element_type_137, _shape_param_1009);  convert_element_type_137 = _shape_param_1009 = None
        view_762: "f32[16, 128, 256]" = torch.ops.aten.view.default(expand_219, _shape_param_1010);  expand_219 = _shape_param_1010 = None
        view_763: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_136, _shape_param_1011);  _shape_param_1011 = None
        permute_299: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        mm_109: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_763, permute_299);  view_763 = permute_299 = None
        view_764: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_109, _shape_param_1012);  mm_109 = _shape_param_1012 = None
        view_765: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_764, _shape_param_1013);  view_764 = _shape_param_1013 = None
        slice_221: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_765, 3, 0, 64)
        unsqueeze_367: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_111, 2);  getitem_111 = None
        unsqueeze_368: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 4);  unsqueeze_367 = None
        expand_220: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_368, _shape_param_1014);  unsqueeze_368 = _shape_param_1014 = None
        clone_137: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_220, memory_format = torch.contiguous_format);  expand_220 = None
        view_766: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_137, _shape_param_1015);  clone_137 = _shape_param_1015 = None
        mul_274: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_221, view_766);  view_766 = None
        slice_222: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_221, 3, 1, 9223372036854775807, 2)
        neg_55: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_222);  slice_222 = None
        unsqueeze_369: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_55, 4);  neg_55 = None
        slice_223: "bf16[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_221, 3, 0, 9223372036854775807, 2);  slice_221 = None
        unsqueeze_370: "bf16[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_223, 4);  slice_223 = None
        cat_110: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_369, unsqueeze_370], -1);  unsqueeze_369 = unsqueeze_370 = None
        view_767: "bf16[1, 128, 16, 64]" = torch.ops.aten.view.default(cat_110, _shape_param_1016);  cat_110 = _shape_param_1016 = None
        unsqueeze_371: "bf16[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_110, 2);  getitem_110 = None
        unsqueeze_372: "bf16[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 4);  unsqueeze_371 = None
        expand_221: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_372, _shape_param_1017);  unsqueeze_372 = _shape_param_1017 = None
        clone_138: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_221, memory_format = torch.contiguous_format);  expand_221 = None
        view_768: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_138, _shape_param_1018);  clone_138 = _shape_param_1018 = None
        mul_275: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_767, view_768);  view_767 = view_768 = None
        add_248: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
        slice_224: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_765, 3, 64, 9223372036854775807);  view_765 = None
        cat_111: "bf16[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_248, slice_224], -1);  add_248 = slice_224 = None
        permute_300: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_111, [0, 2, 1, 3]);  cat_111 = None
        convert_element_type_138: "f32[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_300, torch.float32);  permute_300 = None
        permute_301: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(convert_element_type_138, [0, 1, 3, 2]);  convert_element_type_138 = None
        expand_222: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_301, _shape_param_1019);  permute_301 = _shape_param_1019 = None
        view_769: "f32[16, 256, 128]" = torch.ops.aten.view.default(expand_222, _shape_param_1020);  expand_222 = _shape_param_1020 = None
        bmm_54: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_762, view_769);  view_762 = view_769 = None
        view_770: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_54, _shape_param_1021);  bmm_54 = _shape_param_1021 = None
        div_54: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view_770, 16.0);  view_770 = None
        add_249: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div_54, where);  div_54 = where = None
        amax_27: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add_249, [-1], True)
        sub_55: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_249, amax_27);  add_249 = amax_27 = None
        exp_27: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_28: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
        div_55: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None
        convert_element_type_139: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_55, torch.bfloat16);  div_55 = None
        expand_223: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_139, _shape_param_1022);  convert_element_type_139 = _shape_param_1022 = None
        view_771: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_223, _shape_param_1023);  expand_223 = _shape_param_1023 = None
        view_772: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_136, _shape_param_1024);  _shape_param_1024 = None
        permute_302: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg306_1, [1, 0]);  arg306_1 = None
        mm_110: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_772, permute_302);  view_772 = permute_302 = None
        view_773: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_110, _shape_param_1025);  mm_110 = _shape_param_1025 = None
        view_774: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_773, _shape_param_1026);  view_773 = _shape_param_1026 = None
        permute_303: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_774, [0, 2, 1, 3]);  view_774 = None
        expand_224: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_303, _shape_param_1027);  permute_303 = _shape_param_1027 = None
        view_775: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_224, _shape_param_1028);  expand_224 = _shape_param_1028 = None
        bmm_55: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_771, view_775);  view_771 = view_775 = None
        view_776: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_55, _shape_param_1029);  bmm_55 = _shape_param_1029 = None
        permute_304: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_776, [0, 2, 1, 3]);  view_776 = None
        clone_139: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_304, memory_format = torch.contiguous_format);  permute_304 = None
        view_777: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_139, _shape_param_1030);  clone_139 = _shape_param_1030 = None
        view_778: "bf16[128, 4096]" = torch.ops.aten.view.default(view_777, _shape_param_1031);  view_777 = _shape_param_1031 = None
        permute_305: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        mm_111: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_778, permute_305);  view_778 = permute_305 = None
        view_779: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_111, _shape_param_1032);  mm_111 = _shape_param_1032 = None
        view_780: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_136, _shape_param_1033);  convert_element_type_136 = _shape_param_1033 = None
        permute_306: "bf16[4096, 16384]" = torch.ops.aten.permute.default(arg308_1, [1, 0]);  arg308_1 = None
        addmm_54: "bf16[128, 16384]" = torch.ops.aten.addmm.default(arg309_1, view_780, permute_306);  arg309_1 = view_780 = permute_306 = None
        view_781: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(addmm_54, _shape_param_1034);  addmm_54 = _shape_param_1034 = None
        mul_276: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_781, 0.5)
        pow_28: "bf16[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_781, 3.0)
        mul_277: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_28, 0.044715);  pow_28 = None
        add_250: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_781, mul_277);  view_781 = mul_277 = None
        mul_278: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_250, 0.7978845608028654);  add_250 = None
        tanh_27: "bf16[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_278);  mul_278 = None
        add_251: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_27, 1.0);  tanh_27 = None
        mul_279: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_276, add_251);  mul_276 = add_251 = None
        view_782: "bf16[128, 16384]" = torch.ops.aten.view.default(mul_279, _shape_param_1035);  mul_279 = _shape_param_1035 = None
        permute_307: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
        addmm_55: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg311_1, view_782, permute_307);  arg311_1 = view_782 = permute_307 = None
        view_783: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm_55, _shape_param_1036);  addmm_55 = _shape_param_1036 = None
        add_252: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_779, view_783);  view_779 = view_783 = None
        add_253: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_252, add_244);  add_252 = add_244 = None
        convert_element_type_140: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_253, torch.float32);  add_253 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_140, [2], correction = 0, keepdim = True)
        getitem_112: "f32[1, 128, 1]" = var_mean_28[0]
        getitem_113: "f32[1, 128, 1]" = var_mean_28[1];  var_mean_28 = None
        sub_56: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(convert_element_type_140, getitem_113);  convert_element_type_140 = getitem_113 = None
        add_254: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
        rsqrt_28: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
        mul_280: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_28);  sub_56 = rsqrt_28 = None
        mul_281: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_280, arg312_1);  mul_280 = arg312_1 = None
        add_255: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_281, arg313_1);  mul_281 = arg313_1 = None
        convert_element_type_141: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_255, torch.bfloat16);  add_255 = None
        view_784: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_141, _shape_param_1037);  convert_element_type_141 = _shape_param_1037 = None
        permute_308: "bf16[4096, 2]" = torch.ops.aten.permute.default(arg314_1, [1, 0]);  arg314_1 = None
        addmm_56: "bf16[128, 2]" = torch.ops.aten.addmm.default(arg315_1, view_784, permute_308);  arg315_1 = view_784 = permute_308 = None
        view_785: "bf16[1, 128, 2]" = torch.ops.aten.view.default(addmm_56, _shape_param_1038);  addmm_56 = _shape_param_1038 = None
        split_28 = torch.ops.aten.split.Tensor(view_785, 1, -1);  view_785 = None
        getitem_114: "bf16[1, 128, 1]" = split_28[0]
        getitem_115: "bf16[1, 128, 1]" = split_28[1];  split_28 = None
        clamp_min: "i64[1]" = torch.ops.aten.clamp_min.default(arg316_1, 0);  arg316_1 = None
        clamp_max: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min, 128);  clamp_min = None
        ne: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 128)
        squeeze: "bf16[1, 128]" = torch.ops.aten.squeeze.dim(getitem_114, -1);  getitem_114 = None
        clone_140: "bf16[1, 128]" = torch.ops.aten.clone.default(squeeze, memory_format = torch.contiguous_format);  squeeze = None
        convert_element_type_142: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(clone_140, torch.float32)
        amax_28: "f32[1, 1]" = torch.ops.aten.amax.default(convert_element_type_142, [1], True)
        sub_57: "f32[1, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_142, amax_28);  convert_element_type_142 = amax_28 = None
        exp_28: "f32[1, 128]" = torch.ops.aten.exp.default(sub_57)
        sum_29: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [1], True);  exp_28 = None
        log: "f32[1, 1]" = torch.ops.aten.log.default(sum_29);  sum_29 = None
        sub_58: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_57, log);  sub_57 = log = None
        convert_element_type_143: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(sub_58, torch.bfloat16);  sub_58 = None
        ne_1: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 128)
        full_3: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[1]" = torch.ops.aten.where.self(ne_1, clamp_max, full_3);  ne_1 = full_3 = None
        unsqueeze_373: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather_28: "bf16[1, 1]" = torch.ops.aten.gather.default(convert_element_type_143, 1, unsqueeze_373);  convert_element_type_143 = unsqueeze_373 = None
        squeeze_1: "bf16[1]" = torch.ops.aten.squeeze.dim(gather_28, 1);  gather_28 = None
        neg_56: "bf16[1]" = torch.ops.aten.neg.default(squeeze_1);  squeeze_1 = None
        full_4: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "bf16[1]" = torch.ops.aten.where.self(ne, neg_56, full_4);  ne = neg_56 = full_4 = None
        sum_30: "bf16[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
        ne_2: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max, 128);  clamp_max = None
        sum_31: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_144: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_31, torch.bfloat16);  sum_31 = None
        div_56: "bf16[]" = torch.ops.aten.div.Tensor(sum_30, convert_element_type_144);  sum_30 = convert_element_type_144 = None
        clamp_min_1: "i64[1]" = torch.ops.aten.clamp_min.default(arg317_1, 0);  arg317_1 = None
        clamp_max_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 128);  clamp_min_1 = None
        ne_3: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 128)
        squeeze_2: "bf16[1, 128]" = torch.ops.aten.squeeze.dim(getitem_115, -1);  getitem_115 = None
        clone_141: "bf16[1, 128]" = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
        convert_element_type_145: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(clone_141, torch.float32)
        amax_29: "f32[1, 1]" = torch.ops.aten.amax.default(convert_element_type_145, [1], True)
        sub_59: "f32[1, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_145, amax_29);  convert_element_type_145 = amax_29 = None
        exp_29: "f32[1, 128]" = torch.ops.aten.exp.default(sub_59)
        sum_32: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(exp_29, [1], True);  exp_29 = None
        log_1: "f32[1, 1]" = torch.ops.aten.log.default(sum_32);  sum_32 = None
        sub_60: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_59, log_1);  sub_59 = log_1 = None
        convert_element_type_146: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(sub_60, torch.bfloat16);  sub_60 = None
        ne_4: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 128)
        full_5: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "i64[1]" = torch.ops.aten.where.self(ne_4, clamp_max_1, full_5);  ne_4 = full_5 = None
        unsqueeze_374: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(where_3, 1);  where_3 = None
        gather_29: "bf16[1, 1]" = torch.ops.aten.gather.default(convert_element_type_146, 1, unsqueeze_374);  convert_element_type_146 = unsqueeze_374 = None
        squeeze_3: "bf16[1]" = torch.ops.aten.squeeze.dim(gather_29, 1);  gather_29 = None
        neg_57: "bf16[1]" = torch.ops.aten.neg.default(squeeze_3);  squeeze_3 = None
        full_6: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "bf16[1]" = torch.ops.aten.where.self(ne_3, neg_57, full_6);  ne_3 = neg_57 = full_6 = None
        sum_33: "bf16[]" = torch.ops.aten.sum.default(where_4);  where_4 = None
        ne_5: "b8[1]" = torch.ops.aten.ne.Scalar(clamp_max_1, 128);  clamp_max_1 = None
        sum_34: "i64[]" = torch.ops.aten.sum.default(ne_5);  ne_5 = None
        convert_element_type_147: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_34, torch.bfloat16);  sum_34 = None
        div_57: "bf16[]" = torch.ops.aten.div.Tensor(sum_33, convert_element_type_147);  sum_33 = convert_element_type_147 = None
        add_256: "bf16[]" = torch.ops.aten.add.Tensor(div_56, div_57);  div_56 = div_57 = None
        div_58: "bf16[]" = torch.ops.aten.div.Tensor(add_256, 2);  add_256 = None
        return (clone_140, clone_141, div_58)



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
