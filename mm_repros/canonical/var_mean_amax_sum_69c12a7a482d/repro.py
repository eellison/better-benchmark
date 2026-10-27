"""
Standalone repro captured via capture_hook.
Label: GPTNeoForSequenceClassification
Pattern hash: 69c12a7a482d
Shape hash: 98034570
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
    def forward(self, arg0_1: "f32[50257, 2048]", arg1_1: "i64[32, 128]", arg2_1: "i64[1, 128]", arg3_1: "i64[32, 128]", arg4_1: "f32[2048, 2048]", arg5_1: "f32[2048]", arg6_1: "f32[2048]", arg7_1: "f32[2048, 2048]", arg8_1: "f32[2048, 2048]", arg9_1: "f32[2048, 2048]", arg10_1: "b8[1, 1, 2048, 2048]", arg11_1: "f32[2048]", arg12_1: "f32[2048, 2048]", arg13_1: "f32[2048]", arg14_1: "f32[2048]", arg15_1: "f32[8192]", arg16_1: "f32[8192, 2048]", arg17_1: "f32[2048]", arg18_1: "f32[2048, 8192]", arg19_1: "f32[2048]", arg20_1: "f32[2048]", arg21_1: "f32[2048, 2048]", arg22_1: "f32[2048, 2048]", arg23_1: "f32[2048, 2048]", arg24_1: "b8[1, 1, 2048, 2048]", arg25_1: "f32[2048]", arg26_1: "f32[2048, 2048]", arg27_1: "f32[2048]", arg28_1: "f32[2048]", arg29_1: "f32[8192]", arg30_1: "f32[8192, 2048]", arg31_1: "f32[2048]", arg32_1: "f32[2048, 8192]", arg33_1: "f32[2048]", arg34_1: "f32[2048]", arg35_1: "f32[2048, 2048]", arg36_1: "f32[2048, 2048]", arg37_1: "f32[2048, 2048]", arg38_1: "b8[1, 1, 2048, 2048]", arg39_1: "f32[2048]", arg40_1: "f32[2048, 2048]", arg41_1: "f32[2048]", arg42_1: "f32[2048]", arg43_1: "f32[8192]", arg44_1: "f32[8192, 2048]", arg45_1: "f32[2048]", arg46_1: "f32[2048, 8192]", arg47_1: "f32[2048]", arg48_1: "f32[2048]", arg49_1: "f32[2048, 2048]", arg50_1: "f32[2048, 2048]", arg51_1: "f32[2048, 2048]", arg52_1: "b8[1, 1, 2048, 2048]", arg53_1: "f32[2048]", arg54_1: "f32[2048, 2048]", arg55_1: "f32[2048]", arg56_1: "f32[2048]", arg57_1: "f32[8192]", arg58_1: "f32[8192, 2048]", arg59_1: "f32[2048]", arg60_1: "f32[2048, 8192]", arg61_1: "f32[2048]", arg62_1: "f32[2048]", arg63_1: "f32[2048, 2048]", arg64_1: "f32[2048, 2048]", arg65_1: "f32[2048, 2048]", arg66_1: "b8[1, 1, 2048, 2048]", arg67_1: "f32[2048]", arg68_1: "f32[2048, 2048]", arg69_1: "f32[2048]", arg70_1: "f32[2048]", arg71_1: "f32[8192]", arg72_1: "f32[8192, 2048]", arg73_1: "f32[2048]", arg74_1: "f32[2048, 8192]", arg75_1: "f32[2048]", arg76_1: "f32[2048]", arg77_1: "f32[2048, 2048]", arg78_1: "f32[2048, 2048]", arg79_1: "f32[2048, 2048]", arg80_1: "b8[1, 1, 2048, 2048]", arg81_1: "f32[2048]", arg82_1: "f32[2048, 2048]", arg83_1: "f32[2048]", arg84_1: "f32[2048]", arg85_1: "f32[8192]", arg86_1: "f32[8192, 2048]", arg87_1: "f32[2048]", arg88_1: "f32[2048, 8192]", arg89_1: "f32[2048]", arg90_1: "f32[2048]", arg91_1: "f32[2048, 2048]", arg92_1: "f32[2048, 2048]", arg93_1: "f32[2048, 2048]", arg94_1: "b8[1, 1, 2048, 2048]", arg95_1: "f32[2048]", arg96_1: "f32[2048, 2048]", arg97_1: "f32[2048]", arg98_1: "f32[2048]", arg99_1: "f32[8192]", arg100_1: "f32[8192, 2048]", arg101_1: "f32[2048]", arg102_1: "f32[2048, 8192]", arg103_1: "f32[2048]", arg104_1: "f32[2048]", arg105_1: "f32[2048, 2048]", arg106_1: "f32[2048, 2048]", arg107_1: "f32[2048, 2048]", arg108_1: "b8[1, 1, 2048, 2048]", arg109_1: "f32[2048]", arg110_1: "f32[2048, 2048]", arg111_1: "f32[2048]", arg112_1: "f32[2048]", arg113_1: "f32[8192]", arg114_1: "f32[8192, 2048]", arg115_1: "f32[2048]", arg116_1: "f32[2048, 8192]", arg117_1: "f32[2048]", arg118_1: "f32[2048]", arg119_1: "f32[2048, 2048]", arg120_1: "f32[2048, 2048]", arg121_1: "f32[2048, 2048]", arg122_1: "b8[1, 1, 2048, 2048]", arg123_1: "f32[2048]", arg124_1: "f32[2048, 2048]", arg125_1: "f32[2048]", arg126_1: "f32[2048]", arg127_1: "f32[8192]", arg128_1: "f32[8192, 2048]", arg129_1: "f32[2048]", arg130_1: "f32[2048, 8192]", arg131_1: "f32[2048]", arg132_1: "f32[2048]", arg133_1: "f32[2048, 2048]", arg134_1: "f32[2048, 2048]", arg135_1: "f32[2048, 2048]", arg136_1: "b8[1, 1, 2048, 2048]", arg137_1: "f32[2048]", arg138_1: "f32[2048, 2048]", arg139_1: "f32[2048]", arg140_1: "f32[2048]", arg141_1: "f32[8192]", arg142_1: "f32[8192, 2048]", arg143_1: "f32[2048]", arg144_1: "f32[2048, 8192]", arg145_1: "f32[2048]", arg146_1: "f32[2048]", arg147_1: "f32[2048, 2048]", arg148_1: "f32[2048, 2048]", arg149_1: "f32[2048, 2048]", arg150_1: "b8[1, 1, 2048, 2048]", arg151_1: "f32[2048]", arg152_1: "f32[2048, 2048]", arg153_1: "f32[2048]", arg154_1: "f32[2048]", arg155_1: "f32[8192]", arg156_1: "f32[8192, 2048]", arg157_1: "f32[2048]", arg158_1: "f32[2048, 8192]", arg159_1: "f32[2048]", arg160_1: "f32[2048]", arg161_1: "f32[2048, 2048]", arg162_1: "f32[2048, 2048]", arg163_1: "f32[2048, 2048]", arg164_1: "b8[1, 1, 2048, 2048]", arg165_1: "f32[2048]", arg166_1: "f32[2048, 2048]", arg167_1: "f32[2048]", arg168_1: "f32[2048]", arg169_1: "f32[8192]", arg170_1: "f32[8192, 2048]", arg171_1: "f32[2048]", arg172_1: "f32[2048, 8192]", arg173_1: "f32[2048]", arg174_1: "f32[2048]", arg175_1: "f32[2048, 2048]", arg176_1: "f32[2048, 2048]", arg177_1: "f32[2048, 2048]", arg178_1: "b8[1, 1, 2048, 2048]", arg179_1: "f32[2048]", arg180_1: "f32[2048, 2048]", arg181_1: "f32[2048]", arg182_1: "f32[2048]", arg183_1: "f32[8192]", arg184_1: "f32[8192, 2048]", arg185_1: "f32[2048]", arg186_1: "f32[2048, 8192]", arg187_1: "f32[2048]", arg188_1: "f32[2048]", arg189_1: "f32[2048, 2048]", arg190_1: "f32[2048, 2048]", arg191_1: "f32[2048, 2048]", arg192_1: "b8[1, 1, 2048, 2048]", arg193_1: "f32[2048]", arg194_1: "f32[2048, 2048]", arg195_1: "f32[2048]", arg196_1: "f32[2048]", arg197_1: "f32[8192]", arg198_1: "f32[8192, 2048]", arg199_1: "f32[2048]", arg200_1: "f32[2048, 8192]", arg201_1: "f32[2048]", arg202_1: "f32[2048]", arg203_1: "f32[2048, 2048]", arg204_1: "f32[2048, 2048]", arg205_1: "f32[2048, 2048]", arg206_1: "b8[1, 1, 2048, 2048]", arg207_1: "f32[2048]", arg208_1: "f32[2048, 2048]", arg209_1: "f32[2048]", arg210_1: "f32[2048]", arg211_1: "f32[8192]", arg212_1: "f32[8192, 2048]", arg213_1: "f32[2048]", arg214_1: "f32[2048, 8192]", arg215_1: "f32[2048]", arg216_1: "f32[2048]", arg217_1: "f32[2048, 2048]", arg218_1: "f32[2048, 2048]", arg219_1: "f32[2048, 2048]", arg220_1: "b8[1, 1, 2048, 2048]", arg221_1: "f32[2048]", arg222_1: "f32[2048, 2048]", arg223_1: "f32[2048]", arg224_1: "f32[2048]", arg225_1: "f32[8192]", arg226_1: "f32[8192, 2048]", arg227_1: "f32[2048]", arg228_1: "f32[2048, 8192]", arg229_1: "f32[2048]", arg230_1: "f32[2048]", arg231_1: "f32[2048, 2048]", arg232_1: "f32[2048, 2048]", arg233_1: "f32[2048, 2048]", arg234_1: "b8[1, 1, 2048, 2048]", arg235_1: "f32[2048]", arg236_1: "f32[2048, 2048]", arg237_1: "f32[2048]", arg238_1: "f32[2048]", arg239_1: "f32[8192]", arg240_1: "f32[8192, 2048]", arg241_1: "f32[2048]", arg242_1: "f32[2048, 8192]", arg243_1: "f32[2048]", arg244_1: "f32[2048]", arg245_1: "f32[2048, 2048]", arg246_1: "f32[2048, 2048]", arg247_1: "f32[2048, 2048]", arg248_1: "b8[1, 1, 2048, 2048]", arg249_1: "f32[2048]", arg250_1: "f32[2048, 2048]", arg251_1: "f32[2048]", arg252_1: "f32[2048]", arg253_1: "f32[8192]", arg254_1: "f32[8192, 2048]", arg255_1: "f32[2048]", arg256_1: "f32[2048, 8192]", arg257_1: "f32[2048]", arg258_1: "f32[2048]", arg259_1: "f32[2048, 2048]", arg260_1: "f32[2048, 2048]", arg261_1: "f32[2048, 2048]", arg262_1: "b8[1, 1, 2048, 2048]", arg263_1: "f32[2048]", arg264_1: "f32[2048, 2048]", arg265_1: "f32[2048]", arg266_1: "f32[2048]", arg267_1: "f32[8192]", arg268_1: "f32[8192, 2048]", arg269_1: "f32[2048]", arg270_1: "f32[2048, 8192]", arg271_1: "f32[2048]", arg272_1: "f32[2048]", arg273_1: "f32[2048, 2048]", arg274_1: "f32[2048, 2048]", arg275_1: "f32[2048, 2048]", arg276_1: "b8[1, 1, 2048, 2048]", arg277_1: "f32[2048]", arg278_1: "f32[2048, 2048]", arg279_1: "f32[2048]", arg280_1: "f32[2048]", arg281_1: "f32[8192]", arg282_1: "f32[8192, 2048]", arg283_1: "f32[2048]", arg284_1: "f32[2048, 8192]", arg285_1: "f32[2048]", arg286_1: "f32[2048]", arg287_1: "f32[2048, 2048]", arg288_1: "f32[2048, 2048]", arg289_1: "f32[2048, 2048]", arg290_1: "b8[1, 1, 2048, 2048]", arg291_1: "f32[2048]", arg292_1: "f32[2048, 2048]", arg293_1: "f32[2048]", arg294_1: "f32[2048]", arg295_1: "f32[8192]", arg296_1: "f32[8192, 2048]", arg297_1: "f32[2048]", arg298_1: "f32[2048, 8192]", arg299_1: "f32[2048]", arg300_1: "f32[2048]", arg301_1: "f32[2048, 2048]", arg302_1: "f32[2048, 2048]", arg303_1: "f32[2048, 2048]", arg304_1: "b8[1, 1, 2048, 2048]", arg305_1: "f32[2048]", arg306_1: "f32[2048, 2048]", arg307_1: "f32[2048]", arg308_1: "f32[2048]", arg309_1: "f32[8192]", arg310_1: "f32[8192, 2048]", arg311_1: "f32[2048]", arg312_1: "f32[2048, 8192]", arg313_1: "f32[2048]", arg314_1: "f32[2048]", arg315_1: "f32[2048, 2048]", arg316_1: "f32[2048, 2048]", arg317_1: "f32[2048, 2048]", arg318_1: "b8[1, 1, 2048, 2048]", arg319_1: "f32[2048]", arg320_1: "f32[2048, 2048]", arg321_1: "f32[2048]", arg322_1: "f32[2048]", arg323_1: "f32[8192]", arg324_1: "f32[8192, 2048]", arg325_1: "f32[2048]", arg326_1: "f32[2048, 8192]", arg327_1: "f32[2048]", arg328_1: "f32[2048]", arg329_1: "f32[2048, 2048]", arg330_1: "f32[2048, 2048]", arg331_1: "f32[2048, 2048]", arg332_1: "b8[1, 1, 2048, 2048]", arg333_1: "f32[2048]", arg334_1: "f32[2048, 2048]", arg335_1: "f32[2048]", arg336_1: "f32[2048]", arg337_1: "f32[8192]", arg338_1: "f32[8192, 2048]", arg339_1: "f32[2048]", arg340_1: "f32[2048, 8192]", arg341_1: "f32[2048]", arg342_1: "f32[2048]", arg343_1: "f32[2, 2048]", arg344_1: "i64[32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580):
        # No stacktrace found for following nodes
        embedding: "f32[32, 128, 2048]" = torch.ops.aten.embedding.default(arg0_1, arg1_1);  arg0_1 = None
        iota: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(iota, 1)
        unsqueeze_1: "i64[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "i64[32, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        unsqueeze_3: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        unsqueeze_4: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 3)
        unsqueeze_5: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        full: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_4)
        bitwise_and: "b8[1, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(full, le);  full = le = None
        index: "i64[32, 1, 128, 1]" = torch.ops.aten.index.Tensor(arg3_1, [unsqueeze_2, unsqueeze_4]);  unsqueeze_4 = None
        index_1: "i64[32, 1, 1, 128]" = torch.ops.aten.index.Tensor(arg3_1, [unsqueeze_2, unsqueeze_5]);  arg3_1 = unsqueeze_2 = unsqueeze_5 = None
        eq: "b8[32, 1, 128, 128]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None
        bitwise_and_1: "b8[32, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None
        expand: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(bitwise_and_1, _shape_param_0);  bitwise_and_1 = _shape_param_0 = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_2: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_1, full_2);  expand = full_2 = None
        embedding_1: "f32[1, 128, 2048]" = torch.ops.aten.embedding.default(arg4_1, arg2_1);  arg4_1 = arg2_1 = None
        add: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(embedding, embedding_1)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean[0]
        getitem_1: "f32[32, 128, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add, getitem_1)
        mul: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul, arg5_1);  mul = arg5_1 = None
        add_2: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_1, arg6_1);  mul_1 = arg6_1 = None
        convert_element_type: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        convert_element_type_1: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        permute: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        view: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  convert_element_type_1 = _shape_param_1 = None
        mm: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view, permute)
        view_1: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        convert_element_type_2: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        permute_1: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_2, [1, 0]);  convert_element_type_2 = None
        mm_1: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view, permute_1)
        view_2: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_1, _shape_param_3);  mm_1 = _shape_param_3 = None
        convert_element_type_3: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        permute_2: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0]);  convert_element_type_3 = None
        mm_2: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view, permute_2)
        view_3: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_2, _shape_param_4);  mm_2 = _shape_param_4 = None
        view_4: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_1, _shape_param_5);  view_1 = _shape_param_5 = None
        permute_3: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        view_5: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_2, _shape_param_6);  view_2 = _shape_param_6 = None
        permute_4: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        view_6: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_3, _shape_param_7);  view_3 = _shape_param_7 = None
        permute_5: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        convert_element_type_4: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_4, torch.float32);  permute_4 = None
        permute_6: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_4, [0, 1, 3, 2]);  convert_element_type_4 = None
        convert_element_type_5: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_6, torch.bfloat16);  permute_6 = None
        convert_element_type_6: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_3, torch.bfloat16);  permute_3 = None
        expand_1: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_6, _shape_param_8);  convert_element_type_6 = _shape_param_8 = None
        clone: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_7: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone, _shape_param_9);  clone = _shape_param_9 = None
        expand_2: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_5, _shape_param_10);  convert_element_type_5 = _shape_param_10 = None
        clone_1: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_8: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_11);  clone_1 = _shape_param_11 = None
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_7, view_8)
        view_9: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_12);  _shape_param_12 = None
        slice_1: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg10_1, 2, 0, 128);  arg10_1 = None
        slice_2: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 128);  slice_1 = None
        full_3: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_2, view_9, full_3);  slice_2 = view_9 = None
        add_3: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_1, where);  where_1 = None
        amax: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_3, [-1], True)
        sub_1: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_3, amax);  add_3 = None
        exp: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_7: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand_3: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_7, _shape_param_13);  convert_element_type_7 = _shape_param_13 = None
        view_10: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_3, _shape_param_14);  expand_3 = _shape_param_14 = None
        expand_4: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_5, _shape_param_15);  permute_5 = _shape_param_15 = None
        clone_2: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_11: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_2, _shape_param_16);  clone_2 = _shape_param_16 = None
        bmm_1: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_10, view_11)
        view_12: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_17);  bmm_1 = _shape_param_17 = None
        permute_7: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        clone_3: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_13: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_3, _shape_param_18);  clone_3 = _shape_param_18 = None
        convert_element_type_8: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.bfloat16);  arg11_1 = None
        convert_element_type_9: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.bfloat16);  arg12_1 = None
        view_14: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_13, _shape_param_19);  view_13 = _shape_param_19 = None
        permute_8: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_9, [1, 0]);  convert_element_type_9 = None
        addmm: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_8, view_14, permute_8);  convert_element_type_8 = None
        view_15: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm, _shape_param_20);  addmm = _shape_param_20 = None
        add_4: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_15, add);  view_15 = add = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 128, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_2: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_4, getitem_3);  getitem_3 = None
        mul_2: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_3: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_2, arg13_1);  arg13_1 = None
        add_6: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_3, arg14_1);  mul_3 = arg14_1 = None
        convert_element_type_10: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg15_1, torch.bfloat16);  arg15_1 = None
        convert_element_type_11: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.bfloat16);  arg16_1 = None
        convert_element_type_12: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        view_16: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_12, _shape_param_21);  convert_element_type_12 = _shape_param_21 = None
        permute_9: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_11, [1, 0]);  convert_element_type_11 = None
        addmm_1: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_10, view_16, permute_9);  convert_element_type_10 = None
        view_17: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_1, _shape_param_22);  _shape_param_22 = None
        mul_4: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_17, 0.5)
        convert_element_type_13: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_17, torch.float32)
        pow_1: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_13, 3.0);  convert_element_type_13 = None
        mul_5: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_7: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_17, mul_5);  view_17 = mul_5 = None
        mul_6: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_8: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_4, add_8);  mul_4 = add_8 = None
        convert_element_type_14: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        convert_element_type_15: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.bfloat16);  arg18_1 = None
        convert_element_type_16: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        view_18: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_16, _shape_param_23);  convert_element_type_16 = _shape_param_23 = None
        permute_10: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_15, [1, 0]);  convert_element_type_15 = None
        addmm_2: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_14, view_18, permute_10);  convert_element_type_14 = None
        view_19: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_2, _shape_param_24);  addmm_2 = _shape_param_24 = None
        add_9: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_4, view_19);  add_4 = view_19 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 128, 1]" = var_mean_2[0]
        getitem_5: "f32[32, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        add_10: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_3: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_9, getitem_5);  getitem_5 = None
        mul_8: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_9: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_8, arg19_1);  arg19_1 = None
        add_11: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_9, arg20_1);  mul_9 = arg20_1 = None
        convert_element_type_17: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg21_1, torch.bfloat16);  arg21_1 = None
        convert_element_type_18: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None
        permute_11: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_17, [1, 0]);  convert_element_type_17 = None
        view_20: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_18, _shape_param_25);  convert_element_type_18 = _shape_param_25 = None
        mm_3: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_20, permute_11)
        view_21: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_3, _shape_param_26);  mm_3 = _shape_param_26 = None
        convert_element_type_19: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg22_1, torch.bfloat16);  arg22_1 = None
        permute_12: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_19, [1, 0]);  convert_element_type_19 = None
        mm_4: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_20, permute_12)
        view_22: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_4, _shape_param_27);  mm_4 = _shape_param_27 = None
        convert_element_type_20: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg23_1, torch.bfloat16);  arg23_1 = None
        permute_13: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_20, [1, 0]);  convert_element_type_20 = None
        mm_5: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_20, permute_13)
        view_23: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_5, _shape_param_28);  mm_5 = _shape_param_28 = None
        view_24: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_21, _shape_param_29);  view_21 = _shape_param_29 = None
        permute_14: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        view_25: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_22, _shape_param_30);  view_22 = _shape_param_30 = None
        permute_15: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        view_26: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_23, _shape_param_31);  view_23 = _shape_param_31 = None
        permute_16: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        convert_element_type_21: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_15, torch.float32);  permute_15 = None
        permute_17: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_21, [0, 1, 3, 2]);  convert_element_type_21 = None
        convert_element_type_22: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_17, torch.bfloat16);  permute_17 = None
        convert_element_type_23: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_14, torch.bfloat16);  permute_14 = None
        expand_5: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_23, _shape_param_32);  convert_element_type_23 = _shape_param_32 = None
        clone_4: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_27: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_4, _shape_param_33);  clone_4 = _shape_param_33 = None
        expand_6: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_22, _shape_param_34);  convert_element_type_22 = _shape_param_34 = None
        clone_5: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_28: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_5, _shape_param_35);  clone_5 = _shape_param_35 = None
        bmm_2: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_27, view_28)
        view_29: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_36);  bmm_2 = _shape_param_36 = None
        slice_3: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg24_1, 2, 0, 128);  arg24_1 = None
        slice_4: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 128);  slice_3 = None
        where_2: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_4, view_29, full_3);  slice_4 = view_29 = None
        add_12: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_2, where);  where_2 = None
        amax_1: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_12, [-1], True)
        sub_4: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_12, amax_1);  add_12 = amax_1 = None
        exp_1: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_24: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16)
        expand_7: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_24, _shape_param_37);  convert_element_type_24 = _shape_param_37 = None
        view_30: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_7, _shape_param_38);  expand_7 = _shape_param_38 = None
        expand_8: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_16, _shape_param_39);  permute_16 = _shape_param_39 = None
        clone_6: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_31: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_6, _shape_param_40);  clone_6 = _shape_param_40 = None
        bmm_3: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_30, view_31)
        view_32: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_3, _shape_param_41);  bmm_3 = _shape_param_41 = None
        permute_18: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        clone_7: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_33: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_7, _shape_param_42);  clone_7 = _shape_param_42 = None
        convert_element_type_25: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg25_1, torch.bfloat16);  arg25_1 = None
        convert_element_type_26: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg26_1, torch.bfloat16);  arg26_1 = None
        view_34: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_33, _shape_param_43);  view_33 = _shape_param_43 = None
        permute_19: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_26, [1, 0]);  convert_element_type_26 = None
        addmm_3: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_25, view_34, permute_19);  convert_element_type_25 = None
        view_35: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_3, _shape_param_44);  addmm_3 = _shape_param_44 = None
        add_13: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_35, add_9);  view_35 = add_9 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 128, 1]" = var_mean_3[0]
        getitem_7: "f32[32, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        add_14: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_5: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_13, getitem_7);  getitem_7 = None
        mul_10: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_11: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_10, arg27_1);  arg27_1 = None
        add_15: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_11, arg28_1);  mul_11 = arg28_1 = None
        convert_element_type_27: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg29_1, torch.bfloat16);  arg29_1 = None
        convert_element_type_28: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg30_1, torch.bfloat16);  arg30_1 = None
        convert_element_type_29: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None
        view_36: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_29, _shape_param_45);  convert_element_type_29 = _shape_param_45 = None
        permute_20: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_28, [1, 0]);  convert_element_type_28 = None
        addmm_4: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_27, view_36, permute_20);  convert_element_type_27 = None
        view_37: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_4, _shape_param_46);  _shape_param_46 = None
        mul_12: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_37, 0.5)
        convert_element_type_30: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_37, torch.float32)
        pow_2: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_30, 3.0);  convert_element_type_30 = None
        mul_13: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_16: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_37, mul_13);  view_37 = mul_13 = None
        mul_14: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_16, 0.7978845608028654);  add_16 = None
        tanh_1: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_17: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_12, add_17);  mul_12 = add_17 = None
        convert_element_type_31: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg31_1, torch.bfloat16);  arg31_1 = None
        convert_element_type_32: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg32_1, torch.bfloat16);  arg32_1 = None
        convert_element_type_33: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16);  mul_15 = None
        view_38: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_33, _shape_param_47);  convert_element_type_33 = _shape_param_47 = None
        permute_21: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_32, [1, 0]);  convert_element_type_32 = None
        addmm_5: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_31, view_38, permute_21);  convert_element_type_31 = None
        view_39: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_5, _shape_param_48);  addmm_5 = _shape_param_48 = None
        add_18: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_13, view_39);  add_13 = view_39 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 128, 1]" = var_mean_4[0]
        getitem_9: "f32[32, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        add_19: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_6: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_18, getitem_9);  getitem_9 = None
        mul_16: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_17: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_16, arg33_1);  arg33_1 = None
        add_20: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_17, arg34_1);  mul_17 = arg34_1 = None
        convert_element_type_34: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg35_1, torch.bfloat16);  arg35_1 = None
        convert_element_type_35: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_20, torch.bfloat16);  add_20 = None
        permute_22: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        view_40: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_35, _shape_param_49);  convert_element_type_35 = _shape_param_49 = None
        mm_6: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_40, permute_22)
        view_41: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_6, _shape_param_50);  mm_6 = _shape_param_50 = None
        convert_element_type_36: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg36_1, torch.bfloat16);  arg36_1 = None
        permute_23: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_36, [1, 0]);  convert_element_type_36 = None
        mm_7: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_40, permute_23)
        view_42: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_7, _shape_param_51);  mm_7 = _shape_param_51 = None
        convert_element_type_37: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg37_1, torch.bfloat16);  arg37_1 = None
        permute_24: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_37, [1, 0]);  convert_element_type_37 = None
        mm_8: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_40, permute_24)
        view_43: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_8, _shape_param_52);  mm_8 = _shape_param_52 = None
        view_44: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_41, _shape_param_53);  view_41 = _shape_param_53 = None
        permute_25: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None
        view_45: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_42, _shape_param_54);  view_42 = _shape_param_54 = None
        permute_26: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
        view_46: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_43, _shape_param_55);  view_43 = _shape_param_55 = None
        permute_27: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        convert_element_type_38: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_26, torch.float32);  permute_26 = None
        permute_28: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_38, [0, 1, 3, 2]);  convert_element_type_38 = None
        convert_element_type_39: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_28, torch.bfloat16);  permute_28 = None
        convert_element_type_40: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_25, torch.bfloat16);  permute_25 = None
        expand_9: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_40, _shape_param_56);  convert_element_type_40 = _shape_param_56 = None
        clone_8: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_47: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_8, _shape_param_57);  clone_8 = _shape_param_57 = None
        expand_10: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_39, _shape_param_58);  convert_element_type_39 = _shape_param_58 = None
        clone_9: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_48: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_9, _shape_param_59);  clone_9 = _shape_param_59 = None
        bmm_4: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_47, view_48)
        view_49: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_4, _shape_param_60);  bmm_4 = _shape_param_60 = None
        slice_5: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg38_1, 2, 0, 128);  arg38_1 = None
        slice_6: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_5, 3, 0, 128);  slice_5 = None
        where_3: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_6, view_49, full_3);  slice_6 = view_49 = None
        add_21: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_3, where);  where_3 = None
        amax_2: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_21, [-1], True)
        sub_7: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_21, amax_2);  add_21 = amax_2 = None
        exp_2: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_41: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16)
        expand_11: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_41, _shape_param_61);  convert_element_type_41 = _shape_param_61 = None
        view_50: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_11, _shape_param_62);  expand_11 = _shape_param_62 = None
        expand_12: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_27, _shape_param_63);  permute_27 = _shape_param_63 = None
        clone_10: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_51: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_10, _shape_param_64);  clone_10 = _shape_param_64 = None
        bmm_5: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_50, view_51)
        view_52: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_5, _shape_param_65);  bmm_5 = _shape_param_65 = None
        permute_29: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        clone_11: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_53: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_11, _shape_param_66);  clone_11 = _shape_param_66 = None
        convert_element_type_42: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg39_1, torch.bfloat16);  arg39_1 = None
        convert_element_type_43: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg40_1, torch.bfloat16);  arg40_1 = None
        view_54: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_53, _shape_param_67);  view_53 = _shape_param_67 = None
        permute_30: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_43, [1, 0]);  convert_element_type_43 = None
        addmm_6: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_42, view_54, permute_30);  convert_element_type_42 = None
        view_55: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_6, _shape_param_68);  addmm_6 = _shape_param_68 = None
        add_22: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_55, add_18);  view_55 = add_18 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_10: "f32[32, 128, 1]" = var_mean_5[0]
        getitem_11: "f32[32, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        add_23: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_8: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_22, getitem_11);  getitem_11 = None
        mul_18: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
        mul_19: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_18, arg41_1);  arg41_1 = None
        add_24: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_19, arg42_1);  mul_19 = arg42_1 = None
        convert_element_type_44: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg43_1, torch.bfloat16);  arg43_1 = None
        convert_element_type_45: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg44_1, torch.bfloat16);  arg44_1 = None
        convert_element_type_46: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None
        view_56: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_46, _shape_param_69);  convert_element_type_46 = _shape_param_69 = None
        permute_31: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_45, [1, 0]);  convert_element_type_45 = None
        addmm_7: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_44, view_56, permute_31);  convert_element_type_44 = None
        view_57: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_7, _shape_param_70);  _shape_param_70 = None
        mul_20: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_57, 0.5)
        convert_element_type_47: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_57, torch.float32)
        pow_3: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_47, 3.0);  convert_element_type_47 = None
        mul_21: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_25: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_57, mul_21);  view_57 = mul_21 = None
        mul_22: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_2: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_26: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_20, add_26);  mul_20 = add_26 = None
        convert_element_type_48: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg45_1, torch.bfloat16);  arg45_1 = None
        convert_element_type_49: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg46_1, torch.bfloat16);  arg46_1 = None
        convert_element_type_50: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_23, torch.bfloat16);  mul_23 = None
        view_58: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_50, _shape_param_71);  convert_element_type_50 = _shape_param_71 = None
        permute_32: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_49, [1, 0]);  convert_element_type_49 = None
        addmm_8: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_48, view_58, permute_32);  convert_element_type_48 = None
        view_59: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_8, _shape_param_72);  addmm_8 = _shape_param_72 = None
        add_27: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_22, view_59);  add_22 = view_59 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
        getitem_12: "f32[32, 128, 1]" = var_mean_6[0]
        getitem_13: "f32[32, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        add_28: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_9: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_27, getitem_13);  getitem_13 = None
        mul_24: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
        mul_25: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_24, arg47_1);  arg47_1 = None
        add_29: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_25, arg48_1);  mul_25 = arg48_1 = None
        convert_element_type_51: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg49_1, torch.bfloat16);  arg49_1 = None
        convert_element_type_52: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None
        permute_33: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_51, [1, 0]);  convert_element_type_51 = None
        view_60: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_52, _shape_param_73);  convert_element_type_52 = _shape_param_73 = None
        mm_9: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_60, permute_33)
        view_61: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_9, _shape_param_74);  mm_9 = _shape_param_74 = None
        convert_element_type_53: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg50_1, torch.bfloat16);  arg50_1 = None
        permute_34: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_53, [1, 0]);  convert_element_type_53 = None
        mm_10: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_60, permute_34)
        view_62: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_10, _shape_param_75);  mm_10 = _shape_param_75 = None
        convert_element_type_54: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg51_1, torch.bfloat16);  arg51_1 = None
        permute_35: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_54, [1, 0]);  convert_element_type_54 = None
        mm_11: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_60, permute_35)
        view_63: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_11, _shape_param_76);  mm_11 = _shape_param_76 = None
        view_64: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_61, _shape_param_77);  view_61 = _shape_param_77 = None
        permute_36: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        view_65: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_62, _shape_param_78);  view_62 = _shape_param_78 = None
        permute_37: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        view_66: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_63, _shape_param_79);  view_63 = _shape_param_79 = None
        permute_38: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        convert_element_type_55: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_37, torch.float32);  permute_37 = None
        permute_39: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_55, [0, 1, 3, 2]);  convert_element_type_55 = None
        convert_element_type_56: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_39, torch.bfloat16);  permute_39 = None
        convert_element_type_57: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_36, torch.bfloat16);  permute_36 = None
        expand_13: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_57, _shape_param_80);  convert_element_type_57 = _shape_param_80 = None
        clone_12: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_67: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_12, _shape_param_81);  clone_12 = _shape_param_81 = None
        expand_14: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_56, _shape_param_82);  convert_element_type_56 = _shape_param_82 = None
        clone_13: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_68: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_13, _shape_param_83);  clone_13 = _shape_param_83 = None
        bmm_6: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_67, view_68)
        view_69: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_84);  bmm_6 = _shape_param_84 = None
        slice_7: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg52_1, 2, 0, 128);  arg52_1 = None
        slice_8: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_7, 3, 0, 128);  slice_7 = None
        where_4: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_8, view_69, full_3);  slice_8 = view_69 = None
        add_30: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_4, where);  where_4 = None
        amax_3: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_30, [-1], True)
        sub_10: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_30, amax_3);  add_30 = amax_3 = None
        exp_3: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_58: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16)
        expand_15: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_58, _shape_param_85);  convert_element_type_58 = _shape_param_85 = None
        view_70: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_15, _shape_param_86);  expand_15 = _shape_param_86 = None
        expand_16: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_38, _shape_param_87);  permute_38 = _shape_param_87 = None
        clone_14: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_71: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_14, _shape_param_88);  clone_14 = _shape_param_88 = None
        bmm_7: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_70, view_71)
        view_72: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_7, _shape_param_89);  bmm_7 = _shape_param_89 = None
        permute_40: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        clone_15: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_73: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_15, _shape_param_90);  clone_15 = _shape_param_90 = None
        convert_element_type_59: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg53_1, torch.bfloat16);  arg53_1 = None
        convert_element_type_60: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg54_1, torch.bfloat16);  arg54_1 = None
        view_74: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_73, _shape_param_91);  view_73 = _shape_param_91 = None
        permute_41: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_60, [1, 0]);  convert_element_type_60 = None
        addmm_9: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_59, view_74, permute_41);  convert_element_type_59 = None
        view_75: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_9, _shape_param_92);  addmm_9 = _shape_param_92 = None
        add_31: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_75, add_27);  view_75 = add_27 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 128, 1]" = var_mean_7[0]
        getitem_15: "f32[32, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        add_32: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_11: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_31, getitem_15);  getitem_15 = None
        mul_26: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
        mul_27: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_26, arg55_1);  arg55_1 = None
        add_33: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_27, arg56_1);  mul_27 = arg56_1 = None
        convert_element_type_61: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg57_1, torch.bfloat16);  arg57_1 = None
        convert_element_type_62: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg58_1, torch.bfloat16);  arg58_1 = None
        convert_element_type_63: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16);  add_33 = None
        view_76: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_63, _shape_param_93);  convert_element_type_63 = _shape_param_93 = None
        permute_42: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_62, [1, 0]);  convert_element_type_62 = None
        addmm_10: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_61, view_76, permute_42);  convert_element_type_61 = None
        view_77: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_10, _shape_param_94);  _shape_param_94 = None
        mul_28: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_77, 0.5)
        convert_element_type_64: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_77, torch.float32)
        pow_4: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_64, 3.0);  convert_element_type_64 = None
        mul_29: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_34: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_77, mul_29);  view_77 = mul_29 = None
        mul_30: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_34, 0.7978845608028654);  add_34 = None
        tanh_3: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_35: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_28, add_35);  mul_28 = add_35 = None
        convert_element_type_65: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg59_1, torch.bfloat16);  arg59_1 = None
        convert_element_type_66: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg60_1, torch.bfloat16);  arg60_1 = None
        convert_element_type_67: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_31, torch.bfloat16);  mul_31 = None
        view_78: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_67, _shape_param_95);  convert_element_type_67 = _shape_param_95 = None
        permute_43: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_66, [1, 0]);  convert_element_type_66 = None
        addmm_11: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_65, view_78, permute_43);  convert_element_type_65 = None
        view_79: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_11, _shape_param_96);  addmm_11 = _shape_param_96 = None
        add_36: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_31, view_79);  add_31 = view_79 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 128, 1]" = var_mean_8[0]
        getitem_17: "f32[32, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        add_37: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_12: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_36, getitem_17);  getitem_17 = None
        mul_32: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
        mul_33: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_32, arg61_1);  arg61_1 = None
        add_38: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_33, arg62_1);  mul_33 = arg62_1 = None
        convert_element_type_68: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg63_1, torch.bfloat16);  arg63_1 = None
        convert_element_type_69: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_38, torch.bfloat16);  add_38 = None
        permute_44: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_68, [1, 0]);  convert_element_type_68 = None
        view_80: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_69, _shape_param_97);  convert_element_type_69 = _shape_param_97 = None
        mm_12: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_80, permute_44)
        view_81: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_12, _shape_param_98);  mm_12 = _shape_param_98 = None
        convert_element_type_70: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg64_1, torch.bfloat16);  arg64_1 = None
        permute_45: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_70, [1, 0]);  convert_element_type_70 = None
        mm_13: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_80, permute_45)
        view_82: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_13, _shape_param_99);  mm_13 = _shape_param_99 = None
        convert_element_type_71: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg65_1, torch.bfloat16);  arg65_1 = None
        permute_46: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_71, [1, 0]);  convert_element_type_71 = None
        mm_14: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_80, permute_46)
        view_83: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_14, _shape_param_100);  mm_14 = _shape_param_100 = None
        view_84: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_81, _shape_param_101);  view_81 = _shape_param_101 = None
        permute_47: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        view_85: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_82, _shape_param_102);  view_82 = _shape_param_102 = None
        permute_48: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        view_86: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_83, _shape_param_103);  view_83 = _shape_param_103 = None
        permute_49: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        convert_element_type_72: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_48, torch.float32);  permute_48 = None
        permute_50: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_72, [0, 1, 3, 2]);  convert_element_type_72 = None
        convert_element_type_73: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_50, torch.bfloat16);  permute_50 = None
        convert_element_type_74: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_47, torch.bfloat16);  permute_47 = None
        expand_17: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_74, _shape_param_104);  convert_element_type_74 = _shape_param_104 = None
        clone_16: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_87: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_16, _shape_param_105);  clone_16 = _shape_param_105 = None
        expand_18: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_73, _shape_param_106);  convert_element_type_73 = _shape_param_106 = None
        clone_17: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_88: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_17, _shape_param_107);  clone_17 = _shape_param_107 = None
        bmm_8: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_87, view_88)
        view_89: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_8, _shape_param_108);  bmm_8 = _shape_param_108 = None
        slice_9: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg66_1, 2, 0, 128);  arg66_1 = None
        slice_10: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_9, 3, 0, 128);  slice_9 = None
        where_5: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_10, view_89, full_3);  slice_10 = view_89 = None
        add_39: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_5, where);  where_5 = None
        amax_4: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_39, [-1], True)
        sub_13: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_39, amax_4);  add_39 = amax_4 = None
        exp_4: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_75: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16)
        expand_19: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_75, _shape_param_109);  convert_element_type_75 = _shape_param_109 = None
        view_90: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_19, _shape_param_110);  expand_19 = _shape_param_110 = None
        expand_20: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_49, _shape_param_111);  permute_49 = _shape_param_111 = None
        clone_18: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_91: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_18, _shape_param_112);  clone_18 = _shape_param_112 = None
        bmm_9: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_90, view_91)
        view_92: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_9, _shape_param_113);  bmm_9 = _shape_param_113 = None
        permute_51: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
        clone_19: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_93: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_19, _shape_param_114);  clone_19 = _shape_param_114 = None
        convert_element_type_76: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg67_1, torch.bfloat16);  arg67_1 = None
        convert_element_type_77: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg68_1, torch.bfloat16);  arg68_1 = None
        view_94: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_93, _shape_param_115);  view_93 = _shape_param_115 = None
        permute_52: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_77, [1, 0]);  convert_element_type_77 = None
        addmm_12: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_76, view_94, permute_52);  convert_element_type_76 = None
        view_95: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_12, _shape_param_116);  addmm_12 = _shape_param_116 = None
        add_40: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_95, add_36);  view_95 = add_36 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_18: "f32[32, 128, 1]" = var_mean_9[0]
        getitem_19: "f32[32, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        add_41: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_14: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_40, getitem_19);  getitem_19 = None
        mul_34: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
        mul_35: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_34, arg69_1);  arg69_1 = None
        add_42: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_35, arg70_1);  mul_35 = arg70_1 = None
        convert_element_type_78: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg71_1, torch.bfloat16);  arg71_1 = None
        convert_element_type_79: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg72_1, torch.bfloat16);  arg72_1 = None
        convert_element_type_80: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_42, torch.bfloat16);  add_42 = None
        view_96: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_80, _shape_param_117);  convert_element_type_80 = _shape_param_117 = None
        permute_53: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_79, [1, 0]);  convert_element_type_79 = None
        addmm_13: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_78, view_96, permute_53);  convert_element_type_78 = None
        view_97: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_13, _shape_param_118);  _shape_param_118 = None
        mul_36: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_97, 0.5)
        convert_element_type_81: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_97, torch.float32)
        pow_5: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_81, 3.0);  convert_element_type_81 = None
        mul_37: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_43: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_97, mul_37);  view_97 = mul_37 = None
        mul_38: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_43, 0.7978845608028654);  add_43 = None
        tanh_4: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_44: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_36, add_44);  mul_36 = add_44 = None
        convert_element_type_82: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg73_1, torch.bfloat16);  arg73_1 = None
        convert_element_type_83: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg74_1, torch.bfloat16);  arg74_1 = None
        convert_element_type_84: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None
        view_98: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_84, _shape_param_119);  convert_element_type_84 = _shape_param_119 = None
        permute_54: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_83, [1, 0]);  convert_element_type_83 = None
        addmm_14: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_82, view_98, permute_54);  convert_element_type_82 = None
        view_99: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_14, _shape_param_120);  addmm_14 = _shape_param_120 = None
        add_45: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_40, view_99);  add_40 = view_99 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 128, 1]" = var_mean_10[0]
        getitem_21: "f32[32, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        add_46: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_15: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_45, getitem_21);  getitem_21 = None
        mul_40: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
        mul_41: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_40, arg75_1);  arg75_1 = None
        add_47: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_41, arg76_1);  mul_41 = arg76_1 = None
        convert_element_type_85: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg77_1, torch.bfloat16);  arg77_1 = None
        convert_element_type_86: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None
        permute_55: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_85, [1, 0]);  convert_element_type_85 = None
        view_100: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_86, _shape_param_121);  convert_element_type_86 = _shape_param_121 = None
        mm_15: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_100, permute_55)
        view_101: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_15, _shape_param_122);  mm_15 = _shape_param_122 = None
        convert_element_type_87: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg78_1, torch.bfloat16);  arg78_1 = None
        permute_56: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_87, [1, 0]);  convert_element_type_87 = None
        mm_16: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_100, permute_56)
        view_102: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_16, _shape_param_123);  mm_16 = _shape_param_123 = None
        convert_element_type_88: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg79_1, torch.bfloat16);  arg79_1 = None
        permute_57: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_88, [1, 0]);  convert_element_type_88 = None
        mm_17: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_100, permute_57)
        view_103: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_17, _shape_param_124);  mm_17 = _shape_param_124 = None
        view_104: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_101, _shape_param_125);  view_101 = _shape_param_125 = None
        permute_58: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        view_105: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_102, _shape_param_126);  view_102 = _shape_param_126 = None
        permute_59: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
        view_106: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_103, _shape_param_127);  view_103 = _shape_param_127 = None
        permute_60: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        convert_element_type_89: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_59, torch.float32);  permute_59 = None
        permute_61: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_89, [0, 1, 3, 2]);  convert_element_type_89 = None
        convert_element_type_90: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_61, torch.bfloat16);  permute_61 = None
        convert_element_type_91: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_58, torch.bfloat16);  permute_58 = None
        expand_21: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_91, _shape_param_128);  convert_element_type_91 = _shape_param_128 = None
        clone_20: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_107: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_20, _shape_param_129);  clone_20 = _shape_param_129 = None
        expand_22: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_90, _shape_param_130);  convert_element_type_90 = _shape_param_130 = None
        clone_21: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_108: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_21, _shape_param_131);  clone_21 = _shape_param_131 = None
        bmm_10: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_107, view_108)
        view_109: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_132);  bmm_10 = _shape_param_132 = None
        slice_11: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg80_1, 2, 0, 128);  arg80_1 = None
        slice_12: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_11, 3, 0, 128);  slice_11 = None
        where_6: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_12, view_109, full_3);  slice_12 = view_109 = None
        add_48: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_6, where);  where_6 = None
        amax_5: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_48, [-1], True)
        sub_16: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_48, amax_5);  add_48 = amax_5 = None
        exp_5: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_92: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16)
        expand_23: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_92, _shape_param_133);  convert_element_type_92 = _shape_param_133 = None
        view_110: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_23, _shape_param_134);  expand_23 = _shape_param_134 = None
        expand_24: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_60, _shape_param_135);  permute_60 = _shape_param_135 = None
        clone_22: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_111: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_22, _shape_param_136);  clone_22 = _shape_param_136 = None
        bmm_11: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_110, view_111)
        view_112: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_11, _shape_param_137);  bmm_11 = _shape_param_137 = None
        permute_62: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        clone_23: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_113: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_23, _shape_param_138);  clone_23 = _shape_param_138 = None
        convert_element_type_93: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg81_1, torch.bfloat16);  arg81_1 = None
        convert_element_type_94: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg82_1, torch.bfloat16);  arg82_1 = None
        view_114: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_113, _shape_param_139);  view_113 = _shape_param_139 = None
        permute_63: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_94, [1, 0]);  convert_element_type_94 = None
        addmm_15: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_93, view_114, permute_63);  convert_element_type_93 = None
        view_115: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_15, _shape_param_140);  addmm_15 = _shape_param_140 = None
        add_49: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_115, add_45);  view_115 = add_45 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 128, 1]" = var_mean_11[0]
        getitem_23: "f32[32, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        add_50: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_17: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_49, getitem_23);  getitem_23 = None
        mul_42: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
        mul_43: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_42, arg83_1);  arg83_1 = None
        add_51: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_43, arg84_1);  mul_43 = arg84_1 = None
        convert_element_type_95: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg85_1, torch.bfloat16);  arg85_1 = None
        convert_element_type_96: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg86_1, torch.bfloat16);  arg86_1 = None
        convert_element_type_97: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16);  add_51 = None
        view_116: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_97, _shape_param_141);  convert_element_type_97 = _shape_param_141 = None
        permute_64: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_96, [1, 0]);  convert_element_type_96 = None
        addmm_16: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_95, view_116, permute_64);  convert_element_type_95 = None
        view_117: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_16, _shape_param_142);  _shape_param_142 = None
        mul_44: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_117, 0.5)
        convert_element_type_98: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_117, torch.float32)
        pow_6: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_98, 3.0);  convert_element_type_98 = None
        mul_45: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_52: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_117, mul_45);  view_117 = mul_45 = None
        mul_46: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_52, 0.7978845608028654);  add_52 = None
        tanh_5: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_53: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_44, add_53);  mul_44 = add_53 = None
        convert_element_type_99: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg87_1, torch.bfloat16);  arg87_1 = None
        convert_element_type_100: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg88_1, torch.bfloat16);  arg88_1 = None
        convert_element_type_101: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_47, torch.bfloat16);  mul_47 = None
        view_118: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_101, _shape_param_143);  convert_element_type_101 = _shape_param_143 = None
        permute_65: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_100, [1, 0]);  convert_element_type_100 = None
        addmm_17: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_99, view_118, permute_65);  convert_element_type_99 = None
        view_119: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_17, _shape_param_144);  addmm_17 = _shape_param_144 = None
        add_54: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_49, view_119);  add_49 = view_119 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_54, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 128, 1]" = var_mean_12[0]
        getitem_25: "f32[32, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        add_55: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        sub_18: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_54, getitem_25);  getitem_25 = None
        mul_48: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
        mul_49: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_48, arg89_1);  arg89_1 = None
        add_56: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_49, arg90_1);  mul_49 = arg90_1 = None
        convert_element_type_102: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg91_1, torch.bfloat16);  arg91_1 = None
        convert_element_type_103: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_56, torch.bfloat16);  add_56 = None
        permute_66: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_102, [1, 0]);  convert_element_type_102 = None
        view_120: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_103, _shape_param_145);  convert_element_type_103 = _shape_param_145 = None
        mm_18: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_120, permute_66)
        view_121: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_18, _shape_param_146);  mm_18 = _shape_param_146 = None
        convert_element_type_104: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg92_1, torch.bfloat16);  arg92_1 = None
        permute_67: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_104, [1, 0]);  convert_element_type_104 = None
        mm_19: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_120, permute_67)
        view_122: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_19, _shape_param_147);  mm_19 = _shape_param_147 = None
        convert_element_type_105: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg93_1, torch.bfloat16);  arg93_1 = None
        permute_68: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_105, [1, 0]);  convert_element_type_105 = None
        mm_20: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_120, permute_68)
        view_123: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_20, _shape_param_148);  mm_20 = _shape_param_148 = None
        view_124: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_121, _shape_param_149);  view_121 = _shape_param_149 = None
        permute_69: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        view_125: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_122, _shape_param_150);  view_122 = _shape_param_150 = None
        permute_70: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
        view_126: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_123, _shape_param_151);  view_123 = _shape_param_151 = None
        permute_71: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        convert_element_type_106: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_70, torch.float32);  permute_70 = None
        permute_72: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_106, [0, 1, 3, 2]);  convert_element_type_106 = None
        convert_element_type_107: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_72, torch.bfloat16);  permute_72 = None
        convert_element_type_108: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_69, torch.bfloat16);  permute_69 = None
        expand_25: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_108, _shape_param_152);  convert_element_type_108 = _shape_param_152 = None
        clone_24: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_127: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_24, _shape_param_153);  clone_24 = _shape_param_153 = None
        expand_26: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_107, _shape_param_154);  convert_element_type_107 = _shape_param_154 = None
        clone_25: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_128: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_25, _shape_param_155);  clone_25 = _shape_param_155 = None
        bmm_12: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_127, view_128)
        view_129: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_12, _shape_param_156);  bmm_12 = _shape_param_156 = None
        slice_13: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg94_1, 2, 0, 128);  arg94_1 = None
        slice_14: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_13, 3, 0, 128);  slice_13 = None
        where_7: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_14, view_129, full_3);  slice_14 = view_129 = None
        add_57: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_7, where);  where_7 = None
        amax_6: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_57, [-1], True)
        sub_19: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_57, amax_6);  add_57 = amax_6 = None
        exp_6: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_109: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16)
        expand_27: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_109, _shape_param_157);  convert_element_type_109 = _shape_param_157 = None
        view_130: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_27, _shape_param_158);  expand_27 = _shape_param_158 = None
        expand_28: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_71, _shape_param_159);  permute_71 = _shape_param_159 = None
        clone_26: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_131: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_26, _shape_param_160);  clone_26 = _shape_param_160 = None
        bmm_13: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_130, view_131)
        view_132: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_13, _shape_param_161);  bmm_13 = _shape_param_161 = None
        permute_73: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
        clone_27: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_133: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_27, _shape_param_162);  clone_27 = _shape_param_162 = None
        convert_element_type_110: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg95_1, torch.bfloat16);  arg95_1 = None
        convert_element_type_111: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg96_1, torch.bfloat16);  arg96_1 = None
        view_134: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_133, _shape_param_163);  view_133 = _shape_param_163 = None
        permute_74: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_111, [1, 0]);  convert_element_type_111 = None
        addmm_18: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_110, view_134, permute_74);  convert_element_type_110 = None
        view_135: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_18, _shape_param_164);  addmm_18 = _shape_param_164 = None
        add_58: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_135, add_54);  view_135 = add_54 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_58, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 128, 1]" = var_mean_13[0]
        getitem_27: "f32[32, 128, 1]" = var_mean_13[1];  var_mean_13 = None
        add_59: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_20: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_58, getitem_27);  getitem_27 = None
        mul_50: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
        mul_51: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_50, arg97_1);  arg97_1 = None
        add_60: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_51, arg98_1);  mul_51 = arg98_1 = None
        convert_element_type_112: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg99_1, torch.bfloat16);  arg99_1 = None
        convert_element_type_113: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg100_1, torch.bfloat16);  arg100_1 = None
        convert_element_type_114: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_60, torch.bfloat16);  add_60 = None
        view_136: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_114, _shape_param_165);  convert_element_type_114 = _shape_param_165 = None
        permute_75: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_113, [1, 0]);  convert_element_type_113 = None
        addmm_19: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_112, view_136, permute_75);  convert_element_type_112 = None
        view_137: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_19, _shape_param_166);  _shape_param_166 = None
        mul_52: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_137, 0.5)
        convert_element_type_115: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_137, torch.float32)
        pow_7: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_115, 3.0);  convert_element_type_115 = None
        mul_53: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_61: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_137, mul_53);  view_137 = mul_53 = None
        mul_54: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_61, 0.7978845608028654);  add_61 = None
        tanh_6: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_62: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_52, add_62);  mul_52 = add_62 = None
        convert_element_type_116: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg101_1, torch.bfloat16);  arg101_1 = None
        convert_element_type_117: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg102_1, torch.bfloat16);  arg102_1 = None
        convert_element_type_118: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_55, torch.bfloat16);  mul_55 = None
        view_138: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_118, _shape_param_167);  convert_element_type_118 = _shape_param_167 = None
        permute_76: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_117, [1, 0]);  convert_element_type_117 = None
        addmm_20: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_116, view_138, permute_76);  convert_element_type_116 = None
        view_139: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_20, _shape_param_168);  addmm_20 = _shape_param_168 = None
        add_63: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_58, view_139);  add_58 = view_139 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 128, 1]" = var_mean_14[0]
        getitem_29: "f32[32, 128, 1]" = var_mean_14[1];  var_mean_14 = None
        add_64: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_21: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_63, getitem_29);  getitem_29 = None
        mul_56: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
        mul_57: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_56, arg103_1);  arg103_1 = None
        add_65: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_57, arg104_1);  mul_57 = arg104_1 = None
        convert_element_type_119: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg105_1, torch.bfloat16);  arg105_1 = None
        convert_element_type_120: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None
        permute_77: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_119, [1, 0]);  convert_element_type_119 = None
        view_140: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_120, _shape_param_169);  convert_element_type_120 = _shape_param_169 = None
        mm_21: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_140, permute_77)
        view_141: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_21, _shape_param_170);  mm_21 = _shape_param_170 = None
        convert_element_type_121: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg106_1, torch.bfloat16);  arg106_1 = None
        permute_78: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_121, [1, 0]);  convert_element_type_121 = None
        mm_22: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_140, permute_78)
        view_142: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_22, _shape_param_171);  mm_22 = _shape_param_171 = None
        convert_element_type_122: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg107_1, torch.bfloat16);  arg107_1 = None
        permute_79: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_122, [1, 0]);  convert_element_type_122 = None
        mm_23: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_140, permute_79)
        view_143: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_23, _shape_param_172);  mm_23 = _shape_param_172 = None
        view_144: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_141, _shape_param_173);  view_141 = _shape_param_173 = None
        permute_80: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        view_145: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_142, _shape_param_174);  view_142 = _shape_param_174 = None
        permute_81: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
        view_146: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_143, _shape_param_175);  view_143 = _shape_param_175 = None
        permute_82: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        convert_element_type_123: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_81, torch.float32);  permute_81 = None
        permute_83: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_123, [0, 1, 3, 2]);  convert_element_type_123 = None
        convert_element_type_124: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_83, torch.bfloat16);  permute_83 = None
        convert_element_type_125: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_80, torch.bfloat16);  permute_80 = None
        expand_29: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_125, _shape_param_176);  convert_element_type_125 = _shape_param_176 = None
        clone_28: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_147: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_28, _shape_param_177);  clone_28 = _shape_param_177 = None
        expand_30: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_124, _shape_param_178);  convert_element_type_124 = _shape_param_178 = None
        clone_29: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_148: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_29, _shape_param_179);  clone_29 = _shape_param_179 = None
        bmm_14: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_147, view_148)
        view_149: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_14, _shape_param_180);  bmm_14 = _shape_param_180 = None
        slice_15: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg108_1, 2, 0, 128);  arg108_1 = None
        slice_16: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_15, 3, 0, 128);  slice_15 = None
        where_8: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_16, view_149, full_3);  slice_16 = view_149 = None
        add_66: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_8, where);  where_8 = None
        amax_7: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_66, [-1], True)
        sub_22: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_66, amax_7);  add_66 = amax_7 = None
        exp_7: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_126: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16)
        expand_31: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_126, _shape_param_181);  convert_element_type_126 = _shape_param_181 = None
        view_150: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_31, _shape_param_182);  expand_31 = _shape_param_182 = None
        expand_32: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_82, _shape_param_183);  permute_82 = _shape_param_183 = None
        clone_30: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_151: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_30, _shape_param_184);  clone_30 = _shape_param_184 = None
        bmm_15: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_150, view_151)
        view_152: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_15, _shape_param_185);  bmm_15 = _shape_param_185 = None
        permute_84: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        clone_31: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_153: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_31, _shape_param_186);  clone_31 = _shape_param_186 = None
        convert_element_type_127: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg109_1, torch.bfloat16);  arg109_1 = None
        convert_element_type_128: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg110_1, torch.bfloat16);  arg110_1 = None
        view_154: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_153, _shape_param_187);  view_153 = _shape_param_187 = None
        permute_85: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_128, [1, 0]);  convert_element_type_128 = None
        addmm_21: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_127, view_154, permute_85);  convert_element_type_127 = None
        view_155: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_21, _shape_param_188);  addmm_21 = _shape_param_188 = None
        add_67: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_155, add_63);  view_155 = add_63 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 128, 1]" = var_mean_15[0]
        getitem_31: "f32[32, 128, 1]" = var_mean_15[1];  var_mean_15 = None
        add_68: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_23: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_67, getitem_31);  getitem_31 = None
        mul_58: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
        mul_59: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_58, arg111_1);  arg111_1 = None
        add_69: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_59, arg112_1);  mul_59 = arg112_1 = None
        convert_element_type_129: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg113_1, torch.bfloat16);  arg113_1 = None
        convert_element_type_130: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg114_1, torch.bfloat16);  arg114_1 = None
        convert_element_type_131: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16);  add_69 = None
        view_156: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_131, _shape_param_189);  convert_element_type_131 = _shape_param_189 = None
        permute_86: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_130, [1, 0]);  convert_element_type_130 = None
        addmm_22: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_129, view_156, permute_86);  convert_element_type_129 = None
        view_157: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_22, _shape_param_190);  _shape_param_190 = None
        mul_60: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_157, 0.5)
        convert_element_type_132: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_157, torch.float32)
        pow_8: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_132, 3.0);  convert_element_type_132 = None
        mul_61: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_70: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_157, mul_61);  view_157 = mul_61 = None
        mul_62: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_70, 0.7978845608028654);  add_70 = None
        tanh_7: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_71: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_60, add_71);  mul_60 = add_71 = None
        convert_element_type_133: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg115_1, torch.bfloat16);  arg115_1 = None
        convert_element_type_134: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg116_1, torch.bfloat16);  arg116_1 = None
        convert_element_type_135: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16);  mul_63 = None
        view_158: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_135, _shape_param_191);  convert_element_type_135 = _shape_param_191 = None
        permute_87: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_134, [1, 0]);  convert_element_type_134 = None
        addmm_23: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_133, view_158, permute_87);  convert_element_type_133 = None
        view_159: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_23, _shape_param_192);  addmm_23 = _shape_param_192 = None
        add_72: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_67, view_159);  add_67 = view_159 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 128, 1]" = var_mean_16[0]
        getitem_33: "f32[32, 128, 1]" = var_mean_16[1];  var_mean_16 = None
        add_73: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_24: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_72, getitem_33);  getitem_33 = None
        mul_64: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
        mul_65: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_64, arg117_1);  arg117_1 = None
        add_74: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_65, arg118_1);  mul_65 = arg118_1 = None
        convert_element_type_136: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg119_1, torch.bfloat16);  arg119_1 = None
        convert_element_type_137: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None
        permute_88: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_136, [1, 0]);  convert_element_type_136 = None
        view_160: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_137, _shape_param_193);  convert_element_type_137 = _shape_param_193 = None
        mm_24: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_160, permute_88)
        view_161: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_24, _shape_param_194);  mm_24 = _shape_param_194 = None
        convert_element_type_138: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg120_1, torch.bfloat16);  arg120_1 = None
        permute_89: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_138, [1, 0]);  convert_element_type_138 = None
        mm_25: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_160, permute_89)
        view_162: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_25, _shape_param_195);  mm_25 = _shape_param_195 = None
        convert_element_type_139: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg121_1, torch.bfloat16);  arg121_1 = None
        permute_90: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_139, [1, 0]);  convert_element_type_139 = None
        mm_26: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_160, permute_90)
        view_163: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_26, _shape_param_196);  mm_26 = _shape_param_196 = None
        view_164: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_161, _shape_param_197);  view_161 = _shape_param_197 = None
        permute_91: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
        view_165: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_162, _shape_param_198);  view_162 = _shape_param_198 = None
        permute_92: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        view_166: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_163, _shape_param_199);  view_163 = _shape_param_199 = None
        permute_93: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        convert_element_type_140: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_92, torch.float32);  permute_92 = None
        permute_94: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_140, [0, 1, 3, 2]);  convert_element_type_140 = None
        convert_element_type_141: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_94, torch.bfloat16);  permute_94 = None
        convert_element_type_142: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_91, torch.bfloat16);  permute_91 = None
        expand_33: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_142, _shape_param_200);  convert_element_type_142 = _shape_param_200 = None
        clone_32: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_167: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_32, _shape_param_201);  clone_32 = _shape_param_201 = None
        expand_34: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_141, _shape_param_202);  convert_element_type_141 = _shape_param_202 = None
        clone_33: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_168: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_33, _shape_param_203);  clone_33 = _shape_param_203 = None
        bmm_16: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_167, view_168)
        view_169: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_16, _shape_param_204);  bmm_16 = _shape_param_204 = None
        slice_17: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg122_1, 2, 0, 128);  arg122_1 = None
        slice_18: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_17, 3, 0, 128);  slice_17 = None
        where_9: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_18, view_169, full_3);  slice_18 = view_169 = None
        add_75: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_9, where);  where_9 = None
        amax_8: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_75, [-1], True)
        sub_25: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_75, amax_8);  add_75 = amax_8 = None
        exp_8: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_143: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16)
        expand_35: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_143, _shape_param_205);  convert_element_type_143 = _shape_param_205 = None
        view_170: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_35, _shape_param_206);  expand_35 = _shape_param_206 = None
        expand_36: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_93, _shape_param_207);  permute_93 = _shape_param_207 = None
        clone_34: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_171: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_34, _shape_param_208);  clone_34 = _shape_param_208 = None
        bmm_17: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_170, view_171)
        view_172: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_17, _shape_param_209);  bmm_17 = _shape_param_209 = None
        permute_95: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
        clone_35: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_173: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_35, _shape_param_210);  clone_35 = _shape_param_210 = None
        convert_element_type_144: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg123_1, torch.bfloat16);  arg123_1 = None
        convert_element_type_145: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg124_1, torch.bfloat16);  arg124_1 = None
        view_174: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_173, _shape_param_211);  view_173 = _shape_param_211 = None
        permute_96: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_145, [1, 0]);  convert_element_type_145 = None
        addmm_24: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_144, view_174, permute_96);  convert_element_type_144 = None
        view_175: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_24, _shape_param_212);  addmm_24 = _shape_param_212 = None
        add_76: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_175, add_72);  view_175 = add_72 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
        getitem_34: "f32[32, 128, 1]" = var_mean_17[0]
        getitem_35: "f32[32, 128, 1]" = var_mean_17[1];  var_mean_17 = None
        add_77: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_26: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_76, getitem_35);  getitem_35 = None
        mul_66: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
        mul_67: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_66, arg125_1);  arg125_1 = None
        add_78: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_67, arg126_1);  mul_67 = arg126_1 = None
        convert_element_type_146: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg127_1, torch.bfloat16);  arg127_1 = None
        convert_element_type_147: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg128_1, torch.bfloat16);  arg128_1 = None
        convert_element_type_148: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16);  add_78 = None
        view_176: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_148, _shape_param_213);  convert_element_type_148 = _shape_param_213 = None
        permute_97: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_147, [1, 0]);  convert_element_type_147 = None
        addmm_25: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_146, view_176, permute_97);  convert_element_type_146 = None
        view_177: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_25, _shape_param_214);  _shape_param_214 = None
        mul_68: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_177, 0.5)
        convert_element_type_149: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_177, torch.float32)
        pow_9: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_149, 3.0);  convert_element_type_149 = None
        mul_69: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_79: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_177, mul_69);  view_177 = mul_69 = None
        mul_70: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_79, 0.7978845608028654);  add_79 = None
        tanh_8: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_80: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_68, add_80);  mul_68 = add_80 = None
        convert_element_type_150: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg129_1, torch.bfloat16);  arg129_1 = None
        convert_element_type_151: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg130_1, torch.bfloat16);  arg130_1 = None
        convert_element_type_152: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_71, torch.bfloat16);  mul_71 = None
        view_178: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_152, _shape_param_215);  convert_element_type_152 = _shape_param_215 = None
        permute_98: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_151, [1, 0]);  convert_element_type_151 = None
        addmm_26: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_150, view_178, permute_98);  convert_element_type_150 = None
        view_179: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_26, _shape_param_216);  addmm_26 = _shape_param_216 = None
        add_81: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_76, view_179);  add_76 = view_179 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_36: "f32[32, 128, 1]" = var_mean_18[0]
        getitem_37: "f32[32, 128, 1]" = var_mean_18[1];  var_mean_18 = None
        add_82: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_27: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_81, getitem_37);  getitem_37 = None
        mul_72: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
        mul_73: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_72, arg131_1);  arg131_1 = None
        add_83: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_73, arg132_1);  mul_73 = arg132_1 = None
        convert_element_type_153: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg133_1, torch.bfloat16);  arg133_1 = None
        convert_element_type_154: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None
        permute_99: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_153, [1, 0]);  convert_element_type_153 = None
        view_180: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_154, _shape_param_217);  convert_element_type_154 = _shape_param_217 = None
        mm_27: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_180, permute_99)
        view_181: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_27, _shape_param_218);  mm_27 = _shape_param_218 = None
        convert_element_type_155: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg134_1, torch.bfloat16);  arg134_1 = None
        permute_100: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_155, [1, 0]);  convert_element_type_155 = None
        mm_28: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_180, permute_100)
        view_182: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_28, _shape_param_219);  mm_28 = _shape_param_219 = None
        convert_element_type_156: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg135_1, torch.bfloat16);  arg135_1 = None
        permute_101: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_156, [1, 0]);  convert_element_type_156 = None
        mm_29: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_180, permute_101)
        view_183: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_29, _shape_param_220);  mm_29 = _shape_param_220 = None
        view_184: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_181, _shape_param_221);  view_181 = _shape_param_221 = None
        permute_102: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        view_185: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_182, _shape_param_222);  view_182 = _shape_param_222 = None
        permute_103: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        view_186: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_183, _shape_param_223);  view_183 = _shape_param_223 = None
        permute_104: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        convert_element_type_157: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_103, torch.float32);  permute_103 = None
        permute_105: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_157, [0, 1, 3, 2]);  convert_element_type_157 = None
        convert_element_type_158: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_105, torch.bfloat16);  permute_105 = None
        convert_element_type_159: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_102, torch.bfloat16);  permute_102 = None
        expand_37: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_159, _shape_param_224);  convert_element_type_159 = _shape_param_224 = None
        clone_36: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_187: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_36, _shape_param_225);  clone_36 = _shape_param_225 = None
        expand_38: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_158, _shape_param_226);  convert_element_type_158 = _shape_param_226 = None
        clone_37: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_188: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_37, _shape_param_227);  clone_37 = _shape_param_227 = None
        bmm_18: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_187, view_188)
        view_189: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_18, _shape_param_228);  bmm_18 = _shape_param_228 = None
        slice_19: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg136_1, 2, 0, 128);  arg136_1 = None
        slice_20: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_19, 3, 0, 128);  slice_19 = None
        where_10: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_20, view_189, full_3);  slice_20 = view_189 = None
        add_84: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_10, where);  where_10 = None
        amax_9: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_84, [-1], True)
        sub_28: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_84, amax_9);  add_84 = amax_9 = None
        exp_9: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_160: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16)
        expand_39: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_160, _shape_param_229);  convert_element_type_160 = _shape_param_229 = None
        view_190: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_39, _shape_param_230);  expand_39 = _shape_param_230 = None
        expand_40: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_104, _shape_param_231);  permute_104 = _shape_param_231 = None
        clone_38: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_191: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_38, _shape_param_232);  clone_38 = _shape_param_232 = None
        bmm_19: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_190, view_191)
        view_192: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_19, _shape_param_233);  bmm_19 = _shape_param_233 = None
        permute_106: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        clone_39: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_193: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_39, _shape_param_234);  clone_39 = _shape_param_234 = None
        convert_element_type_161: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg137_1, torch.bfloat16);  arg137_1 = None
        convert_element_type_162: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg138_1, torch.bfloat16);  arg138_1 = None
        view_194: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_193, _shape_param_235);  view_193 = _shape_param_235 = None
        permute_107: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_162, [1, 0]);  convert_element_type_162 = None
        addmm_27: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_161, view_194, permute_107);  convert_element_type_161 = None
        view_195: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_27, _shape_param_236);  addmm_27 = _shape_param_236 = None
        add_85: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_195, add_81);  view_195 = add_81 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_38: "f32[32, 128, 1]" = var_mean_19[0]
        getitem_39: "f32[32, 128, 1]" = var_mean_19[1];  var_mean_19 = None
        add_86: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_29: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_85, getitem_39);  getitem_39 = None
        mul_74: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
        mul_75: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_74, arg139_1);  arg139_1 = None
        add_87: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_75, arg140_1);  mul_75 = arg140_1 = None
        convert_element_type_163: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg141_1, torch.bfloat16);  arg141_1 = None
        convert_element_type_164: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg142_1, torch.bfloat16);  arg142_1 = None
        convert_element_type_165: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16);  add_87 = None
        view_196: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_165, _shape_param_237);  convert_element_type_165 = _shape_param_237 = None
        permute_108: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_164, [1, 0]);  convert_element_type_164 = None
        addmm_28: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_163, view_196, permute_108);  convert_element_type_163 = None
        view_197: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_28, _shape_param_238);  _shape_param_238 = None
        mul_76: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
        convert_element_type_166: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_197, torch.float32)
        pow_10: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_166, 3.0);  convert_element_type_166 = None
        mul_77: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_88: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_197, mul_77);  view_197 = mul_77 = None
        mul_78: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_88, 0.7978845608028654);  add_88 = None
        tanh_9: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_89: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_76, add_89);  mul_76 = add_89 = None
        convert_element_type_167: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg143_1, torch.bfloat16);  arg143_1 = None
        convert_element_type_168: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg144_1, torch.bfloat16);  arg144_1 = None
        convert_element_type_169: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None
        view_198: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_169, _shape_param_239);  convert_element_type_169 = _shape_param_239 = None
        permute_109: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_168, [1, 0]);  convert_element_type_168 = None
        addmm_29: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_167, view_198, permute_109);  convert_element_type_167 = None
        view_199: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_29, _shape_param_240);  addmm_29 = _shape_param_240 = None
        add_90: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_85, view_199);  add_85 = view_199 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_90, [2], correction = 0, keepdim = True)
        getitem_40: "f32[32, 128, 1]" = var_mean_20[0]
        getitem_41: "f32[32, 128, 1]" = var_mean_20[1];  var_mean_20 = None
        add_91: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        sub_30: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_90, getitem_41);  getitem_41 = None
        mul_80: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
        mul_81: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_80, arg145_1);  arg145_1 = None
        add_92: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_81, arg146_1);  mul_81 = arg146_1 = None
        convert_element_type_170: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg147_1, torch.bfloat16);  arg147_1 = None
        convert_element_type_171: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_92, torch.bfloat16);  add_92 = None
        permute_110: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_170, [1, 0]);  convert_element_type_170 = None
        view_200: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_171, _shape_param_241);  convert_element_type_171 = _shape_param_241 = None
        mm_30: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_200, permute_110)
        view_201: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_30, _shape_param_242);  mm_30 = _shape_param_242 = None
        convert_element_type_172: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg148_1, torch.bfloat16);  arg148_1 = None
        permute_111: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_172, [1, 0]);  convert_element_type_172 = None
        mm_31: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_200, permute_111)
        view_202: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_31, _shape_param_243);  mm_31 = _shape_param_243 = None
        convert_element_type_173: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg149_1, torch.bfloat16);  arg149_1 = None
        permute_112: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_173, [1, 0]);  convert_element_type_173 = None
        mm_32: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_200, permute_112)
        view_203: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_32, _shape_param_244);  mm_32 = _shape_param_244 = None
        view_204: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_201, _shape_param_245);  view_201 = _shape_param_245 = None
        permute_113: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
        view_205: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_202, _shape_param_246);  view_202 = _shape_param_246 = None
        permute_114: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
        view_206: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_203, _shape_param_247);  view_203 = _shape_param_247 = None
        permute_115: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        convert_element_type_174: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_114, torch.float32);  permute_114 = None
        permute_116: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_174, [0, 1, 3, 2]);  convert_element_type_174 = None
        convert_element_type_175: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_116, torch.bfloat16);  permute_116 = None
        convert_element_type_176: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_113, torch.bfloat16);  permute_113 = None
        expand_41: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_176, _shape_param_248);  convert_element_type_176 = _shape_param_248 = None
        clone_40: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_207: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_40, _shape_param_249);  clone_40 = _shape_param_249 = None
        expand_42: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_175, _shape_param_250);  convert_element_type_175 = _shape_param_250 = None
        clone_41: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_208: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_41, _shape_param_251);  clone_41 = _shape_param_251 = None
        bmm_20: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_207, view_208)
        view_209: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_20, _shape_param_252);  bmm_20 = _shape_param_252 = None
        slice_21: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg150_1, 2, 0, 128);  arg150_1 = None
        slice_22: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_21, 3, 0, 128);  slice_21 = None
        where_11: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_22, view_209, full_3);  slice_22 = view_209 = None
        add_93: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_11, where);  where_11 = None
        amax_10: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_93, [-1], True)
        sub_31: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_93, amax_10);  add_93 = amax_10 = None
        exp_10: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_177: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16)
        expand_43: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_177, _shape_param_253);  convert_element_type_177 = _shape_param_253 = None
        view_210: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_43, _shape_param_254);  expand_43 = _shape_param_254 = None
        expand_44: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_115, _shape_param_255);  permute_115 = _shape_param_255 = None
        clone_42: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_211: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_42, _shape_param_256);  clone_42 = _shape_param_256 = None
        bmm_21: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_210, view_211)
        view_212: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_21, _shape_param_257);  bmm_21 = _shape_param_257 = None
        permute_117: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        clone_43: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_213: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_43, _shape_param_258);  clone_43 = _shape_param_258 = None
        convert_element_type_178: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg151_1, torch.bfloat16);  arg151_1 = None
        convert_element_type_179: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg152_1, torch.bfloat16);  arg152_1 = None
        view_214: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_213, _shape_param_259);  view_213 = _shape_param_259 = None
        permute_118: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_179, [1, 0]);  convert_element_type_179 = None
        addmm_30: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_178, view_214, permute_118);  convert_element_type_178 = None
        view_215: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_30, _shape_param_260);  addmm_30 = _shape_param_260 = None
        add_94: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_215, add_90);  view_215 = add_90 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 128, 1]" = var_mean_21[0]
        getitem_43: "f32[32, 128, 1]" = var_mean_21[1];  var_mean_21 = None
        add_95: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        sub_32: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_94, getitem_43);  getitem_43 = None
        mul_82: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
        mul_83: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_82, arg153_1);  arg153_1 = None
        add_96: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_83, arg154_1);  mul_83 = arg154_1 = None
        convert_element_type_180: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg155_1, torch.bfloat16);  arg155_1 = None
        convert_element_type_181: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg156_1, torch.bfloat16);  arg156_1 = None
        convert_element_type_182: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_96, torch.bfloat16);  add_96 = None
        view_216: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_182, _shape_param_261);  convert_element_type_182 = _shape_param_261 = None
        permute_119: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_181, [1, 0]);  convert_element_type_181 = None
        addmm_31: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_180, view_216, permute_119);  convert_element_type_180 = None
        view_217: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_31, _shape_param_262);  _shape_param_262 = None
        mul_84: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        convert_element_type_183: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_217, torch.float32)
        pow_11: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_183, 3.0);  convert_element_type_183 = None
        mul_85: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_97: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_217, mul_85);  view_217 = mul_85 = None
        mul_86: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_97, 0.7978845608028654);  add_97 = None
        tanh_10: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_98: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_84, add_98);  mul_84 = add_98 = None
        convert_element_type_184: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg157_1, torch.bfloat16);  arg157_1 = None
        convert_element_type_185: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg158_1, torch.bfloat16);  arg158_1 = None
        convert_element_type_186: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_87, torch.bfloat16);  mul_87 = None
        view_218: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_186, _shape_param_263);  convert_element_type_186 = _shape_param_263 = None
        permute_120: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_185, [1, 0]);  convert_element_type_185 = None
        addmm_32: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_184, view_218, permute_120);  convert_element_type_184 = None
        view_219: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_32, _shape_param_264);  addmm_32 = _shape_param_264 = None
        add_99: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_94, view_219);  add_94 = view_219 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 128, 1]" = var_mean_22[0]
        getitem_45: "f32[32, 128, 1]" = var_mean_22[1];  var_mean_22 = None
        add_100: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        sub_33: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_99, getitem_45);  getitem_45 = None
        mul_88: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
        mul_89: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_88, arg159_1);  arg159_1 = None
        add_101: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_89, arg160_1);  mul_89 = arg160_1 = None
        convert_element_type_187: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg161_1, torch.bfloat16);  arg161_1 = None
        convert_element_type_188: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_101, torch.bfloat16);  add_101 = None
        permute_121: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_187, [1, 0]);  convert_element_type_187 = None
        view_220: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_188, _shape_param_265);  convert_element_type_188 = _shape_param_265 = None
        mm_33: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_220, permute_121)
        view_221: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_33, _shape_param_266);  mm_33 = _shape_param_266 = None
        convert_element_type_189: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg162_1, torch.bfloat16);  arg162_1 = None
        permute_122: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_189, [1, 0]);  convert_element_type_189 = None
        mm_34: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_220, permute_122)
        view_222: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_34, _shape_param_267);  mm_34 = _shape_param_267 = None
        convert_element_type_190: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg163_1, torch.bfloat16);  arg163_1 = None
        permute_123: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_190, [1, 0]);  convert_element_type_190 = None
        mm_35: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_220, permute_123)
        view_223: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_35, _shape_param_268);  mm_35 = _shape_param_268 = None
        view_224: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_221, _shape_param_269);  view_221 = _shape_param_269 = None
        permute_124: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
        view_225: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_222, _shape_param_270);  view_222 = _shape_param_270 = None
        permute_125: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
        view_226: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_223, _shape_param_271);  view_223 = _shape_param_271 = None
        permute_126: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        convert_element_type_191: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_125, torch.float32);  permute_125 = None
        permute_127: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_191, [0, 1, 3, 2]);  convert_element_type_191 = None
        convert_element_type_192: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_127, torch.bfloat16);  permute_127 = None
        convert_element_type_193: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_124, torch.bfloat16);  permute_124 = None
        expand_45: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_193, _shape_param_272);  convert_element_type_193 = _shape_param_272 = None
        clone_44: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_227: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_44, _shape_param_273);  clone_44 = _shape_param_273 = None
        expand_46: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_192, _shape_param_274);  convert_element_type_192 = _shape_param_274 = None
        clone_45: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_228: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_45, _shape_param_275);  clone_45 = _shape_param_275 = None
        bmm_22: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_227, view_228)
        view_229: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_276);  bmm_22 = _shape_param_276 = None
        slice_23: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg164_1, 2, 0, 128);  arg164_1 = None
        slice_24: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_23, 3, 0, 128);  slice_23 = None
        where_12: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_24, view_229, full_3);  slice_24 = view_229 = None
        add_102: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_12, where);  where_12 = None
        amax_11: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_102, [-1], True)
        sub_34: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_102, amax_11);  add_102 = amax_11 = None
        exp_11: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_194: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16)
        expand_47: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_194, _shape_param_277);  convert_element_type_194 = _shape_param_277 = None
        view_230: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_47, _shape_param_278);  expand_47 = _shape_param_278 = None
        expand_48: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_126, _shape_param_279);  permute_126 = _shape_param_279 = None
        clone_46: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_231: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_46, _shape_param_280);  clone_46 = _shape_param_280 = None
        bmm_23: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_230, view_231)
        view_232: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_23, _shape_param_281);  bmm_23 = _shape_param_281 = None
        permute_128: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None
        clone_47: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_233: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_47, _shape_param_282);  clone_47 = _shape_param_282 = None
        convert_element_type_195: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg165_1, torch.bfloat16);  arg165_1 = None
        convert_element_type_196: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg166_1, torch.bfloat16);  arg166_1 = None
        view_234: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_233, _shape_param_283);  view_233 = _shape_param_283 = None
        permute_129: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_196, [1, 0]);  convert_element_type_196 = None
        addmm_33: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_195, view_234, permute_129);  convert_element_type_195 = None
        view_235: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_33, _shape_param_284);  addmm_33 = _shape_param_284 = None
        add_103: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_235, add_99);  view_235 = add_99 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 128, 1]" = var_mean_23[0]
        getitem_47: "f32[32, 128, 1]" = var_mean_23[1];  var_mean_23 = None
        add_104: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        sub_35: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_103, getitem_47);  getitem_47 = None
        mul_90: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
        mul_91: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_90, arg167_1);  arg167_1 = None
        add_105: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_91, arg168_1);  mul_91 = arg168_1 = None
        convert_element_type_197: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg169_1, torch.bfloat16);  arg169_1 = None
        convert_element_type_198: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg170_1, torch.bfloat16);  arg170_1 = None
        convert_element_type_199: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_105, torch.bfloat16);  add_105 = None
        view_236: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_199, _shape_param_285);  convert_element_type_199 = _shape_param_285 = None
        permute_130: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_198, [1, 0]);  convert_element_type_198 = None
        addmm_34: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_197, view_236, permute_130);  convert_element_type_197 = None
        view_237: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_34, _shape_param_286);  _shape_param_286 = None
        mul_92: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_237, 0.5)
        convert_element_type_200: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_237, torch.float32)
        pow_12: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_200, 3.0);  convert_element_type_200 = None
        mul_93: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_106: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_237, mul_93);  view_237 = mul_93 = None
        mul_94: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_106, 0.7978845608028654);  add_106 = None
        tanh_11: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_107: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_92, add_107);  mul_92 = add_107 = None
        convert_element_type_201: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg171_1, torch.bfloat16);  arg171_1 = None
        convert_element_type_202: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg172_1, torch.bfloat16);  arg172_1 = None
        convert_element_type_203: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_95, torch.bfloat16);  mul_95 = None
        view_238: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_203, _shape_param_287);  convert_element_type_203 = _shape_param_287 = None
        permute_131: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_202, [1, 0]);  convert_element_type_202 = None
        addmm_35: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_201, view_238, permute_131);  convert_element_type_201 = None
        view_239: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_35, _shape_param_288);  addmm_35 = _shape_param_288 = None
        add_108: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_103, view_239);  add_103 = view_239 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_108, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 128, 1]" = var_mean_24[0]
        getitem_49: "f32[32, 128, 1]" = var_mean_24[1];  var_mean_24 = None
        add_109: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        sub_36: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_108, getitem_49);  getitem_49 = None
        mul_96: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
        mul_97: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_96, arg173_1);  arg173_1 = None
        add_110: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_97, arg174_1);  mul_97 = arg174_1 = None
        convert_element_type_204: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg175_1, torch.bfloat16);  arg175_1 = None
        convert_element_type_205: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_110, torch.bfloat16);  add_110 = None
        permute_132: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_204, [1, 0]);  convert_element_type_204 = None
        view_240: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_205, _shape_param_289);  convert_element_type_205 = _shape_param_289 = None
        mm_36: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_240, permute_132)
        view_241: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_36, _shape_param_290);  mm_36 = _shape_param_290 = None
        convert_element_type_206: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg176_1, torch.bfloat16);  arg176_1 = None
        permute_133: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_206, [1, 0]);  convert_element_type_206 = None
        mm_37: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_240, permute_133)
        view_242: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_37, _shape_param_291);  mm_37 = _shape_param_291 = None
        convert_element_type_207: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg177_1, torch.bfloat16);  arg177_1 = None
        permute_134: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_207, [1, 0]);  convert_element_type_207 = None
        mm_38: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_240, permute_134)
        view_243: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_38, _shape_param_292);  mm_38 = _shape_param_292 = None
        view_244: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_241, _shape_param_293);  view_241 = _shape_param_293 = None
        permute_135: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
        view_245: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_242, _shape_param_294);  view_242 = _shape_param_294 = None
        permute_136: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None
        view_246: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_243, _shape_param_295);  view_243 = _shape_param_295 = None
        permute_137: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        convert_element_type_208: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_136, torch.float32);  permute_136 = None
        permute_138: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_208, [0, 1, 3, 2]);  convert_element_type_208 = None
        convert_element_type_209: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_138, torch.bfloat16);  permute_138 = None
        convert_element_type_210: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_135, torch.bfloat16);  permute_135 = None
        expand_49: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_210, _shape_param_296);  convert_element_type_210 = _shape_param_296 = None
        clone_48: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_247: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_48, _shape_param_297);  clone_48 = _shape_param_297 = None
        expand_50: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_209, _shape_param_298);  convert_element_type_209 = _shape_param_298 = None
        clone_49: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_248: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_49, _shape_param_299);  clone_49 = _shape_param_299 = None
        bmm_24: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_247, view_248)
        view_249: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_24, _shape_param_300);  bmm_24 = _shape_param_300 = None
        slice_25: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg178_1, 2, 0, 128);  arg178_1 = None
        slice_26: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_25, 3, 0, 128);  slice_25 = None
        where_13: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_26, view_249, full_3);  slice_26 = view_249 = None
        add_111: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_13, where);  where_13 = None
        amax_12: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_111, [-1], True)
        sub_37: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_111, amax_12);  add_111 = amax_12 = None
        exp_12: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_13: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_211: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16)
        expand_51: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_211, _shape_param_301);  convert_element_type_211 = _shape_param_301 = None
        view_250: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_51, _shape_param_302);  expand_51 = _shape_param_302 = None
        expand_52: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_137, _shape_param_303);  permute_137 = _shape_param_303 = None
        clone_50: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_251: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_50, _shape_param_304);  clone_50 = _shape_param_304 = None
        bmm_25: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_250, view_251)
        view_252: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_25, _shape_param_305);  bmm_25 = _shape_param_305 = None
        permute_139: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
        clone_51: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        view_253: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_51, _shape_param_306);  clone_51 = _shape_param_306 = None
        convert_element_type_212: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg179_1, torch.bfloat16);  arg179_1 = None
        convert_element_type_213: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg180_1, torch.bfloat16);  arg180_1 = None
        view_254: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_253, _shape_param_307);  view_253 = _shape_param_307 = None
        permute_140: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_213, [1, 0]);  convert_element_type_213 = None
        addmm_36: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_212, view_254, permute_140);  convert_element_type_212 = None
        view_255: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_36, _shape_param_308);  addmm_36 = _shape_param_308 = None
        add_112: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_255, add_108);  view_255 = add_108 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_112, [2], correction = 0, keepdim = True)
        getitem_50: "f32[32, 128, 1]" = var_mean_25[0]
        getitem_51: "f32[32, 128, 1]" = var_mean_25[1];  var_mean_25 = None
        add_113: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_25: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        sub_38: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_112, getitem_51);  getitem_51 = None
        mul_98: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = None
        mul_99: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_98, arg181_1);  arg181_1 = None
        add_114: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_99, arg182_1);  mul_99 = arg182_1 = None
        convert_element_type_214: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg183_1, torch.bfloat16);  arg183_1 = None
        convert_element_type_215: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg184_1, torch.bfloat16);  arg184_1 = None
        convert_element_type_216: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_114, torch.bfloat16);  add_114 = None
        view_256: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_216, _shape_param_309);  convert_element_type_216 = _shape_param_309 = None
        permute_141: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_215, [1, 0]);  convert_element_type_215 = None
        addmm_37: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_214, view_256, permute_141);  convert_element_type_214 = None
        view_257: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_37, _shape_param_310);  _shape_param_310 = None
        mul_100: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_257, 0.5)
        convert_element_type_217: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_257, torch.float32)
        pow_13: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_217, 3.0);  convert_element_type_217 = None
        mul_101: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_115: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_257, mul_101);  view_257 = mul_101 = None
        mul_102: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_115, 0.7978845608028654);  add_115 = None
        tanh_12: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_102);  mul_102 = None
        add_116: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_103: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_100, add_116);  mul_100 = add_116 = None
        convert_element_type_218: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg185_1, torch.bfloat16);  arg185_1 = None
        convert_element_type_219: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg186_1, torch.bfloat16);  arg186_1 = None
        convert_element_type_220: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_103, torch.bfloat16);  mul_103 = None
        view_258: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_220, _shape_param_311);  convert_element_type_220 = _shape_param_311 = None
        permute_142: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_219, [1, 0]);  convert_element_type_219 = None
        addmm_38: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_218, view_258, permute_142);  convert_element_type_218 = None
        view_259: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_38, _shape_param_312);  addmm_38 = _shape_param_312 = None
        add_117: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_112, view_259);  add_112 = view_259 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
        getitem_52: "f32[32, 128, 1]" = var_mean_26[0]
        getitem_53: "f32[32, 128, 1]" = var_mean_26[1];  var_mean_26 = None
        add_118: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_26: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_39: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_117, getitem_53);  getitem_53 = None
        mul_104: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = None
        mul_105: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_104, arg187_1);  arg187_1 = None
        add_119: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_105, arg188_1);  mul_105 = arg188_1 = None
        convert_element_type_221: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg189_1, torch.bfloat16);  arg189_1 = None
        convert_element_type_222: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None
        permute_143: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_221, [1, 0]);  convert_element_type_221 = None
        view_260: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_222, _shape_param_313);  convert_element_type_222 = _shape_param_313 = None
        mm_39: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_260, permute_143)
        view_261: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_39, _shape_param_314);  mm_39 = _shape_param_314 = None
        convert_element_type_223: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg190_1, torch.bfloat16);  arg190_1 = None
        permute_144: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_223, [1, 0]);  convert_element_type_223 = None
        mm_40: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_260, permute_144)
        view_262: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_40, _shape_param_315);  mm_40 = _shape_param_315 = None
        convert_element_type_224: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg191_1, torch.bfloat16);  arg191_1 = None
        permute_145: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_224, [1, 0]);  convert_element_type_224 = None
        mm_41: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_260, permute_145)
        view_263: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_41, _shape_param_316);  mm_41 = _shape_param_316 = None
        view_264: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_261, _shape_param_317);  view_261 = _shape_param_317 = None
        permute_146: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None
        view_265: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_262, _shape_param_318);  view_262 = _shape_param_318 = None
        permute_147: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
        view_266: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_263, _shape_param_319);  view_263 = _shape_param_319 = None
        permute_148: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None
        convert_element_type_225: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_147, torch.float32);  permute_147 = None
        permute_149: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_225, [0, 1, 3, 2]);  convert_element_type_225 = None
        convert_element_type_226: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_149, torch.bfloat16);  permute_149 = None
        convert_element_type_227: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_146, torch.bfloat16);  permute_146 = None
        expand_53: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_227, _shape_param_320);  convert_element_type_227 = _shape_param_320 = None
        clone_52: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_267: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_52, _shape_param_321);  clone_52 = _shape_param_321 = None
        expand_54: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_226, _shape_param_322);  convert_element_type_226 = _shape_param_322 = None
        clone_53: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_268: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_53, _shape_param_323);  clone_53 = _shape_param_323 = None
        bmm_26: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_267, view_268)
        view_269: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_26, _shape_param_324);  bmm_26 = _shape_param_324 = None
        slice_27: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg192_1, 2, 0, 128);  arg192_1 = None
        slice_28: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_27, 3, 0, 128);  slice_27 = None
        where_14: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_28, view_269, full_3);  slice_28 = view_269 = None
        add_120: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_14, where);  where_14 = None
        amax_13: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_120, [-1], True)
        sub_40: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_120, amax_13);  add_120 = amax_13 = None
        exp_13: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        sum_14: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_228: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16)
        expand_55: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_228, _shape_param_325);  convert_element_type_228 = _shape_param_325 = None
        view_270: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_55, _shape_param_326);  expand_55 = _shape_param_326 = None
        expand_56: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_148, _shape_param_327);  permute_148 = _shape_param_327 = None
        clone_54: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_271: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_54, _shape_param_328);  clone_54 = _shape_param_328 = None
        bmm_27: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_270, view_271)
        view_272: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_27, _shape_param_329);  bmm_27 = _shape_param_329 = None
        permute_150: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
        clone_55: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_273: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_55, _shape_param_330);  clone_55 = _shape_param_330 = None
        convert_element_type_229: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg193_1, torch.bfloat16);  arg193_1 = None
        convert_element_type_230: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg194_1, torch.bfloat16);  arg194_1 = None
        view_274: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_273, _shape_param_331);  view_273 = _shape_param_331 = None
        permute_151: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_230, [1, 0]);  convert_element_type_230 = None
        addmm_39: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_229, view_274, permute_151);  convert_element_type_229 = None
        view_275: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_39, _shape_param_332);  addmm_39 = _shape_param_332 = None
        add_121: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_275, add_117);  view_275 = add_117 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
        getitem_54: "f32[32, 128, 1]" = var_mean_27[0]
        getitem_55: "f32[32, 128, 1]" = var_mean_27[1];  var_mean_27 = None
        add_122: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_27: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_41: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_121, getitem_55);  getitem_55 = None
        mul_106: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = None
        mul_107: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_106, arg195_1);  arg195_1 = None
        add_123: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_107, arg196_1);  mul_107 = arg196_1 = None
        convert_element_type_231: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg197_1, torch.bfloat16);  arg197_1 = None
        convert_element_type_232: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg198_1, torch.bfloat16);  arg198_1 = None
        convert_element_type_233: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_123, torch.bfloat16);  add_123 = None
        view_276: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_233, _shape_param_333);  convert_element_type_233 = _shape_param_333 = None
        permute_152: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_232, [1, 0]);  convert_element_type_232 = None
        addmm_40: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_231, view_276, permute_152);  convert_element_type_231 = None
        view_277: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_40, _shape_param_334);  _shape_param_334 = None
        mul_108: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_277, 0.5)
        convert_element_type_234: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_277, torch.float32)
        pow_14: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_234, 3.0);  convert_element_type_234 = None
        mul_109: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_124: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_277, mul_109);  view_277 = mul_109 = None
        mul_110: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_124, 0.7978845608028654);  add_124 = None
        tanh_13: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_110);  mul_110 = None
        add_125: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_111: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_108, add_125);  mul_108 = add_125 = None
        convert_element_type_235: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg199_1, torch.bfloat16);  arg199_1 = None
        convert_element_type_236: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg200_1, torch.bfloat16);  arg200_1 = None
        convert_element_type_237: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_111, torch.bfloat16);  mul_111 = None
        view_278: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_237, _shape_param_335);  convert_element_type_237 = _shape_param_335 = None
        permute_153: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_236, [1, 0]);  convert_element_type_236 = None
        addmm_41: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_235, view_278, permute_153);  convert_element_type_235 = None
        view_279: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_41, _shape_param_336);  addmm_41 = _shape_param_336 = None
        add_126: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_121, view_279);  add_121 = view_279 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add_126, [2], correction = 0, keepdim = True)
        getitem_56: "f32[32, 128, 1]" = var_mean_28[0]
        getitem_57: "f32[32, 128, 1]" = var_mean_28[1];  var_mean_28 = None
        add_127: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_28: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        sub_42: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_126, getitem_57);  getitem_57 = None
        mul_112: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = None
        mul_113: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_112, arg201_1);  arg201_1 = None
        add_128: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_113, arg202_1);  mul_113 = arg202_1 = None
        convert_element_type_238: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg203_1, torch.bfloat16);  arg203_1 = None
        convert_element_type_239: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_128, torch.bfloat16);  add_128 = None
        permute_154: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_238, [1, 0]);  convert_element_type_238 = None
        view_280: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_239, _shape_param_337);  convert_element_type_239 = _shape_param_337 = None
        mm_42: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_280, permute_154)
        view_281: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_42, _shape_param_338);  mm_42 = _shape_param_338 = None
        convert_element_type_240: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg204_1, torch.bfloat16);  arg204_1 = None
        permute_155: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_240, [1, 0]);  convert_element_type_240 = None
        mm_43: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_280, permute_155)
        view_282: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_43, _shape_param_339);  mm_43 = _shape_param_339 = None
        convert_element_type_241: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg205_1, torch.bfloat16);  arg205_1 = None
        permute_156: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_241, [1, 0]);  convert_element_type_241 = None
        mm_44: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_280, permute_156)
        view_283: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_44, _shape_param_340);  mm_44 = _shape_param_340 = None
        view_284: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_281, _shape_param_341);  view_281 = _shape_param_341 = None
        permute_157: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None
        view_285: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_282, _shape_param_342);  view_282 = _shape_param_342 = None
        permute_158: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
        view_286: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_283, _shape_param_343);  view_283 = _shape_param_343 = None
        permute_159: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
        convert_element_type_242: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_158, torch.float32);  permute_158 = None
        permute_160: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_242, [0, 1, 3, 2]);  convert_element_type_242 = None
        convert_element_type_243: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_160, torch.bfloat16);  permute_160 = None
        convert_element_type_244: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_157, torch.bfloat16);  permute_157 = None
        expand_57: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_244, _shape_param_344);  convert_element_type_244 = _shape_param_344 = None
        clone_56: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_287: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_56, _shape_param_345);  clone_56 = _shape_param_345 = None
        expand_58: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_243, _shape_param_346);  convert_element_type_243 = _shape_param_346 = None
        clone_57: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_288: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_57, _shape_param_347);  clone_57 = _shape_param_347 = None
        bmm_28: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_287, view_288)
        view_289: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_28, _shape_param_348);  bmm_28 = _shape_param_348 = None
        slice_29: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg206_1, 2, 0, 128);  arg206_1 = None
        slice_30: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_29, 3, 0, 128);  slice_29 = None
        where_15: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_30, view_289, full_3);  slice_30 = view_289 = None
        add_129: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_15, where);  where_15 = None
        amax_14: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_129, [-1], True)
        sub_43: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_129, amax_14);  add_129 = amax_14 = None
        exp_14: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_15: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_245: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16)
        expand_59: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_245, _shape_param_349);  convert_element_type_245 = _shape_param_349 = None
        view_290: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_59, _shape_param_350);  expand_59 = _shape_param_350 = None
        expand_60: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_159, _shape_param_351);  permute_159 = _shape_param_351 = None
        clone_58: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_291: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_58, _shape_param_352);  clone_58 = _shape_param_352 = None
        bmm_29: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_290, view_291)
        view_292: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_29, _shape_param_353);  bmm_29 = _shape_param_353 = None
        permute_161: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
        clone_59: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_293: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_59, _shape_param_354);  clone_59 = _shape_param_354 = None
        convert_element_type_246: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg207_1, torch.bfloat16);  arg207_1 = None
        convert_element_type_247: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg208_1, torch.bfloat16);  arg208_1 = None
        view_294: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_293, _shape_param_355);  view_293 = _shape_param_355 = None
        permute_162: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_247, [1, 0]);  convert_element_type_247 = None
        addmm_42: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_246, view_294, permute_162);  convert_element_type_246 = None
        view_295: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_42, _shape_param_356);  addmm_42 = _shape_param_356 = None
        add_130: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_295, add_126);  view_295 = add_126 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(add_130, [2], correction = 0, keepdim = True)
        getitem_58: "f32[32, 128, 1]" = var_mean_29[0]
        getitem_59: "f32[32, 128, 1]" = var_mean_29[1];  var_mean_29 = None
        add_131: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_29: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        sub_44: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_130, getitem_59);  getitem_59 = None
        mul_114: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = None
        mul_115: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_114, arg209_1);  arg209_1 = None
        add_132: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_115, arg210_1);  mul_115 = arg210_1 = None
        convert_element_type_248: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg211_1, torch.bfloat16);  arg211_1 = None
        convert_element_type_249: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg212_1, torch.bfloat16);  arg212_1 = None
        convert_element_type_250: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_132, torch.bfloat16);  add_132 = None
        view_296: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_250, _shape_param_357);  convert_element_type_250 = _shape_param_357 = None
        permute_163: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_249, [1, 0]);  convert_element_type_249 = None
        addmm_43: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_248, view_296, permute_163);  convert_element_type_248 = None
        view_297: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_43, _shape_param_358);  _shape_param_358 = None
        mul_116: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_297, 0.5)
        convert_element_type_251: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_297, torch.float32)
        pow_15: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_251, 3.0);  convert_element_type_251 = None
        mul_117: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_133: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_297, mul_117);  view_297 = mul_117 = None
        mul_118: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_133, 0.7978845608028654);  add_133 = None
        tanh_14: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_134: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_119: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_116, add_134);  mul_116 = add_134 = None
        convert_element_type_252: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg213_1, torch.bfloat16);  arg213_1 = None
        convert_element_type_253: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg214_1, torch.bfloat16);  arg214_1 = None
        convert_element_type_254: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_119, torch.bfloat16);  mul_119 = None
        view_298: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_254, _shape_param_359);  convert_element_type_254 = _shape_param_359 = None
        permute_164: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_253, [1, 0]);  convert_element_type_253 = None
        addmm_44: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_252, view_298, permute_164);  convert_element_type_252 = None
        view_299: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_44, _shape_param_360);  addmm_44 = _shape_param_360 = None
        add_135: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_130, view_299);  add_130 = view_299 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(add_135, [2], correction = 0, keepdim = True)
        getitem_60: "f32[32, 128, 1]" = var_mean_30[0]
        getitem_61: "f32[32, 128, 1]" = var_mean_30[1];  var_mean_30 = None
        add_136: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_30: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        sub_45: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_135, getitem_61);  getitem_61 = None
        mul_120: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = None
        mul_121: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_120, arg215_1);  arg215_1 = None
        add_137: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_121, arg216_1);  mul_121 = arg216_1 = None
        convert_element_type_255: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg217_1, torch.bfloat16);  arg217_1 = None
        convert_element_type_256: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_137, torch.bfloat16);  add_137 = None
        permute_165: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_255, [1, 0]);  convert_element_type_255 = None
        view_300: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_256, _shape_param_361);  convert_element_type_256 = _shape_param_361 = None
        mm_45: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_300, permute_165)
        view_301: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_45, _shape_param_362);  mm_45 = _shape_param_362 = None
        convert_element_type_257: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg218_1, torch.bfloat16);  arg218_1 = None
        permute_166: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_257, [1, 0]);  convert_element_type_257 = None
        mm_46: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_300, permute_166)
        view_302: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_46, _shape_param_363);  mm_46 = _shape_param_363 = None
        convert_element_type_258: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg219_1, torch.bfloat16);  arg219_1 = None
        permute_167: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_258, [1, 0]);  convert_element_type_258 = None
        mm_47: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_300, permute_167)
        view_303: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_47, _shape_param_364);  mm_47 = _shape_param_364 = None
        view_304: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_301, _shape_param_365);  view_301 = _shape_param_365 = None
        permute_168: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None
        view_305: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_302, _shape_param_366);  view_302 = _shape_param_366 = None
        permute_169: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_305, [0, 2, 1, 3]);  view_305 = None
        view_306: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_303, _shape_param_367);  view_303 = _shape_param_367 = None
        permute_170: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None
        convert_element_type_259: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_169, torch.float32);  permute_169 = None
        permute_171: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_259, [0, 1, 3, 2]);  convert_element_type_259 = None
        convert_element_type_260: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_171, torch.bfloat16);  permute_171 = None
        convert_element_type_261: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_168, torch.bfloat16);  permute_168 = None
        expand_61: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_261, _shape_param_368);  convert_element_type_261 = _shape_param_368 = None
        clone_60: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_307: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_60, _shape_param_369);  clone_60 = _shape_param_369 = None
        expand_62: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_260, _shape_param_370);  convert_element_type_260 = _shape_param_370 = None
        clone_61: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_308: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_61, _shape_param_371);  clone_61 = _shape_param_371 = None
        bmm_30: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_307, view_308)
        view_309: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_30, _shape_param_372);  bmm_30 = _shape_param_372 = None
        slice_31: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg220_1, 2, 0, 128);  arg220_1 = None
        slice_32: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_31, 3, 0, 128);  slice_31 = None
        where_16: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_32, view_309, full_3);  slice_32 = view_309 = None
        add_138: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_16, where);  where_16 = None
        amax_15: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_138, [-1], True)
        sub_46: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_138, amax_15);  add_138 = amax_15 = None
        exp_15: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_16: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_262: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16)
        expand_63: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_262, _shape_param_373);  convert_element_type_262 = _shape_param_373 = None
        view_310: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_63, _shape_param_374);  expand_63 = _shape_param_374 = None
        expand_64: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_170, _shape_param_375);  permute_170 = _shape_param_375 = None
        clone_62: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_311: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_62, _shape_param_376);  clone_62 = _shape_param_376 = None
        bmm_31: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_310, view_311)
        view_312: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_31, _shape_param_377);  bmm_31 = _shape_param_377 = None
        permute_172: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None
        clone_63: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_313: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_63, _shape_param_378);  clone_63 = _shape_param_378 = None
        convert_element_type_263: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg221_1, torch.bfloat16);  arg221_1 = None
        convert_element_type_264: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg222_1, torch.bfloat16);  arg222_1 = None
        view_314: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_313, _shape_param_379);  view_313 = _shape_param_379 = None
        permute_173: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_264, [1, 0]);  convert_element_type_264 = None
        addmm_45: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_263, view_314, permute_173);  convert_element_type_263 = None
        view_315: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_45, _shape_param_380);  addmm_45 = _shape_param_380 = None
        add_139: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_315, add_135);  view_315 = add_135 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(add_139, [2], correction = 0, keepdim = True)
        getitem_62: "f32[32, 128, 1]" = var_mean_31[0]
        getitem_63: "f32[32, 128, 1]" = var_mean_31[1];  var_mean_31 = None
        add_140: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_31: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        sub_47: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_139, getitem_63);  getitem_63 = None
        mul_122: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = None
        mul_123: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_122, arg223_1);  arg223_1 = None
        add_141: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_123, arg224_1);  mul_123 = arg224_1 = None
        convert_element_type_265: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg225_1, torch.bfloat16);  arg225_1 = None
        convert_element_type_266: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg226_1, torch.bfloat16);  arg226_1 = None
        convert_element_type_267: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_141, torch.bfloat16);  add_141 = None
        view_316: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_267, _shape_param_381);  convert_element_type_267 = _shape_param_381 = None
        permute_174: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_266, [1, 0]);  convert_element_type_266 = None
        addmm_46: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_265, view_316, permute_174);  convert_element_type_265 = None
        view_317: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_46, _shape_param_382);  _shape_param_382 = None
        mul_124: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_317, 0.5)
        convert_element_type_268: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_317, torch.float32)
        pow_16: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_268, 3.0);  convert_element_type_268 = None
        mul_125: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_142: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_317, mul_125);  view_317 = mul_125 = None
        mul_126: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_142, 0.7978845608028654);  add_142 = None
        tanh_15: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_126);  mul_126 = None
        add_143: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_127: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_124, add_143);  mul_124 = add_143 = None
        convert_element_type_269: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg227_1, torch.bfloat16);  arg227_1 = None
        convert_element_type_270: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg228_1, torch.bfloat16);  arg228_1 = None
        convert_element_type_271: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_127, torch.bfloat16);  mul_127 = None
        view_318: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_271, _shape_param_383);  convert_element_type_271 = _shape_param_383 = None
        permute_175: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_270, [1, 0]);  convert_element_type_270 = None
        addmm_47: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_269, view_318, permute_175);  convert_element_type_269 = None
        view_319: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_47, _shape_param_384);  addmm_47 = _shape_param_384 = None
        add_144: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_139, view_319);  add_139 = view_319 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(add_144, [2], correction = 0, keepdim = True)
        getitem_64: "f32[32, 128, 1]" = var_mean_32[0]
        getitem_65: "f32[32, 128, 1]" = var_mean_32[1];  var_mean_32 = None
        add_145: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_32: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
        sub_48: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_144, getitem_65);  getitem_65 = None
        mul_128: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = None
        mul_129: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_128, arg229_1);  arg229_1 = None
        add_146: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_129, arg230_1);  mul_129 = arg230_1 = None
        convert_element_type_272: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg231_1, torch.bfloat16);  arg231_1 = None
        convert_element_type_273: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_146, torch.bfloat16);  add_146 = None
        permute_176: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_272, [1, 0]);  convert_element_type_272 = None
        view_320: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_273, _shape_param_385);  convert_element_type_273 = _shape_param_385 = None
        mm_48: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_320, permute_176)
        view_321: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_48, _shape_param_386);  mm_48 = _shape_param_386 = None
        convert_element_type_274: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg232_1, torch.bfloat16);  arg232_1 = None
        permute_177: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_274, [1, 0]);  convert_element_type_274 = None
        mm_49: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_320, permute_177)
        view_322: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_49, _shape_param_387);  mm_49 = _shape_param_387 = None
        convert_element_type_275: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg233_1, torch.bfloat16);  arg233_1 = None
        permute_178: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_275, [1, 0]);  convert_element_type_275 = None
        mm_50: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_320, permute_178)
        view_323: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_50, _shape_param_388);  mm_50 = _shape_param_388 = None
        view_324: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_321, _shape_param_389);  view_321 = _shape_param_389 = None
        permute_179: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
        view_325: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_322, _shape_param_390);  view_322 = _shape_param_390 = None
        permute_180: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_325, [0, 2, 1, 3]);  view_325 = None
        view_326: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_323, _shape_param_391);  view_323 = _shape_param_391 = None
        permute_181: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None
        convert_element_type_276: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_180, torch.float32);  permute_180 = None
        permute_182: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_276, [0, 1, 3, 2]);  convert_element_type_276 = None
        convert_element_type_277: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_182, torch.bfloat16);  permute_182 = None
        convert_element_type_278: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_179, torch.bfloat16);  permute_179 = None
        expand_65: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_278, _shape_param_392);  convert_element_type_278 = _shape_param_392 = None
        clone_64: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_327: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_64, _shape_param_393);  clone_64 = _shape_param_393 = None
        expand_66: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_277, _shape_param_394);  convert_element_type_277 = _shape_param_394 = None
        clone_65: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_328: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_65, _shape_param_395);  clone_65 = _shape_param_395 = None
        bmm_32: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_327, view_328)
        view_329: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_32, _shape_param_396);  bmm_32 = _shape_param_396 = None
        slice_33: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg234_1, 2, 0, 128);  arg234_1 = None
        slice_34: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_33, 3, 0, 128);  slice_33 = None
        where_17: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_34, view_329, full_3);  slice_34 = view_329 = None
        add_147: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_17, where);  where_17 = None
        amax_16: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_147, [-1], True)
        sub_49: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_147, amax_16);  add_147 = amax_16 = None
        exp_16: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_17: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_279: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16)
        expand_67: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_279, _shape_param_397);  convert_element_type_279 = _shape_param_397 = None
        view_330: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_67, _shape_param_398);  expand_67 = _shape_param_398 = None
        expand_68: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_181, _shape_param_399);  permute_181 = _shape_param_399 = None
        clone_66: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_331: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_66, _shape_param_400);  clone_66 = _shape_param_400 = None
        bmm_33: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_330, view_331)
        view_332: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_33, _shape_param_401);  bmm_33 = _shape_param_401 = None
        permute_183: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
        clone_67: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        view_333: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_67, _shape_param_402);  clone_67 = _shape_param_402 = None
        convert_element_type_280: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg235_1, torch.bfloat16);  arg235_1 = None
        convert_element_type_281: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg236_1, torch.bfloat16);  arg236_1 = None
        view_334: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_333, _shape_param_403);  view_333 = _shape_param_403 = None
        permute_184: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_281, [1, 0]);  convert_element_type_281 = None
        addmm_48: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_280, view_334, permute_184);  convert_element_type_280 = None
        view_335: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_48, _shape_param_404);  addmm_48 = _shape_param_404 = None
        add_148: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_335, add_144);  view_335 = add_144 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(add_148, [2], correction = 0, keepdim = True)
        getitem_66: "f32[32, 128, 1]" = var_mean_33[0]
        getitem_67: "f32[32, 128, 1]" = var_mean_33[1];  var_mean_33 = None
        add_149: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_33: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        sub_50: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_148, getitem_67);  getitem_67 = None
        mul_130: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = None
        mul_131: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_130, arg237_1);  arg237_1 = None
        add_150: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_131, arg238_1);  mul_131 = arg238_1 = None
        convert_element_type_282: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg239_1, torch.bfloat16);  arg239_1 = None
        convert_element_type_283: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg240_1, torch.bfloat16);  arg240_1 = None
        convert_element_type_284: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_150, torch.bfloat16);  add_150 = None
        view_336: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_284, _shape_param_405);  convert_element_type_284 = _shape_param_405 = None
        permute_185: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_283, [1, 0]);  convert_element_type_283 = None
        addmm_49: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_282, view_336, permute_185);  convert_element_type_282 = None
        view_337: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_49, _shape_param_406);  _shape_param_406 = None
        mul_132: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_337, 0.5)
        convert_element_type_285: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_337, torch.float32)
        pow_17: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_285, 3.0);  convert_element_type_285 = None
        mul_133: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_151: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_337, mul_133);  view_337 = mul_133 = None
        mul_134: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_151, 0.7978845608028654);  add_151 = None
        tanh_16: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_134);  mul_134 = None
        add_152: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_16, 1.0);  tanh_16 = None
        mul_135: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_132, add_152);  mul_132 = add_152 = None
        convert_element_type_286: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg241_1, torch.bfloat16);  arg241_1 = None
        convert_element_type_287: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg242_1, torch.bfloat16);  arg242_1 = None
        convert_element_type_288: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_135, torch.bfloat16);  mul_135 = None
        view_338: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_288, _shape_param_407);  convert_element_type_288 = _shape_param_407 = None
        permute_186: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_287, [1, 0]);  convert_element_type_287 = None
        addmm_50: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_286, view_338, permute_186);  convert_element_type_286 = None
        view_339: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_50, _shape_param_408);  addmm_50 = _shape_param_408 = None
        add_153: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_148, view_339);  add_148 = view_339 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(add_153, [2], correction = 0, keepdim = True)
        getitem_68: "f32[32, 128, 1]" = var_mean_34[0]
        getitem_69: "f32[32, 128, 1]" = var_mean_34[1];  var_mean_34 = None
        add_154: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_34: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_51: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_153, getitem_69);  getitem_69 = None
        mul_136: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = None
        mul_137: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_136, arg243_1);  arg243_1 = None
        add_155: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_137, arg244_1);  mul_137 = arg244_1 = None
        convert_element_type_289: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg245_1, torch.bfloat16);  arg245_1 = None
        convert_element_type_290: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_155, torch.bfloat16);  add_155 = None
        permute_187: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_289, [1, 0]);  convert_element_type_289 = None
        view_340: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_290, _shape_param_409);  convert_element_type_290 = _shape_param_409 = None
        mm_51: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_340, permute_187)
        view_341: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_51, _shape_param_410);  mm_51 = _shape_param_410 = None
        convert_element_type_291: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg246_1, torch.bfloat16);  arg246_1 = None
        permute_188: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_291, [1, 0]);  convert_element_type_291 = None
        mm_52: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_340, permute_188)
        view_342: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_52, _shape_param_411);  mm_52 = _shape_param_411 = None
        convert_element_type_292: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg247_1, torch.bfloat16);  arg247_1 = None
        permute_189: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_292, [1, 0]);  convert_element_type_292 = None
        mm_53: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_340, permute_189)
        view_343: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_53, _shape_param_412);  mm_53 = _shape_param_412 = None
        view_344: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_341, _shape_param_413);  view_341 = _shape_param_413 = None
        permute_190: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
        view_345: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_342, _shape_param_414);  view_342 = _shape_param_414 = None
        permute_191: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
        view_346: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_343, _shape_param_415);  view_343 = _shape_param_415 = None
        permute_192: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
        convert_element_type_293: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_191, torch.float32);  permute_191 = None
        permute_193: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_293, [0, 1, 3, 2]);  convert_element_type_293 = None
        convert_element_type_294: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_193, torch.bfloat16);  permute_193 = None
        convert_element_type_295: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_190, torch.bfloat16);  permute_190 = None
        expand_69: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_295, _shape_param_416);  convert_element_type_295 = _shape_param_416 = None
        clone_68: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_347: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_68, _shape_param_417);  clone_68 = _shape_param_417 = None
        expand_70: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_294, _shape_param_418);  convert_element_type_294 = _shape_param_418 = None
        clone_69: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_348: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_69, _shape_param_419);  clone_69 = _shape_param_419 = None
        bmm_34: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_347, view_348)
        view_349: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_34, _shape_param_420);  bmm_34 = _shape_param_420 = None
        slice_35: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg248_1, 2, 0, 128);  arg248_1 = None
        slice_36: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_35, 3, 0, 128);  slice_35 = None
        where_18: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_36, view_349, full_3);  slice_36 = view_349 = None
        add_156: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_18, where);  where_18 = None
        amax_17: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_156, [-1], True)
        sub_52: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_156, amax_17);  add_156 = amax_17 = None
        exp_17: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_18: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_17: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_296: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16)
        expand_71: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_296, _shape_param_421);  convert_element_type_296 = _shape_param_421 = None
        view_350: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_71, _shape_param_422);  expand_71 = _shape_param_422 = None
        expand_72: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_192, _shape_param_423);  permute_192 = _shape_param_423 = None
        clone_70: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_351: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_70, _shape_param_424);  clone_70 = _shape_param_424 = None
        bmm_35: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_350, view_351)
        view_352: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_35, _shape_param_425);  bmm_35 = _shape_param_425 = None
        permute_194: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None
        clone_71: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_353: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_71, _shape_param_426);  clone_71 = _shape_param_426 = None
        convert_element_type_297: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg249_1, torch.bfloat16);  arg249_1 = None
        convert_element_type_298: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg250_1, torch.bfloat16);  arg250_1 = None
        view_354: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_353, _shape_param_427);  view_353 = _shape_param_427 = None
        permute_195: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_298, [1, 0]);  convert_element_type_298 = None
        addmm_51: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_297, view_354, permute_195);  convert_element_type_297 = None
        view_355: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_51, _shape_param_428);  addmm_51 = _shape_param_428 = None
        add_157: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_355, add_153);  view_355 = add_153 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
        getitem_70: "f32[32, 128, 1]" = var_mean_35[0]
        getitem_71: "f32[32, 128, 1]" = var_mean_35[1];  var_mean_35 = None
        add_158: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
        rsqrt_35: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        sub_53: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_157, getitem_71);  getitem_71 = None
        mul_138: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = None
        mul_139: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_138, arg251_1);  arg251_1 = None
        add_159: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_139, arg252_1);  mul_139 = arg252_1 = None
        convert_element_type_299: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg253_1, torch.bfloat16);  arg253_1 = None
        convert_element_type_300: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg254_1, torch.bfloat16);  arg254_1 = None
        convert_element_type_301: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_159, torch.bfloat16);  add_159 = None
        view_356: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_301, _shape_param_429);  convert_element_type_301 = _shape_param_429 = None
        permute_196: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_300, [1, 0]);  convert_element_type_300 = None
        addmm_52: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_299, view_356, permute_196);  convert_element_type_299 = None
        view_357: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_52, _shape_param_430);  _shape_param_430 = None
        mul_140: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_357, 0.5)
        convert_element_type_302: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_357, torch.float32)
        pow_18: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_302, 3.0);  convert_element_type_302 = None
        mul_141: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_160: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_357, mul_141);  view_357 = mul_141 = None
        mul_142: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_160, 0.7978845608028654);  add_160 = None
        tanh_17: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_142);  mul_142 = None
        add_161: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_17, 1.0);  tanh_17 = None
        mul_143: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_140, add_161);  mul_140 = add_161 = None
        convert_element_type_303: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg255_1, torch.bfloat16);  arg255_1 = None
        convert_element_type_304: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg256_1, torch.bfloat16);  arg256_1 = None
        convert_element_type_305: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_143, torch.bfloat16);  mul_143 = None
        view_358: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_305, _shape_param_431);  convert_element_type_305 = _shape_param_431 = None
        permute_197: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_304, [1, 0]);  convert_element_type_304 = None
        addmm_53: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_303, view_358, permute_197);  convert_element_type_303 = None
        view_359: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_53, _shape_param_432);  addmm_53 = _shape_param_432 = None
        add_162: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_157, view_359);  add_157 = view_359 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(add_162, [2], correction = 0, keepdim = True)
        getitem_72: "f32[32, 128, 1]" = var_mean_36[0]
        getitem_73: "f32[32, 128, 1]" = var_mean_36[1];  var_mean_36 = None
        add_163: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_36: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        sub_54: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_162, getitem_73);  getitem_73 = None
        mul_144: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = None
        mul_145: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_144, arg257_1);  arg257_1 = None
        add_164: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_145, arg258_1);  mul_145 = arg258_1 = None
        convert_element_type_306: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg259_1, torch.bfloat16);  arg259_1 = None
        convert_element_type_307: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_164, torch.bfloat16);  add_164 = None
        permute_198: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_306, [1, 0]);  convert_element_type_306 = None
        view_360: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_307, _shape_param_433);  convert_element_type_307 = _shape_param_433 = None
        mm_54: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_360, permute_198)
        view_361: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_54, _shape_param_434);  mm_54 = _shape_param_434 = None
        convert_element_type_308: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg260_1, torch.bfloat16);  arg260_1 = None
        permute_199: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_308, [1, 0]);  convert_element_type_308 = None
        mm_55: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_360, permute_199)
        view_362: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_55, _shape_param_435);  mm_55 = _shape_param_435 = None
        convert_element_type_309: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg261_1, torch.bfloat16);  arg261_1 = None
        permute_200: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_309, [1, 0]);  convert_element_type_309 = None
        mm_56: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_360, permute_200)
        view_363: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_56, _shape_param_436);  mm_56 = _shape_param_436 = None
        view_364: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_361, _shape_param_437);  view_361 = _shape_param_437 = None
        permute_201: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None
        view_365: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_362, _shape_param_438);  view_362 = _shape_param_438 = None
        permute_202: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_365, [0, 2, 1, 3]);  view_365 = None
        view_366: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_363, _shape_param_439);  view_363 = _shape_param_439 = None
        permute_203: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
        convert_element_type_310: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_202, torch.float32);  permute_202 = None
        permute_204: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_310, [0, 1, 3, 2]);  convert_element_type_310 = None
        convert_element_type_311: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_204, torch.bfloat16);  permute_204 = None
        convert_element_type_312: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_201, torch.bfloat16);  permute_201 = None
        expand_73: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_312, _shape_param_440);  convert_element_type_312 = _shape_param_440 = None
        clone_72: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_367: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_72, _shape_param_441);  clone_72 = _shape_param_441 = None
        expand_74: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_311, _shape_param_442);  convert_element_type_311 = _shape_param_442 = None
        clone_73: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_368: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_73, _shape_param_443);  clone_73 = _shape_param_443 = None
        bmm_36: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_367, view_368)
        view_369: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_36, _shape_param_444);  bmm_36 = _shape_param_444 = None
        slice_37: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg262_1, 2, 0, 128);  arg262_1 = None
        slice_38: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_37, 3, 0, 128);  slice_37 = None
        where_19: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_38, view_369, full_3);  slice_38 = view_369 = None
        add_165: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_19, where);  where_19 = None
        amax_18: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_165, [-1], True)
        sub_55: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_165, amax_18);  add_165 = amax_18 = None
        exp_18: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_19: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_313: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16)
        expand_75: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_313, _shape_param_445);  convert_element_type_313 = _shape_param_445 = None
        view_370: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_75, _shape_param_446);  expand_75 = _shape_param_446 = None
        expand_76: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_203, _shape_param_447);  permute_203 = _shape_param_447 = None
        clone_74: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_371: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_74, _shape_param_448);  clone_74 = _shape_param_448 = None
        bmm_37: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_370, view_371)
        view_372: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_37, _shape_param_449);  bmm_37 = _shape_param_449 = None
        permute_205: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
        clone_75: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_373: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_75, _shape_param_450);  clone_75 = _shape_param_450 = None
        convert_element_type_314: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg263_1, torch.bfloat16);  arg263_1 = None
        convert_element_type_315: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg264_1, torch.bfloat16);  arg264_1 = None
        view_374: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_373, _shape_param_451);  view_373 = _shape_param_451 = None
        permute_206: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_315, [1, 0]);  convert_element_type_315 = None
        addmm_54: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_314, view_374, permute_206);  convert_element_type_314 = None
        view_375: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_54, _shape_param_452);  addmm_54 = _shape_param_452 = None
        add_166: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_375, add_162);  view_375 = add_162 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(add_166, [2], correction = 0, keepdim = True)
        getitem_74: "f32[32, 128, 1]" = var_mean_37[0]
        getitem_75: "f32[32, 128, 1]" = var_mean_37[1];  var_mean_37 = None
        add_167: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_37: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
        sub_56: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_166, getitem_75);  getitem_75 = None
        mul_146: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = None
        mul_147: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_146, arg265_1);  arg265_1 = None
        add_168: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_147, arg266_1);  mul_147 = arg266_1 = None
        convert_element_type_316: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg267_1, torch.bfloat16);  arg267_1 = None
        convert_element_type_317: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg268_1, torch.bfloat16);  arg268_1 = None
        convert_element_type_318: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_168, torch.bfloat16);  add_168 = None
        view_376: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_318, _shape_param_453);  convert_element_type_318 = _shape_param_453 = None
        permute_207: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_317, [1, 0]);  convert_element_type_317 = None
        addmm_55: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_316, view_376, permute_207);  convert_element_type_316 = None
        view_377: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_55, _shape_param_454);  _shape_param_454 = None
        mul_148: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_377, 0.5)
        convert_element_type_319: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_377, torch.float32)
        pow_19: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_319, 3.0);  convert_element_type_319 = None
        mul_149: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_169: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_377, mul_149);  view_377 = mul_149 = None
        mul_150: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_169, 0.7978845608028654);  add_169 = None
        tanh_18: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_150);  mul_150 = None
        add_170: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_18, 1.0);  tanh_18 = None
        mul_151: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_148, add_170);  mul_148 = add_170 = None
        convert_element_type_320: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg269_1, torch.bfloat16);  arg269_1 = None
        convert_element_type_321: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg270_1, torch.bfloat16);  arg270_1 = None
        convert_element_type_322: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_151, torch.bfloat16);  mul_151 = None
        view_378: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_322, _shape_param_455);  convert_element_type_322 = _shape_param_455 = None
        permute_208: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_321, [1, 0]);  convert_element_type_321 = None
        addmm_56: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_320, view_378, permute_208);  convert_element_type_320 = None
        view_379: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_56, _shape_param_456);  addmm_56 = _shape_param_456 = None
        add_171: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_166, view_379);  add_166 = view_379 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(add_171, [2], correction = 0, keepdim = True)
        getitem_76: "f32[32, 128, 1]" = var_mean_38[0]
        getitem_77: "f32[32, 128, 1]" = var_mean_38[1];  var_mean_38 = None
        add_172: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_38: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        sub_57: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_171, getitem_77);  getitem_77 = None
        mul_152: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = None
        mul_153: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_152, arg271_1);  arg271_1 = None
        add_173: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_153, arg272_1);  mul_153 = arg272_1 = None
        convert_element_type_323: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg273_1, torch.bfloat16);  arg273_1 = None
        convert_element_type_324: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_173, torch.bfloat16);  add_173 = None
        permute_209: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_323, [1, 0]);  convert_element_type_323 = None
        view_380: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_324, _shape_param_457);  convert_element_type_324 = _shape_param_457 = None
        mm_57: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_380, permute_209)
        view_381: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_57, _shape_param_458);  mm_57 = _shape_param_458 = None
        convert_element_type_325: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg274_1, torch.bfloat16);  arg274_1 = None
        permute_210: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_325, [1, 0]);  convert_element_type_325 = None
        mm_58: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_380, permute_210)
        view_382: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_58, _shape_param_459);  mm_58 = _shape_param_459 = None
        convert_element_type_326: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg275_1, torch.bfloat16);  arg275_1 = None
        permute_211: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_326, [1, 0]);  convert_element_type_326 = None
        mm_59: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_380, permute_211)
        view_383: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_59, _shape_param_460);  mm_59 = _shape_param_460 = None
        view_384: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_381, _shape_param_461);  view_381 = _shape_param_461 = None
        permute_212: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
        view_385: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_382, _shape_param_462);  view_382 = _shape_param_462 = None
        permute_213: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_385, [0, 2, 1, 3]);  view_385 = None
        view_386: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_383, _shape_param_463);  view_383 = _shape_param_463 = None
        permute_214: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None
        convert_element_type_327: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_213, torch.float32);  permute_213 = None
        permute_215: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_327, [0, 1, 3, 2]);  convert_element_type_327 = None
        convert_element_type_328: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_215, torch.bfloat16);  permute_215 = None
        convert_element_type_329: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_212, torch.bfloat16);  permute_212 = None
        expand_77: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_329, _shape_param_464);  convert_element_type_329 = _shape_param_464 = None
        clone_76: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_387: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_76, _shape_param_465);  clone_76 = _shape_param_465 = None
        expand_78: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_328, _shape_param_466);  convert_element_type_328 = _shape_param_466 = None
        clone_77: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
        view_388: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_77, _shape_param_467);  clone_77 = _shape_param_467 = None
        bmm_38: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_387, view_388)
        view_389: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_38, _shape_param_468);  bmm_38 = _shape_param_468 = None
        slice_39: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg276_1, 2, 0, 128);  arg276_1 = None
        slice_40: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_39, 3, 0, 128);  slice_39 = None
        where_20: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_40, view_389, full_3);  slice_40 = view_389 = None
        add_174: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_20, where);  where_20 = None
        amax_19: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_174, [-1], True)
        sub_58: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_174, amax_19);  add_174 = amax_19 = None
        exp_19: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        sum_20: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_19: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        convert_element_type_330: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16)
        expand_79: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_330, _shape_param_469);  convert_element_type_330 = _shape_param_469 = None
        view_390: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_79, _shape_param_470);  expand_79 = _shape_param_470 = None
        expand_80: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_214, _shape_param_471);  permute_214 = _shape_param_471 = None
        clone_78: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_391: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_78, _shape_param_472);  clone_78 = _shape_param_472 = None
        bmm_39: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_390, view_391)
        view_392: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_39, _shape_param_473);  bmm_39 = _shape_param_473 = None
        permute_216: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None
        clone_79: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
        view_393: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_79, _shape_param_474);  clone_79 = _shape_param_474 = None
        convert_element_type_331: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg277_1, torch.bfloat16);  arg277_1 = None
        convert_element_type_332: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg278_1, torch.bfloat16);  arg278_1 = None
        view_394: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_393, _shape_param_475);  view_393 = _shape_param_475 = None
        permute_217: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_332, [1, 0]);  convert_element_type_332 = None
        addmm_57: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_331, view_394, permute_217);  convert_element_type_331 = None
        view_395: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_57, _shape_param_476);  addmm_57 = _shape_param_476 = None
        add_175: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_395, add_171);  view_395 = add_171 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(add_175, [2], correction = 0, keepdim = True)
        getitem_78: "f32[32, 128, 1]" = var_mean_39[0]
        getitem_79: "f32[32, 128, 1]" = var_mean_39[1];  var_mean_39 = None
        add_176: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_39: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
        sub_59: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_175, getitem_79);  getitem_79 = None
        mul_154: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = None
        mul_155: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_154, arg279_1);  arg279_1 = None
        add_177: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_155, arg280_1);  mul_155 = arg280_1 = None
        convert_element_type_333: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg281_1, torch.bfloat16);  arg281_1 = None
        convert_element_type_334: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg282_1, torch.bfloat16);  arg282_1 = None
        convert_element_type_335: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_177, torch.bfloat16);  add_177 = None
        view_396: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_335, _shape_param_477);  convert_element_type_335 = _shape_param_477 = None
        permute_218: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_334, [1, 0]);  convert_element_type_334 = None
        addmm_58: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_333, view_396, permute_218);  convert_element_type_333 = None
        view_397: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_58, _shape_param_478);  _shape_param_478 = None
        mul_156: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_397, 0.5)
        convert_element_type_336: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_397, torch.float32)
        pow_20: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_336, 3.0);  convert_element_type_336 = None
        mul_157: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_178: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_397, mul_157);  view_397 = mul_157 = None
        mul_158: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_178, 0.7978845608028654);  add_178 = None
        tanh_19: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
        add_179: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_19, 1.0);  tanh_19 = None
        mul_159: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_156, add_179);  mul_156 = add_179 = None
        convert_element_type_337: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg283_1, torch.bfloat16);  arg283_1 = None
        convert_element_type_338: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg284_1, torch.bfloat16);  arg284_1 = None
        convert_element_type_339: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_159, torch.bfloat16);  mul_159 = None
        view_398: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_339, _shape_param_479);  convert_element_type_339 = _shape_param_479 = None
        permute_219: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_338, [1, 0]);  convert_element_type_338 = None
        addmm_59: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_337, view_398, permute_219);  convert_element_type_337 = None
        view_399: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_59, _shape_param_480);  addmm_59 = _shape_param_480 = None
        add_180: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_175, view_399);  add_175 = view_399 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(add_180, [2], correction = 0, keepdim = True)
        getitem_80: "f32[32, 128, 1]" = var_mean_40[0]
        getitem_81: "f32[32, 128, 1]" = var_mean_40[1];  var_mean_40 = None
        add_181: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_40: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_60: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_180, getitem_81);  getitem_81 = None
        mul_160: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = None
        mul_161: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_160, arg285_1);  arg285_1 = None
        add_182: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_161, arg286_1);  mul_161 = arg286_1 = None
        convert_element_type_340: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg287_1, torch.bfloat16);  arg287_1 = None
        convert_element_type_341: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_182, torch.bfloat16);  add_182 = None
        permute_220: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_340, [1, 0]);  convert_element_type_340 = None
        view_400: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_341, _shape_param_481);  convert_element_type_341 = _shape_param_481 = None
        mm_60: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_400, permute_220)
        view_401: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_60, _shape_param_482);  mm_60 = _shape_param_482 = None
        convert_element_type_342: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg288_1, torch.bfloat16);  arg288_1 = None
        permute_221: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_342, [1, 0]);  convert_element_type_342 = None
        mm_61: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_400, permute_221)
        view_402: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_61, _shape_param_483);  mm_61 = _shape_param_483 = None
        convert_element_type_343: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg289_1, torch.bfloat16);  arg289_1 = None
        permute_222: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_343, [1, 0]);  convert_element_type_343 = None
        mm_62: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_400, permute_222)
        view_403: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_62, _shape_param_484);  mm_62 = _shape_param_484 = None
        view_404: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_401, _shape_param_485);  view_401 = _shape_param_485 = None
        permute_223: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
        view_405: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_402, _shape_param_486);  view_402 = _shape_param_486 = None
        permute_224: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_405, [0, 2, 1, 3]);  view_405 = None
        view_406: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_403, _shape_param_487);  view_403 = _shape_param_487 = None
        permute_225: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
        convert_element_type_344: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_224, torch.float32);  permute_224 = None
        permute_226: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_344, [0, 1, 3, 2]);  convert_element_type_344 = None
        convert_element_type_345: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_226, torch.bfloat16);  permute_226 = None
        convert_element_type_346: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_223, torch.bfloat16);  permute_223 = None
        expand_81: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_346, _shape_param_488);  convert_element_type_346 = _shape_param_488 = None
        clone_80: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_407: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_80, _shape_param_489);  clone_80 = _shape_param_489 = None
        expand_82: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_345, _shape_param_490);  convert_element_type_345 = _shape_param_490 = None
        clone_81: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_408: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_81, _shape_param_491);  clone_81 = _shape_param_491 = None
        bmm_40: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_407, view_408)
        view_409: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_40, _shape_param_492);  bmm_40 = _shape_param_492 = None
        slice_41: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg290_1, 2, 0, 128);  arg290_1 = None
        slice_42: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_41, 3, 0, 128);  slice_41 = None
        where_21: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_42, view_409, full_3);  slice_42 = view_409 = None
        add_183: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_21, where);  where_21 = None
        amax_20: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_183, [-1], True)
        sub_61: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_183, amax_20);  add_183 = amax_20 = None
        exp_20: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_21: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_347: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16)
        expand_83: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_347, _shape_param_493);  convert_element_type_347 = _shape_param_493 = None
        view_410: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_83, _shape_param_494);  expand_83 = _shape_param_494 = None
        expand_84: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_225, _shape_param_495);  permute_225 = _shape_param_495 = None
        clone_82: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_411: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_82, _shape_param_496);  clone_82 = _shape_param_496 = None
        bmm_41: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_410, view_411)
        view_412: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_41, _shape_param_497);  bmm_41 = _shape_param_497 = None
        permute_227: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
        clone_83: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_413: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_83, _shape_param_498);  clone_83 = _shape_param_498 = None
        convert_element_type_348: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg291_1, torch.bfloat16);  arg291_1 = None
        convert_element_type_349: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg292_1, torch.bfloat16);  arg292_1 = None
        view_414: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_413, _shape_param_499);  view_413 = _shape_param_499 = None
        permute_228: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_349, [1, 0]);  convert_element_type_349 = None
        addmm_60: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_348, view_414, permute_228);  convert_element_type_348 = None
        view_415: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_60, _shape_param_500);  addmm_60 = _shape_param_500 = None
        add_184: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_415, add_180);  view_415 = add_180 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(add_184, [2], correction = 0, keepdim = True)
        getitem_82: "f32[32, 128, 1]" = var_mean_41[0]
        getitem_83: "f32[32, 128, 1]" = var_mean_41[1];  var_mean_41 = None
        add_185: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_41: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
        sub_62: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_184, getitem_83);  getitem_83 = None
        mul_162: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = None
        mul_163: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_162, arg293_1);  arg293_1 = None
        add_186: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_163, arg294_1);  mul_163 = arg294_1 = None
        convert_element_type_350: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg295_1, torch.bfloat16);  arg295_1 = None
        convert_element_type_351: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg296_1, torch.bfloat16);  arg296_1 = None
        convert_element_type_352: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_186, torch.bfloat16);  add_186 = None
        view_416: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_352, _shape_param_501);  convert_element_type_352 = _shape_param_501 = None
        permute_229: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_351, [1, 0]);  convert_element_type_351 = None
        addmm_61: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_350, view_416, permute_229);  convert_element_type_350 = None
        view_417: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_61, _shape_param_502);  _shape_param_502 = None
        mul_164: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_417, 0.5)
        convert_element_type_353: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_417, torch.float32)
        pow_21: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_353, 3.0);  convert_element_type_353 = None
        mul_165: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_187: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_417, mul_165);  view_417 = mul_165 = None
        mul_166: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_187, 0.7978845608028654);  add_187 = None
        tanh_20: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_166);  mul_166 = None
        add_188: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_20, 1.0);  tanh_20 = None
        mul_167: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_164, add_188);  mul_164 = add_188 = None
        convert_element_type_354: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg297_1, torch.bfloat16);  arg297_1 = None
        convert_element_type_355: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg298_1, torch.bfloat16);  arg298_1 = None
        convert_element_type_356: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_167, torch.bfloat16);  mul_167 = None
        view_418: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_356, _shape_param_503);  convert_element_type_356 = _shape_param_503 = None
        permute_230: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_355, [1, 0]);  convert_element_type_355 = None
        addmm_62: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_354, view_418, permute_230);  convert_element_type_354 = None
        view_419: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_62, _shape_param_504);  addmm_62 = _shape_param_504 = None
        add_189: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_184, view_419);  add_184 = view_419 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(add_189, [2], correction = 0, keepdim = True)
        getitem_84: "f32[32, 128, 1]" = var_mean_42[0]
        getitem_85: "f32[32, 128, 1]" = var_mean_42[1];  var_mean_42 = None
        add_190: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
        rsqrt_42: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
        sub_63: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_189, getitem_85);  getitem_85 = None
        mul_168: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = None
        mul_169: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_168, arg299_1);  arg299_1 = None
        add_191: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_169, arg300_1);  mul_169 = arg300_1 = None
        convert_element_type_357: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg301_1, torch.bfloat16);  arg301_1 = None
        convert_element_type_358: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_191, torch.bfloat16);  add_191 = None
        permute_231: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_357, [1, 0]);  convert_element_type_357 = None
        view_420: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_358, _shape_param_505);  convert_element_type_358 = _shape_param_505 = None
        mm_63: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_420, permute_231)
        view_421: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_63, _shape_param_506);  mm_63 = _shape_param_506 = None
        convert_element_type_359: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg302_1, torch.bfloat16);  arg302_1 = None
        permute_232: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_359, [1, 0]);  convert_element_type_359 = None
        mm_64: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_420, permute_232)
        view_422: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_64, _shape_param_507);  mm_64 = _shape_param_507 = None
        convert_element_type_360: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg303_1, torch.bfloat16);  arg303_1 = None
        permute_233: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_360, [1, 0]);  convert_element_type_360 = None
        mm_65: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_420, permute_233)
        view_423: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_65, _shape_param_508);  mm_65 = _shape_param_508 = None
        view_424: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_421, _shape_param_509);  view_421 = _shape_param_509 = None
        permute_234: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
        view_425: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_422, _shape_param_510);  view_422 = _shape_param_510 = None
        permute_235: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None
        view_426: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_423, _shape_param_511);  view_423 = _shape_param_511 = None
        permute_236: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
        convert_element_type_361: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_235, torch.float32);  permute_235 = None
        permute_237: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_361, [0, 1, 3, 2]);  convert_element_type_361 = None
        convert_element_type_362: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_237, torch.bfloat16);  permute_237 = None
        convert_element_type_363: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_234, torch.bfloat16);  permute_234 = None
        expand_85: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_363, _shape_param_512);  convert_element_type_363 = _shape_param_512 = None
        clone_84: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_427: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_84, _shape_param_513);  clone_84 = _shape_param_513 = None
        expand_86: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_362, _shape_param_514);  convert_element_type_362 = _shape_param_514 = None
        clone_85: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
        view_428: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_85, _shape_param_515);  clone_85 = _shape_param_515 = None
        bmm_42: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_427, view_428)
        view_429: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_42, _shape_param_516);  bmm_42 = _shape_param_516 = None
        slice_43: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg304_1, 2, 0, 128);  arg304_1 = None
        slice_44: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_43, 3, 0, 128);  slice_43 = None
        where_22: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_44, view_429, full_3);  slice_44 = view_429 = None
        add_192: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_22, where);  where_22 = None
        amax_21: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_192, [-1], True)
        sub_64: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_192, amax_21);  add_192 = amax_21 = None
        exp_21: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        sum_22: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_21: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        convert_element_type_364: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16)
        expand_87: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_364, _shape_param_517);  convert_element_type_364 = _shape_param_517 = None
        view_430: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_87, _shape_param_518);  expand_87 = _shape_param_518 = None
        expand_88: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_236, _shape_param_519);  permute_236 = _shape_param_519 = None
        clone_86: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_431: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_86, _shape_param_520);  clone_86 = _shape_param_520 = None
        bmm_43: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_430, view_431)
        view_432: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_43, _shape_param_521);  bmm_43 = _shape_param_521 = None
        permute_238: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
        clone_87: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
        view_433: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_87, _shape_param_522);  clone_87 = _shape_param_522 = None
        convert_element_type_365: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg305_1, torch.bfloat16);  arg305_1 = None
        convert_element_type_366: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg306_1, torch.bfloat16);  arg306_1 = None
        view_434: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_433, _shape_param_523);  view_433 = _shape_param_523 = None
        permute_239: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_366, [1, 0]);  convert_element_type_366 = None
        addmm_63: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_365, view_434, permute_239);  convert_element_type_365 = None
        view_435: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_63, _shape_param_524);  addmm_63 = _shape_param_524 = None
        add_193: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_435, add_189);  view_435 = add_189 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(add_193, [2], correction = 0, keepdim = True)
        getitem_86: "f32[32, 128, 1]" = var_mean_43[0]
        getitem_87: "f32[32, 128, 1]" = var_mean_43[1];  var_mean_43 = None
        add_194: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_43: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
        sub_65: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_193, getitem_87);  getitem_87 = None
        mul_170: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = None
        mul_171: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_170, arg307_1);  arg307_1 = None
        add_195: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_171, arg308_1);  mul_171 = arg308_1 = None
        convert_element_type_367: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg309_1, torch.bfloat16);  arg309_1 = None
        convert_element_type_368: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg310_1, torch.bfloat16);  arg310_1 = None
        convert_element_type_369: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_195, torch.bfloat16);  add_195 = None
        view_436: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_369, _shape_param_525);  convert_element_type_369 = _shape_param_525 = None
        permute_240: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_368, [1, 0]);  convert_element_type_368 = None
        addmm_64: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_367, view_436, permute_240);  convert_element_type_367 = None
        view_437: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_64, _shape_param_526);  _shape_param_526 = None
        mul_172: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_437, 0.5)
        convert_element_type_370: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_437, torch.float32)
        pow_22: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_370, 3.0);  convert_element_type_370 = None
        mul_173: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_196: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_437, mul_173);  view_437 = mul_173 = None
        mul_174: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_196, 0.7978845608028654);  add_196 = None
        tanh_21: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_174);  mul_174 = None
        add_197: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_21, 1.0);  tanh_21 = None
        mul_175: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_172, add_197);  mul_172 = add_197 = None
        convert_element_type_371: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg311_1, torch.bfloat16);  arg311_1 = None
        convert_element_type_372: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg312_1, torch.bfloat16);  arg312_1 = None
        convert_element_type_373: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_175, torch.bfloat16);  mul_175 = None
        view_438: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_373, _shape_param_527);  convert_element_type_373 = _shape_param_527 = None
        permute_241: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_372, [1, 0]);  convert_element_type_372 = None
        addmm_65: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_371, view_438, permute_241);  convert_element_type_371 = None
        view_439: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_65, _shape_param_528);  addmm_65 = _shape_param_528 = None
        add_198: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_193, view_439);  add_193 = view_439 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(add_198, [2], correction = 0, keepdim = True)
        getitem_88: "f32[32, 128, 1]" = var_mean_44[0]
        getitem_89: "f32[32, 128, 1]" = var_mean_44[1];  var_mean_44 = None
        add_199: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_44: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        sub_66: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_198, getitem_89);  getitem_89 = None
        mul_176: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = None
        mul_177: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_176, arg313_1);  arg313_1 = None
        add_200: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_177, arg314_1);  mul_177 = arg314_1 = None
        convert_element_type_374: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg315_1, torch.bfloat16);  arg315_1 = None
        convert_element_type_375: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_200, torch.bfloat16);  add_200 = None
        permute_242: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_374, [1, 0]);  convert_element_type_374 = None
        view_440: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_375, _shape_param_529);  convert_element_type_375 = _shape_param_529 = None
        mm_66: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_440, permute_242)
        view_441: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_66, _shape_param_530);  mm_66 = _shape_param_530 = None
        convert_element_type_376: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg316_1, torch.bfloat16);  arg316_1 = None
        permute_243: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_376, [1, 0]);  convert_element_type_376 = None
        mm_67: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_440, permute_243)
        view_442: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_67, _shape_param_531);  mm_67 = _shape_param_531 = None
        convert_element_type_377: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg317_1, torch.bfloat16);  arg317_1 = None
        permute_244: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_377, [1, 0]);  convert_element_type_377 = None
        mm_68: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_440, permute_244)
        view_443: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_68, _shape_param_532);  mm_68 = _shape_param_532 = None
        view_444: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_441, _shape_param_533);  view_441 = _shape_param_533 = None
        permute_245: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_444, [0, 2, 1, 3]);  view_444 = None
        view_445: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_442, _shape_param_534);  view_442 = _shape_param_534 = None
        permute_246: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
        view_446: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_443, _shape_param_535);  view_443 = _shape_param_535 = None
        permute_247: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_446, [0, 2, 1, 3]);  view_446 = None
        convert_element_type_378: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_246, torch.float32);  permute_246 = None
        permute_248: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_378, [0, 1, 3, 2]);  convert_element_type_378 = None
        convert_element_type_379: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_248, torch.bfloat16);  permute_248 = None
        convert_element_type_380: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_245, torch.bfloat16);  permute_245 = None
        expand_89: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_380, _shape_param_536);  convert_element_type_380 = _shape_param_536 = None
        clone_88: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_447: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_88, _shape_param_537);  clone_88 = _shape_param_537 = None
        expand_90: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_379, _shape_param_538);  convert_element_type_379 = _shape_param_538 = None
        clone_89: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_448: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_89, _shape_param_539);  clone_89 = _shape_param_539 = None
        bmm_44: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_447, view_448)
        view_449: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_44, _shape_param_540);  bmm_44 = _shape_param_540 = None
        slice_45: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg318_1, 2, 0, 128);  arg318_1 = None
        slice_46: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_45, 3, 0, 128);  slice_45 = None
        where_23: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_46, view_449, full_3);  slice_46 = view_449 = None
        add_201: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_23, where);  where_23 = None
        amax_22: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_201, [-1], True)
        sub_67: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_201, amax_22);  add_201 = amax_22 = None
        exp_22: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_23: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_381: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16)
        expand_91: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_381, _shape_param_541);  convert_element_type_381 = _shape_param_541 = None
        view_450: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_91, _shape_param_542);  expand_91 = _shape_param_542 = None
        expand_92: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_247, _shape_param_543);  permute_247 = _shape_param_543 = None
        clone_90: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_451: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_90, _shape_param_544);  clone_90 = _shape_param_544 = None
        bmm_45: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_450, view_451)
        view_452: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_45, _shape_param_545);  bmm_45 = _shape_param_545 = None
        permute_249: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None
        clone_91: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_453: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_91, _shape_param_546);  clone_91 = _shape_param_546 = None
        convert_element_type_382: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg319_1, torch.bfloat16);  arg319_1 = None
        convert_element_type_383: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg320_1, torch.bfloat16);  arg320_1 = None
        view_454: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_453, _shape_param_547);  view_453 = _shape_param_547 = None
        permute_250: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_383, [1, 0]);  convert_element_type_383 = None
        addmm_66: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_382, view_454, permute_250);  convert_element_type_382 = None
        view_455: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_66, _shape_param_548);  addmm_66 = _shape_param_548 = None
        add_202: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_455, add_198);  view_455 = add_198 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(add_202, [2], correction = 0, keepdim = True)
        getitem_90: "f32[32, 128, 1]" = var_mean_45[0]
        getitem_91: "f32[32, 128, 1]" = var_mean_45[1];  var_mean_45 = None
        add_203: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_45: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        sub_68: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_202, getitem_91);  getitem_91 = None
        mul_178: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = None
        mul_179: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_178, arg321_1);  arg321_1 = None
        add_204: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_179, arg322_1);  mul_179 = arg322_1 = None
        convert_element_type_384: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg323_1, torch.bfloat16);  arg323_1 = None
        convert_element_type_385: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg324_1, torch.bfloat16);  arg324_1 = None
        convert_element_type_386: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_204, torch.bfloat16);  add_204 = None
        view_456: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_386, _shape_param_549);  convert_element_type_386 = _shape_param_549 = None
        permute_251: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_385, [1, 0]);  convert_element_type_385 = None
        addmm_67: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_384, view_456, permute_251);  convert_element_type_384 = None
        view_457: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_67, _shape_param_550);  _shape_param_550 = None
        mul_180: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_457, 0.5)
        convert_element_type_387: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_457, torch.float32)
        pow_23: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_387, 3.0);  convert_element_type_387 = None
        mul_181: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_205: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_457, mul_181);  view_457 = mul_181 = None
        mul_182: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_205, 0.7978845608028654);  add_205 = None
        tanh_22: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_182);  mul_182 = None
        add_206: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_22, 1.0);  tanh_22 = None
        mul_183: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_180, add_206);  mul_180 = add_206 = None
        convert_element_type_388: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg325_1, torch.bfloat16);  arg325_1 = None
        convert_element_type_389: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg326_1, torch.bfloat16);  arg326_1 = None
        convert_element_type_390: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_183, torch.bfloat16);  mul_183 = None
        view_458: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_390, _shape_param_551);  convert_element_type_390 = _shape_param_551 = None
        permute_252: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_389, [1, 0]);  convert_element_type_389 = None
        addmm_68: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_388, view_458, permute_252);  convert_element_type_388 = None
        view_459: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_68, _shape_param_552);  addmm_68 = _shape_param_552 = None
        add_207: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_202, view_459);  add_202 = view_459 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(add_207, [2], correction = 0, keepdim = True)
        getitem_92: "f32[32, 128, 1]" = var_mean_46[0]
        getitem_93: "f32[32, 128, 1]" = var_mean_46[1];  var_mean_46 = None
        add_208: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_46: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
        sub_69: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_207, getitem_93);  getitem_93 = None
        mul_184: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = None
        mul_185: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_184, arg327_1);  arg327_1 = None
        add_209: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_185, arg328_1);  mul_185 = arg328_1 = None
        convert_element_type_391: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg329_1, torch.bfloat16);  arg329_1 = None
        convert_element_type_392: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_209, torch.bfloat16);  add_209 = None
        permute_253: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_391, [1, 0]);  convert_element_type_391 = None
        view_460: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_392, _shape_param_553);  convert_element_type_392 = _shape_param_553 = None
        mm_69: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_460, permute_253)
        view_461: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_69, _shape_param_554);  mm_69 = _shape_param_554 = None
        convert_element_type_393: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg330_1, torch.bfloat16);  arg330_1 = None
        permute_254: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_393, [1, 0]);  convert_element_type_393 = None
        mm_70: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_460, permute_254)
        view_462: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_70, _shape_param_555);  mm_70 = _shape_param_555 = None
        convert_element_type_394: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg331_1, torch.bfloat16);  arg331_1 = None
        permute_255: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_394, [1, 0]);  convert_element_type_394 = None
        mm_71: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_460, permute_255)
        view_463: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_71, _shape_param_556);  mm_71 = _shape_param_556 = None
        view_464: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_461, _shape_param_557);  view_461 = _shape_param_557 = None
        permute_256: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_464, [0, 2, 1, 3]);  view_464 = None
        view_465: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_462, _shape_param_558);  view_462 = _shape_param_558 = None
        permute_257: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_465, [0, 2, 1, 3]);  view_465 = None
        view_466: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_463, _shape_param_559);  view_463 = _shape_param_559 = None
        permute_258: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None
        convert_element_type_395: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_257, torch.float32);  permute_257 = None
        permute_259: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_395, [0, 1, 3, 2]);  convert_element_type_395 = None
        convert_element_type_396: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_259, torch.bfloat16);  permute_259 = None
        convert_element_type_397: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_256, torch.bfloat16);  permute_256 = None
        expand_93: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_397, _shape_param_560);  convert_element_type_397 = _shape_param_560 = None
        clone_92: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_467: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_92, _shape_param_561);  clone_92 = _shape_param_561 = None
        expand_94: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_396, _shape_param_562);  convert_element_type_396 = _shape_param_562 = None
        clone_93: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
        view_468: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_93, _shape_param_563);  clone_93 = _shape_param_563 = None
        bmm_46: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_467, view_468)
        view_469: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_46, _shape_param_564);  bmm_46 = _shape_param_564 = None
        slice_47: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg332_1, 2, 0, 128);  arg332_1 = None
        slice_48: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_47, 3, 0, 128);  slice_47 = None
        where_24: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_48, view_469, full_3);  slice_48 = view_469 = full_3 = None
        add_210: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_24, where);  where_24 = where = None
        amax_23: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_210, [-1], True)
        sub_70: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_210, amax_23);  add_210 = amax_23 = None
        exp_23: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
        sum_24: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_23: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        convert_element_type_398: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16)
        expand_95: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_398, _shape_param_565);  convert_element_type_398 = _shape_param_565 = None
        view_470: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_95, _shape_param_566);  expand_95 = _shape_param_566 = None
        expand_96: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_258, _shape_param_567);  permute_258 = _shape_param_567 = None
        clone_94: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_471: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_94, _shape_param_568);  clone_94 = _shape_param_568 = None
        bmm_47: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_470, view_471)
        view_472: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_47, _shape_param_569);  bmm_47 = _shape_param_569 = None
        permute_260: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
        clone_95: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_473: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_95, _shape_param_570);  clone_95 = _shape_param_570 = None
        convert_element_type_399: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg333_1, torch.bfloat16);  arg333_1 = None
        convert_element_type_400: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg334_1, torch.bfloat16);  arg334_1 = None
        view_474: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_473, _shape_param_571);  view_473 = _shape_param_571 = None
        permute_261: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_400, [1, 0]);  convert_element_type_400 = None
        addmm_69: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_399, view_474, permute_261);  convert_element_type_399 = None
        view_475: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_69, _shape_param_572);  addmm_69 = _shape_param_572 = None
        add_211: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_475, add_207);  view_475 = add_207 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(add_211, [2], correction = 0, keepdim = True)
        getitem_94: "f32[32, 128, 1]" = var_mean_47[0]
        getitem_95: "f32[32, 128, 1]" = var_mean_47[1];  var_mean_47 = None
        add_212: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_47: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        sub_71: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_211, getitem_95);  getitem_95 = None
        mul_186: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = None
        mul_187: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_186, arg335_1);  arg335_1 = None
        add_213: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_187, arg336_1);  mul_187 = arg336_1 = None
        convert_element_type_401: "bf16[8192]" = torch.ops.prims.convert_element_type.default(arg337_1, torch.bfloat16);  arg337_1 = None
        convert_element_type_402: "bf16[8192, 2048]" = torch.ops.prims.convert_element_type.default(arg338_1, torch.bfloat16);  arg338_1 = None
        convert_element_type_403: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_213, torch.bfloat16);  add_213 = None
        view_476: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_403, _shape_param_573);  convert_element_type_403 = _shape_param_573 = None
        permute_262: "bf16[2048, 8192]" = torch.ops.aten.permute.default(convert_element_type_402, [1, 0]);  convert_element_type_402 = None
        addmm_70: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(convert_element_type_401, view_476, permute_262);  convert_element_type_401 = None
        view_477: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_70, _shape_param_574);  _shape_param_574 = None
        mul_188: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_477, 0.5)
        convert_element_type_404: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_477, torch.float32)
        pow_24: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_404, 3.0);  convert_element_type_404 = None
        mul_189: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_214: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_477, mul_189);  view_477 = mul_189 = None
        mul_190: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_214, 0.7978845608028654);  add_214 = None
        tanh_23: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_190);  mul_190 = None
        add_215: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_23, 1.0);  tanh_23 = None
        mul_191: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_188, add_215);  mul_188 = add_215 = None
        convert_element_type_405: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg339_1, torch.bfloat16);  arg339_1 = None
        convert_element_type_406: "bf16[2048, 8192]" = torch.ops.prims.convert_element_type.default(arg340_1, torch.bfloat16);  arg340_1 = None
        convert_element_type_407: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_191, torch.bfloat16);  mul_191 = None
        view_478: "bf16[4096, 8192]" = torch.ops.aten.view.default(convert_element_type_407, _shape_param_575);  convert_element_type_407 = _shape_param_575 = None
        permute_263: "bf16[8192, 2048]" = torch.ops.aten.permute.default(convert_element_type_406, [1, 0]);  convert_element_type_406 = None
        addmm_71: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(convert_element_type_405, view_478, permute_263);  convert_element_type_405 = None
        view_479: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_71, _shape_param_576);  addmm_71 = _shape_param_576 = None
        add_216: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_211, view_479);  add_211 = view_479 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(add_216, [2], correction = 0, keepdim = True)
        getitem_96: "f32[32, 128, 1]" = var_mean_48[0]
        getitem_97: "f32[32, 128, 1]" = var_mean_48[1];  var_mean_48 = None
        add_217: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_48: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
        sub_72: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_216, getitem_97);  add_216 = getitem_97 = None
        mul_192: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = None
        mul_193: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_192, arg341_1);  arg341_1 = None
        add_218: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_193, arg342_1);  mul_193 = arg342_1 = None
        convert_element_type_408: "bf16[2, 2048]" = torch.ops.prims.convert_element_type.default(arg343_1, torch.bfloat16);  arg343_1 = None
        convert_element_type_409: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_218, torch.bfloat16);  add_218 = None
        permute_264: "bf16[2048, 2]" = torch.ops.aten.permute.default(convert_element_type_408, [1, 0]);  convert_element_type_408 = None
        view_480: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_409, _shape_param_577);  convert_element_type_409 = _shape_param_577 = None
        mm_72: "bf16[4096, 2]" = torch.ops.aten.mm.default(view_480, permute_264)
        view_481: "bf16[32, 128, 2]" = torch.ops.aten.view.default(mm_72, _shape_param_578);  mm_72 = _shape_param_578 = None
        ne: "b8[32, 128]" = torch.ops.aten.ne.Scalar(arg1_1, 0);  arg1_1 = None
        convert_element_type_410: "i32[32, 128]" = torch.ops.prims.convert_element_type.default(ne, torch.int32);  ne = None
        iota_1: "i32[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        mul_194: "i32[32, 128]" = torch.ops.aten.mul.Tensor(iota_1, convert_element_type_410);  iota_1 = convert_element_type_410 = None
        argmax: "i64[32]" = torch.ops.aten.argmax.default(mul_194, -1);  mul_194 = None
        index_2: "bf16[32, 2]" = torch.ops.aten.index.Tensor(view_481, [iota, argmax]);  view_481 = None
        convert_element_type_411: "f32[32, 2]" = torch.ops.prims.convert_element_type.default(index_2, torch.float32)
        amax_24: "f32[32, 1]" = torch.ops.aten.amax.default(convert_element_type_411, [1], True)
        sub_73: "f32[32, 2]" = torch.ops.aten.sub.Tensor(convert_element_type_411, amax_24);  convert_element_type_411 = amax_24 = None
        exp_24: "f32[32, 2]" = torch.ops.aten.exp.default(sub_73)
        sum_25: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[32, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_74: "f32[32, 2]" = torch.ops.aten.sub.Tensor(sub_73, log);  sub_73 = log = None
        convert_element_type_412: "bf16[32, 2]" = torch.ops.prims.convert_element_type.default(sub_74, torch.bfloat16);  sub_74 = None
        convert_element_type_413: "f32[32, 2]" = torch.ops.prims.convert_element_type.default(convert_element_type_412, torch.float32);  convert_element_type_412 = None
        ne_1: "b8[32]" = torch.ops.aten.ne.Scalar(arg344_1, -100)
        full_4: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "i64[32]" = torch.ops.aten.where.self(ne_1, arg344_1, full_4)
        unsqueeze_6: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(where_25, 1);  where_25 = None
        gather: "f32[32, 1]" = torch.ops.aten.gather.default(convert_element_type_413, 1, unsqueeze_6);  convert_element_type_413 = unsqueeze_6 = None
        squeeze: "f32[32]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[32]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_26: "f32[32]" = torch.ops.aten.where.self(ne_1, neg, full_1);  neg = full_1 = None
        sum_26: "i64[]" = torch.ops.aten.sum.default(ne_1);  ne_1 = None
        convert_element_type_414: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27: "f32[]" = torch.ops.aten.sum.default(where_26);  where_26 = None
        div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_414);  sum_27 = None
        unsqueeze_7: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(arg344_1, 1);  arg344_1 = None
        ne_2: "b8[32, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_7, -100)
        where_27: "i64[32, 1]" = torch.ops.aten.where.self(ne_2, unsqueeze_7, full_4);  unsqueeze_7 = full_4 = None
        iota_2: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_482: "i64[1, 2]" = torch.ops.aten.view.default(iota_2, _shape_param_579);  iota_2 = _shape_param_579 = None
        expand_97: "i64[32, 2]" = torch.ops.aten.expand.default(where_27, _shape_param_580);  where_27 = _shape_param_580 = None
        eq_1: "b8[32, 2]" = torch.ops.aten.eq.Tensor(expand_97, view_482);  expand_97 = view_482 = None
        permute_265: "bf16[2, 2048]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
        div_25: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 2048);  rsqrt_48 = None
        permute_266: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        permute_267: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
        div_26: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 2048);  rsqrt_47 = None
        permute_268: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
        permute_269: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_470, [0, 2, 1]);  view_470 = None
        permute_270: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_471, [0, 2, 1]);  view_471 = None
        permute_271: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_467, [0, 2, 1]);  view_467 = None
        permute_272: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_468, [0, 2, 1]);  view_468 = None
        permute_273: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
        permute_274: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
        permute_275: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
        div_27: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 2048);  rsqrt_46 = None
        permute_276: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
        permute_277: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
        div_28: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 2048);  rsqrt_45 = None
        permute_278: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
        permute_279: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_450, [0, 2, 1]);  view_450 = None
        permute_280: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_451, [0, 2, 1]);  view_451 = None
        permute_281: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_447, [0, 2, 1]);  view_447 = None
        permute_282: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_448, [0, 2, 1]);  view_448 = None
        permute_283: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
        permute_284: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
        permute_285: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        div_29: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 2048);  rsqrt_44 = None
        permute_286: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        permute_287: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
        div_30: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 2048);  rsqrt_43 = None
        permute_288: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
        permute_289: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_430, [0, 2, 1]);  view_430 = None
        permute_290: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_431, [0, 2, 1]);  view_431 = None
        permute_291: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_427, [0, 2, 1]);  view_427 = None
        permute_292: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_428, [0, 2, 1]);  view_428 = None
        permute_293: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        permute_294: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
        permute_295: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
        div_31: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 2048);  rsqrt_42 = None
        permute_296: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
        permute_297: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
        div_32: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 2048);  rsqrt_41 = None
        permute_298: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
        permute_299: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_410, [0, 2, 1]);  view_410 = None
        permute_300: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_411, [0, 2, 1]);  view_411 = None
        permute_301: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_407, [0, 2, 1]);  view_407 = None
        permute_302: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_408, [0, 2, 1]);  view_408 = None
        permute_303: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
        permute_304: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
        permute_305: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
        div_33: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 2048);  rsqrt_40 = None
        permute_306: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
        permute_307: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
        div_34: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 2048);  rsqrt_39 = None
        permute_308: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
        permute_309: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_390, [0, 2, 1]);  view_390 = None
        permute_310: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_391, [0, 2, 1]);  view_391 = None
        permute_311: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_387, [0, 2, 1]);  view_387 = None
        permute_312: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_388, [0, 2, 1]);  view_388 = None
        permute_313: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        permute_314: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
        permute_315: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
        div_35: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 2048);  rsqrt_38 = None
        permute_316: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
        permute_317: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
        div_36: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 2048);  rsqrt_37 = None
        permute_318: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
        permute_319: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_370, [0, 2, 1]);  view_370 = None
        permute_320: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_371, [0, 2, 1]);  view_371 = None
        permute_321: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_367, [0, 2, 1]);  view_367 = None
        permute_322: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_368, [0, 2, 1]);  view_368 = None
        permute_323: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
        permute_324: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
        permute_325: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
        div_37: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 2048);  rsqrt_36 = None
        permute_326: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
        permute_327: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
        div_38: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 2048);  rsqrt_35 = None
        permute_328: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        permute_329: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_350, [0, 2, 1]);  view_350 = None
        permute_330: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None
        permute_331: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_347, [0, 2, 1]);  view_347 = None
        permute_332: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_348, [0, 2, 1]);  view_348 = None
        permute_333: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
        permute_334: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
        permute_335: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
        div_39: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 2048);  rsqrt_34 = None
        permute_336: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        permute_337: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        div_40: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 2048);  rsqrt_33 = None
        permute_338: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
        permute_339: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_330, [0, 2, 1]);  view_330 = None
        permute_340: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_331, [0, 2, 1]);  view_331 = None
        permute_341: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_327, [0, 2, 1]);  view_327 = None
        permute_342: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_328, [0, 2, 1]);  view_328 = None
        permute_343: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        permute_344: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
        permute_345: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        div_41: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 2048);  rsqrt_32 = None
        permute_346: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
        permute_347: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        div_42: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 2048);  rsqrt_31 = None
        permute_348: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        permute_349: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_310, [0, 2, 1]);  view_310 = None
        permute_350: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None
        permute_351: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_307, [0, 2, 1]);  view_307 = None
        permute_352: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_308, [0, 2, 1]);  view_308 = None
        permute_353: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        permute_354: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
        permute_355: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        div_43: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 2048);  rsqrt_30 = None
        permute_356: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
        permute_357: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        div_44: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 2048);  rsqrt_29 = None
        permute_358: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
        permute_359: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_290, [0, 2, 1]);  view_290 = None
        permute_360: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_291, [0, 2, 1]);  view_291 = None
        permute_361: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None
        permute_362: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_288, [0, 2, 1]);  view_288 = None
        permute_363: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        permute_364: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
        permute_365: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
        div_45: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 2048);  rsqrt_28 = None
        permute_366: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        permute_367: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        div_46: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 2048);  rsqrt_27 = None
        permute_368: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        permute_369: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_270, [0, 2, 1]);  view_270 = None
        permute_370: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_271, [0, 2, 1]);  view_271 = None
        permute_371: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
        permute_372: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_268, [0, 2, 1]);  view_268 = None
        permute_373: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        permute_374: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        permute_375: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        div_47: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 2048);  rsqrt_26 = None
        permute_376: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        permute_377: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        div_48: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 2048);  rsqrt_25 = None
        permute_378: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        permute_379: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_250, [0, 2, 1]);  view_250 = None
        permute_380: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
        permute_381: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_247, [0, 2, 1]);  view_247 = None
        permute_382: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_248, [0, 2, 1]);  view_248 = None
        permute_383: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        permute_384: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        permute_385: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        div_49: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 2048);  rsqrt_24 = None
        permute_386: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        permute_387: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        div_50: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 2048);  rsqrt_23 = None
        permute_388: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        permute_389: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
        permute_390: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
        permute_391: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_227, [0, 2, 1]);  view_227 = None
        permute_392: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None
        permute_393: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        permute_394: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        permute_395: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        div_51: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 2048);  rsqrt_22 = None
        permute_396: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        permute_397: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        div_52: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 2048);  rsqrt_21 = None
        permute_398: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        permute_399: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        permute_400: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
        permute_401: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        permute_402: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
        permute_403: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        permute_404: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        permute_405: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        div_53: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 2048);  rsqrt_20 = None
        permute_406: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        permute_407: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        div_54: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 2048);  rsqrt_19 = None
        permute_408: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        permute_409: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
        permute_410: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
        permute_411: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
        permute_412: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        permute_413: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        permute_414: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
        permute_415: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        div_55: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 2048);  rsqrt_18 = None
        permute_416: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        permute_417: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        div_56: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 2048);  rsqrt_17 = None
        permute_418: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        permute_419: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_170, [0, 2, 1]);  view_170 = None
        permute_420: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
        permute_421: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
        permute_422: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
        permute_423: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        permute_424: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        permute_425: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        div_57: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 2048);  rsqrt_16 = None
        permute_426: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        permute_427: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        div_58: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 2048);  rsqrt_15 = None
        permute_428: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        permute_429: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_150, [0, 2, 1]);  view_150 = None
        permute_430: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_151, [0, 2, 1]);  view_151 = None
        permute_431: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
        permute_432: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_148, [0, 2, 1]);  view_148 = None
        permute_433: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        permute_434: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        permute_435: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        div_59: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 2048);  rsqrt_14 = None
        permute_436: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        permute_437: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        div_60: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 2048);  rsqrt_13 = None
        permute_438: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        permute_439: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
        permute_440: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None
        permute_441: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_127, [0, 2, 1]);  view_127 = None
        permute_442: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None
        permute_443: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        permute_444: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        permute_445: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        div_61: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 2048);  rsqrt_12 = None
        permute_446: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        permute_447: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        div_62: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 2048);  rsqrt_11 = None
        permute_448: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        permute_449: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
        permute_450: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
        permute_451: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
        permute_452: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_108, [0, 2, 1]);  view_108 = None
        permute_453: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        permute_454: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        permute_455: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        div_63: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 2048);  rsqrt_10 = None
        permute_456: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        permute_457: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        div_64: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 2048);  rsqrt_9 = None
        permute_458: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        permute_459: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_90, [0, 2, 1]);  view_90 = None
        permute_460: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_461: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
        permute_462: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_88, [0, 2, 1]);  view_88 = None
        permute_463: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        permute_464: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        permute_465: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        div_65: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 2048);  rsqrt_8 = None
        permute_466: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        permute_467: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        div_66: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 2048);  rsqrt_7 = None
        permute_468: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        permute_469: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None
        permute_470: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None
        permute_471: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_67, [0, 2, 1]);  view_67 = None
        permute_472: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
        permute_473: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        permute_474: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        permute_475: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        div_67: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 2048);  rsqrt_6 = None
        permute_476: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        permute_477: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        div_68: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 2048);  rsqrt_5 = None
        permute_478: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        permute_479: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
        permute_480: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
        permute_481: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_47, [0, 2, 1]);  view_47 = None
        permute_482: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_48, [0, 2, 1]);  view_48 = None
        permute_483: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        permute_484: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        permute_485: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        div_69: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 2048);  rsqrt_4 = None
        permute_486: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        permute_487: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        div_70: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 2048);  rsqrt_3 = None
        permute_488: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        permute_489: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
        permute_490: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        permute_491: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
        permute_492: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
        permute_493: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        permute_494: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_495: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        div_71: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 2048);  rsqrt_2 = None
        permute_496: "bf16[2048, 8192]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_497: "bf16[8192, 2048]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        div_72: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 2048);  rsqrt_1 = None
        permute_498: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_499: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        permute_500: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        permute_501: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        permute_502: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
        permute_503: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_504: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        permute_505: "bf16[2048, 2048]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (embedding, iota, embedding_1, getitem_1, rsqrt, view, bmm, amax, sum_1, view_14, mul_2, view_16, addmm_1, view_18, mul_8, view_20, div_1, view_34, mul_10, view_36, addmm_4, view_38, mul_16, view_40, div_2, view_54, mul_18, view_56, addmm_7, view_58, mul_24, view_60, div_3, view_74, mul_26, view_76, addmm_10, view_78, mul_32, view_80, div_4, view_94, mul_34, view_96, addmm_13, view_98, mul_40, view_100, div_5, view_114, mul_42, view_116, addmm_16, view_118, mul_48, view_120, div_6, view_134, mul_50, view_136, addmm_19, view_138, mul_56, view_140, div_7, view_154, mul_58, view_156, addmm_22, view_158, mul_64, view_160, div_8, view_174, mul_66, view_176, addmm_25, view_178, mul_72, view_180, div_9, view_194, mul_74, view_196, addmm_28, view_198, mul_80, view_200, div_10, view_214, mul_82, view_216, addmm_31, view_218, mul_88, view_220, div_11, view_234, mul_90, view_236, addmm_34, view_238, mul_96, view_240, div_12, view_254, mul_98, view_256, addmm_37, view_258, mul_104, view_260, div_13, view_274, mul_106, view_276, addmm_40, view_278, mul_112, view_280, div_14, view_294, mul_114, view_296, addmm_43, view_298, mul_120, view_300, div_15, view_314, mul_122, view_316, addmm_46, view_318, mul_128, view_320, div_16, view_334, mul_130, view_336, addmm_49, view_338, mul_136, view_340, div_17, view_354, mul_138, view_356, addmm_52, view_358, mul_144, view_360, div_18, view_374, mul_146, view_376, addmm_55, view_378, mul_152, view_380, div_19, view_394, mul_154, view_396, addmm_58, view_398, mul_160, view_400, div_20, view_414, mul_162, view_416, addmm_61, view_418, mul_168, view_420, div_21, view_434, mul_170, view_436, addmm_64, view_438, mul_176, view_440, div_22, view_454, mul_178, view_456, addmm_67, view_458, mul_184, view_460, div_23, view_474, mul_186, view_476, addmm_70, view_478, mul_192, view_480, argmax, index_2, convert_element_type_414, div_24, ne_2, eq_1, permute_265, div_25, permute_266, permute_267, div_26, permute_268, permute_269, permute_270, permute_271, permute_272, permute_273, permute_274, permute_275, div_27, permute_276, permute_277, div_28, permute_278, permute_279, permute_280, permute_281, permute_282, permute_283, permute_284, permute_285, div_29, permute_286, permute_287, div_30, permute_288, permute_289, permute_290, permute_291, permute_292, permute_293, permute_294, permute_295, div_31, permute_296, permute_297, div_32, permute_298, permute_299, permute_300, permute_301, permute_302, permute_303, permute_304, permute_305, div_33, permute_306, permute_307, div_34, permute_308, permute_309, permute_310, permute_311, permute_312, permute_313, permute_314, permute_315, div_35, permute_316, permute_317, div_36, permute_318, permute_319, permute_320, permute_321, permute_322, permute_323, permute_324, permute_325, div_37, permute_326, permute_327, div_38, permute_328, permute_329, permute_330, permute_331, permute_332, permute_333, permute_334, permute_335, div_39, permute_336, permute_337, div_40, permute_338, permute_339, permute_340, permute_341, permute_342, permute_343, permute_344, permute_345, div_41, permute_346, permute_347, div_42, permute_348, permute_349, permute_350, permute_351, permute_352, permute_353, permute_354, permute_355, div_43, permute_356, permute_357, div_44, permute_358, permute_359, permute_360, permute_361, permute_362, permute_363, permute_364, permute_365, div_45, permute_366, permute_367, div_46, permute_368, permute_369, permute_370, permute_371, permute_372, permute_373, permute_374, permute_375, div_47, permute_376, permute_377, div_48, permute_378, permute_379, permute_380, permute_381, permute_382, permute_383, permute_384, permute_385, div_49, permute_386, permute_387, div_50, permute_388, permute_389, permute_390, permute_391, permute_392, permute_393, permute_394, permute_395, div_51, permute_396, permute_397, div_52, permute_398, permute_399, permute_400, permute_401, permute_402, permute_403, permute_404, permute_405, div_53, permute_406, permute_407, div_54, permute_408, permute_409, permute_410, permute_411, permute_412, permute_413, permute_414, permute_415, div_55, permute_416, permute_417, div_56, permute_418, permute_419, permute_420, permute_421, permute_422, permute_423, permute_424, permute_425, div_57, permute_426, permute_427, div_58, permute_428, permute_429, permute_430, permute_431, permute_432, permute_433, permute_434, permute_435, div_59, permute_436, permute_437, div_60, permute_438, permute_439, permute_440, permute_441, permute_442, permute_443, permute_444, permute_445, div_61, permute_446, permute_447, div_62, permute_448, permute_449, permute_450, permute_451, permute_452, permute_453, permute_454, permute_455, div_63, permute_456, permute_457, div_64, permute_458, permute_459, permute_460, permute_461, permute_462, permute_463, permute_464, permute_465, div_65, permute_466, permute_467, div_66, permute_468, permute_469, permute_470, permute_471, permute_472, permute_473, permute_474, permute_475, div_67, permute_476, permute_477, div_68, permute_478, permute_479, permute_480, permute_481, permute_482, permute_483, permute_484, permute_485, div_69, permute_486, permute_487, div_70, permute_488, permute_489, permute_490, permute_491, permute_492, permute_493, permute_494, permute_495, div_71, permute_496, permute_497, div_72, permute_498, permute_499, permute_500, permute_501, permute_502, permute_503, permute_504, permute_505)



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
