"""
Standalone repro captured via capture_hook.
Label: GPTNeoForSequenceClassification
Pattern hash: 9c489d7750bc
Shape hash: 2f03d16e
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
    def forward(self, arg0_1: "bf16[50257, 2048]", arg1_1: "i64[32, 128]", arg2_1: "bf16[2048, 2048]", arg3_1: "i64[1, 128]", arg4_1: "b8[1, 1, 2048, 2048]", arg5_1: "bf16[2048]", arg6_1: "bf16[2048]", arg7_1: "bf16[2048, 2048]", arg8_1: "bf16[2048, 2048]", arg9_1: "i64[32, 128]", arg10_1: "bf16[2048, 2048]", arg11_1: "bf16[2048, 2048]", arg12_1: "bf16[2048]", arg13_1: "bf16[2048]", arg14_1: "bf16[2048]", arg15_1: "bf16[8192, 2048]", arg16_1: "bf16[8192]", arg17_1: "bf16[2048, 8192]", arg18_1: "bf16[2048]", arg19_1: "b8[1, 1, 2048, 2048]", arg20_1: "bf16[2048]", arg21_1: "bf16[2048]", arg22_1: "bf16[2048, 2048]", arg23_1: "bf16[2048, 2048]", arg24_1: "bf16[2048, 2048]", arg25_1: "bf16[2048, 2048]", arg26_1: "bf16[2048]", arg27_1: "bf16[2048]", arg28_1: "bf16[2048]", arg29_1: "bf16[8192, 2048]", arg30_1: "bf16[8192]", arg31_1: "bf16[2048, 8192]", arg32_1: "bf16[2048]", arg33_1: "b8[1, 1, 2048, 2048]", arg34_1: "bf16[2048]", arg35_1: "bf16[2048]", arg36_1: "bf16[2048, 2048]", arg37_1: "bf16[2048, 2048]", arg38_1: "bf16[2048, 2048]", arg39_1: "bf16[2048, 2048]", arg40_1: "bf16[2048]", arg41_1: "bf16[2048]", arg42_1: "bf16[2048]", arg43_1: "bf16[8192, 2048]", arg44_1: "bf16[8192]", arg45_1: "bf16[2048, 8192]", arg46_1: "bf16[2048]", arg47_1: "b8[1, 1, 2048, 2048]", arg48_1: "bf16[2048]", arg49_1: "bf16[2048]", arg50_1: "bf16[2048, 2048]", arg51_1: "bf16[2048, 2048]", arg52_1: "bf16[2048, 2048]", arg53_1: "bf16[2048, 2048]", arg54_1: "bf16[2048]", arg55_1: "bf16[2048]", arg56_1: "bf16[2048]", arg57_1: "bf16[8192, 2048]", arg58_1: "bf16[8192]", arg59_1: "bf16[2048, 8192]", arg60_1: "bf16[2048]", arg61_1: "b8[1, 1, 2048, 2048]", arg62_1: "bf16[2048]", arg63_1: "bf16[2048]", arg64_1: "bf16[2048, 2048]", arg65_1: "bf16[2048, 2048]", arg66_1: "bf16[2048, 2048]", arg67_1: "bf16[2048, 2048]", arg68_1: "bf16[2048]", arg69_1: "bf16[2048]", arg70_1: "bf16[2048]", arg71_1: "bf16[8192, 2048]", arg72_1: "bf16[8192]", arg73_1: "bf16[2048, 8192]", arg74_1: "bf16[2048]", arg75_1: "b8[1, 1, 2048, 2048]", arg76_1: "bf16[2048]", arg77_1: "bf16[2048]", arg78_1: "bf16[2048, 2048]", arg79_1: "bf16[2048, 2048]", arg80_1: "bf16[2048, 2048]", arg81_1: "bf16[2048, 2048]", arg82_1: "bf16[2048]", arg83_1: "bf16[2048]", arg84_1: "bf16[2048]", arg85_1: "bf16[8192, 2048]", arg86_1: "bf16[8192]", arg87_1: "bf16[2048, 8192]", arg88_1: "bf16[2048]", arg89_1: "b8[1, 1, 2048, 2048]", arg90_1: "bf16[2048]", arg91_1: "bf16[2048]", arg92_1: "bf16[2048, 2048]", arg93_1: "bf16[2048, 2048]", arg94_1: "bf16[2048, 2048]", arg95_1: "bf16[2048, 2048]", arg96_1: "bf16[2048]", arg97_1: "bf16[2048]", arg98_1: "bf16[2048]", arg99_1: "bf16[8192, 2048]", arg100_1: "bf16[8192]", arg101_1: "bf16[2048, 8192]", arg102_1: "bf16[2048]", arg103_1: "b8[1, 1, 2048, 2048]", arg104_1: "bf16[2048]", arg105_1: "bf16[2048]", arg106_1: "bf16[2048, 2048]", arg107_1: "bf16[2048, 2048]", arg108_1: "bf16[2048, 2048]", arg109_1: "bf16[2048, 2048]", arg110_1: "bf16[2048]", arg111_1: "bf16[2048]", arg112_1: "bf16[2048]", arg113_1: "bf16[8192, 2048]", arg114_1: "bf16[8192]", arg115_1: "bf16[2048, 8192]", arg116_1: "bf16[2048]", arg117_1: "b8[1, 1, 2048, 2048]", arg118_1: "bf16[2048]", arg119_1: "bf16[2048]", arg120_1: "bf16[2048, 2048]", arg121_1: "bf16[2048, 2048]", arg122_1: "bf16[2048, 2048]", arg123_1: "bf16[2048, 2048]", arg124_1: "bf16[2048]", arg125_1: "bf16[2048]", arg126_1: "bf16[2048]", arg127_1: "bf16[8192, 2048]", arg128_1: "bf16[8192]", arg129_1: "bf16[2048, 8192]", arg130_1: "bf16[2048]", arg131_1: "b8[1, 1, 2048, 2048]", arg132_1: "bf16[2048]", arg133_1: "bf16[2048]", arg134_1: "bf16[2048, 2048]", arg135_1: "bf16[2048, 2048]", arg136_1: "bf16[2048, 2048]", arg137_1: "bf16[2048, 2048]", arg138_1: "bf16[2048]", arg139_1: "bf16[2048]", arg140_1: "bf16[2048]", arg141_1: "bf16[8192, 2048]", arg142_1: "bf16[8192]", arg143_1: "bf16[2048, 8192]", arg144_1: "bf16[2048]", arg145_1: "b8[1, 1, 2048, 2048]", arg146_1: "bf16[2048]", arg147_1: "bf16[2048]", arg148_1: "bf16[2048, 2048]", arg149_1: "bf16[2048, 2048]", arg150_1: "bf16[2048, 2048]", arg151_1: "bf16[2048, 2048]", arg152_1: "bf16[2048]", arg153_1: "bf16[2048]", arg154_1: "bf16[2048]", arg155_1: "bf16[8192, 2048]", arg156_1: "bf16[8192]", arg157_1: "bf16[2048, 8192]", arg158_1: "bf16[2048]", arg159_1: "b8[1, 1, 2048, 2048]", arg160_1: "bf16[2048]", arg161_1: "bf16[2048]", arg162_1: "bf16[2048, 2048]", arg163_1: "bf16[2048, 2048]", arg164_1: "bf16[2048, 2048]", arg165_1: "bf16[2048, 2048]", arg166_1: "bf16[2048]", arg167_1: "bf16[2048]", arg168_1: "bf16[2048]", arg169_1: "bf16[8192, 2048]", arg170_1: "bf16[8192]", arg171_1: "bf16[2048, 8192]", arg172_1: "bf16[2048]", arg173_1: "b8[1, 1, 2048, 2048]", arg174_1: "bf16[2048]", arg175_1: "bf16[2048]", arg176_1: "bf16[2048, 2048]", arg177_1: "bf16[2048, 2048]", arg178_1: "bf16[2048, 2048]", arg179_1: "bf16[2048, 2048]", arg180_1: "bf16[2048]", arg181_1: "bf16[2048]", arg182_1: "bf16[2048]", arg183_1: "bf16[8192, 2048]", arg184_1: "bf16[8192]", arg185_1: "bf16[2048, 8192]", arg186_1: "bf16[2048]", arg187_1: "b8[1, 1, 2048, 2048]", arg188_1: "bf16[2048]", arg189_1: "bf16[2048]", arg190_1: "bf16[2048, 2048]", arg191_1: "bf16[2048, 2048]", arg192_1: "bf16[2048, 2048]", arg193_1: "bf16[2048, 2048]", arg194_1: "bf16[2048]", arg195_1: "bf16[2048]", arg196_1: "bf16[2048]", arg197_1: "bf16[8192, 2048]", arg198_1: "bf16[8192]", arg199_1: "bf16[2048, 8192]", arg200_1: "bf16[2048]", arg201_1: "b8[1, 1, 2048, 2048]", arg202_1: "bf16[2048]", arg203_1: "bf16[2048]", arg204_1: "bf16[2048, 2048]", arg205_1: "bf16[2048, 2048]", arg206_1: "bf16[2048, 2048]", arg207_1: "bf16[2048, 2048]", arg208_1: "bf16[2048]", arg209_1: "bf16[2048]", arg210_1: "bf16[2048]", arg211_1: "bf16[8192, 2048]", arg212_1: "bf16[8192]", arg213_1: "bf16[2048, 8192]", arg214_1: "bf16[2048]", arg215_1: "b8[1, 1, 2048, 2048]", arg216_1: "bf16[2048]", arg217_1: "bf16[2048]", arg218_1: "bf16[2048, 2048]", arg219_1: "bf16[2048, 2048]", arg220_1: "bf16[2048, 2048]", arg221_1: "bf16[2048, 2048]", arg222_1: "bf16[2048]", arg223_1: "bf16[2048]", arg224_1: "bf16[2048]", arg225_1: "bf16[8192, 2048]", arg226_1: "bf16[8192]", arg227_1: "bf16[2048, 8192]", arg228_1: "bf16[2048]", arg229_1: "b8[1, 1, 2048, 2048]", arg230_1: "bf16[2048]", arg231_1: "bf16[2048]", arg232_1: "bf16[2048, 2048]", arg233_1: "bf16[2048, 2048]", arg234_1: "bf16[2048, 2048]", arg235_1: "bf16[2048, 2048]", arg236_1: "bf16[2048]", arg237_1: "bf16[2048]", arg238_1: "bf16[2048]", arg239_1: "bf16[8192, 2048]", arg240_1: "bf16[8192]", arg241_1: "bf16[2048, 8192]", arg242_1: "bf16[2048]", arg243_1: "b8[1, 1, 2048, 2048]", arg244_1: "bf16[2048]", arg245_1: "bf16[2048]", arg246_1: "bf16[2048, 2048]", arg247_1: "bf16[2048, 2048]", arg248_1: "bf16[2048, 2048]", arg249_1: "bf16[2048, 2048]", arg250_1: "bf16[2048]", arg251_1: "bf16[2048]", arg252_1: "bf16[2048]", arg253_1: "bf16[8192, 2048]", arg254_1: "bf16[8192]", arg255_1: "bf16[2048, 8192]", arg256_1: "bf16[2048]", arg257_1: "b8[1, 1, 2048, 2048]", arg258_1: "bf16[2048]", arg259_1: "bf16[2048]", arg260_1: "bf16[2048, 2048]", arg261_1: "bf16[2048, 2048]", arg262_1: "bf16[2048, 2048]", arg263_1: "bf16[2048, 2048]", arg264_1: "bf16[2048]", arg265_1: "bf16[2048]", arg266_1: "bf16[2048]", arg267_1: "bf16[8192, 2048]", arg268_1: "bf16[8192]", arg269_1: "bf16[2048, 8192]", arg270_1: "bf16[2048]", arg271_1: "b8[1, 1, 2048, 2048]", arg272_1: "bf16[2048]", arg273_1: "bf16[2048]", arg274_1: "bf16[2048, 2048]", arg275_1: "bf16[2048, 2048]", arg276_1: "bf16[2048, 2048]", arg277_1: "bf16[2048, 2048]", arg278_1: "bf16[2048]", arg279_1: "bf16[2048]", arg280_1: "bf16[2048]", arg281_1: "bf16[8192, 2048]", arg282_1: "bf16[8192]", arg283_1: "bf16[2048, 8192]", arg284_1: "bf16[2048]", arg285_1: "b8[1, 1, 2048, 2048]", arg286_1: "bf16[2048]", arg287_1: "bf16[2048]", arg288_1: "bf16[2048, 2048]", arg289_1: "bf16[2048, 2048]", arg290_1: "bf16[2048, 2048]", arg291_1: "bf16[2048, 2048]", arg292_1: "bf16[2048]", arg293_1: "bf16[2048]", arg294_1: "bf16[2048]", arg295_1: "bf16[8192, 2048]", arg296_1: "bf16[8192]", arg297_1: "bf16[2048, 8192]", arg298_1: "bf16[2048]", arg299_1: "b8[1, 1, 2048, 2048]", arg300_1: "bf16[2048]", arg301_1: "bf16[2048]", arg302_1: "bf16[2048, 2048]", arg303_1: "bf16[2048, 2048]", arg304_1: "bf16[2048, 2048]", arg305_1: "bf16[2048, 2048]", arg306_1: "bf16[2048]", arg307_1: "bf16[2048]", arg308_1: "bf16[2048]", arg309_1: "bf16[8192, 2048]", arg310_1: "bf16[8192]", arg311_1: "bf16[2048, 8192]", arg312_1: "bf16[2048]", arg313_1: "b8[1, 1, 2048, 2048]", arg314_1: "bf16[2048]", arg315_1: "bf16[2048]", arg316_1: "bf16[2048, 2048]", arg317_1: "bf16[2048, 2048]", arg318_1: "bf16[2048, 2048]", arg319_1: "bf16[2048, 2048]", arg320_1: "bf16[2048]", arg321_1: "bf16[2048]", arg322_1: "bf16[2048]", arg323_1: "bf16[8192, 2048]", arg324_1: "bf16[8192]", arg325_1: "bf16[2048, 8192]", arg326_1: "bf16[2048]", arg327_1: "b8[1, 1, 2048, 2048]", arg328_1: "bf16[2048]", arg329_1: "bf16[2048]", arg330_1: "bf16[2048, 2048]", arg331_1: "bf16[2048, 2048]", arg332_1: "bf16[2048, 2048]", arg333_1: "bf16[2048, 2048]", arg334_1: "bf16[2048]", arg335_1: "bf16[2048]", arg336_1: "bf16[2048]", arg337_1: "bf16[8192, 2048]", arg338_1: "bf16[8192]", arg339_1: "bf16[2048, 8192]", arg340_1: "bf16[2048]", arg341_1: "i64[32]", arg342_1: "bf16[2048]", arg343_1: "bf16[2048]", arg344_1: "bf16[2, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626):
        # No stacktrace found for following nodes
        embedding: "bf16[32, 128, 2048]" = torch.ops.aten.embedding.default(arg0_1, arg1_1);  arg0_1 = None
        embedding_1: "bf16[1, 128, 2048]" = torch.ops.aten.embedding.default(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
        add: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        convert_element_type: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean[0]
        getitem_1: "f32[32, 128, 1]" = var_mean[1];  var_mean = None
        slice_1: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg4_1, 2, 0, 128);  arg4_1 = None
        slice_2: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 128);  slice_1 = None
        sub: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul, arg5_1);  mul = arg5_1 = None
        add_2: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_1, arg6_1);  mul_1 = arg6_1 = None
        convert_element_type_1: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        view: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_0);  _shape_param_0 = None
        permute: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        mm: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        view_2: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute_1: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        convert_element_type_2: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        expand: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_2, _shape_param_3);  convert_element_type_2 = _shape_param_3 = None
        clone: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_3: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        view_4: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_5);  _shape_param_5 = None
        permute_2: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        mm_1: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
        view_5: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_1, _shape_param_6);  mm_1 = _shape_param_6 = None
        view_6: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_5, _shape_param_7);  view_5 = _shape_param_7 = None
        permute_3: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        convert_element_type_3: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_3, torch.float32);  permute_3 = None
        permute_4: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_3, [0, 1, 3, 2]);  convert_element_type_3 = None
        expand_1: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_4, _shape_param_8);  permute_4 = _shape_param_8 = None
        clone_1: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_7: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_9);  clone_1 = _shape_param_9 = None
        bmm: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_3, view_7);  view_3 = view_7 = None
        view_8: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_10);  bmm = _shape_param_10 = None
        full: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_2, view_8, full);  slice_2 = view_8 = full = None
        full_1: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_3: "i64[128]" = torch.ops.aten.add.Tensor(iota, 0);  iota = None
        unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_3, 0);  add_3 = None
        unsqueeze_1: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        iota_1: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_4: "i64[128]" = torch.ops.aten.add.Tensor(iota_1, 0);  iota_1 = None
        unsqueeze_3: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_4, 0);  add_4 = None
        unsqueeze_4: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_2, unsqueeze_5)
        bitwise_and: "b8[1, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(full_1, le);  full_1 = le = None
        iota_2: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_6: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(iota_2, 1);  iota_2 = None
        unsqueeze_7: "i64[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 2);  unsqueeze_6 = None
        unsqueeze_8: "i64[32, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3);  unsqueeze_7 = None
        index: "i64[32, 1, 128, 1]" = torch.ops.aten.index.Tensor(arg9_1, [unsqueeze_8, unsqueeze_5]);  unsqueeze_5 = None
        index_1: "i64[32, 1, 1, 128]" = torch.ops.aten.index.Tensor(arg9_1, [unsqueeze_8, unsqueeze_2]);  arg9_1 = unsqueeze_8 = unsqueeze_2 = None
        eq: "b8[32, 1, 128, 128]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None
        bitwise_and_1: "b8[32, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None
        expand_2: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(bitwise_and_1, _shape_param_11);  bitwise_and_1 = _shape_param_11 = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_3: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_2, full_2, full_3);  expand_2 = full_2 = full_3 = None
        add_5: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where, where_1);  where = None
        amax: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_5, [-1], True)
        sub_1: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_5, amax);  add_5 = amax = None
        exp: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_4: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand_3: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_4, _shape_param_12);  convert_element_type_4 = _shape_param_12 = None
        view_9: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_3, _shape_param_13);  expand_3 = _shape_param_13 = None
        view_10: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_14);  convert_element_type_1 = _shape_param_14 = None
        permute_5: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        mm_2: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_10, permute_5);  view_10 = permute_5 = None
        view_11: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_2, _shape_param_15);  mm_2 = _shape_param_15 = None
        view_12: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_11, _shape_param_16);  view_11 = _shape_param_16 = None
        permute_6: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        expand_4: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_6, _shape_param_17);  permute_6 = _shape_param_17 = None
        clone_2: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_13: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_2, _shape_param_18);  clone_2 = _shape_param_18 = None
        bmm_1: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_9, view_13);  view_9 = view_13 = None
        view_14: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_19);  bmm_1 = _shape_param_19 = None
        permute_7: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_3: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_15: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_3, _shape_param_20);  clone_3 = _shape_param_20 = None
        view_16: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_15, _shape_param_21);  view_15 = _shape_param_21 = None
        permute_8: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg12_1, view_16, permute_8);  arg12_1 = view_16 = permute_8 = None
        view_17: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm, _shape_param_22);  addmm = _shape_param_22 = None
        add_6: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_17, add);  view_17 = add = None
        convert_element_type_5: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_6, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_5, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 128, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_5, getitem_3);  convert_element_type_5 = getitem_3 = None
        add_7: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_2: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_3: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_2, arg13_1);  mul_2 = arg13_1 = None
        add_8: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_3, arg14_1);  mul_3 = arg14_1 = None
        convert_element_type_6: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None
        view_18: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_6, _shape_param_23);  convert_element_type_6 = _shape_param_23 = None
        permute_9: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_1: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg16_1, view_18, permute_9);  arg16_1 = view_18 = permute_9 = None
        view_19: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_1, _shape_param_24);  addmm_1 = _shape_param_24 = None
        mul_4: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        pow_1: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_19, 3.0)
        mul_5: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_9: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_19, mul_5);  view_19 = mul_5 = None
        mul_6: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_9, 0.7978845608028654);  add_9 = None
        tanh: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_10: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_7: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_4, add_10);  mul_4 = add_10 = None
        view_20: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_7, _shape_param_25);  mul_7 = _shape_param_25 = None
        permute_10: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_2: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg18_1, view_20, permute_10);  arg18_1 = view_20 = permute_10 = None
        view_21: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_2, _shape_param_26);  addmm_2 = _shape_param_26 = None
        add_11: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_6, view_21);  add_6 = view_21 = None
        convert_element_type_7: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_11, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_7, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 128, 1]" = var_mean_2[0]
        getitem_5: "f32[32, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        slice_3: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg19_1, 2, 0, 128);  arg19_1 = None
        slice_4: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 128);  slice_3 = None
        sub_3: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_7, getitem_5);  convert_element_type_7 = getitem_5 = None
        add_12: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_8: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_9: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_8, arg20_1);  mul_8 = arg20_1 = None
        add_13: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_9, arg21_1);  mul_9 = arg21_1 = None
        convert_element_type_8: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None
        view_22: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_8, _shape_param_27);  _shape_param_27 = None
        permute_11: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        mm_3: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_22, permute_11);  view_22 = permute_11 = None
        view_23: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_3, _shape_param_28);  mm_3 = _shape_param_28 = None
        view_24: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_23, _shape_param_29);  view_23 = _shape_param_29 = None
        permute_12: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        convert_element_type_9: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_12, torch.float32);  permute_12 = None
        expand_5: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_9, _shape_param_30);  convert_element_type_9 = _shape_param_30 = None
        clone_4: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_25: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_4, _shape_param_31);  clone_4 = _shape_param_31 = None
        view_26: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_8, _shape_param_32);  _shape_param_32 = None
        permute_13: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        mm_4: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_26, permute_13);  view_26 = permute_13 = None
        view_27: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_4, _shape_param_33);  mm_4 = _shape_param_33 = None
        view_28: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_27, _shape_param_34);  view_27 = _shape_param_34 = None
        permute_14: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        convert_element_type_10: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_14, torch.float32);  permute_14 = None
        permute_15: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_10, [0, 1, 3, 2]);  convert_element_type_10 = None
        expand_6: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_15, _shape_param_35);  permute_15 = _shape_param_35 = None
        clone_5: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_29: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_5, _shape_param_36);  clone_5 = _shape_param_36 = None
        bmm_2: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_25, view_29);  view_25 = view_29 = None
        view_30: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_37);  bmm_2 = _shape_param_37 = None
        full_4: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_4, view_30, full_4);  slice_4 = view_30 = full_4 = None
        add_14: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_2, where_1);  where_2 = None
        amax_1: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_14, [-1], True)
        sub_4: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_14, amax_1);  add_14 = amax_1 = None
        exp_1: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_11: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        expand_7: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_11, _shape_param_38);  convert_element_type_11 = _shape_param_38 = None
        view_31: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_7, _shape_param_39);  expand_7 = _shape_param_39 = None
        view_32: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_8, _shape_param_40);  convert_element_type_8 = _shape_param_40 = None
        permute_16: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        mm_5: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_32, permute_16);  view_32 = permute_16 = None
        view_33: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_5, _shape_param_41);  mm_5 = _shape_param_41 = None
        view_34: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_33, _shape_param_42);  view_33 = _shape_param_42 = None
        permute_17: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        expand_8: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_17, _shape_param_43);  permute_17 = _shape_param_43 = None
        clone_6: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_35: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_6, _shape_param_44);  clone_6 = _shape_param_44 = None
        bmm_3: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_31, view_35);  view_31 = view_35 = None
        view_36: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_3, _shape_param_45);  bmm_3 = _shape_param_45 = None
        permute_18: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_7: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_37: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_7, _shape_param_46);  clone_7 = _shape_param_46 = None
        view_38: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_37, _shape_param_47);  view_37 = _shape_param_47 = None
        permute_19: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_3: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg26_1, view_38, permute_19);  arg26_1 = view_38 = permute_19 = None
        view_39: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_3, _shape_param_48);  addmm_3 = _shape_param_48 = None
        add_15: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_39, add_11);  view_39 = add_11 = None
        convert_element_type_12: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_15, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_12, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 128, 1]" = var_mean_3[0]
        getitem_7: "f32[32, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_12, getitem_7);  convert_element_type_12 = getitem_7 = None
        add_16: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_10: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_11: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_10, arg27_1);  mul_10 = arg27_1 = None
        add_17: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_11, arg28_1);  mul_11 = arg28_1 = None
        convert_element_type_13: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None
        view_40: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_13, _shape_param_49);  convert_element_type_13 = _shape_param_49 = None
        permute_20: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_4: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg30_1, view_40, permute_20);  arg30_1 = view_40 = permute_20 = None
        view_41: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_4, _shape_param_50);  addmm_4 = _shape_param_50 = None
        mul_12: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        pow_2: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_41, 3.0)
        mul_13: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_18: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_41, mul_13);  view_41 = mul_13 = None
        mul_14: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_18, 0.7978845608028654);  add_18 = None
        tanh_1: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_19: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
        mul_15: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_12, add_19);  mul_12 = add_19 = None
        view_42: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_15, _shape_param_51);  mul_15 = _shape_param_51 = None
        permute_21: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_5: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg32_1, view_42, permute_21);  arg32_1 = view_42 = permute_21 = None
        view_43: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_5, _shape_param_52);  addmm_5 = _shape_param_52 = None
        add_20: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_15, view_43);  add_15 = view_43 = None
        convert_element_type_14: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_20, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_14, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 128, 1]" = var_mean_4[0]
        getitem_9: "f32[32, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        slice_5: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg33_1, 2, 0, 128);  arg33_1 = None
        slice_6: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_5, 3, 0, 128);  slice_5 = None
        sub_6: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_14, getitem_9);  convert_element_type_14 = getitem_9 = None
        add_21: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        mul_16: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_17: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_16, arg34_1);  mul_16 = arg34_1 = None
        add_22: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_17, arg35_1);  mul_17 = arg35_1 = None
        convert_element_type_15: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16);  add_22 = None
        view_44: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_15, _shape_param_53);  _shape_param_53 = None
        permute_22: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
        mm_6: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_44, permute_22);  view_44 = permute_22 = None
        view_45: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_6, _shape_param_54);  mm_6 = _shape_param_54 = None
        view_46: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_45, _shape_param_55);  view_45 = _shape_param_55 = None
        permute_23: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        convert_element_type_16: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_23, torch.float32);  permute_23 = None
        expand_9: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_16, _shape_param_56);  convert_element_type_16 = _shape_param_56 = None
        clone_8: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_47: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_8, _shape_param_57);  clone_8 = _shape_param_57 = None
        view_48: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_15, _shape_param_58);  _shape_param_58 = None
        permute_24: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        mm_7: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_48, permute_24);  view_48 = permute_24 = None
        view_49: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_7, _shape_param_59);  mm_7 = _shape_param_59 = None
        view_50: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_49, _shape_param_60);  view_49 = _shape_param_60 = None
        permute_25: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        convert_element_type_17: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_25, torch.float32);  permute_25 = None
        permute_26: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_17, [0, 1, 3, 2]);  convert_element_type_17 = None
        expand_10: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_26, _shape_param_61);  permute_26 = _shape_param_61 = None
        clone_9: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_51: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_9, _shape_param_62);  clone_9 = _shape_param_62 = None
        bmm_4: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_47, view_51);  view_47 = view_51 = None
        view_52: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_4, _shape_param_63);  bmm_4 = _shape_param_63 = None
        full_5: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_6, view_52, full_5);  slice_6 = view_52 = full_5 = None
        add_23: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_3, where_1);  where_3 = None
        amax_2: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_23, [-1], True)
        sub_7: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_23, amax_2);  add_23 = amax_2 = None
        exp_2: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_18: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        expand_11: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_18, _shape_param_64);  convert_element_type_18 = _shape_param_64 = None
        view_53: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_11, _shape_param_65);  expand_11 = _shape_param_65 = None
        view_54: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_15, _shape_param_66);  convert_element_type_15 = _shape_param_66 = None
        permute_27: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg38_1, [1, 0]);  arg38_1 = None
        mm_8: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_54, permute_27);  view_54 = permute_27 = None
        view_55: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_8, _shape_param_67);  mm_8 = _shape_param_67 = None
        view_56: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_55, _shape_param_68);  view_55 = _shape_param_68 = None
        permute_28: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        expand_12: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_28, _shape_param_69);  permute_28 = _shape_param_69 = None
        clone_10: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_57: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_10, _shape_param_70);  clone_10 = _shape_param_70 = None
        bmm_5: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_53, view_57);  view_53 = view_57 = None
        view_58: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_5, _shape_param_71);  bmm_5 = _shape_param_71 = None
        permute_29: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_11: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_59: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_11, _shape_param_72);  clone_11 = _shape_param_72 = None
        view_60: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_59, _shape_param_73);  view_59 = _shape_param_73 = None
        permute_30: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_6: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg40_1, view_60, permute_30);  arg40_1 = view_60 = permute_30 = None
        view_61: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_6, _shape_param_74);  addmm_6 = _shape_param_74 = None
        add_24: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_61, add_20);  view_61 = add_20 = None
        convert_element_type_19: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_24, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_19, [2], correction = 0, keepdim = True)
        getitem_10: "f32[32, 128, 1]" = var_mean_5[0]
        getitem_11: "f32[32, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_19, getitem_11);  convert_element_type_19 = getitem_11 = None
        add_25: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_18: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_19: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_18, arg41_1);  mul_18 = arg41_1 = None
        add_26: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_19, arg42_1);  mul_19 = arg42_1 = None
        convert_element_type_20: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_26, torch.bfloat16);  add_26 = None
        view_62: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_20, _shape_param_75);  convert_element_type_20 = _shape_param_75 = None
        permute_31: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_7: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg44_1, view_62, permute_31);  arg44_1 = view_62 = permute_31 = None
        view_63: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_7, _shape_param_76);  addmm_7 = _shape_param_76 = None
        mul_20: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        pow_3: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_63, 3.0)
        mul_21: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_27: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_63, mul_21);  view_63 = mul_21 = None
        mul_22: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_27, 0.7978845608028654);  add_27 = None
        tanh_2: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_28: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
        mul_23: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_20, add_28);  mul_20 = add_28 = None
        view_64: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_23, _shape_param_77);  mul_23 = _shape_param_77 = None
        permute_32: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_8: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg46_1, view_64, permute_32);  arg46_1 = view_64 = permute_32 = None
        view_65: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_8, _shape_param_78);  addmm_8 = _shape_param_78 = None
        add_29: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_24, view_65);  add_24 = view_65 = None
        convert_element_type_21: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_29, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_21, [2], correction = 0, keepdim = True)
        getitem_12: "f32[32, 128, 1]" = var_mean_6[0]
        getitem_13: "f32[32, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        slice_7: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg47_1, 2, 0, 128);  arg47_1 = None
        slice_8: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_7, 3, 0, 128);  slice_7 = None
        sub_9: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_21, getitem_13);  convert_element_type_21 = getitem_13 = None
        add_30: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_24: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_25: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_24, arg48_1);  mul_24 = arg48_1 = None
        add_31: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_25, arg49_1);  mul_25 = arg49_1 = None
        convert_element_type_22: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None
        view_66: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_22, _shape_param_79);  _shape_param_79 = None
        permute_33: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
        mm_9: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_66, permute_33);  view_66 = permute_33 = None
        view_67: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_9, _shape_param_80);  mm_9 = _shape_param_80 = None
        view_68: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_67, _shape_param_81);  view_67 = _shape_param_81 = None
        permute_34: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        convert_element_type_23: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_34, torch.float32);  permute_34 = None
        expand_13: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_23, _shape_param_82);  convert_element_type_23 = _shape_param_82 = None
        clone_12: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_69: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_12, _shape_param_83);  clone_12 = _shape_param_83 = None
        view_70: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_22, _shape_param_84);  _shape_param_84 = None
        permute_35: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        mm_10: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_70, permute_35);  view_70 = permute_35 = None
        view_71: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_10, _shape_param_85);  mm_10 = _shape_param_85 = None
        view_72: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_71, _shape_param_86);  view_71 = _shape_param_86 = None
        permute_36: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        convert_element_type_24: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_36, torch.float32);  permute_36 = None
        permute_37: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_24, [0, 1, 3, 2]);  convert_element_type_24 = None
        expand_14: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_37, _shape_param_87);  permute_37 = _shape_param_87 = None
        clone_13: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_73: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_13, _shape_param_88);  clone_13 = _shape_param_88 = None
        bmm_6: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_69, view_73);  view_69 = view_73 = None
        view_74: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_89);  bmm_6 = _shape_param_89 = None
        full_6: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_8, view_74, full_6);  slice_8 = view_74 = full_6 = None
        add_32: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_4, where_1);  where_4 = None
        amax_3: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_32, [-1], True)
        sub_10: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_32, amax_3);  add_32 = amax_3 = None
        exp_3: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_25: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        expand_15: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_25, _shape_param_90);  convert_element_type_25 = _shape_param_90 = None
        view_75: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_15, _shape_param_91);  expand_15 = _shape_param_91 = None
        view_76: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_22, _shape_param_92);  convert_element_type_22 = _shape_param_92 = None
        permute_38: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
        mm_11: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_76, permute_38);  view_76 = permute_38 = None
        view_77: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_11, _shape_param_93);  mm_11 = _shape_param_93 = None
        view_78: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_77, _shape_param_94);  view_77 = _shape_param_94 = None
        permute_39: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
        expand_16: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_39, _shape_param_95);  permute_39 = _shape_param_95 = None
        clone_14: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_79: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_14, _shape_param_96);  clone_14 = _shape_param_96 = None
        bmm_7: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_75, view_79);  view_75 = view_79 = None
        view_80: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_7, _shape_param_97);  bmm_7 = _shape_param_97 = None
        permute_40: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        clone_15: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_81: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_15, _shape_param_98);  clone_15 = _shape_param_98 = None
        view_82: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_81, _shape_param_99);  view_81 = _shape_param_99 = None
        permute_41: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_9: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg54_1, view_82, permute_41);  arg54_1 = view_82 = permute_41 = None
        view_83: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_9, _shape_param_100);  addmm_9 = _shape_param_100 = None
        add_33: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_83, add_29);  view_83 = add_29 = None
        convert_element_type_26: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_33, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_26, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 128, 1]" = var_mean_7[0]
        getitem_15: "f32[32, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_26, getitem_15);  convert_element_type_26 = getitem_15 = None
        add_34: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_26: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_27: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_26, arg55_1);  mul_26 = arg55_1 = None
        add_35: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_27, arg56_1);  mul_27 = arg56_1 = None
        convert_element_type_27: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None
        view_84: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_27, _shape_param_101);  convert_element_type_27 = _shape_param_101 = None
        permute_42: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_10: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg58_1, view_84, permute_42);  arg58_1 = view_84 = permute_42 = None
        view_85: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_10, _shape_param_102);  addmm_10 = _shape_param_102 = None
        mul_28: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
        pow_4: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_85, 3.0)
        mul_29: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_36: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_85, mul_29);  view_85 = mul_29 = None
        mul_30: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_36, 0.7978845608028654);  add_36 = None
        tanh_3: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_37: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
        mul_31: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_28, add_37);  mul_28 = add_37 = None
        view_86: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_31, _shape_param_103);  mul_31 = _shape_param_103 = None
        permute_43: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_11: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg60_1, view_86, permute_43);  arg60_1 = view_86 = permute_43 = None
        view_87: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_11, _shape_param_104);  addmm_11 = _shape_param_104 = None
        add_38: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_33, view_87);  add_33 = view_87 = None
        convert_element_type_28: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_38, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_28, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 128, 1]" = var_mean_8[0]
        getitem_17: "f32[32, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        slice_9: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg61_1, 2, 0, 128);  arg61_1 = None
        slice_10: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_9, 3, 0, 128);  slice_9 = None
        sub_12: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_28, getitem_17);  convert_element_type_28 = getitem_17 = None
        add_39: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_32: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_33: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_32, arg62_1);  mul_32 = arg62_1 = None
        add_40: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_33, arg63_1);  mul_33 = arg63_1 = None
        convert_element_type_29: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16);  add_40 = None
        view_88: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_29, _shape_param_105);  _shape_param_105 = None
        permute_44: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
        mm_12: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_88, permute_44);  view_88 = permute_44 = None
        view_89: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_12, _shape_param_106);  mm_12 = _shape_param_106 = None
        view_90: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_89, _shape_param_107);  view_89 = _shape_param_107 = None
        permute_45: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        convert_element_type_30: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_45, torch.float32);  permute_45 = None
        expand_17: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_30, _shape_param_108);  convert_element_type_30 = _shape_param_108 = None
        clone_16: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_91: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_16, _shape_param_109);  clone_16 = _shape_param_109 = None
        view_92: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_29, _shape_param_110);  _shape_param_110 = None
        permute_46: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        mm_13: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_92, permute_46);  view_92 = permute_46 = None
        view_93: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_13, _shape_param_111);  mm_13 = _shape_param_111 = None
        view_94: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_93, _shape_param_112);  view_93 = _shape_param_112 = None
        permute_47: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        convert_element_type_31: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_47, torch.float32);  permute_47 = None
        permute_48: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_31, [0, 1, 3, 2]);  convert_element_type_31 = None
        expand_18: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_48, _shape_param_113);  permute_48 = _shape_param_113 = None
        clone_17: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_95: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_17, _shape_param_114);  clone_17 = _shape_param_114 = None
        bmm_8: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_91, view_95);  view_91 = view_95 = None
        view_96: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_8, _shape_param_115);  bmm_8 = _shape_param_115 = None
        full_7: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_10, view_96, full_7);  slice_10 = view_96 = full_7 = None
        add_41: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_5, where_1);  where_5 = None
        amax_4: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_41, [-1], True)
        sub_13: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_41, amax_4);  add_41 = amax_4 = None
        exp_4: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_32: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        expand_19: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_32, _shape_param_116);  convert_element_type_32 = _shape_param_116 = None
        view_97: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_19, _shape_param_117);  expand_19 = _shape_param_117 = None
        view_98: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_29, _shape_param_118);  convert_element_type_29 = _shape_param_118 = None
        permute_49: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        mm_14: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_98, permute_49);  view_98 = permute_49 = None
        view_99: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_14, _shape_param_119);  mm_14 = _shape_param_119 = None
        view_100: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_99, _shape_param_120);  view_99 = _shape_param_120 = None
        permute_50: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        expand_20: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_50, _shape_param_121);  permute_50 = _shape_param_121 = None
        clone_18: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_101: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_18, _shape_param_122);  clone_18 = _shape_param_122 = None
        bmm_9: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_97, view_101);  view_97 = view_101 = None
        view_102: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_9, _shape_param_123);  bmm_9 = _shape_param_123 = None
        permute_51: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_19: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_103: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_19, _shape_param_124);  clone_19 = _shape_param_124 = None
        view_104: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_103, _shape_param_125);  view_103 = _shape_param_125 = None
        permute_52: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_12: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg68_1, view_104, permute_52);  arg68_1 = view_104 = permute_52 = None
        view_105: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_12, _shape_param_126);  addmm_12 = _shape_param_126 = None
        add_42: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_105, add_38);  view_105 = add_38 = None
        convert_element_type_33: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_42, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_33, [2], correction = 0, keepdim = True)
        getitem_18: "f32[32, 128, 1]" = var_mean_9[0]
        getitem_19: "f32[32, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_33, getitem_19);  convert_element_type_33 = getitem_19 = None
        add_43: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_34: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_35: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_34, arg69_1);  mul_34 = arg69_1 = None
        add_44: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_35, arg70_1);  mul_35 = arg70_1 = None
        convert_element_type_34: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None
        view_106: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_34, _shape_param_127);  convert_element_type_34 = _shape_param_127 = None
        permute_53: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_13: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg72_1, view_106, permute_53);  arg72_1 = view_106 = permute_53 = None
        view_107: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_13, _shape_param_128);  addmm_13 = _shape_param_128 = None
        mul_36: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
        pow_5: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_107, 3.0)
        mul_37: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_45: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_107, mul_37);  view_107 = mul_37 = None
        mul_38: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_45, 0.7978845608028654);  add_45 = None
        tanh_4: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_46: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
        mul_39: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_36, add_46);  mul_36 = add_46 = None
        view_108: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_39, _shape_param_129);  mul_39 = _shape_param_129 = None
        permute_54: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_14: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg74_1, view_108, permute_54);  arg74_1 = view_108 = permute_54 = None
        view_109: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_14, _shape_param_130);  addmm_14 = _shape_param_130 = None
        add_47: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_42, view_109);  add_42 = view_109 = None
        convert_element_type_35: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_47, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_35, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 128, 1]" = var_mean_10[0]
        getitem_21: "f32[32, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        slice_11: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg75_1, 2, 0, 128);  arg75_1 = None
        slice_12: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_11, 3, 0, 128);  slice_11 = None
        sub_15: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_35, getitem_21);  convert_element_type_35 = getitem_21 = None
        add_48: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_40: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_41: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_40, arg76_1);  mul_40 = arg76_1 = None
        add_49: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_41, arg77_1);  mul_41 = arg77_1 = None
        convert_element_type_36: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16);  add_49 = None
        view_110: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_36, _shape_param_131);  _shape_param_131 = None
        permute_55: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
        mm_15: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_110, permute_55);  view_110 = permute_55 = None
        view_111: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_15, _shape_param_132);  mm_15 = _shape_param_132 = None
        view_112: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_111, _shape_param_133);  view_111 = _shape_param_133 = None
        permute_56: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        convert_element_type_37: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_56, torch.float32);  permute_56 = None
        expand_21: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_37, _shape_param_134);  convert_element_type_37 = _shape_param_134 = None
        clone_20: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_113: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_20, _shape_param_135);  clone_20 = _shape_param_135 = None
        view_114: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_36, _shape_param_136);  _shape_param_136 = None
        permute_57: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        mm_16: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_114, permute_57);  view_114 = permute_57 = None
        view_115: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_16, _shape_param_137);  mm_16 = _shape_param_137 = None
        view_116: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_115, _shape_param_138);  view_115 = _shape_param_138 = None
        permute_58: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        convert_element_type_38: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_58, torch.float32);  permute_58 = None
        permute_59: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_38, [0, 1, 3, 2]);  convert_element_type_38 = None
        expand_22: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_59, _shape_param_139);  permute_59 = _shape_param_139 = None
        clone_21: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_117: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_21, _shape_param_140);  clone_21 = _shape_param_140 = None
        bmm_10: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_113, view_117);  view_113 = view_117 = None
        view_118: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_141);  bmm_10 = _shape_param_141 = None
        full_8: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_12, view_118, full_8);  slice_12 = view_118 = full_8 = None
        add_50: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_6, where_1);  where_6 = None
        amax_5: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_50, [-1], True)
        sub_16: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_50, amax_5);  add_50 = amax_5 = None
        exp_5: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_39: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        expand_23: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_39, _shape_param_142);  convert_element_type_39 = _shape_param_142 = None
        view_119: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_23, _shape_param_143);  expand_23 = _shape_param_143 = None
        view_120: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_36, _shape_param_144);  convert_element_type_36 = _shape_param_144 = None
        permute_60: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
        mm_17: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_120, permute_60);  view_120 = permute_60 = None
        view_121: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_17, _shape_param_145);  mm_17 = _shape_param_145 = None
        view_122: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_121, _shape_param_146);  view_121 = _shape_param_146 = None
        permute_61: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        expand_24: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_61, _shape_param_147);  permute_61 = _shape_param_147 = None
        clone_22: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_123: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_22, _shape_param_148);  clone_22 = _shape_param_148 = None
        bmm_11: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_119, view_123);  view_119 = view_123 = None
        view_124: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_11, _shape_param_149);  bmm_11 = _shape_param_149 = None
        permute_62: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        clone_23: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_125: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_23, _shape_param_150);  clone_23 = _shape_param_150 = None
        view_126: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_125, _shape_param_151);  view_125 = _shape_param_151 = None
        permute_63: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_15: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg82_1, view_126, permute_63);  arg82_1 = view_126 = permute_63 = None
        view_127: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_15, _shape_param_152);  addmm_15 = _shape_param_152 = None
        add_51: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_127, add_47);  view_127 = add_47 = None
        convert_element_type_40: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_51, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_40, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 128, 1]" = var_mean_11[0]
        getitem_23: "f32[32, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_40, getitem_23);  convert_element_type_40 = getitem_23 = None
        add_52: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_42: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_43: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_42, arg83_1);  mul_42 = arg83_1 = None
        add_53: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_43, arg84_1);  mul_43 = arg84_1 = None
        convert_element_type_41: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16);  add_53 = None
        view_128: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_41, _shape_param_153);  convert_element_type_41 = _shape_param_153 = None
        permute_64: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_16: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg86_1, view_128, permute_64);  arg86_1 = view_128 = permute_64 = None
        view_129: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_16, _shape_param_154);  addmm_16 = _shape_param_154 = None
        mul_44: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        pow_6: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_129, 3.0)
        mul_45: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_54: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_129, mul_45);  view_129 = mul_45 = None
        mul_46: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_54, 0.7978845608028654);  add_54 = None
        tanh_5: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_55: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
        mul_47: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_44, add_55);  mul_44 = add_55 = None
        view_130: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_47, _shape_param_155);  mul_47 = _shape_param_155 = None
        permute_65: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_17: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg88_1, view_130, permute_65);  arg88_1 = view_130 = permute_65 = None
        view_131: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_17, _shape_param_156);  addmm_17 = _shape_param_156 = None
        add_56: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_51, view_131);  add_51 = view_131 = None
        convert_element_type_42: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_56, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_42, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 128, 1]" = var_mean_12[0]
        getitem_25: "f32[32, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        slice_13: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg89_1, 2, 0, 128);  arg89_1 = None
        slice_14: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_13, 3, 0, 128);  slice_13 = None
        sub_18: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_42, getitem_25);  convert_element_type_42 = getitem_25 = None
        add_57: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        mul_48: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_49: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_48, arg90_1);  mul_48 = arg90_1 = None
        add_58: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_49, arg91_1);  mul_49 = arg91_1 = None
        convert_element_type_43: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_58, torch.bfloat16);  add_58 = None
        view_132: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_43, _shape_param_157);  _shape_param_157 = None
        permute_66: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        mm_18: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_132, permute_66);  view_132 = permute_66 = None
        view_133: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_18, _shape_param_158);  mm_18 = _shape_param_158 = None
        view_134: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_133, _shape_param_159);  view_133 = _shape_param_159 = None
        permute_67: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        convert_element_type_44: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_67, torch.float32);  permute_67 = None
        expand_25: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_44, _shape_param_160);  convert_element_type_44 = _shape_param_160 = None
        clone_24: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_135: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_24, _shape_param_161);  clone_24 = _shape_param_161 = None
        view_136: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_43, _shape_param_162);  _shape_param_162 = None
        permute_68: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        mm_19: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_136, permute_68);  view_136 = permute_68 = None
        view_137: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_19, _shape_param_163);  mm_19 = _shape_param_163 = None
        view_138: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_137, _shape_param_164);  view_137 = _shape_param_164 = None
        permute_69: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        convert_element_type_45: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_69, torch.float32);  permute_69 = None
        permute_70: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_45, [0, 1, 3, 2]);  convert_element_type_45 = None
        expand_26: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_70, _shape_param_165);  permute_70 = _shape_param_165 = None
        clone_25: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_139: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_25, _shape_param_166);  clone_25 = _shape_param_166 = None
        bmm_12: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_135, view_139);  view_135 = view_139 = None
        view_140: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_12, _shape_param_167);  bmm_12 = _shape_param_167 = None
        full_9: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_14, view_140, full_9);  slice_14 = view_140 = full_9 = None
        add_59: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_7, where_1);  where_7 = None
        amax_6: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_59, [-1], True)
        sub_19: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_59, amax_6);  add_59 = amax_6 = None
        exp_6: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_46: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        expand_27: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_46, _shape_param_168);  convert_element_type_46 = _shape_param_168 = None
        view_141: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_27, _shape_param_169);  expand_27 = _shape_param_169 = None
        view_142: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_43, _shape_param_170);  convert_element_type_43 = _shape_param_170 = None
        permute_71: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        mm_20: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_142, permute_71);  view_142 = permute_71 = None
        view_143: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_20, _shape_param_171);  mm_20 = _shape_param_171 = None
        view_144: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_143, _shape_param_172);  view_143 = _shape_param_172 = None
        permute_72: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        expand_28: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_72, _shape_param_173);  permute_72 = _shape_param_173 = None
        clone_26: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_145: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_26, _shape_param_174);  clone_26 = _shape_param_174 = None
        bmm_13: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_141, view_145);  view_141 = view_145 = None
        view_146: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_13, _shape_param_175);  bmm_13 = _shape_param_175 = None
        permute_73: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_27: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_147: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_27, _shape_param_176);  clone_27 = _shape_param_176 = None
        view_148: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_147, _shape_param_177);  view_147 = _shape_param_177 = None
        permute_74: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_18: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg96_1, view_148, permute_74);  arg96_1 = view_148 = permute_74 = None
        view_149: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_18, _shape_param_178);  addmm_18 = _shape_param_178 = None
        add_60: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_149, add_56);  view_149 = add_56 = None
        convert_element_type_47: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_60, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_47, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 128, 1]" = var_mean_13[0]
        getitem_27: "f32[32, 128, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_20: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_47, getitem_27);  convert_element_type_47 = getitem_27 = None
        add_61: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        mul_50: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_51: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_50, arg97_1);  mul_50 = arg97_1 = None
        add_62: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_51, arg98_1);  mul_51 = arg98_1 = None
        convert_element_type_48: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_62, torch.bfloat16);  add_62 = None
        view_150: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_48, _shape_param_179);  convert_element_type_48 = _shape_param_179 = None
        permute_75: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_19: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg100_1, view_150, permute_75);  arg100_1 = view_150 = permute_75 = None
        view_151: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_19, _shape_param_180);  addmm_19 = _shape_param_180 = None
        mul_52: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
        pow_7: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_151, 3.0)
        mul_53: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_63: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_151, mul_53);  view_151 = mul_53 = None
        mul_54: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_63, 0.7978845608028654);  add_63 = None
        tanh_6: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_64: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
        mul_55: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_52, add_64);  mul_52 = add_64 = None
        view_152: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_55, _shape_param_181);  mul_55 = _shape_param_181 = None
        permute_76: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_20: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg102_1, view_152, permute_76);  arg102_1 = view_152 = permute_76 = None
        view_153: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_20, _shape_param_182);  addmm_20 = _shape_param_182 = None
        add_65: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_60, view_153);  add_60 = view_153 = None
        convert_element_type_49: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_65, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_49, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 128, 1]" = var_mean_14[0]
        getitem_29: "f32[32, 128, 1]" = var_mean_14[1];  var_mean_14 = None
        slice_15: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg103_1, 2, 0, 128);  arg103_1 = None
        slice_16: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_15, 3, 0, 128);  slice_15 = None
        sub_21: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_49, getitem_29);  convert_element_type_49 = getitem_29 = None
        add_66: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        mul_56: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_57: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_56, arg104_1);  mul_56 = arg104_1 = None
        add_67: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_57, arg105_1);  mul_57 = arg105_1 = None
        convert_element_type_50: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_67, torch.bfloat16);  add_67 = None
        view_154: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_50, _shape_param_183);  _shape_param_183 = None
        permute_77: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        mm_21: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_154, permute_77);  view_154 = permute_77 = None
        view_155: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_21, _shape_param_184);  mm_21 = _shape_param_184 = None
        view_156: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_155, _shape_param_185);  view_155 = _shape_param_185 = None
        permute_78: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
        convert_element_type_51: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_78, torch.float32);  permute_78 = None
        expand_29: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_51, _shape_param_186);  convert_element_type_51 = _shape_param_186 = None
        clone_28: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_157: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_28, _shape_param_187);  clone_28 = _shape_param_187 = None
        view_158: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_50, _shape_param_188);  _shape_param_188 = None
        permute_79: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        mm_22: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_158, permute_79);  view_158 = permute_79 = None
        view_159: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_22, _shape_param_189);  mm_22 = _shape_param_189 = None
        view_160: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_159, _shape_param_190);  view_159 = _shape_param_190 = None
        permute_80: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
        convert_element_type_52: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_80, torch.float32);  permute_80 = None
        permute_81: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_52, [0, 1, 3, 2]);  convert_element_type_52 = None
        expand_30: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_81, _shape_param_191);  permute_81 = _shape_param_191 = None
        clone_29: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_161: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_29, _shape_param_192);  clone_29 = _shape_param_192 = None
        bmm_14: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_157, view_161);  view_157 = view_161 = None
        view_162: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_14, _shape_param_193);  bmm_14 = _shape_param_193 = None
        full_10: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_16, view_162, full_10);  slice_16 = view_162 = full_10 = None
        add_68: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_8, where_1);  where_8 = None
        amax_7: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_68, [-1], True)
        sub_22: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_68, amax_7);  add_68 = amax_7 = None
        exp_7: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_53: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        expand_31: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_53, _shape_param_194);  convert_element_type_53 = _shape_param_194 = None
        view_163: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_31, _shape_param_195);  expand_31 = _shape_param_195 = None
        view_164: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_50, _shape_param_196);  convert_element_type_50 = _shape_param_196 = None
        permute_82: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        mm_23: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_164, permute_82);  view_164 = permute_82 = None
        view_165: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_23, _shape_param_197);  mm_23 = _shape_param_197 = None
        view_166: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_165, _shape_param_198);  view_165 = _shape_param_198 = None
        permute_83: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        expand_32: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_83, _shape_param_199);  permute_83 = _shape_param_199 = None
        clone_30: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_167: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_30, _shape_param_200);  clone_30 = _shape_param_200 = None
        bmm_15: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_163, view_167);  view_163 = view_167 = None
        view_168: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_15, _shape_param_201);  bmm_15 = _shape_param_201 = None
        permute_84: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_31: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_169: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_31, _shape_param_202);  clone_31 = _shape_param_202 = None
        view_170: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_169, _shape_param_203);  view_169 = _shape_param_203 = None
        permute_85: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_21: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg110_1, view_170, permute_85);  arg110_1 = view_170 = permute_85 = None
        view_171: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_21, _shape_param_204);  addmm_21 = _shape_param_204 = None
        add_69: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_171, add_65);  view_171 = add_65 = None
        convert_element_type_54: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_69, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_54, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 128, 1]" = var_mean_15[0]
        getitem_31: "f32[32, 128, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_23: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_54, getitem_31);  convert_element_type_54 = getitem_31 = None
        add_70: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_58: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_59: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_58, arg111_1);  mul_58 = arg111_1 = None
        add_71: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_59, arg112_1);  mul_59 = arg112_1 = None
        convert_element_type_55: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16);  add_71 = None
        view_172: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_55, _shape_param_205);  convert_element_type_55 = _shape_param_205 = None
        permute_86: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_22: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg114_1, view_172, permute_86);  arg114_1 = view_172 = permute_86 = None
        view_173: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_22, _shape_param_206);  addmm_22 = _shape_param_206 = None
        mul_60: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        pow_8: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_173, 3.0)
        mul_61: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_72: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_173, mul_61);  view_173 = mul_61 = None
        mul_62: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_72, 0.7978845608028654);  add_72 = None
        tanh_7: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_73: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
        mul_63: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_60, add_73);  mul_60 = add_73 = None
        view_174: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_63, _shape_param_207);  mul_63 = _shape_param_207 = None
        permute_87: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_23: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg116_1, view_174, permute_87);  arg116_1 = view_174 = permute_87 = None
        view_175: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_23, _shape_param_208);  addmm_23 = _shape_param_208 = None
        add_74: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_69, view_175);  add_69 = view_175 = None
        convert_element_type_56: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_74, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_56, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 128, 1]" = var_mean_16[0]
        getitem_33: "f32[32, 128, 1]" = var_mean_16[1];  var_mean_16 = None
        slice_17: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg117_1, 2, 0, 128);  arg117_1 = None
        slice_18: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_17, 3, 0, 128);  slice_17 = None
        sub_24: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_56, getitem_33);  convert_element_type_56 = getitem_33 = None
        add_75: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_64: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_65: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_64, arg118_1);  mul_64 = arg118_1 = None
        add_76: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_65, arg119_1);  mul_65 = arg119_1 = None
        convert_element_type_57: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_76, torch.bfloat16);  add_76 = None
        view_176: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_57, _shape_param_209);  _shape_param_209 = None
        permute_88: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
        mm_24: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_176, permute_88);  view_176 = permute_88 = None
        view_177: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_24, _shape_param_210);  mm_24 = _shape_param_210 = None
        view_178: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_177, _shape_param_211);  view_177 = _shape_param_211 = None
        permute_89: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        convert_element_type_58: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_89, torch.float32);  permute_89 = None
        expand_33: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_58, _shape_param_212);  convert_element_type_58 = _shape_param_212 = None
        clone_32: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_179: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_32, _shape_param_213);  clone_32 = _shape_param_213 = None
        view_180: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_57, _shape_param_214);  _shape_param_214 = None
        permute_90: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        mm_25: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_180, permute_90);  view_180 = permute_90 = None
        view_181: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_25, _shape_param_215);  mm_25 = _shape_param_215 = None
        view_182: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_181, _shape_param_216);  view_181 = _shape_param_216 = None
        permute_91: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        convert_element_type_59: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_91, torch.float32);  permute_91 = None
        permute_92: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_59, [0, 1, 3, 2]);  convert_element_type_59 = None
        expand_34: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_92, _shape_param_217);  permute_92 = _shape_param_217 = None
        clone_33: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_183: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_33, _shape_param_218);  clone_33 = _shape_param_218 = None
        bmm_16: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_179, view_183);  view_179 = view_183 = None
        view_184: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_16, _shape_param_219);  bmm_16 = _shape_param_219 = None
        full_11: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_18, view_184, full_11);  slice_18 = view_184 = full_11 = None
        add_77: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_9, where_1);  where_9 = None
        amax_8: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_77, [-1], True)
        sub_25: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_77, amax_8);  add_77 = amax_8 = None
        exp_8: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_60: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        expand_35: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_60, _shape_param_220);  convert_element_type_60 = _shape_param_220 = None
        view_185: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_35, _shape_param_221);  expand_35 = _shape_param_221 = None
        view_186: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_57, _shape_param_222);  convert_element_type_57 = _shape_param_222 = None
        permute_93: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
        mm_26: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_186, permute_93);  view_186 = permute_93 = None
        view_187: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_26, _shape_param_223);  mm_26 = _shape_param_223 = None
        view_188: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_187, _shape_param_224);  view_187 = _shape_param_224 = None
        permute_94: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
        expand_36: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_94, _shape_param_225);  permute_94 = _shape_param_225 = None
        clone_34: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_189: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_34, _shape_param_226);  clone_34 = _shape_param_226 = None
        bmm_17: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_185, view_189);  view_185 = view_189 = None
        view_190: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_17, _shape_param_227);  bmm_17 = _shape_param_227 = None
        permute_95: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_35: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_191: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_35, _shape_param_228);  clone_35 = _shape_param_228 = None
        view_192: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_191, _shape_param_229);  view_191 = _shape_param_229 = None
        permute_96: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_24: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg124_1, view_192, permute_96);  arg124_1 = view_192 = permute_96 = None
        view_193: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_24, _shape_param_230);  addmm_24 = _shape_param_230 = None
        add_78: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_193, add_74);  view_193 = add_74 = None
        convert_element_type_61: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_78, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_61, [2], correction = 0, keepdim = True)
        getitem_34: "f32[32, 128, 1]" = var_mean_17[0]
        getitem_35: "f32[32, 128, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_26: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_61, getitem_35);  convert_element_type_61 = getitem_35 = None
        add_79: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        mul_66: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_67: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_66, arg125_1);  mul_66 = arg125_1 = None
        add_80: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_67, arg126_1);  mul_67 = arg126_1 = None
        convert_element_type_62: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16);  add_80 = None
        view_194: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_62, _shape_param_231);  convert_element_type_62 = _shape_param_231 = None
        permute_97: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_25: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg128_1, view_194, permute_97);  arg128_1 = view_194 = permute_97 = None
        view_195: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_25, _shape_param_232);  addmm_25 = _shape_param_232 = None
        mul_68: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
        pow_9: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_195, 3.0)
        mul_69: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_81: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_195, mul_69);  view_195 = mul_69 = None
        mul_70: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_81, 0.7978845608028654);  add_81 = None
        tanh_8: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_82: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
        mul_71: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_68, add_82);  mul_68 = add_82 = None
        view_196: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_71, _shape_param_233);  mul_71 = _shape_param_233 = None
        permute_98: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_26: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg130_1, view_196, permute_98);  arg130_1 = view_196 = permute_98 = None
        view_197: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_26, _shape_param_234);  addmm_26 = _shape_param_234 = None
        add_83: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_78, view_197);  add_78 = view_197 = None
        convert_element_type_63: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_83, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_63, [2], correction = 0, keepdim = True)
        getitem_36: "f32[32, 128, 1]" = var_mean_18[0]
        getitem_37: "f32[32, 128, 1]" = var_mean_18[1];  var_mean_18 = None
        slice_19: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg131_1, 2, 0, 128);  arg131_1 = None
        slice_20: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_19, 3, 0, 128);  slice_19 = None
        sub_27: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_63, getitem_37);  convert_element_type_63 = getitem_37 = None
        add_84: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_72: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_73: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_72, arg132_1);  mul_72 = arg132_1 = None
        add_85: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_73, arg133_1);  mul_73 = arg133_1 = None
        convert_element_type_64: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16);  add_85 = None
        view_198: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_64, _shape_param_235);  _shape_param_235 = None
        permute_99: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
        mm_27: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_198, permute_99);  view_198 = permute_99 = None
        view_199: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_27, _shape_param_236);  mm_27 = _shape_param_236 = None
        view_200: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_199, _shape_param_237);  view_199 = _shape_param_237 = None
        permute_100: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
        convert_element_type_65: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_100, torch.float32);  permute_100 = None
        expand_37: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_65, _shape_param_238);  convert_element_type_65 = _shape_param_238 = None
        clone_36: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_201: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_36, _shape_param_239);  clone_36 = _shape_param_239 = None
        view_202: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_64, _shape_param_240);  _shape_param_240 = None
        permute_101: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        mm_28: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_202, permute_101);  view_202 = permute_101 = None
        view_203: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_28, _shape_param_241);  mm_28 = _shape_param_241 = None
        view_204: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_203, _shape_param_242);  view_203 = _shape_param_242 = None
        permute_102: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
        convert_element_type_66: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_102, torch.float32);  permute_102 = None
        permute_103: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_66, [0, 1, 3, 2]);  convert_element_type_66 = None
        expand_38: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_103, _shape_param_243);  permute_103 = _shape_param_243 = None
        clone_37: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_205: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_37, _shape_param_244);  clone_37 = _shape_param_244 = None
        bmm_18: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_201, view_205);  view_201 = view_205 = None
        view_206: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_18, _shape_param_245);  bmm_18 = _shape_param_245 = None
        full_12: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_20, view_206, full_12);  slice_20 = view_206 = full_12 = None
        add_86: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_10, where_1);  where_10 = None
        amax_9: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_86, [-1], True)
        sub_28: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_86, amax_9);  add_86 = amax_9 = None
        exp_9: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_67: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        expand_39: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_67, _shape_param_246);  convert_element_type_67 = _shape_param_246 = None
        view_207: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_39, _shape_param_247);  expand_39 = _shape_param_247 = None
        view_208: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_64, _shape_param_248);  convert_element_type_64 = _shape_param_248 = None
        permute_104: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        mm_29: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_208, permute_104);  view_208 = permute_104 = None
        view_209: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_29, _shape_param_249);  mm_29 = _shape_param_249 = None
        view_210: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_209, _shape_param_250);  view_209 = _shape_param_250 = None
        permute_105: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
        expand_40: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_105, _shape_param_251);  permute_105 = _shape_param_251 = None
        clone_38: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_211: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_38, _shape_param_252);  clone_38 = _shape_param_252 = None
        bmm_19: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_207, view_211);  view_207 = view_211 = None
        view_212: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_19, _shape_param_253);  bmm_19 = _shape_param_253 = None
        permute_106: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        clone_39: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_213: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_39, _shape_param_254);  clone_39 = _shape_param_254 = None
        view_214: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_213, _shape_param_255);  view_213 = _shape_param_255 = None
        permute_107: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_27: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg138_1, view_214, permute_107);  arg138_1 = view_214 = permute_107 = None
        view_215: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_27, _shape_param_256);  addmm_27 = _shape_param_256 = None
        add_87: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_215, add_83);  view_215 = add_83 = None
        convert_element_type_68: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_87, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_68, [2], correction = 0, keepdim = True)
        getitem_38: "f32[32, 128, 1]" = var_mean_19[0]
        getitem_39: "f32[32, 128, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_29: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_68, getitem_39);  convert_element_type_68 = getitem_39 = None
        add_88: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_74: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_75: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_74, arg139_1);  mul_74 = arg139_1 = None
        add_89: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_75, arg140_1);  mul_75 = arg140_1 = None
        convert_element_type_69: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16);  add_89 = None
        view_216: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_69, _shape_param_257);  convert_element_type_69 = _shape_param_257 = None
        permute_108: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_28: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg142_1, view_216, permute_108);  arg142_1 = view_216 = permute_108 = None
        view_217: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_28, _shape_param_258);  addmm_28 = _shape_param_258 = None
        mul_76: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        pow_10: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_217, 3.0)
        mul_77: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_90: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_217, mul_77);  view_217 = mul_77 = None
        mul_78: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_90, 0.7978845608028654);  add_90 = None
        tanh_9: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_91: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
        mul_79: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_76, add_91);  mul_76 = add_91 = None
        view_218: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_79, _shape_param_259);  mul_79 = _shape_param_259 = None
        permute_109: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_29: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg144_1, view_218, permute_109);  arg144_1 = view_218 = permute_109 = None
        view_219: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_29, _shape_param_260);  addmm_29 = _shape_param_260 = None
        add_92: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_87, view_219);  add_87 = view_219 = None
        convert_element_type_70: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_92, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_70, [2], correction = 0, keepdim = True)
        getitem_40: "f32[32, 128, 1]" = var_mean_20[0]
        getitem_41: "f32[32, 128, 1]" = var_mean_20[1];  var_mean_20 = None
        slice_21: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg145_1, 2, 0, 128);  arg145_1 = None
        slice_22: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_21, 3, 0, 128);  slice_21 = None
        sub_30: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_70, getitem_41);  convert_element_type_70 = getitem_41 = None
        add_93: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_80: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_81: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_80, arg146_1);  mul_80 = arg146_1 = None
        add_94: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_81, arg147_1);  mul_81 = arg147_1 = None
        convert_element_type_71: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16);  add_94 = None
        view_220: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_71, _shape_param_261);  _shape_param_261 = None
        permute_110: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        mm_30: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_220, permute_110);  view_220 = permute_110 = None
        view_221: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_30, _shape_param_262);  mm_30 = _shape_param_262 = None
        view_222: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_221, _shape_param_263);  view_221 = _shape_param_263 = None
        permute_111: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
        convert_element_type_72: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_111, torch.float32);  permute_111 = None
        expand_41: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_72, _shape_param_264);  convert_element_type_72 = _shape_param_264 = None
        clone_40: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_223: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_40, _shape_param_265);  clone_40 = _shape_param_265 = None
        view_224: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_71, _shape_param_266);  _shape_param_266 = None
        permute_112: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        mm_31: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_224, permute_112);  view_224 = permute_112 = None
        view_225: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_31, _shape_param_267);  mm_31 = _shape_param_267 = None
        view_226: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_225, _shape_param_268);  view_225 = _shape_param_268 = None
        permute_113: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        convert_element_type_73: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_113, torch.float32);  permute_113 = None
        permute_114: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_73, [0, 1, 3, 2]);  convert_element_type_73 = None
        expand_42: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_114, _shape_param_269);  permute_114 = _shape_param_269 = None
        clone_41: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_227: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_41, _shape_param_270);  clone_41 = _shape_param_270 = None
        bmm_20: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_223, view_227);  view_223 = view_227 = None
        view_228: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_20, _shape_param_271);  bmm_20 = _shape_param_271 = None
        full_13: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_22, view_228, full_13);  slice_22 = view_228 = full_13 = None
        add_95: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_11, where_1);  where_11 = None
        amax_10: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_95, [-1], True)
        sub_31: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_95, amax_10);  add_95 = amax_10 = None
        exp_10: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_74: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        expand_43: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_74, _shape_param_272);  convert_element_type_74 = _shape_param_272 = None
        view_229: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_43, _shape_param_273);  expand_43 = _shape_param_273 = None
        view_230: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_71, _shape_param_274);  convert_element_type_71 = _shape_param_274 = None
        permute_115: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        mm_32: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_230, permute_115);  view_230 = permute_115 = None
        view_231: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_32, _shape_param_275);  mm_32 = _shape_param_275 = None
        view_232: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_231, _shape_param_276);  view_231 = _shape_param_276 = None
        permute_116: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None
        expand_44: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_116, _shape_param_277);  permute_116 = _shape_param_277 = None
        clone_42: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_233: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_42, _shape_param_278);  clone_42 = _shape_param_278 = None
        bmm_21: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_229, view_233);  view_229 = view_233 = None
        view_234: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_21, _shape_param_279);  bmm_21 = _shape_param_279 = None
        permute_117: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_43: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_235: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_43, _shape_param_280);  clone_43 = _shape_param_280 = None
        view_236: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_235, _shape_param_281);  view_235 = _shape_param_281 = None
        permute_118: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_30: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg152_1, view_236, permute_118);  arg152_1 = view_236 = permute_118 = None
        view_237: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_30, _shape_param_282);  addmm_30 = _shape_param_282 = None
        add_96: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_237, add_92);  view_237 = add_92 = None
        convert_element_type_75: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_96, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_75, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 128, 1]" = var_mean_21[0]
        getitem_43: "f32[32, 128, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_32: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_75, getitem_43);  convert_element_type_75 = getitem_43 = None
        add_97: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        mul_82: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_83: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_82, arg153_1);  mul_82 = arg153_1 = None
        add_98: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_83, arg154_1);  mul_83 = arg154_1 = None
        convert_element_type_76: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_98, torch.bfloat16);  add_98 = None
        view_238: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_76, _shape_param_283);  convert_element_type_76 = _shape_param_283 = None
        permute_119: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_31: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg156_1, view_238, permute_119);  arg156_1 = view_238 = permute_119 = None
        view_239: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_31, _shape_param_284);  addmm_31 = _shape_param_284 = None
        mul_84: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        pow_11: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_239, 3.0)
        mul_85: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_99: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_239, mul_85);  view_239 = mul_85 = None
        mul_86: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_99, 0.7978845608028654);  add_99 = None
        tanh_10: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_100: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
        mul_87: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_84, add_100);  mul_84 = add_100 = None
        view_240: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_87, _shape_param_285);  mul_87 = _shape_param_285 = None
        permute_120: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_32: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg158_1, view_240, permute_120);  arg158_1 = view_240 = permute_120 = None
        view_241: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_32, _shape_param_286);  addmm_32 = _shape_param_286 = None
        add_101: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_96, view_241);  add_96 = view_241 = None
        convert_element_type_77: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_101, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_77, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 128, 1]" = var_mean_22[0]
        getitem_45: "f32[32, 128, 1]" = var_mean_22[1];  var_mean_22 = None
        slice_23: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg159_1, 2, 0, 128);  arg159_1 = None
        slice_24: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_23, 3, 0, 128);  slice_23 = None
        sub_33: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_77, getitem_45);  convert_element_type_77 = getitem_45 = None
        add_102: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_88: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
        mul_89: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_88, arg160_1);  mul_88 = arg160_1 = None
        add_103: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_89, arg161_1);  mul_89 = arg161_1 = None
        convert_element_type_78: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16);  add_103 = None
        view_242: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_78, _shape_param_287);  _shape_param_287 = None
        permute_121: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
        mm_33: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_242, permute_121);  view_242 = permute_121 = None
        view_243: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_33, _shape_param_288);  mm_33 = _shape_param_288 = None
        view_244: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_243, _shape_param_289);  view_243 = _shape_param_289 = None
        permute_122: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
        convert_element_type_79: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_122, torch.float32);  permute_122 = None
        expand_45: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_79, _shape_param_290);  convert_element_type_79 = _shape_param_290 = None
        clone_44: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_245: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_44, _shape_param_291);  clone_44 = _shape_param_291 = None
        view_246: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_78, _shape_param_292);  _shape_param_292 = None
        permute_123: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        mm_34: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_246, permute_123);  view_246 = permute_123 = None
        view_247: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_34, _shape_param_293);  mm_34 = _shape_param_293 = None
        view_248: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_247, _shape_param_294);  view_247 = _shape_param_294 = None
        permute_124: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
        convert_element_type_80: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_124, torch.float32);  permute_124 = None
        permute_125: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_80, [0, 1, 3, 2]);  convert_element_type_80 = None
        expand_46: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_125, _shape_param_295);  permute_125 = _shape_param_295 = None
        clone_45: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_249: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_45, _shape_param_296);  clone_45 = _shape_param_296 = None
        bmm_22: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_245, view_249);  view_245 = view_249 = None
        view_250: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_297);  bmm_22 = _shape_param_297 = None
        full_14: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_24, view_250, full_14);  slice_24 = view_250 = full_14 = None
        add_104: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_12, where_1);  where_12 = None
        amax_11: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_104, [-1], True)
        sub_34: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_104, amax_11);  add_104 = amax_11 = None
        exp_11: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_81: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        expand_47: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_81, _shape_param_298);  convert_element_type_81 = _shape_param_298 = None
        view_251: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_47, _shape_param_299);  expand_47 = _shape_param_299 = None
        view_252: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_78, _shape_param_300);  convert_element_type_78 = _shape_param_300 = None
        permute_126: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        mm_35: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_252, permute_126);  view_252 = permute_126 = None
        view_253: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_35, _shape_param_301);  mm_35 = _shape_param_301 = None
        view_254: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_253, _shape_param_302);  view_253 = _shape_param_302 = None
        permute_127: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_254, [0, 2, 1, 3]);  view_254 = None
        expand_48: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_127, _shape_param_303);  permute_127 = _shape_param_303 = None
        clone_46: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_255: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_46, _shape_param_304);  clone_46 = _shape_param_304 = None
        bmm_23: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_251, view_255);  view_251 = view_255 = None
        view_256: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_23, _shape_param_305);  bmm_23 = _shape_param_305 = None
        permute_128: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        clone_47: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_257: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_47, _shape_param_306);  clone_47 = _shape_param_306 = None
        view_258: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_257, _shape_param_307);  view_257 = _shape_param_307 = None
        permute_129: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_33: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg166_1, view_258, permute_129);  arg166_1 = view_258 = permute_129 = None
        view_259: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_33, _shape_param_308);  addmm_33 = _shape_param_308 = None
        add_105: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_259, add_101);  view_259 = add_101 = None
        convert_element_type_82: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_105, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_82, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 128, 1]" = var_mean_23[0]
        getitem_47: "f32[32, 128, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_35: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_82, getitem_47);  convert_element_type_82 = getitem_47 = None
        add_106: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        mul_90: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_91: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_90, arg167_1);  mul_90 = arg167_1 = None
        add_107: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_91, arg168_1);  mul_91 = arg168_1 = None
        convert_element_type_83: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_107, torch.bfloat16);  add_107 = None
        view_260: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_83, _shape_param_309);  convert_element_type_83 = _shape_param_309 = None
        permute_130: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_34: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg170_1, view_260, permute_130);  arg170_1 = view_260 = permute_130 = None
        view_261: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_34, _shape_param_310);  addmm_34 = _shape_param_310 = None
        mul_92: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        pow_12: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_261, 3.0)
        mul_93: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_108: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_261, mul_93);  view_261 = mul_93 = None
        mul_94: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_108, 0.7978845608028654);  add_108 = None
        tanh_11: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_109: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
        mul_95: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_92, add_109);  mul_92 = add_109 = None
        view_262: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_95, _shape_param_311);  mul_95 = _shape_param_311 = None
        permute_131: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_35: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg172_1, view_262, permute_131);  arg172_1 = view_262 = permute_131 = None
        view_263: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_35, _shape_param_312);  addmm_35 = _shape_param_312 = None
        add_110: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_105, view_263);  add_105 = view_263 = None
        convert_element_type_84: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_110, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_84, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 128, 1]" = var_mean_24[0]
        getitem_49: "f32[32, 128, 1]" = var_mean_24[1];  var_mean_24 = None
        slice_25: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg173_1, 2, 0, 128);  arg173_1 = None
        slice_26: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_25, 3, 0, 128);  slice_25 = None
        sub_36: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_84, getitem_49);  convert_element_type_84 = getitem_49 = None
        add_111: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        mul_96: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_97: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_96, arg174_1);  mul_96 = arg174_1 = None
        add_112: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_97, arg175_1);  mul_97 = arg175_1 = None
        convert_element_type_85: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16);  add_112 = None
        view_264: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_85, _shape_param_313);  _shape_param_313 = None
        permute_132: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        mm_36: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_264, permute_132);  view_264 = permute_132 = None
        view_265: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_36, _shape_param_314);  mm_36 = _shape_param_314 = None
        view_266: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_265, _shape_param_315);  view_265 = _shape_param_315 = None
        permute_133: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None
        convert_element_type_86: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_133, torch.float32);  permute_133 = None
        expand_49: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_86, _shape_param_316);  convert_element_type_86 = _shape_param_316 = None
        clone_48: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_267: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_48, _shape_param_317);  clone_48 = _shape_param_317 = None
        view_268: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_85, _shape_param_318);  _shape_param_318 = None
        permute_134: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        mm_37: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_268, permute_134);  view_268 = permute_134 = None
        view_269: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_37, _shape_param_319);  mm_37 = _shape_param_319 = None
        view_270: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_269, _shape_param_320);  view_269 = _shape_param_320 = None
        permute_135: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
        convert_element_type_87: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_135, torch.float32);  permute_135 = None
        permute_136: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_87, [0, 1, 3, 2]);  convert_element_type_87 = None
        expand_50: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_136, _shape_param_321);  permute_136 = _shape_param_321 = None
        clone_49: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_271: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_49, _shape_param_322);  clone_49 = _shape_param_322 = None
        bmm_24: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_267, view_271);  view_267 = view_271 = None
        view_272: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_24, _shape_param_323);  bmm_24 = _shape_param_323 = None
        full_15: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_26, view_272, full_15);  slice_26 = view_272 = full_15 = None
        add_113: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_13, where_1);  where_13 = None
        amax_12: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_113, [-1], True)
        sub_37: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_113, amax_12);  add_113 = amax_12 = None
        exp_12: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_13: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_88: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None
        expand_51: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_88, _shape_param_324);  convert_element_type_88 = _shape_param_324 = None
        view_273: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_51, _shape_param_325);  expand_51 = _shape_param_325 = None
        view_274: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_85, _shape_param_326);  convert_element_type_85 = _shape_param_326 = None
        permute_137: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        mm_38: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_274, permute_137);  view_274 = permute_137 = None
        view_275: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_38, _shape_param_327);  mm_38 = _shape_param_327 = None
        view_276: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_275, _shape_param_328);  view_275 = _shape_param_328 = None
        permute_138: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_276, [0, 2, 1, 3]);  view_276 = None
        expand_52: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_138, _shape_param_329);  permute_138 = _shape_param_329 = None
        clone_50: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_277: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_50, _shape_param_330);  clone_50 = _shape_param_330 = None
        bmm_25: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_273, view_277);  view_273 = view_277 = None
        view_278: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_25, _shape_param_331);  bmm_25 = _shape_param_331 = None
        permute_139: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
        clone_51: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        view_279: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_51, _shape_param_332);  clone_51 = _shape_param_332 = None
        view_280: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_279, _shape_param_333);  view_279 = _shape_param_333 = None
        permute_140: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_36: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg180_1, view_280, permute_140);  arg180_1 = view_280 = permute_140 = None
        view_281: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_36, _shape_param_334);  addmm_36 = _shape_param_334 = None
        add_114: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_281, add_110);  view_281 = add_110 = None
        convert_element_type_89: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_114, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_89, [2], correction = 0, keepdim = True)
        getitem_50: "f32[32, 128, 1]" = var_mean_25[0]
        getitem_51: "f32[32, 128, 1]" = var_mean_25[1];  var_mean_25 = None
        sub_38: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_89, getitem_51);  convert_element_type_89 = getitem_51 = None
        add_115: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_25: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_98: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
        mul_99: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_98, arg181_1);  mul_98 = arg181_1 = None
        add_116: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_99, arg182_1);  mul_99 = arg182_1 = None
        convert_element_type_90: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_116, torch.bfloat16);  add_116 = None
        view_282: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_90, _shape_param_335);  convert_element_type_90 = _shape_param_335 = None
        permute_141: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_37: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg184_1, view_282, permute_141);  arg184_1 = view_282 = permute_141 = None
        view_283: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_37, _shape_param_336);  addmm_37 = _shape_param_336 = None
        mul_100: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_283, 0.5)
        pow_13: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_283, 3.0)
        mul_101: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_117: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_283, mul_101);  view_283 = mul_101 = None
        mul_102: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_117, 0.7978845608028654);  add_117 = None
        tanh_12: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_102);  mul_102 = None
        add_118: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
        mul_103: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_100, add_118);  mul_100 = add_118 = None
        view_284: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_103, _shape_param_337);  mul_103 = _shape_param_337 = None
        permute_142: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_38: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg186_1, view_284, permute_142);  arg186_1 = view_284 = permute_142 = None
        view_285: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_38, _shape_param_338);  addmm_38 = _shape_param_338 = None
        add_119: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_114, view_285);  add_114 = view_285 = None
        convert_element_type_91: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_119, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_91, [2], correction = 0, keepdim = True)
        getitem_52: "f32[32, 128, 1]" = var_mean_26[0]
        getitem_53: "f32[32, 128, 1]" = var_mean_26[1];  var_mean_26 = None
        slice_27: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg187_1, 2, 0, 128);  arg187_1 = None
        slice_28: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_27, 3, 0, 128);  slice_27 = None
        sub_39: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_91, getitem_53);  convert_element_type_91 = getitem_53 = None
        add_120: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_26: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        mul_104: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = rsqrt_26 = None
        mul_105: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_104, arg188_1);  mul_104 = arg188_1 = None
        add_121: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_105, arg189_1);  mul_105 = arg189_1 = None
        convert_element_type_92: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_121, torch.bfloat16);  add_121 = None
        view_286: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_92, _shape_param_339);  _shape_param_339 = None
        permute_143: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        mm_39: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_286, permute_143);  view_286 = permute_143 = None
        view_287: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_39, _shape_param_340);  mm_39 = _shape_param_340 = None
        view_288: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_287, _shape_param_341);  view_287 = _shape_param_341 = None
        permute_144: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
        convert_element_type_93: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_144, torch.float32);  permute_144 = None
        expand_53: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_93, _shape_param_342);  convert_element_type_93 = _shape_param_342 = None
        clone_52: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_289: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_52, _shape_param_343);  clone_52 = _shape_param_343 = None
        view_290: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_92, _shape_param_344);  _shape_param_344 = None
        permute_145: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        mm_40: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_290, permute_145);  view_290 = permute_145 = None
        view_291: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_40, _shape_param_345);  mm_40 = _shape_param_345 = None
        view_292: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_291, _shape_param_346);  view_291 = _shape_param_346 = None
        permute_146: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
        convert_element_type_94: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_146, torch.float32);  permute_146 = None
        permute_147: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_94, [0, 1, 3, 2]);  convert_element_type_94 = None
        expand_54: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_147, _shape_param_347);  permute_147 = _shape_param_347 = None
        clone_53: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_293: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_53, _shape_param_348);  clone_53 = _shape_param_348 = None
        bmm_26: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_289, view_293);  view_289 = view_293 = None
        view_294: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_26, _shape_param_349);  bmm_26 = _shape_param_349 = None
        full_16: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_28, view_294, full_16);  slice_28 = view_294 = full_16 = None
        add_122: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_14, where_1);  where_14 = None
        amax_13: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_122, [-1], True)
        sub_40: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_122, amax_13);  add_122 = amax_13 = None
        exp_13: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        sum_14: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_95: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None
        expand_55: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_95, _shape_param_350);  convert_element_type_95 = _shape_param_350 = None
        view_295: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_55, _shape_param_351);  expand_55 = _shape_param_351 = None
        view_296: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_92, _shape_param_352);  convert_element_type_92 = _shape_param_352 = None
        permute_148: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
        mm_41: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_296, permute_148);  view_296 = permute_148 = None
        view_297: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_41, _shape_param_353);  mm_41 = _shape_param_353 = None
        view_298: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_297, _shape_param_354);  view_297 = _shape_param_354 = None
        permute_149: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_298, [0, 2, 1, 3]);  view_298 = None
        expand_56: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_149, _shape_param_355);  permute_149 = _shape_param_355 = None
        clone_54: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_299: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_54, _shape_param_356);  clone_54 = _shape_param_356 = None
        bmm_27: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_295, view_299);  view_295 = view_299 = None
        view_300: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_27, _shape_param_357);  bmm_27 = _shape_param_357 = None
        permute_150: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
        clone_55: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_301: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_55, _shape_param_358);  clone_55 = _shape_param_358 = None
        view_302: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_301, _shape_param_359);  view_301 = _shape_param_359 = None
        permute_151: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_39: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg194_1, view_302, permute_151);  arg194_1 = view_302 = permute_151 = None
        view_303: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_39, _shape_param_360);  addmm_39 = _shape_param_360 = None
        add_123: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_303, add_119);  view_303 = add_119 = None
        convert_element_type_96: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_123, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_96, [2], correction = 0, keepdim = True)
        getitem_54: "f32[32, 128, 1]" = var_mean_27[0]
        getitem_55: "f32[32, 128, 1]" = var_mean_27[1];  var_mean_27 = None
        sub_41: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_96, getitem_55);  convert_element_type_96 = getitem_55 = None
        add_124: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_27: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        mul_106: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = rsqrt_27 = None
        mul_107: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_106, arg195_1);  mul_106 = arg195_1 = None
        add_125: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_107, arg196_1);  mul_107 = arg196_1 = None
        convert_element_type_97: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_125, torch.bfloat16);  add_125 = None
        view_304: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_97, _shape_param_361);  convert_element_type_97 = _shape_param_361 = None
        permute_152: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_40: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg198_1, view_304, permute_152);  arg198_1 = view_304 = permute_152 = None
        view_305: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_40, _shape_param_362);  addmm_40 = _shape_param_362 = None
        mul_108: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_305, 0.5)
        pow_14: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_305, 3.0)
        mul_109: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_126: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_305, mul_109);  view_305 = mul_109 = None
        mul_110: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_126, 0.7978845608028654);  add_126 = None
        tanh_13: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_110);  mul_110 = None
        add_127: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
        mul_111: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_108, add_127);  mul_108 = add_127 = None
        view_306: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_111, _shape_param_363);  mul_111 = _shape_param_363 = None
        permute_153: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_41: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg200_1, view_306, permute_153);  arg200_1 = view_306 = permute_153 = None
        view_307: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_41, _shape_param_364);  addmm_41 = _shape_param_364 = None
        add_128: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_123, view_307);  add_123 = view_307 = None
        convert_element_type_98: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_128, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_98, [2], correction = 0, keepdim = True)
        getitem_56: "f32[32, 128, 1]" = var_mean_28[0]
        getitem_57: "f32[32, 128, 1]" = var_mean_28[1];  var_mean_28 = None
        slice_29: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg201_1, 2, 0, 128);  arg201_1 = None
        slice_30: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_29, 3, 0, 128);  slice_29 = None
        sub_42: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_98, getitem_57);  convert_element_type_98 = getitem_57 = None
        add_129: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_28: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_112: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = rsqrt_28 = None
        mul_113: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_112, arg202_1);  mul_112 = arg202_1 = None
        add_130: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_113, arg203_1);  mul_113 = arg203_1 = None
        convert_element_type_99: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_130, torch.bfloat16);  add_130 = None
        view_308: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_99, _shape_param_365);  _shape_param_365 = None
        permute_154: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        mm_42: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_308, permute_154);  view_308 = permute_154 = None
        view_309: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_42, _shape_param_366);  mm_42 = _shape_param_366 = None
        view_310: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_309, _shape_param_367);  view_309 = _shape_param_367 = None
        permute_155: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
        convert_element_type_100: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_155, torch.float32);  permute_155 = None
        expand_57: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_100, _shape_param_368);  convert_element_type_100 = _shape_param_368 = None
        clone_56: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_311: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_56, _shape_param_369);  clone_56 = _shape_param_369 = None
        view_312: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_99, _shape_param_370);  _shape_param_370 = None
        permute_156: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        mm_43: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_312, permute_156);  view_312 = permute_156 = None
        view_313: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_43, _shape_param_371);  mm_43 = _shape_param_371 = None
        view_314: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_313, _shape_param_372);  view_313 = _shape_param_372 = None
        permute_157: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
        convert_element_type_101: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_157, torch.float32);  permute_157 = None
        permute_158: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_101, [0, 1, 3, 2]);  convert_element_type_101 = None
        expand_58: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_158, _shape_param_373);  permute_158 = _shape_param_373 = None
        clone_57: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_315: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_57, _shape_param_374);  clone_57 = _shape_param_374 = None
        bmm_28: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_311, view_315);  view_311 = view_315 = None
        view_316: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_28, _shape_param_375);  bmm_28 = _shape_param_375 = None
        full_17: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_30, view_316, full_17);  slice_30 = view_316 = full_17 = None
        add_131: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_15, where_1);  where_15 = None
        amax_14: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_131, [-1], True)
        sub_43: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_131, amax_14);  add_131 = amax_14 = None
        exp_14: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_15: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_102: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None
        expand_59: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_102, _shape_param_376);  convert_element_type_102 = _shape_param_376 = None
        view_317: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_59, _shape_param_377);  expand_59 = _shape_param_377 = None
        view_318: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_99, _shape_param_378);  convert_element_type_99 = _shape_param_378 = None
        permute_159: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        mm_44: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_318, permute_159);  view_318 = permute_159 = None
        view_319: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_44, _shape_param_379);  mm_44 = _shape_param_379 = None
        view_320: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_319, _shape_param_380);  view_319 = _shape_param_380 = None
        permute_160: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_320, [0, 2, 1, 3]);  view_320 = None
        expand_60: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_160, _shape_param_381);  permute_160 = _shape_param_381 = None
        clone_58: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_321: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_58, _shape_param_382);  clone_58 = _shape_param_382 = None
        bmm_29: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_317, view_321);  view_317 = view_321 = None
        view_322: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_29, _shape_param_383);  bmm_29 = _shape_param_383 = None
        permute_161: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None
        clone_59: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_323: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_59, _shape_param_384);  clone_59 = _shape_param_384 = None
        view_324: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_323, _shape_param_385);  view_323 = _shape_param_385 = None
        permute_162: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        addmm_42: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg208_1, view_324, permute_162);  arg208_1 = view_324 = permute_162 = None
        view_325: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_42, _shape_param_386);  addmm_42 = _shape_param_386 = None
        add_132: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_325, add_128);  view_325 = add_128 = None
        convert_element_type_103: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_132, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_103, [2], correction = 0, keepdim = True)
        getitem_58: "f32[32, 128, 1]" = var_mean_29[0]
        getitem_59: "f32[32, 128, 1]" = var_mean_29[1];  var_mean_29 = None
        sub_44: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_103, getitem_59);  convert_element_type_103 = getitem_59 = None
        add_133: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_29: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        mul_114: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
        mul_115: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_114, arg209_1);  mul_114 = arg209_1 = None
        add_134: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_115, arg210_1);  mul_115 = arg210_1 = None
        convert_element_type_104: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_134, torch.bfloat16);  add_134 = None
        view_326: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_104, _shape_param_387);  convert_element_type_104 = _shape_param_387 = None
        permute_163: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_43: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg212_1, view_326, permute_163);  arg212_1 = view_326 = permute_163 = None
        view_327: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_43, _shape_param_388);  addmm_43 = _shape_param_388 = None
        mul_116: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_327, 0.5)
        pow_15: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_327, 3.0)
        mul_117: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_135: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_327, mul_117);  view_327 = mul_117 = None
        mul_118: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_135, 0.7978845608028654);  add_135 = None
        tanh_14: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_136: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
        mul_119: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_116, add_136);  mul_116 = add_136 = None
        view_328: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_119, _shape_param_389);  mul_119 = _shape_param_389 = None
        permute_164: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        addmm_44: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg214_1, view_328, permute_164);  arg214_1 = view_328 = permute_164 = None
        view_329: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_44, _shape_param_390);  addmm_44 = _shape_param_390 = None
        add_137: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_132, view_329);  add_132 = view_329 = None
        convert_element_type_105: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_137, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_105, [2], correction = 0, keepdim = True)
        getitem_60: "f32[32, 128, 1]" = var_mean_30[0]
        getitem_61: "f32[32, 128, 1]" = var_mean_30[1];  var_mean_30 = None
        slice_31: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg215_1, 2, 0, 128);  arg215_1 = None
        slice_32: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_31, 3, 0, 128);  slice_31 = None
        sub_45: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_105, getitem_61);  convert_element_type_105 = getitem_61 = None
        add_138: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_30: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        mul_120: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = rsqrt_30 = None
        mul_121: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_120, arg216_1);  mul_120 = arg216_1 = None
        add_139: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_121, arg217_1);  mul_121 = arg217_1 = None
        convert_element_type_106: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_139, torch.bfloat16);  add_139 = None
        view_330: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_106, _shape_param_391);  _shape_param_391 = None
        permute_165: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        mm_45: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_330, permute_165);  view_330 = permute_165 = None
        view_331: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_45, _shape_param_392);  mm_45 = _shape_param_392 = None
        view_332: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_331, _shape_param_393);  view_331 = _shape_param_393 = None
        permute_166: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
        convert_element_type_107: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_166, torch.float32);  permute_166 = None
        expand_61: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_107, _shape_param_394);  convert_element_type_107 = _shape_param_394 = None
        clone_60: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_333: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_60, _shape_param_395);  clone_60 = _shape_param_395 = None
        view_334: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_106, _shape_param_396);  _shape_param_396 = None
        permute_167: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        mm_46: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_334, permute_167);  view_334 = permute_167 = None
        view_335: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_46, _shape_param_397);  mm_46 = _shape_param_397 = None
        view_336: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_335, _shape_param_398);  view_335 = _shape_param_398 = None
        permute_168: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None
        convert_element_type_108: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_168, torch.float32);  permute_168 = None
        permute_169: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_108, [0, 1, 3, 2]);  convert_element_type_108 = None
        expand_62: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_169, _shape_param_399);  permute_169 = _shape_param_399 = None
        clone_61: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_337: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_61, _shape_param_400);  clone_61 = _shape_param_400 = None
        bmm_30: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_333, view_337);  view_333 = view_337 = None
        view_338: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_30, _shape_param_401);  bmm_30 = _shape_param_401 = None
        full_18: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_32, view_338, full_18);  slice_32 = view_338 = full_18 = None
        add_140: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_16, where_1);  where_16 = None
        amax_15: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_140, [-1], True)
        sub_46: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_140, amax_15);  add_140 = amax_15 = None
        exp_15: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_16: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_109: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None
        expand_63: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_109, _shape_param_402);  convert_element_type_109 = _shape_param_402 = None
        view_339: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_63, _shape_param_403);  expand_63 = _shape_param_403 = None
        view_340: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_106, _shape_param_404);  convert_element_type_106 = _shape_param_404 = None
        permute_170: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        mm_47: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_340, permute_170);  view_340 = permute_170 = None
        view_341: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_47, _shape_param_405);  mm_47 = _shape_param_405 = None
        view_342: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_341, _shape_param_406);  view_341 = _shape_param_406 = None
        permute_171: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None
        expand_64: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_171, _shape_param_407);  permute_171 = _shape_param_407 = None
        clone_62: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_343: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_62, _shape_param_408);  clone_62 = _shape_param_408 = None
        bmm_31: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_339, view_343);  view_339 = view_343 = None
        view_344: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_31, _shape_param_409);  bmm_31 = _shape_param_409 = None
        permute_172: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
        clone_63: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_345: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_63, _shape_param_410);  clone_63 = _shape_param_410 = None
        view_346: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_345, _shape_param_411);  view_345 = _shape_param_411 = None
        permute_173: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        addmm_45: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg222_1, view_346, permute_173);  arg222_1 = view_346 = permute_173 = None
        view_347: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_45, _shape_param_412);  addmm_45 = _shape_param_412 = None
        add_141: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_347, add_137);  view_347 = add_137 = None
        convert_element_type_110: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_141, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_110, [2], correction = 0, keepdim = True)
        getitem_62: "f32[32, 128, 1]" = var_mean_31[0]
        getitem_63: "f32[32, 128, 1]" = var_mean_31[1];  var_mean_31 = None
        sub_47: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_110, getitem_63);  convert_element_type_110 = getitem_63 = None
        add_142: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_31: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        mul_122: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
        mul_123: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_122, arg223_1);  mul_122 = arg223_1 = None
        add_143: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_123, arg224_1);  mul_123 = arg224_1 = None
        convert_element_type_111: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16);  add_143 = None
        view_348: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_111, _shape_param_413);  convert_element_type_111 = _shape_param_413 = None
        permute_174: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_46: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg226_1, view_348, permute_174);  arg226_1 = view_348 = permute_174 = None
        view_349: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_46, _shape_param_414);  addmm_46 = _shape_param_414 = None
        mul_124: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_349, 0.5)
        pow_16: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_349, 3.0)
        mul_125: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_144: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_349, mul_125);  view_349 = mul_125 = None
        mul_126: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_144, 0.7978845608028654);  add_144 = None
        tanh_15: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_126);  mul_126 = None
        add_145: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
        mul_127: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_124, add_145);  mul_124 = add_145 = None
        view_350: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_127, _shape_param_415);  mul_127 = _shape_param_415 = None
        permute_175: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        addmm_47: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg228_1, view_350, permute_175);  arg228_1 = view_350 = permute_175 = None
        view_351: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_47, _shape_param_416);  addmm_47 = _shape_param_416 = None
        add_146: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_141, view_351);  add_141 = view_351 = None
        convert_element_type_112: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_146, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_112, [2], correction = 0, keepdim = True)
        getitem_64: "f32[32, 128, 1]" = var_mean_32[0]
        getitem_65: "f32[32, 128, 1]" = var_mean_32[1];  var_mean_32 = None
        slice_33: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg229_1, 2, 0, 128);  arg229_1 = None
        slice_34: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_33, 3, 0, 128);  slice_33 = None
        sub_48: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_112, getitem_65);  convert_element_type_112 = getitem_65 = None
        add_147: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_32: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
        mul_128: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = rsqrt_32 = None
        mul_129: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_128, arg230_1);  mul_128 = arg230_1 = None
        add_148: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_129, arg231_1);  mul_129 = arg231_1 = None
        convert_element_type_113: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_148, torch.bfloat16);  add_148 = None
        view_352: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_113, _shape_param_417);  _shape_param_417 = None
        permute_176: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        mm_48: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_352, permute_176);  view_352 = permute_176 = None
        view_353: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_48, _shape_param_418);  mm_48 = _shape_param_418 = None
        view_354: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_353, _shape_param_419);  view_353 = _shape_param_419 = None
        permute_177: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
        convert_element_type_114: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_177, torch.float32);  permute_177 = None
        expand_65: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_114, _shape_param_420);  convert_element_type_114 = _shape_param_420 = None
        clone_64: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_355: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_64, _shape_param_421);  clone_64 = _shape_param_421 = None
        view_356: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_113, _shape_param_422);  _shape_param_422 = None
        permute_178: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        mm_49: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_356, permute_178);  view_356 = permute_178 = None
        view_357: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_49, _shape_param_423);  mm_49 = _shape_param_423 = None
        view_358: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_357, _shape_param_424);  view_357 = _shape_param_424 = None
        permute_179: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
        convert_element_type_115: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_179, torch.float32);  permute_179 = None
        permute_180: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_115, [0, 1, 3, 2]);  convert_element_type_115 = None
        expand_66: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_180, _shape_param_425);  permute_180 = _shape_param_425 = None
        clone_65: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_359: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_65, _shape_param_426);  clone_65 = _shape_param_426 = None
        bmm_32: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_355, view_359);  view_355 = view_359 = None
        view_360: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_32, _shape_param_427);  bmm_32 = _shape_param_427 = None
        full_19: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_17: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_34, view_360, full_19);  slice_34 = view_360 = full_19 = None
        add_149: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_17, where_1);  where_17 = None
        amax_16: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_149, [-1], True)
        sub_49: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_149, amax_16);  add_149 = amax_16 = None
        exp_16: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_17: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_116: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None
        expand_67: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_116, _shape_param_428);  convert_element_type_116 = _shape_param_428 = None
        view_361: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_67, _shape_param_429);  expand_67 = _shape_param_429 = None
        view_362: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_113, _shape_param_430);  convert_element_type_113 = _shape_param_430 = None
        permute_181: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
        mm_50: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_362, permute_181);  view_362 = permute_181 = None
        view_363: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_50, _shape_param_431);  mm_50 = _shape_param_431 = None
        view_364: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_363, _shape_param_432);  view_363 = _shape_param_432 = None
        permute_182: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None
        expand_68: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_182, _shape_param_433);  permute_182 = _shape_param_433 = None
        clone_66: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_365: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_66, _shape_param_434);  clone_66 = _shape_param_434 = None
        bmm_33: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_361, view_365);  view_361 = view_365 = None
        view_366: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_33, _shape_param_435);  bmm_33 = _shape_param_435 = None
        permute_183: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
        clone_67: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        view_367: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_67, _shape_param_436);  clone_67 = _shape_param_436 = None
        view_368: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_367, _shape_param_437);  view_367 = _shape_param_437 = None
        permute_184: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_48: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg236_1, view_368, permute_184);  arg236_1 = view_368 = permute_184 = None
        view_369: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_48, _shape_param_438);  addmm_48 = _shape_param_438 = None
        add_150: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_369, add_146);  view_369 = add_146 = None
        convert_element_type_117: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_150, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_117, [2], correction = 0, keepdim = True)
        getitem_66: "f32[32, 128, 1]" = var_mean_33[0]
        getitem_67: "f32[32, 128, 1]" = var_mean_33[1];  var_mean_33 = None
        sub_50: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_117, getitem_67);  convert_element_type_117 = getitem_67 = None
        add_151: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_33: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        mul_130: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = rsqrt_33 = None
        mul_131: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_130, arg237_1);  mul_130 = arg237_1 = None
        add_152: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_131, arg238_1);  mul_131 = arg238_1 = None
        convert_element_type_118: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_152, torch.bfloat16);  add_152 = None
        view_370: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_118, _shape_param_439);  convert_element_type_118 = _shape_param_439 = None
        permute_185: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_49: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg240_1, view_370, permute_185);  arg240_1 = view_370 = permute_185 = None
        view_371: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_49, _shape_param_440);  addmm_49 = _shape_param_440 = None
        mul_132: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_371, 0.5)
        pow_17: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_371, 3.0)
        mul_133: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_153: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_371, mul_133);  view_371 = mul_133 = None
        mul_134: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_153, 0.7978845608028654);  add_153 = None
        tanh_16: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_134);  mul_134 = None
        add_154: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_16, 1.0);  tanh_16 = None
        mul_135: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_132, add_154);  mul_132 = add_154 = None
        view_372: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_135, _shape_param_441);  mul_135 = _shape_param_441 = None
        permute_186: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_50: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg242_1, view_372, permute_186);  arg242_1 = view_372 = permute_186 = None
        view_373: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_50, _shape_param_442);  addmm_50 = _shape_param_442 = None
        add_155: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_150, view_373);  add_150 = view_373 = None
        convert_element_type_119: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_155, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_119, [2], correction = 0, keepdim = True)
        getitem_68: "f32[32, 128, 1]" = var_mean_34[0]
        getitem_69: "f32[32, 128, 1]" = var_mean_34[1];  var_mean_34 = None
        slice_35: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg243_1, 2, 0, 128);  arg243_1 = None
        slice_36: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_35, 3, 0, 128);  slice_35 = None
        sub_51: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_119, getitem_69);  convert_element_type_119 = getitem_69 = None
        add_156: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_34: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        mul_136: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = rsqrt_34 = None
        mul_137: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_136, arg244_1);  mul_136 = arg244_1 = None
        add_157: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_137, arg245_1);  mul_137 = arg245_1 = None
        convert_element_type_120: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_157, torch.bfloat16);  add_157 = None
        view_374: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_120, _shape_param_443);  _shape_param_443 = None
        permute_187: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
        mm_51: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_374, permute_187);  view_374 = permute_187 = None
        view_375: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_51, _shape_param_444);  mm_51 = _shape_param_444 = None
        view_376: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_375, _shape_param_445);  view_375 = _shape_param_445 = None
        permute_188: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        convert_element_type_121: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_188, torch.float32);  permute_188 = None
        expand_69: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_121, _shape_param_446);  convert_element_type_121 = _shape_param_446 = None
        clone_68: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_377: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_68, _shape_param_447);  clone_68 = _shape_param_447 = None
        view_378: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_120, _shape_param_448);  _shape_param_448 = None
        permute_189: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        mm_52: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_378, permute_189);  view_378 = permute_189 = None
        view_379: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_52, _shape_param_449);  mm_52 = _shape_param_449 = None
        view_380: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_379, _shape_param_450);  view_379 = _shape_param_450 = None
        permute_190: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_380, [0, 2, 1, 3]);  view_380 = None
        convert_element_type_122: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_190, torch.float32);  permute_190 = None
        permute_191: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_122, [0, 1, 3, 2]);  convert_element_type_122 = None
        expand_70: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_191, _shape_param_451);  permute_191 = _shape_param_451 = None
        clone_69: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_381: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_69, _shape_param_452);  clone_69 = _shape_param_452 = None
        bmm_34: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_377, view_381);  view_377 = view_381 = None
        view_382: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_34, _shape_param_453);  bmm_34 = _shape_param_453 = None
        full_20: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_36, view_382, full_20);  slice_36 = view_382 = full_20 = None
        add_158: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_18, where_1);  where_18 = None
        amax_17: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_158, [-1], True)
        sub_52: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_158, amax_17);  add_158 = amax_17 = None
        exp_17: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_18: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_17: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_123: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None
        expand_71: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_123, _shape_param_454);  convert_element_type_123 = _shape_param_454 = None
        view_383: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_71, _shape_param_455);  expand_71 = _shape_param_455 = None
        view_384: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_120, _shape_param_456);  convert_element_type_120 = _shape_param_456 = None
        permute_192: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
        mm_53: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_384, permute_192);  view_384 = permute_192 = None
        view_385: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_53, _shape_param_457);  mm_53 = _shape_param_457 = None
        view_386: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_385, _shape_param_458);  view_385 = _shape_param_458 = None
        permute_193: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None
        expand_72: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_193, _shape_param_459);  permute_193 = _shape_param_459 = None
        clone_70: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_387: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_70, _shape_param_460);  clone_70 = _shape_param_460 = None
        bmm_35: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_383, view_387);  view_383 = view_387 = None
        view_388: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_35, _shape_param_461);  bmm_35 = _shape_param_461 = None
        permute_194: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
        clone_71: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_389: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_71, _shape_param_462);  clone_71 = _shape_param_462 = None
        view_390: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_389, _shape_param_463);  view_389 = _shape_param_463 = None
        permute_195: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_51: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg250_1, view_390, permute_195);  arg250_1 = view_390 = permute_195 = None
        view_391: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_51, _shape_param_464);  addmm_51 = _shape_param_464 = None
        add_159: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_391, add_155);  view_391 = add_155 = None
        convert_element_type_124: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_159, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_124, [2], correction = 0, keepdim = True)
        getitem_70: "f32[32, 128, 1]" = var_mean_35[0]
        getitem_71: "f32[32, 128, 1]" = var_mean_35[1];  var_mean_35 = None
        sub_53: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_124, getitem_71);  convert_element_type_124 = getitem_71 = None
        add_160: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
        rsqrt_35: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        mul_138: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = rsqrt_35 = None
        mul_139: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_138, arg251_1);  mul_138 = arg251_1 = None
        add_161: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_139, arg252_1);  mul_139 = arg252_1 = None
        convert_element_type_125: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_161, torch.bfloat16);  add_161 = None
        view_392: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_125, _shape_param_465);  convert_element_type_125 = _shape_param_465 = None
        permute_196: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_52: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg254_1, view_392, permute_196);  arg254_1 = view_392 = permute_196 = None
        view_393: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_52, _shape_param_466);  addmm_52 = _shape_param_466 = None
        mul_140: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_393, 0.5)
        pow_18: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_393, 3.0)
        mul_141: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_162: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_393, mul_141);  view_393 = mul_141 = None
        mul_142: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_162, 0.7978845608028654);  add_162 = None
        tanh_17: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_142);  mul_142 = None
        add_163: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_17, 1.0);  tanh_17 = None
        mul_143: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_140, add_163);  mul_140 = add_163 = None
        view_394: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_143, _shape_param_467);  mul_143 = _shape_param_467 = None
        permute_197: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_53: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg256_1, view_394, permute_197);  arg256_1 = view_394 = permute_197 = None
        view_395: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_53, _shape_param_468);  addmm_53 = _shape_param_468 = None
        add_164: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_159, view_395);  add_159 = view_395 = None
        convert_element_type_126: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_164, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_126, [2], correction = 0, keepdim = True)
        getitem_72: "f32[32, 128, 1]" = var_mean_36[0]
        getitem_73: "f32[32, 128, 1]" = var_mean_36[1];  var_mean_36 = None
        slice_37: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg257_1, 2, 0, 128);  arg257_1 = None
        slice_38: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_37, 3, 0, 128);  slice_37 = None
        sub_54: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_126, getitem_73);  convert_element_type_126 = getitem_73 = None
        add_165: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_36: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        mul_144: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = rsqrt_36 = None
        mul_145: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_144, arg258_1);  mul_144 = arg258_1 = None
        add_166: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_145, arg259_1);  mul_145 = arg259_1 = None
        convert_element_type_127: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_166, torch.bfloat16);  add_166 = None
        view_396: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_127, _shape_param_469);  _shape_param_469 = None
        permute_198: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
        mm_54: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_396, permute_198);  view_396 = permute_198 = None
        view_397: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_54, _shape_param_470);  mm_54 = _shape_param_470 = None
        view_398: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_397, _shape_param_471);  view_397 = _shape_param_471 = None
        permute_199: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
        convert_element_type_128: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_199, torch.float32);  permute_199 = None
        expand_73: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_128, _shape_param_472);  convert_element_type_128 = _shape_param_472 = None
        clone_72: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_399: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_72, _shape_param_473);  clone_72 = _shape_param_473 = None
        view_400: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_127, _shape_param_474);  _shape_param_474 = None
        permute_200: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        mm_55: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_400, permute_200);  view_400 = permute_200 = None
        view_401: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_55, _shape_param_475);  mm_55 = _shape_param_475 = None
        view_402: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_401, _shape_param_476);  view_401 = _shape_param_476 = None
        permute_201: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        convert_element_type_129: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_201, torch.float32);  permute_201 = None
        permute_202: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_129, [0, 1, 3, 2]);  convert_element_type_129 = None
        expand_74: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_202, _shape_param_477);  permute_202 = _shape_param_477 = None
        clone_73: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_403: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_73, _shape_param_478);  clone_73 = _shape_param_478 = None
        bmm_36: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_399, view_403);  view_399 = view_403 = None
        view_404: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_36, _shape_param_479);  bmm_36 = _shape_param_479 = None
        full_21: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_38, view_404, full_21);  slice_38 = view_404 = full_21 = None
        add_167: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_19, where_1);  where_19 = None
        amax_18: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_167, [-1], True)
        sub_55: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_167, amax_18);  add_167 = amax_18 = None
        exp_18: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_19: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_130: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None
        expand_75: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_130, _shape_param_480);  convert_element_type_130 = _shape_param_480 = None
        view_405: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_75, _shape_param_481);  expand_75 = _shape_param_481 = None
        view_406: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_127, _shape_param_482);  convert_element_type_127 = _shape_param_482 = None
        permute_203: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        mm_56: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_406, permute_203);  view_406 = permute_203 = None
        view_407: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_56, _shape_param_483);  mm_56 = _shape_param_483 = None
        view_408: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_407, _shape_param_484);  view_407 = _shape_param_484 = None
        permute_204: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_408, [0, 2, 1, 3]);  view_408 = None
        expand_76: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_204, _shape_param_485);  permute_204 = _shape_param_485 = None
        clone_74: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_409: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_74, _shape_param_486);  clone_74 = _shape_param_486 = None
        bmm_37: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_405, view_409);  view_405 = view_409 = None
        view_410: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_37, _shape_param_487);  bmm_37 = _shape_param_487 = None
        permute_205: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
        clone_75: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_411: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_75, _shape_param_488);  clone_75 = _shape_param_488 = None
        view_412: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_411, _shape_param_489);  view_411 = _shape_param_489 = None
        permute_206: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_54: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg264_1, view_412, permute_206);  arg264_1 = view_412 = permute_206 = None
        view_413: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_54, _shape_param_490);  addmm_54 = _shape_param_490 = None
        add_168: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_413, add_164);  view_413 = add_164 = None
        convert_element_type_131: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_168, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_131, [2], correction = 0, keepdim = True)
        getitem_74: "f32[32, 128, 1]" = var_mean_37[0]
        getitem_75: "f32[32, 128, 1]" = var_mean_37[1];  var_mean_37 = None
        sub_56: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_131, getitem_75);  convert_element_type_131 = getitem_75 = None
        add_169: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_37: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
        mul_146: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = rsqrt_37 = None
        mul_147: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_146, arg265_1);  mul_146 = arg265_1 = None
        add_170: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_147, arg266_1);  mul_147 = arg266_1 = None
        convert_element_type_132: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_170, torch.bfloat16);  add_170 = None
        view_414: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_132, _shape_param_491);  convert_element_type_132 = _shape_param_491 = None
        permute_207: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        addmm_55: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg268_1, view_414, permute_207);  arg268_1 = view_414 = permute_207 = None
        view_415: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_55, _shape_param_492);  addmm_55 = _shape_param_492 = None
        mul_148: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_415, 0.5)
        pow_19: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_415, 3.0)
        mul_149: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_171: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_415, mul_149);  view_415 = mul_149 = None
        mul_150: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_171, 0.7978845608028654);  add_171 = None
        tanh_18: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_150);  mul_150 = None
        add_172: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_18, 1.0);  tanh_18 = None
        mul_151: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_148, add_172);  mul_148 = add_172 = None
        view_416: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_151, _shape_param_493);  mul_151 = _shape_param_493 = None
        permute_208: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        addmm_56: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg270_1, view_416, permute_208);  arg270_1 = view_416 = permute_208 = None
        view_417: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_56, _shape_param_494);  addmm_56 = _shape_param_494 = None
        add_173: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_168, view_417);  add_168 = view_417 = None
        convert_element_type_133: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_173, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_133, [2], correction = 0, keepdim = True)
        getitem_76: "f32[32, 128, 1]" = var_mean_38[0]
        getitem_77: "f32[32, 128, 1]" = var_mean_38[1];  var_mean_38 = None
        slice_39: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg271_1, 2, 0, 128);  arg271_1 = None
        slice_40: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_39, 3, 0, 128);  slice_39 = None
        sub_57: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_133, getitem_77);  convert_element_type_133 = getitem_77 = None
        add_174: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_38: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        mul_152: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = rsqrt_38 = None
        mul_153: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_152, arg272_1);  mul_152 = arg272_1 = None
        add_175: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_153, arg273_1);  mul_153 = arg273_1 = None
        convert_element_type_134: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_175, torch.bfloat16);  add_175 = None
        view_418: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_134, _shape_param_495);  _shape_param_495 = None
        permute_209: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
        mm_57: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_418, permute_209);  view_418 = permute_209 = None
        view_419: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_57, _shape_param_496);  mm_57 = _shape_param_496 = None
        view_420: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_419, _shape_param_497);  view_419 = _shape_param_497 = None
        permute_210: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
        convert_element_type_135: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_210, torch.float32);  permute_210 = None
        expand_77: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_135, _shape_param_498);  convert_element_type_135 = _shape_param_498 = None
        clone_76: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_421: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_76, _shape_param_499);  clone_76 = _shape_param_499 = None
        view_422: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_134, _shape_param_500);  _shape_param_500 = None
        permute_211: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        mm_58: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_422, permute_211);  view_422 = permute_211 = None
        view_423: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_58, _shape_param_501);  mm_58 = _shape_param_501 = None
        view_424: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_423, _shape_param_502);  view_423 = _shape_param_502 = None
        permute_212: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
        convert_element_type_136: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_212, torch.float32);  permute_212 = None
        permute_213: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_136, [0, 1, 3, 2]);  convert_element_type_136 = None
        expand_78: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_213, _shape_param_503);  permute_213 = _shape_param_503 = None
        clone_77: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
        view_425: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_77, _shape_param_504);  clone_77 = _shape_param_504 = None
        bmm_38: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_421, view_425);  view_421 = view_425 = None
        view_426: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_38, _shape_param_505);  bmm_38 = _shape_param_505 = None
        full_22: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_40, view_426, full_22);  slice_40 = view_426 = full_22 = None
        add_176: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_20, where_1);  where_20 = None
        amax_19: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_176, [-1], True)
        sub_58: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_176, amax_19);  add_176 = amax_19 = None
        exp_19: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        sum_20: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_19: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        convert_element_type_137: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None
        expand_79: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_137, _shape_param_506);  convert_element_type_137 = _shape_param_506 = None
        view_427: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_79, _shape_param_507);  expand_79 = _shape_param_507 = None
        view_428: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_134, _shape_param_508);  convert_element_type_134 = _shape_param_508 = None
        permute_214: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
        mm_59: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_428, permute_214);  view_428 = permute_214 = None
        view_429: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_59, _shape_param_509);  mm_59 = _shape_param_509 = None
        view_430: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_429, _shape_param_510);  view_429 = _shape_param_510 = None
        permute_215: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
        expand_80: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_215, _shape_param_511);  permute_215 = _shape_param_511 = None
        clone_78: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_431: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_78, _shape_param_512);  clone_78 = _shape_param_512 = None
        bmm_39: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_427, view_431);  view_427 = view_431 = None
        view_432: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_39, _shape_param_513);  bmm_39 = _shape_param_513 = None
        permute_216: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
        clone_79: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
        view_433: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_79, _shape_param_514);  clone_79 = _shape_param_514 = None
        view_434: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_433, _shape_param_515);  view_433 = _shape_param_515 = None
        permute_217: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        addmm_57: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg278_1, view_434, permute_217);  arg278_1 = view_434 = permute_217 = None
        view_435: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_57, _shape_param_516);  addmm_57 = _shape_param_516 = None
        add_177: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_435, add_173);  view_435 = add_173 = None
        convert_element_type_138: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_177, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_138, [2], correction = 0, keepdim = True)
        getitem_78: "f32[32, 128, 1]" = var_mean_39[0]
        getitem_79: "f32[32, 128, 1]" = var_mean_39[1];  var_mean_39 = None
        sub_59: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_138, getitem_79);  convert_element_type_138 = getitem_79 = None
        add_178: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_39: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        mul_154: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = rsqrt_39 = None
        mul_155: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_154, arg279_1);  mul_154 = arg279_1 = None
        add_179: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_155, arg280_1);  mul_155 = arg280_1 = None
        convert_element_type_139: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_179, torch.bfloat16);  add_179 = None
        view_436: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_139, _shape_param_517);  convert_element_type_139 = _shape_param_517 = None
        permute_218: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_58: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg282_1, view_436, permute_218);  arg282_1 = view_436 = permute_218 = None
        view_437: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_58, _shape_param_518);  addmm_58 = _shape_param_518 = None
        mul_156: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_437, 0.5)
        pow_20: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_437, 3.0)
        mul_157: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_180: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_437, mul_157);  view_437 = mul_157 = None
        mul_158: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_180, 0.7978845608028654);  add_180 = None
        tanh_19: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
        add_181: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_19, 1.0);  tanh_19 = None
        mul_159: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_156, add_181);  mul_156 = add_181 = None
        view_438: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_159, _shape_param_519);  mul_159 = _shape_param_519 = None
        permute_219: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        addmm_59: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg284_1, view_438, permute_219);  arg284_1 = view_438 = permute_219 = None
        view_439: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_59, _shape_param_520);  addmm_59 = _shape_param_520 = None
        add_182: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_177, view_439);  add_177 = view_439 = None
        convert_element_type_140: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_182, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_140, [2], correction = 0, keepdim = True)
        getitem_80: "f32[32, 128, 1]" = var_mean_40[0]
        getitem_81: "f32[32, 128, 1]" = var_mean_40[1];  var_mean_40 = None
        slice_41: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg285_1, 2, 0, 128);  arg285_1 = None
        slice_42: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_41, 3, 0, 128);  slice_41 = None
        sub_60: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_140, getitem_81);  convert_element_type_140 = getitem_81 = None
        add_183: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_40: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
        mul_160: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = rsqrt_40 = None
        mul_161: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_160, arg286_1);  mul_160 = arg286_1 = None
        add_184: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_161, arg287_1);  mul_161 = arg287_1 = None
        convert_element_type_141: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_184, torch.bfloat16);  add_184 = None
        view_440: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_141, _shape_param_521);  _shape_param_521 = None
        permute_220: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg288_1, [1, 0]);  arg288_1 = None
        mm_60: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_440, permute_220);  view_440 = permute_220 = None
        view_441: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_60, _shape_param_522);  mm_60 = _shape_param_522 = None
        view_442: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_441, _shape_param_523);  view_441 = _shape_param_523 = None
        permute_221: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
        convert_element_type_142: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_221, torch.float32);  permute_221 = None
        expand_81: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_142, _shape_param_524);  convert_element_type_142 = _shape_param_524 = None
        clone_80: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_443: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_80, _shape_param_525);  clone_80 = _shape_param_525 = None
        view_444: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_141, _shape_param_526);  _shape_param_526 = None
        permute_222: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        mm_61: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_444, permute_222);  view_444 = permute_222 = None
        view_445: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_61, _shape_param_527);  mm_61 = _shape_param_527 = None
        view_446: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_445, _shape_param_528);  view_445 = _shape_param_528 = None
        permute_223: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_446, [0, 2, 1, 3]);  view_446 = None
        convert_element_type_143: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_223, torch.float32);  permute_223 = None
        permute_224: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_143, [0, 1, 3, 2]);  convert_element_type_143 = None
        expand_82: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_224, _shape_param_529);  permute_224 = _shape_param_529 = None
        clone_81: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_447: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_81, _shape_param_530);  clone_81 = _shape_param_530 = None
        bmm_40: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_443, view_447);  view_443 = view_447 = None
        view_448: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_40, _shape_param_531);  bmm_40 = _shape_param_531 = None
        full_23: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_21: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_42, view_448, full_23);  slice_42 = view_448 = full_23 = None
        add_185: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_21, where_1);  where_21 = None
        amax_20: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_185, [-1], True)
        sub_61: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_185, amax_20);  add_185 = amax_20 = None
        exp_20: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_21: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_144: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None
        expand_83: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_144, _shape_param_532);  convert_element_type_144 = _shape_param_532 = None
        view_449: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_83, _shape_param_533);  expand_83 = _shape_param_533 = None
        view_450: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_141, _shape_param_534);  convert_element_type_141 = _shape_param_534 = None
        permute_225: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
        mm_62: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_450, permute_225);  view_450 = permute_225 = None
        view_451: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_62, _shape_param_535);  mm_62 = _shape_param_535 = None
        view_452: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_451, _shape_param_536);  view_451 = _shape_param_536 = None
        permute_226: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None
        expand_84: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_226, _shape_param_537);  permute_226 = _shape_param_537 = None
        clone_82: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_453: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_82, _shape_param_538);  clone_82 = _shape_param_538 = None
        bmm_41: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_449, view_453);  view_449 = view_453 = None
        view_454: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_41, _shape_param_539);  bmm_41 = _shape_param_539 = None
        permute_227: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
        clone_83: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_455: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_83, _shape_param_540);  clone_83 = _shape_param_540 = None
        view_456: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_455, _shape_param_541);  view_455 = _shape_param_541 = None
        permute_228: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
        addmm_60: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg292_1, view_456, permute_228);  arg292_1 = view_456 = permute_228 = None
        view_457: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_60, _shape_param_542);  addmm_60 = _shape_param_542 = None
        add_186: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_457, add_182);  view_457 = add_182 = None
        convert_element_type_145: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_186, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_145, [2], correction = 0, keepdim = True)
        getitem_82: "f32[32, 128, 1]" = var_mean_41[0]
        getitem_83: "f32[32, 128, 1]" = var_mean_41[1];  var_mean_41 = None
        sub_62: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_145, getitem_83);  convert_element_type_145 = getitem_83 = None
        add_187: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_41: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
        mul_162: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = rsqrt_41 = None
        mul_163: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_162, arg293_1);  mul_162 = arg293_1 = None
        add_188: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_163, arg294_1);  mul_163 = arg294_1 = None
        convert_element_type_146: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_188, torch.bfloat16);  add_188 = None
        view_458: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_146, _shape_param_543);  convert_element_type_146 = _shape_param_543 = None
        permute_229: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_61: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg296_1, view_458, permute_229);  arg296_1 = view_458 = permute_229 = None
        view_459: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_61, _shape_param_544);  addmm_61 = _shape_param_544 = None
        mul_164: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_459, 0.5)
        pow_21: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_459, 3.0)
        mul_165: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_189: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_459, mul_165);  view_459 = mul_165 = None
        mul_166: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_189, 0.7978845608028654);  add_189 = None
        tanh_20: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_166);  mul_166 = None
        add_190: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_20, 1.0);  tanh_20 = None
        mul_167: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_164, add_190);  mul_164 = add_190 = None
        view_460: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_167, _shape_param_545);  mul_167 = _shape_param_545 = None
        permute_230: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_62: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg298_1, view_460, permute_230);  arg298_1 = view_460 = permute_230 = None
        view_461: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_62, _shape_param_546);  addmm_62 = _shape_param_546 = None
        add_191: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_186, view_461);  add_186 = view_461 = None
        convert_element_type_147: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_191, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_147, [2], correction = 0, keepdim = True)
        getitem_84: "f32[32, 128, 1]" = var_mean_42[0]
        getitem_85: "f32[32, 128, 1]" = var_mean_42[1];  var_mean_42 = None
        slice_43: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg299_1, 2, 0, 128);  arg299_1 = None
        slice_44: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_43, 3, 0, 128);  slice_43 = None
        sub_63: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_147, getitem_85);  convert_element_type_147 = getitem_85 = None
        add_192: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
        rsqrt_42: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        mul_168: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = rsqrt_42 = None
        mul_169: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_168, arg300_1);  mul_168 = arg300_1 = None
        add_193: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_169, arg301_1);  mul_169 = arg301_1 = None
        convert_element_type_148: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_193, torch.bfloat16);  add_193 = None
        view_462: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_148, _shape_param_547);  _shape_param_547 = None
        permute_231: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg302_1, [1, 0]);  arg302_1 = None
        mm_63: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_462, permute_231);  view_462 = permute_231 = None
        view_463: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_63, _shape_param_548);  mm_63 = _shape_param_548 = None
        view_464: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_463, _shape_param_549);  view_463 = _shape_param_549 = None
        permute_232: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_464, [0, 2, 1, 3]);  view_464 = None
        convert_element_type_149: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_232, torch.float32);  permute_232 = None
        expand_85: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_149, _shape_param_550);  convert_element_type_149 = _shape_param_550 = None
        clone_84: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_465: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_84, _shape_param_551);  clone_84 = _shape_param_551 = None
        view_466: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_148, _shape_param_552);  _shape_param_552 = None
        permute_233: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        mm_64: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_466, permute_233);  view_466 = permute_233 = None
        view_467: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_64, _shape_param_553);  mm_64 = _shape_param_553 = None
        view_468: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_467, _shape_param_554);  view_467 = _shape_param_554 = None
        permute_234: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
        convert_element_type_150: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_234, torch.float32);  permute_234 = None
        permute_235: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_150, [0, 1, 3, 2]);  convert_element_type_150 = None
        expand_86: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_235, _shape_param_555);  permute_235 = _shape_param_555 = None
        clone_85: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
        view_469: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_85, _shape_param_556);  clone_85 = _shape_param_556 = None
        bmm_42: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_465, view_469);  view_465 = view_469 = None
        view_470: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_42, _shape_param_557);  bmm_42 = _shape_param_557 = None
        full_24: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_44, view_470, full_24);  slice_44 = view_470 = full_24 = None
        add_194: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_22, where_1);  where_22 = None
        amax_21: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_194, [-1], True)
        sub_64: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_194, amax_21);  add_194 = amax_21 = None
        exp_21: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        sum_22: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_21: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        convert_element_type_151: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None
        expand_87: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_151, _shape_param_558);  convert_element_type_151 = _shape_param_558 = None
        view_471: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_87, _shape_param_559);  expand_87 = _shape_param_559 = None
        view_472: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_148, _shape_param_560);  convert_element_type_148 = _shape_param_560 = None
        permute_236: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg304_1, [1, 0]);  arg304_1 = None
        mm_65: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_472, permute_236);  view_472 = permute_236 = None
        view_473: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_65, _shape_param_561);  mm_65 = _shape_param_561 = None
        view_474: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_473, _shape_param_562);  view_473 = _shape_param_562 = None
        permute_237: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_474, [0, 2, 1, 3]);  view_474 = None
        expand_88: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_237, _shape_param_563);  permute_237 = _shape_param_563 = None
        clone_86: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_475: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_86, _shape_param_564);  clone_86 = _shape_param_564 = None
        bmm_43: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_471, view_475);  view_471 = view_475 = None
        view_476: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_43, _shape_param_565);  bmm_43 = _shape_param_565 = None
        permute_238: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
        clone_87: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
        view_477: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_87, _shape_param_566);  clone_87 = _shape_param_566 = None
        view_478: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_477, _shape_param_567);  view_477 = _shape_param_567 = None
        permute_239: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_63: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg306_1, view_478, permute_239);  arg306_1 = view_478 = permute_239 = None
        view_479: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_63, _shape_param_568);  addmm_63 = _shape_param_568 = None
        add_195: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_479, add_191);  view_479 = add_191 = None
        convert_element_type_152: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_195, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_152, [2], correction = 0, keepdim = True)
        getitem_86: "f32[32, 128, 1]" = var_mean_43[0]
        getitem_87: "f32[32, 128, 1]" = var_mean_43[1];  var_mean_43 = None
        sub_65: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_152, getitem_87);  convert_element_type_152 = getitem_87 = None
        add_196: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_43: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        mul_170: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = rsqrt_43 = None
        mul_171: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_170, arg307_1);  mul_170 = arg307_1 = None
        add_197: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_171, arg308_1);  mul_171 = arg308_1 = None
        convert_element_type_153: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_197, torch.bfloat16);  add_197 = None
        view_480: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_153, _shape_param_569);  convert_element_type_153 = _shape_param_569 = None
        permute_240: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        addmm_64: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg310_1, view_480, permute_240);  arg310_1 = view_480 = permute_240 = None
        view_481: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_64, _shape_param_570);  addmm_64 = _shape_param_570 = None
        mul_172: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_481, 0.5)
        pow_22: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_481, 3.0)
        mul_173: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_198: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_481, mul_173);  view_481 = mul_173 = None
        mul_174: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_198, 0.7978845608028654);  add_198 = None
        tanh_21: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_174);  mul_174 = None
        add_199: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_21, 1.0);  tanh_21 = None
        mul_175: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_172, add_199);  mul_172 = add_199 = None
        view_482: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_175, _shape_param_571);  mul_175 = _shape_param_571 = None
        permute_241: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_65: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg312_1, view_482, permute_241);  arg312_1 = view_482 = permute_241 = None
        view_483: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_65, _shape_param_572);  addmm_65 = _shape_param_572 = None
        add_200: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_195, view_483);  add_195 = view_483 = None
        convert_element_type_154: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_200, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_154, [2], correction = 0, keepdim = True)
        getitem_88: "f32[32, 128, 1]" = var_mean_44[0]
        getitem_89: "f32[32, 128, 1]" = var_mean_44[1];  var_mean_44 = None
        slice_45: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg313_1, 2, 0, 128);  arg313_1 = None
        slice_46: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_45, 3, 0, 128);  slice_45 = None
        sub_66: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_154, getitem_89);  convert_element_type_154 = getitem_89 = None
        add_201: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_44: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
        mul_176: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = rsqrt_44 = None
        mul_177: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_176, arg314_1);  mul_176 = arg314_1 = None
        add_202: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_177, arg315_1);  mul_177 = arg315_1 = None
        convert_element_type_155: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_202, torch.bfloat16);  add_202 = None
        view_484: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_155, _shape_param_573);  _shape_param_573 = None
        permute_242: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg316_1, [1, 0]);  arg316_1 = None
        mm_66: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_484, permute_242);  view_484 = permute_242 = None
        view_485: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_66, _shape_param_574);  mm_66 = _shape_param_574 = None
        view_486: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_485, _shape_param_575);  view_485 = _shape_param_575 = None
        permute_243: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
        convert_element_type_156: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_243, torch.float32);  permute_243 = None
        expand_89: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_156, _shape_param_576);  convert_element_type_156 = _shape_param_576 = None
        clone_88: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_487: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_88, _shape_param_577);  clone_88 = _shape_param_577 = None
        view_488: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_155, _shape_param_578);  _shape_param_578 = None
        permute_244: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        mm_67: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_488, permute_244);  view_488 = permute_244 = None
        view_489: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_67, _shape_param_579);  mm_67 = _shape_param_579 = None
        view_490: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_489, _shape_param_580);  view_489 = _shape_param_580 = None
        permute_245: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_490, [0, 2, 1, 3]);  view_490 = None
        convert_element_type_157: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_245, torch.float32);  permute_245 = None
        permute_246: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_157, [0, 1, 3, 2]);  convert_element_type_157 = None
        expand_90: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_246, _shape_param_581);  permute_246 = _shape_param_581 = None
        clone_89: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_491: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_89, _shape_param_582);  clone_89 = _shape_param_582 = None
        bmm_44: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_487, view_491);  view_487 = view_491 = None
        view_492: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_44, _shape_param_583);  bmm_44 = _shape_param_583 = None
        full_25: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_23: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_46, view_492, full_25);  slice_46 = view_492 = full_25 = None
        add_203: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_23, where_1);  where_23 = None
        amax_22: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_203, [-1], True)
        sub_67: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_203, amax_22);  add_203 = amax_22 = None
        exp_22: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_23: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_158: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None
        expand_91: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_158, _shape_param_584);  convert_element_type_158 = _shape_param_584 = None
        view_493: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_91, _shape_param_585);  expand_91 = _shape_param_585 = None
        view_494: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_155, _shape_param_586);  convert_element_type_155 = _shape_param_586 = None
        permute_247: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg318_1, [1, 0]);  arg318_1 = None
        mm_68: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_494, permute_247);  view_494 = permute_247 = None
        view_495: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_68, _shape_param_587);  mm_68 = _shape_param_587 = None
        view_496: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_495, _shape_param_588);  view_495 = _shape_param_588 = None
        permute_248: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_496, [0, 2, 1, 3]);  view_496 = None
        expand_92: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_248, _shape_param_589);  permute_248 = _shape_param_589 = None
        clone_90: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_497: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_90, _shape_param_590);  clone_90 = _shape_param_590 = None
        bmm_45: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_493, view_497);  view_493 = view_497 = None
        view_498: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_45, _shape_param_591);  bmm_45 = _shape_param_591 = None
        permute_249: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
        clone_91: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_499: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_91, _shape_param_592);  clone_91 = _shape_param_592 = None
        view_500: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_499, _shape_param_593);  view_499 = _shape_param_593 = None
        permute_250: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
        addmm_66: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg320_1, view_500, permute_250);  arg320_1 = view_500 = permute_250 = None
        view_501: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_66, _shape_param_594);  addmm_66 = _shape_param_594 = None
        add_204: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_501, add_200);  view_501 = add_200 = None
        convert_element_type_159: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_204, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_159, [2], correction = 0, keepdim = True)
        getitem_90: "f32[32, 128, 1]" = var_mean_45[0]
        getitem_91: "f32[32, 128, 1]" = var_mean_45[1];  var_mean_45 = None
        sub_68: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_159, getitem_91);  convert_element_type_159 = getitem_91 = None
        add_205: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_45: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
        mul_178: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = rsqrt_45 = None
        mul_179: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_178, arg321_1);  mul_178 = arg321_1 = None
        add_206: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_179, arg322_1);  mul_179 = arg322_1 = None
        convert_element_type_160: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_206, torch.bfloat16);  add_206 = None
        view_502: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_160, _shape_param_595);  convert_element_type_160 = _shape_param_595 = None
        permute_251: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
        addmm_67: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg324_1, view_502, permute_251);  arg324_1 = view_502 = permute_251 = None
        view_503: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_67, _shape_param_596);  addmm_67 = _shape_param_596 = None
        mul_180: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_503, 0.5)
        pow_23: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_503, 3.0)
        mul_181: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_207: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_503, mul_181);  view_503 = mul_181 = None
        mul_182: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_207, 0.7978845608028654);  add_207 = None
        tanh_22: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_182);  mul_182 = None
        add_208: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_22, 1.0);  tanh_22 = None
        mul_183: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_180, add_208);  mul_180 = add_208 = None
        view_504: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_183, _shape_param_597);  mul_183 = _shape_param_597 = None
        permute_252: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg325_1, [1, 0]);  arg325_1 = None
        addmm_68: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg326_1, view_504, permute_252);  arg326_1 = view_504 = permute_252 = None
        view_505: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_68, _shape_param_598);  addmm_68 = _shape_param_598 = None
        add_209: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_204, view_505);  add_204 = view_505 = None
        convert_element_type_161: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_209, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_161, [2], correction = 0, keepdim = True)
        getitem_92: "f32[32, 128, 1]" = var_mean_46[0]
        getitem_93: "f32[32, 128, 1]" = var_mean_46[1];  var_mean_46 = None
        slice_47: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg327_1, 2, 0, 128);  arg327_1 = None
        slice_48: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_47, 3, 0, 128);  slice_47 = None
        sub_69: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_161, getitem_93);  convert_element_type_161 = getitem_93 = None
        add_210: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_46: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
        mul_184: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = rsqrt_46 = None
        mul_185: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_184, arg328_1);  mul_184 = arg328_1 = None
        add_211: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_185, arg329_1);  mul_185 = arg329_1 = None
        convert_element_type_162: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_211, torch.bfloat16);  add_211 = None
        view_506: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_162, _shape_param_599);  _shape_param_599 = None
        permute_253: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg330_1, [1, 0]);  arg330_1 = None
        mm_69: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_506, permute_253);  view_506 = permute_253 = None
        view_507: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_69, _shape_param_600);  mm_69 = _shape_param_600 = None
        view_508: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_507, _shape_param_601);  view_507 = _shape_param_601 = None
        permute_254: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
        convert_element_type_163: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_254, torch.float32);  permute_254 = None
        expand_93: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_163, _shape_param_602);  convert_element_type_163 = _shape_param_602 = None
        clone_92: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_509: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_92, _shape_param_603);  clone_92 = _shape_param_603 = None
        view_510: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_162, _shape_param_604);  _shape_param_604 = None
        permute_255: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        mm_70: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_510, permute_255);  view_510 = permute_255 = None
        view_511: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_70, _shape_param_605);  mm_70 = _shape_param_605 = None
        view_512: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_511, _shape_param_606);  view_511 = _shape_param_606 = None
        permute_256: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
        convert_element_type_164: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_256, torch.float32);  permute_256 = None
        permute_257: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_164, [0, 1, 3, 2]);  convert_element_type_164 = None
        expand_94: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_257, _shape_param_607);  permute_257 = _shape_param_607 = None
        clone_93: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
        view_513: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_93, _shape_param_608);  clone_93 = _shape_param_608 = None
        bmm_46: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_509, view_513);  view_509 = view_513 = None
        view_514: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_46, _shape_param_609);  bmm_46 = _shape_param_609 = None
        full_26: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_48, view_514, full_26);  slice_48 = view_514 = full_26 = None
        add_212: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_24, where_1);  where_24 = where_1 = None
        amax_23: "f32[32, 16, 128, 1]" = torch.ops.aten.amax.default(add_212, [-1], True)
        sub_70: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_212, amax_23);  add_212 = amax_23 = None
        exp_23: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
        sum_24: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_23: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        convert_element_type_165: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None
        expand_95: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_165, _shape_param_610);  convert_element_type_165 = _shape_param_610 = None
        view_515: "bf16[512, 128, 128]" = torch.ops.aten.view.default(expand_95, _shape_param_611);  expand_95 = _shape_param_611 = None
        view_516: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_162, _shape_param_612);  convert_element_type_162 = _shape_param_612 = None
        permute_258: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg332_1, [1, 0]);  arg332_1 = None
        mm_71: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_516, permute_258);  view_516 = permute_258 = None
        view_517: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_71, _shape_param_613);  mm_71 = _shape_param_613 = None
        view_518: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_517, _shape_param_614);  view_517 = _shape_param_614 = None
        permute_259: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_518, [0, 2, 1, 3]);  view_518 = None
        expand_96: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_259, _shape_param_615);  permute_259 = _shape_param_615 = None
        clone_94: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_519: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_94, _shape_param_616);  clone_94 = _shape_param_616 = None
        bmm_47: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_515, view_519);  view_515 = view_519 = None
        view_520: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_47, _shape_param_617);  bmm_47 = _shape_param_617 = None
        permute_260: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_520, [0, 2, 1, 3]);  view_520 = None
        clone_95: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_521: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_95, _shape_param_618);  clone_95 = _shape_param_618 = None
        view_522: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_521, _shape_param_619);  view_521 = _shape_param_619 = None
        permute_261: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
        addmm_69: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg334_1, view_522, permute_261);  arg334_1 = view_522 = permute_261 = None
        view_523: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_69, _shape_param_620);  addmm_69 = _shape_param_620 = None
        add_213: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_523, add_209);  view_523 = add_209 = None
        convert_element_type_166: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_213, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_166, [2], correction = 0, keepdim = True)
        getitem_94: "f32[32, 128, 1]" = var_mean_47[0]
        getitem_95: "f32[32, 128, 1]" = var_mean_47[1];  var_mean_47 = None
        sub_71: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_166, getitem_95);  convert_element_type_166 = getitem_95 = None
        add_214: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_47: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
        mul_186: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = rsqrt_47 = None
        mul_187: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_186, arg335_1);  mul_186 = arg335_1 = None
        add_215: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_187, arg336_1);  mul_187 = arg336_1 = None
        convert_element_type_167: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_215, torch.bfloat16);  add_215 = None
        view_524: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_167, _shape_param_621);  convert_element_type_167 = _shape_param_621 = None
        permute_262: "bf16[2048, 8192]" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        addmm_70: "bf16[4096, 8192]" = torch.ops.aten.addmm.default(arg338_1, view_524, permute_262);  arg338_1 = view_524 = permute_262 = None
        view_525: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(addmm_70, _shape_param_622);  addmm_70 = _shape_param_622 = None
        mul_188: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_525, 0.5)
        pow_24: "bf16[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_525, 3.0)
        mul_189: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_216: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_525, mul_189);  view_525 = mul_189 = None
        mul_190: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_216, 0.7978845608028654);  add_216 = None
        tanh_23: "bf16[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_190);  mul_190 = None
        add_217: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_23, 1.0);  tanh_23 = None
        mul_191: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_188, add_217);  mul_188 = add_217 = None
        view_526: "bf16[4096, 8192]" = torch.ops.aten.view.default(mul_191, _shape_param_623);  mul_191 = _shape_param_623 = None
        permute_263: "bf16[8192, 2048]" = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
        addmm_71: "bf16[4096, 2048]" = torch.ops.aten.addmm.default(arg340_1, view_526, permute_263);  arg340_1 = view_526 = permute_263 = None
        view_527: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(addmm_71, _shape_param_624);  addmm_71 = _shape_param_624 = None
        add_218: "bf16[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_213, view_527);  add_213 = view_527 = None
        convert_element_type_168: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_218, torch.float32);  add_218 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_168, [2], correction = 0, keepdim = True)
        getitem_96: "f32[32, 128, 1]" = var_mean_48[0]
        getitem_97: "f32[32, 128, 1]" = var_mean_48[1];  var_mean_48 = None
        ne: "b8[32]" = torch.ops.aten.ne.Scalar(arg341_1, -100)
        sub_72: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_168, getitem_97);  convert_element_type_168 = getitem_97 = None
        add_219: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_48: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
        mul_192: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = rsqrt_48 = None
        mul_193: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_192, arg342_1);  mul_192 = arg342_1 = None
        add_220: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_193, arg343_1);  mul_193 = arg343_1 = None
        convert_element_type_169: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_220, torch.bfloat16);  add_220 = None
        view_528: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_169, _shape_param_625);  convert_element_type_169 = _shape_param_625 = None
        permute_264: "bf16[2048, 2]" = torch.ops.aten.permute.default(arg344_1, [1, 0]);  arg344_1 = None
        mm_72: "bf16[4096, 2]" = torch.ops.aten.mm.default(view_528, permute_264);  view_528 = permute_264 = None
        view_529: "bf16[32, 128, 2]" = torch.ops.aten.view.default(mm_72, _shape_param_626);  mm_72 = _shape_param_626 = None
        iota_3: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        iota_4: "i32[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        ne_1: "b8[32, 128]" = torch.ops.aten.ne.Scalar(arg1_1, 0);  arg1_1 = None
        convert_element_type_170: "i32[32, 128]" = torch.ops.prims.convert_element_type.default(ne_1, torch.int32);  ne_1 = None
        mul_194: "i32[32, 128]" = torch.ops.aten.mul.Tensor(iota_4, convert_element_type_170);  iota_4 = convert_element_type_170 = None
        argmax: "i64[32]" = torch.ops.aten.argmax.default(mul_194, -1);  mul_194 = None
        index_2: "bf16[32, 2]" = torch.ops.aten.index.Tensor(view_529, [iota_3, argmax]);  view_529 = iota_3 = argmax = None
        convert_element_type_171: "f32[32, 2]" = torch.ops.prims.convert_element_type.default(index_2, torch.float32)
        amax_24: "f32[32, 1]" = torch.ops.aten.amax.default(convert_element_type_171, [1], True)
        sub_73: "f32[32, 2]" = torch.ops.aten.sub.Tensor(convert_element_type_171, amax_24);  convert_element_type_171 = amax_24 = None
        exp_24: "f32[32, 2]" = torch.ops.aten.exp.default(sub_73)
        sum_25: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[32, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_74: "f32[32, 2]" = torch.ops.aten.sub.Tensor(sub_73, log);  sub_73 = log = None
        convert_element_type_172: "bf16[32, 2]" = torch.ops.prims.convert_element_type.default(sub_74, torch.bfloat16);  sub_74 = None
        ne_2: "b8[32]" = torch.ops.aten.ne.Scalar(arg341_1, -100)
        full_27: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "i64[32]" = torch.ops.aten.where.self(ne_2, arg341_1, full_27);  ne_2 = full_27 = None
        unsqueeze_9: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(where_25, 1);  where_25 = None
        gather: "bf16[32, 1]" = torch.ops.aten.gather.default(convert_element_type_172, 1, unsqueeze_9);  convert_element_type_172 = unsqueeze_9 = None
        squeeze: "bf16[32]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[32]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_28: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "bf16[32]" = torch.ops.aten.where.self(ne, neg, full_28);  ne = neg = full_28 = None
        sum_26: "bf16[]" = torch.ops.aten.sum.default(where_26);  where_26 = None
        ne_3: "b8[32]" = torch.ops.aten.ne.Scalar(arg341_1, -100);  arg341_1 = None
        sum_27: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
        convert_element_type_173: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_27, torch.bfloat16);  sum_27 = None
        div_24: "bf16[]" = torch.ops.aten.div.Tensor(sum_26, convert_element_type_173);  sum_26 = convert_element_type_173 = None
        return (index_2, div_24)



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
