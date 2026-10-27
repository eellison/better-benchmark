"""
Standalone repro captured via capture_hook.
Label: swin_base_patch4_window7_224
Pattern hash: b6ac47dff992
Shape hash: 565e1333
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
    def forward(self, arg0_1: "bf16[128, 128, 56, 56]", arg1_1: "bf16[128]", arg2_1: "bf16[128]", arg3_1: "bf16[128]", arg4_1: "bf16[128]", arg5_1: "bf16[384, 128]", arg6_1: "bf16[384]", arg7_1: "i64[49, 49]", arg8_1: "bf16[169, 4]", arg9_1: "bf16[128, 128]", arg10_1: "bf16[128]", arg11_1: "bf16[128]", arg12_1: "bf16[128]", arg13_1: "bf16[512, 128]", arg14_1: "bf16[512]", arg15_1: "bf16[128, 512]", arg16_1: "bf16[128]", arg17_1: "bf16[128]", arg18_1: "bf16[128]", arg19_1: "bf16[384, 128]", arg20_1: "bf16[384]", arg21_1: "i64[49, 49]", arg22_1: "bf16[169, 4]", arg23_1: "bf16[64, 49, 49]", arg24_1: "bf16[128, 128]", arg25_1: "bf16[128]", arg26_1: "bf16[128]", arg27_1: "bf16[128]", arg28_1: "bf16[512, 128]", arg29_1: "bf16[512]", arg30_1: "bf16[128, 512]", arg31_1: "bf16[128]", arg32_1: "bf16[512]", arg33_1: "bf16[512]", arg34_1: "bf16[256, 512]", arg35_1: "bf16[256]", arg36_1: "bf16[256]", arg37_1: "bf16[768, 256]", arg38_1: "bf16[768]", arg39_1: "i64[49, 49]", arg40_1: "bf16[169, 8]", arg41_1: "bf16[256, 256]", arg42_1: "bf16[256]", arg43_1: "bf16[256]", arg44_1: "bf16[256]", arg45_1: "bf16[1024, 256]", arg46_1: "bf16[1024]", arg47_1: "bf16[256, 1024]", arg48_1: "bf16[256]", arg49_1: "bf16[256]", arg50_1: "bf16[256]", arg51_1: "bf16[768, 256]", arg52_1: "bf16[768]", arg53_1: "i64[49, 49]", arg54_1: "bf16[169, 8]", arg55_1: "bf16[16, 49, 49]", arg56_1: "bf16[256, 256]", arg57_1: "bf16[256]", arg58_1: "bf16[256]", arg59_1: "bf16[256]", arg60_1: "bf16[1024, 256]", arg61_1: "bf16[1024]", arg62_1: "bf16[256, 1024]", arg63_1: "bf16[256]", arg64_1: "bf16[1024]", arg65_1: "bf16[1024]", arg66_1: "bf16[512, 1024]", arg67_1: "bf16[512]", arg68_1: "bf16[512]", arg69_1: "bf16[1536, 512]", arg70_1: "bf16[1536]", arg71_1: "i64[49, 49]", arg72_1: "bf16[169, 16]", arg73_1: "bf16[512, 512]", arg74_1: "bf16[512]", arg75_1: "bf16[512]", arg76_1: "bf16[512]", arg77_1: "bf16[2048, 512]", arg78_1: "bf16[2048]", arg79_1: "bf16[512, 2048]", arg80_1: "bf16[512]", arg81_1: "bf16[512]", arg82_1: "bf16[512]", arg83_1: "bf16[1536, 512]", arg84_1: "bf16[1536]", arg85_1: "i64[49, 49]", arg86_1: "bf16[169, 16]", arg87_1: "bf16[4, 49, 49]", arg88_1: "bf16[512, 512]", arg89_1: "bf16[512]", arg90_1: "bf16[512]", arg91_1: "bf16[512]", arg92_1: "bf16[2048, 512]", arg93_1: "bf16[2048]", arg94_1: "bf16[512, 2048]", arg95_1: "bf16[512]", arg96_1: "bf16[512]", arg97_1: "bf16[512]", arg98_1: "bf16[1536, 512]", arg99_1: "bf16[1536]", arg100_1: "i64[49, 49]", arg101_1: "bf16[169, 16]", arg102_1: "bf16[512, 512]", arg103_1: "bf16[512]", arg104_1: "bf16[512]", arg105_1: "bf16[512]", arg106_1: "bf16[2048, 512]", arg107_1: "bf16[2048]", arg108_1: "bf16[512, 2048]", arg109_1: "bf16[512]", arg110_1: "bf16[512]", arg111_1: "bf16[512]", arg112_1: "bf16[1536, 512]", arg113_1: "bf16[1536]", arg114_1: "i64[49, 49]", arg115_1: "bf16[169, 16]", arg116_1: "bf16[4, 49, 49]", arg117_1: "bf16[512, 512]", arg118_1: "bf16[512]", arg119_1: "bf16[512]", arg120_1: "bf16[512]", arg121_1: "bf16[2048, 512]", arg122_1: "bf16[2048]", arg123_1: "bf16[512, 2048]", arg124_1: "bf16[512]", arg125_1: "bf16[512]", arg126_1: "bf16[512]", arg127_1: "bf16[1536, 512]", arg128_1: "bf16[1536]", arg129_1: "i64[49, 49]", arg130_1: "bf16[169, 16]", arg131_1: "bf16[512, 512]", arg132_1: "bf16[512]", arg133_1: "bf16[512]", arg134_1: "bf16[512]", arg135_1: "bf16[2048, 512]", arg136_1: "bf16[2048]", arg137_1: "bf16[512, 2048]", arg138_1: "bf16[512]", arg139_1: "bf16[512]", arg140_1: "bf16[512]", arg141_1: "bf16[1536, 512]", arg142_1: "bf16[1536]", arg143_1: "i64[49, 49]", arg144_1: "bf16[169, 16]", arg145_1: "bf16[4, 49, 49]", arg146_1: "bf16[512, 512]", arg147_1: "bf16[512]", arg148_1: "bf16[512]", arg149_1: "bf16[512]", arg150_1: "bf16[2048, 512]", arg151_1: "bf16[2048]", arg152_1: "bf16[512, 2048]", arg153_1: "bf16[512]", arg154_1: "bf16[512]", arg155_1: "bf16[512]", arg156_1: "bf16[1536, 512]", arg157_1: "bf16[1536]", arg158_1: "i64[49, 49]", arg159_1: "bf16[169, 16]", arg160_1: "bf16[512, 512]", arg161_1: "bf16[512]", arg162_1: "bf16[512]", arg163_1: "bf16[512]", arg164_1: "bf16[2048, 512]", arg165_1: "bf16[2048]", arg166_1: "bf16[512, 2048]", arg167_1: "bf16[512]", arg168_1: "bf16[512]", arg169_1: "bf16[512]", arg170_1: "bf16[1536, 512]", arg171_1: "bf16[1536]", arg172_1: "i64[49, 49]", arg173_1: "bf16[169, 16]", arg174_1: "bf16[4, 49, 49]", arg175_1: "bf16[512, 512]", arg176_1: "bf16[512]", arg177_1: "bf16[512]", arg178_1: "bf16[512]", arg179_1: "bf16[2048, 512]", arg180_1: "bf16[2048]", arg181_1: "bf16[512, 2048]", arg182_1: "bf16[512]", arg183_1: "bf16[512]", arg184_1: "bf16[512]", arg185_1: "bf16[1536, 512]", arg186_1: "bf16[1536]", arg187_1: "i64[49, 49]", arg188_1: "bf16[169, 16]", arg189_1: "bf16[512, 512]", arg190_1: "bf16[512]", arg191_1: "bf16[512]", arg192_1: "bf16[512]", arg193_1: "bf16[2048, 512]", arg194_1: "bf16[2048]", arg195_1: "bf16[512, 2048]", arg196_1: "bf16[512]", arg197_1: "bf16[512]", arg198_1: "bf16[512]", arg199_1: "bf16[1536, 512]", arg200_1: "bf16[1536]", arg201_1: "i64[49, 49]", arg202_1: "bf16[169, 16]", arg203_1: "bf16[4, 49, 49]", arg204_1: "bf16[512, 512]", arg205_1: "bf16[512]", arg206_1: "bf16[512]", arg207_1: "bf16[512]", arg208_1: "bf16[2048, 512]", arg209_1: "bf16[2048]", arg210_1: "bf16[512, 2048]", arg211_1: "bf16[512]", arg212_1: "bf16[512]", arg213_1: "bf16[512]", arg214_1: "bf16[1536, 512]", arg215_1: "bf16[1536]", arg216_1: "i64[49, 49]", arg217_1: "bf16[169, 16]", arg218_1: "bf16[512, 512]", arg219_1: "bf16[512]", arg220_1: "bf16[512]", arg221_1: "bf16[512]", arg222_1: "bf16[2048, 512]", arg223_1: "bf16[2048]", arg224_1: "bf16[512, 2048]", arg225_1: "bf16[512]", arg226_1: "bf16[512]", arg227_1: "bf16[512]", arg228_1: "bf16[1536, 512]", arg229_1: "bf16[1536]", arg230_1: "i64[49, 49]", arg231_1: "bf16[169, 16]", arg232_1: "bf16[4, 49, 49]", arg233_1: "bf16[512, 512]", arg234_1: "bf16[512]", arg235_1: "bf16[512]", arg236_1: "bf16[512]", arg237_1: "bf16[2048, 512]", arg238_1: "bf16[2048]", arg239_1: "bf16[512, 2048]", arg240_1: "bf16[512]", arg241_1: "bf16[512]", arg242_1: "bf16[512]", arg243_1: "bf16[1536, 512]", arg244_1: "bf16[1536]", arg245_1: "i64[49, 49]", arg246_1: "bf16[169, 16]", arg247_1: "bf16[512, 512]", arg248_1: "bf16[512]", arg249_1: "bf16[512]", arg250_1: "bf16[512]", arg251_1: "bf16[2048, 512]", arg252_1: "bf16[2048]", arg253_1: "bf16[512, 2048]", arg254_1: "bf16[512]", arg255_1: "bf16[512]", arg256_1: "bf16[512]", arg257_1: "bf16[1536, 512]", arg258_1: "bf16[1536]", arg259_1: "i64[49, 49]", arg260_1: "bf16[169, 16]", arg261_1: "bf16[4, 49, 49]", arg262_1: "bf16[512, 512]", arg263_1: "bf16[512]", arg264_1: "bf16[512]", arg265_1: "bf16[512]", arg266_1: "bf16[2048, 512]", arg267_1: "bf16[2048]", arg268_1: "bf16[512, 2048]", arg269_1: "bf16[512]", arg270_1: "bf16[512]", arg271_1: "bf16[512]", arg272_1: "bf16[1536, 512]", arg273_1: "bf16[1536]", arg274_1: "i64[49, 49]", arg275_1: "bf16[169, 16]", arg276_1: "bf16[512, 512]", arg277_1: "bf16[512]", arg278_1: "bf16[512]", arg279_1: "bf16[512]", arg280_1: "bf16[2048, 512]", arg281_1: "bf16[2048]", arg282_1: "bf16[512, 2048]", arg283_1: "bf16[512]", arg284_1: "bf16[512]", arg285_1: "bf16[512]", arg286_1: "bf16[1536, 512]", arg287_1: "bf16[1536]", arg288_1: "i64[49, 49]", arg289_1: "bf16[169, 16]", arg290_1: "bf16[4, 49, 49]", arg291_1: "bf16[512, 512]", arg292_1: "bf16[512]", arg293_1: "bf16[512]", arg294_1: "bf16[512]", arg295_1: "bf16[2048, 512]", arg296_1: "bf16[2048]", arg297_1: "bf16[512, 2048]", arg298_1: "bf16[512]", arg299_1: "bf16[512]", arg300_1: "bf16[512]", arg301_1: "bf16[1536, 512]", arg302_1: "bf16[1536]", arg303_1: "i64[49, 49]", arg304_1: "bf16[169, 16]", arg305_1: "bf16[512, 512]", arg306_1: "bf16[512]", arg307_1: "bf16[512]", arg308_1: "bf16[512]", arg309_1: "bf16[2048, 512]", arg310_1: "bf16[2048]", arg311_1: "bf16[512, 2048]", arg312_1: "bf16[512]", arg313_1: "bf16[512]", arg314_1: "bf16[512]", arg315_1: "bf16[1536, 512]", arg316_1: "bf16[1536]", arg317_1: "i64[49, 49]", arg318_1: "bf16[169, 16]", arg319_1: "bf16[4, 49, 49]", arg320_1: "bf16[512, 512]", arg321_1: "bf16[512]", arg322_1: "bf16[512]", arg323_1: "bf16[512]", arg324_1: "bf16[2048, 512]", arg325_1: "bf16[2048]", arg326_1: "bf16[512, 2048]", arg327_1: "bf16[512]", arg328_1: "bf16[2048]", arg329_1: "bf16[2048]", arg330_1: "bf16[1024, 2048]", arg331_1: "bf16[1024]", arg332_1: "bf16[1024]", arg333_1: "bf16[3072, 1024]", arg334_1: "bf16[3072]", arg335_1: "i64[49, 49]", arg336_1: "bf16[169, 32]", arg337_1: "bf16[1024, 1024]", arg338_1: "bf16[1024]", arg339_1: "bf16[1024]", arg340_1: "bf16[1024]", arg341_1: "bf16[4096, 1024]", arg342_1: "bf16[4096]", arg343_1: "bf16[1024, 4096]", arg344_1: "bf16[1024]", arg345_1: "bf16[1024]", arg346_1: "bf16[1024]", arg347_1: "bf16[3072, 1024]", arg348_1: "bf16[3072]", arg349_1: "i64[49, 49]", arg350_1: "bf16[169, 32]", arg351_1: "bf16[1024, 1024]", arg352_1: "bf16[1024]", arg353_1: "bf16[1024]", arg354_1: "bf16[1024]", arg355_1: "bf16[4096, 1024]", arg356_1: "bf16[4096]", arg357_1: "bf16[1024, 4096]", arg358_1: "bf16[1024]", arg359_1: "bf16[1024]", arg360_1: "bf16[1024]", arg361_1: "bf16[1000, 1024]", arg362_1: "bf16[1000]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631, _shape_param_632, _shape_param_633, _shape_param_634, _shape_param_635, _shape_param_636, _shape_param_637, _shape_param_638, _shape_param_639, _shape_param_640, _shape_param_641, _shape_param_642, _shape_param_643, _shape_param_644, _shape_param_645, _shape_param_646, _shape_param_647, _shape_param_648, _shape_param_649, _shape_param_650, _shape_param_651, _shape_param_652, _shape_param_653, _shape_param_654, _shape_param_655, _shape_param_656, _shape_param_657, _shape_param_658, _shape_param_659, _shape_param_660, _shape_param_661, _shape_param_662, _shape_param_663, _shape_param_664, _shape_param_665, _shape_param_666, _shape_param_667, _shape_param_668, _shape_param_669, _shape_param_670, _shape_param_671, _shape_param_672, _shape_param_673, _shape_param_674, _shape_param_675, _shape_param_676, _shape_param_677, _shape_param_678, _shape_param_679, _shape_param_680, _shape_param_681, _shape_param_682, _shape_param_683, _shape_param_684, _shape_param_685, _shape_param_686, _shape_param_687, _shape_param_688, _shape_param_689, _shape_param_690, _shape_param_691, _shape_param_692, _shape_param_693, _shape_param_694, _shape_param_695, _shape_param_696, _shape_param_697, _shape_param_698, _shape_param_699, _shape_param_700, _shape_param_701, _shape_param_702, _shape_param_703, _shape_param_704, _shape_param_705, _shape_param_706, _shape_param_707, _shape_param_708, _shape_param_709, _shape_param_710, _shape_param_711, _shape_param_712, _shape_param_713, _shape_param_714, _shape_param_715, _shape_param_716, _shape_param_717, _shape_param_718, _shape_param_719, _shape_param_720, _shape_param_721, _shape_param_722, _shape_param_723, _shape_param_724, _shape_param_725, _shape_param_726, _shape_param_727, _shape_param_728, _shape_param_729, _shape_param_730, _shape_param_731, _shape_param_732, _shape_param_733, _shape_param_734, _shape_param_735, _shape_param_736, _shape_param_737, _shape_param_738, _shape_param_739, _shape_param_740, _shape_param_741, _shape_param_742, _shape_param_743, _shape_param_744, _shape_param_745, _shape_param_746, _shape_param_747, _shape_param_748, _shape_param_749, _shape_param_750, _shape_param_751, _shape_param_752, _shape_param_753, _shape_param_754, _shape_param_755, _shape_param_756, _shape_param_757, _shape_param_758, _shape_param_759, _shape_param_760, _shape_param_761, _shape_param_762, _shape_param_763, _shape_param_764, _shape_param_765, _shape_param_766, _shape_param_767, _shape_param_768, _shape_param_769, _shape_param_770, _shape_param_771, _shape_param_772, _shape_param_773, _shape_param_774, _shape_param_775, _shape_param_776, _shape_param_777, _shape_param_778, _shape_param_779, _shape_param_780, _shape_param_781, _shape_param_782, _shape_param_783, _shape_param_784, _shape_param_785, _shape_param_786, _shape_param_787, _shape_param_788, _shape_param_789, _shape_param_790, _shape_param_791, _shape_param_792, _shape_param_793, _shape_param_794, _shape_param_795, _shape_param_796, _shape_param_797, _shape_param_798, _shape_param_799, _shape_param_800, _shape_param_801, _shape_param_802, _shape_param_803, _shape_param_804, _shape_param_805, _shape_param_806, _shape_param_807, _shape_param_808, _shape_param_809, _shape_param_810, _shape_param_811, _shape_param_812, _shape_param_813, _shape_param_814, _shape_param_815, _shape_param_816, _shape_param_817, _shape_param_818, _shape_param_819, _shape_param_820, _shape_param_821, _shape_param_822, _shape_param_823, _shape_param_824, _shape_param_825):
        # No stacktrace found for following nodes
        permute: "bf16[128, 56, 56, 128]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 3, 1]);  arg0_1 = None
        convert_element_type: "f32[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 56, 56, 1]" = var_mean[0]
        getitem_1: "f32[128, 56, 56, 1]" = var_mean[1];  var_mean = None
        sub: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add: "f32[128, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        add_1: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
        convert_element_type_1: "bf16[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        convert_element_type_2: "f32[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_2, [3], correction = 0, keepdim = True)
        getitem_2: "f32[128, 56, 56, 1]" = var_mean_1[0]
        getitem_3: "f32[128, 56, 56, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_2, getitem_3);  convert_element_type_2 = getitem_3 = None
        add_2: "f32[128, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[128, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_2: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_2, arg3_1);  mul_2 = arg3_1 = None
        add_3: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_3, arg4_1);  mul_3 = arg4_1 = None
        convert_element_type_3: "bf16[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_0);  convert_element_type_3 = _shape_param_0 = None
        permute_1: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view, [0, 1, 3, 2, 4, 5]);  view = None
        clone: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_1: "bf16[8192, 7, 7, 128]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        view_2: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        view_3: "bf16[401408, 128]" = torch.ops.aten.view.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        permute_2: "bf16[128, 384]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm: "bf16[401408, 384]" = torch.ops.aten.addmm.default(arg6_1, view_3, permute_2);  arg6_1 = view_3 = permute_2 = None
        view_4: "bf16[8192, 49, 384]" = torch.ops.aten.view.default(addmm, _shape_param_4);  addmm = _shape_param_4 = None
        view_5: "bf16[8192, 49, 3, 4, 32]" = torch.ops.aten.view.default(view_4, _shape_param_5);  view_4 = _shape_param_5 = None
        permute_3: "bf16[3, 8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_5, [2, 0, 3, 1, 4]);  view_5 = None
        unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
        getitem_4: "bf16[8192, 4, 49, 32]" = unbind[0]
        getitem_5: "bf16[8192, 4, 49, 32]" = unbind[1]
        getitem_6: "bf16[8192, 4, 49, 32]" = unbind[2];  unbind = None
        mul_4: "bf16[8192, 4, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_4, 0.1767766952966369);  getitem_4 = None
        expand: "bf16[8192, 4, 49, 32]" = torch.ops.aten.expand.default(mul_4, _shape_param_6);  mul_4 = _shape_param_6 = None
        clone_1: "bf16[8192, 4, 49, 32]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_6: "bf16[32768, 49, 32]" = torch.ops.aten.view.default(clone_1, _shape_param_7);  clone_1 = _shape_param_7 = None
        constant_pad_nd: "bf16[32768, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_6, _shape_param_8);  view_6 = _shape_param_8 = None
        permute_4: "bf16[8192, 4, 32, 49]" = torch.ops.aten.permute.default(getitem_5, [0, 1, 3, 2]);  getitem_5 = None
        expand_1: "bf16[8192, 4, 32, 49]" = torch.ops.aten.expand.default(permute_4, _shape_param_9);  permute_4 = _shape_param_9 = None
        clone_2: "bf16[8192, 4, 32, 49]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_7: "bf16[32768, 32, 49]" = torch.ops.aten.view.default(clone_2, _shape_param_10);  clone_2 = _shape_param_10 = None
        constant_pad_nd_1: "bf16[32768, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_7, _shape_param_11);  view_7 = _shape_param_11 = None
        bmm: "bf16[32768, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        slice_1: "bf16[32768, 49, 56]" = torch.ops.aten.slice.Tensor(bmm, 1, 0, -7);  bmm = None
        slice_2: "bf16[32768, 49, 49]" = torch.ops.aten.slice.Tensor(slice_1, 2, 0, -7);  slice_1 = None
        view_8: "bf16[8192, 4, 49, 49]" = torch.ops.aten.view.default(slice_2, _shape_param_12);  slice_2 = _shape_param_12 = None
        view_9: "i64[2401]" = torch.ops.aten.view.default(arg7_1, [-1]);  arg7_1 = None
        index: "bf16[2401, 4]" = torch.ops.aten.index.Tensor(arg8_1, [view_9]);  arg8_1 = view_9 = None
        view_10: "bf16[49, 49, 4]" = torch.ops.aten.view.default(index, _shape_param_13);  index = _shape_param_13 = None
        permute_5: "bf16[4, 49, 49]" = torch.ops.aten.permute.default(view_10, [2, 0, 1]);  view_10 = None
        clone_3: "bf16[4, 49, 49]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        unsqueeze: "bf16[1, 4, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_3, 0);  clone_3 = None
        add_4: "bf16[8192, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_8, unsqueeze);  view_8 = unsqueeze = None
        convert_element_type_4: "f32[8192, 4, 49, 49]" = torch.ops.prims.convert_element_type.default(add_4, torch.float32);  add_4 = None
        amax: "f32[8192, 4, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_4, [-1], True)
        sub_2: "f32[8192, 4, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_4, amax);  convert_element_type_4 = amax = None
        exp: "f32[8192, 4, 49, 49]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[8192, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8192, 4, 49, 49]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_5: "bf16[8192, 4, 49, 49]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand_2: "bf16[8192, 4, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_5, _shape_param_14);  convert_element_type_5 = _shape_param_14 = None
        view_11: "bf16[32768, 49, 49]" = torch.ops.aten.view.default(expand_2, _shape_param_15);  expand_2 = _shape_param_15 = None
        constant_pad_nd_2: "bf16[32768, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_11, _shape_param_16);  view_11 = _shape_param_16 = None
        expand_3: "bf16[8192, 4, 49, 32]" = torch.ops.aten.expand.default(getitem_6, _shape_param_17);  getitem_6 = _shape_param_17 = None
        clone_4: "bf16[8192, 4, 49, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_12: "bf16[32768, 49, 32]" = torch.ops.aten.view.default(clone_4, _shape_param_18);  clone_4 = _shape_param_18 = None
        constant_pad_nd_3: "bf16[32768, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_12, _shape_param_19);  view_12 = _shape_param_19 = None
        bmm_1: "bf16[32768, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_2, constant_pad_nd_3);  constant_pad_nd_2 = constant_pad_nd_3 = None
        slice_3: "bf16[32768, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_1, 1, 0, -7);  bmm_1 = None
        view_13: "bf16[8192, 4, 49, 32]" = torch.ops.aten.view.default(slice_3, _shape_param_20);  slice_3 = _shape_param_20 = None
        permute_6: "bf16[8192, 49, 4, 32]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        clone_5: "bf16[8192, 49, 4, 32]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_14: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(clone_5, _shape_param_21);  clone_5 = _shape_param_21 = None
        view_15: "bf16[401408, 128]" = torch.ops.aten.view.default(view_14, _shape_param_22);  view_14 = _shape_param_22 = None
        permute_7: "bf16[128, 128]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_1: "bf16[401408, 128]" = torch.ops.aten.addmm.default(arg10_1, view_15, permute_7);  arg10_1 = view_15 = permute_7 = None
        view_16: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(addmm_1, _shape_param_23);  addmm_1 = _shape_param_23 = None
        view_17: "bf16[8192, 7, 7, 128]" = torch.ops.aten.view.default(view_16, _shape_param_24);  view_16 = _shape_param_24 = None
        view_18: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_17, _shape_param_25);  view_17 = _shape_param_25 = None
        permute_8: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_18, [0, 1, 3, 2, 4, 5]);  view_18 = None
        clone_6: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_19: "bf16[128, 56, 56, 128]" = torch.ops.aten.view.default(clone_6, _shape_param_26);  clone_6 = _shape_param_26 = None
        add_5: "bf16[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(convert_element_type_1, view_19);  convert_element_type_1 = view_19 = None
        view_20: "bf16[128, 3136, 128]" = torch.ops.aten.view.default(add_5, _shape_param_27);  add_5 = _shape_param_27 = None
        convert_element_type_6: "f32[128, 3136, 128]" = torch.ops.prims.convert_element_type.default(view_20, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_6, [2], correction = 0, keepdim = True)
        getitem_7: "f32[128, 3136, 1]" = var_mean_2[0]
        getitem_8: "f32[128, 3136, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_6, getitem_8);  convert_element_type_6 = getitem_8 = None
        add_6: "f32[128, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_7, 1e-05);  getitem_7 = None
        rsqrt_2: "f32[128, 3136, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        mul_5: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_6: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_5, arg11_1);  mul_5 = arg11_1 = None
        add_7: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(mul_6, arg12_1);  mul_6 = arg12_1 = None
        convert_element_type_7: "bf16[128, 3136, 128]" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None
        view_21: "bf16[401408, 128]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_28);  convert_element_type_7 = _shape_param_28 = None
        permute_9: "bf16[128, 512]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_2: "bf16[401408, 512]" = torch.ops.aten.addmm.default(arg14_1, view_21, permute_9);  arg14_1 = view_21 = permute_9 = None
        view_22: "bf16[128, 3136, 512]" = torch.ops.aten.view.default(addmm_2, _shape_param_29);  addmm_2 = _shape_param_29 = None
        convert_element_type_8: "f32[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(view_22, torch.float32);  view_22 = None
        mul_7: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 0.5)
        mul_8: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 0.7071067811865476);  convert_element_type_8 = None
        erf: "f32[128, 3136, 512]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_8: "f32[128, 3136, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_7, add_8);  mul_7 = add_8 = None
        convert_element_type_9: "bf16[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        view_23: "bf16[401408, 512]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_30);  convert_element_type_9 = _shape_param_30 = None
        permute_10: "bf16[512, 128]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_3: "bf16[401408, 128]" = torch.ops.aten.addmm.default(arg16_1, view_23, permute_10);  arg16_1 = view_23 = permute_10 = None
        view_24: "bf16[128, 3136, 128]" = torch.ops.aten.view.default(addmm_3, _shape_param_31);  addmm_3 = _shape_param_31 = None
        add_9: "bf16[128, 3136, 128]" = torch.ops.aten.add.Tensor(view_20, view_24);  view_20 = view_24 = None
        view_25: "bf16[128, 56, 56, 128]" = torch.ops.aten.view.default(add_9, _shape_param_32);  add_9 = _shape_param_32 = None
        convert_element_type_10: "f32[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(view_25, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_10, [3], correction = 0, keepdim = True)
        getitem_9: "f32[128, 56, 56, 1]" = var_mean_3[0]
        getitem_10: "f32[128, 56, 56, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_4: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_10, getitem_10);  convert_element_type_10 = getitem_10 = None
        add_10: "f32[128, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_3: "f32[128, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_10: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
        mul_11: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_10, arg17_1);  mul_10 = arg17_1 = None
        add_11: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_11, arg18_1);  mul_11 = arg18_1 = None
        convert_element_type_11: "bf16[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None
        iota: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_12: "i64[56]" = torch.ops.aten.add.Tensor(iota, 3);  iota = None
        fmod: "i64[56]" = torch.ops.aten.fmod.Scalar(add_12, 56);  add_12 = None
        index_1: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(convert_element_type_11, [None, fmod]);  convert_element_type_11 = fmod = None
        iota_1: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_13: "i64[56]" = torch.ops.aten.add.Tensor(iota_1, 3);  iota_1 = None
        fmod_1: "i64[56]" = torch.ops.aten.fmod.Scalar(add_13, 56);  add_13 = None
        index_2: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index_1, [None, None, fmod_1]);  index_1 = fmod_1 = None
        view_26: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(index_2, _shape_param_33);  index_2 = _shape_param_33 = None
        permute_11: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_26, [0, 1, 3, 2, 4, 5]);  view_26 = None
        clone_7: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        view_27: "bf16[8192, 7, 7, 128]" = torch.ops.aten.view.default(clone_7, _shape_param_34);  clone_7 = _shape_param_34 = None
        view_28: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(view_27, _shape_param_35);  view_27 = _shape_param_35 = None
        view_29: "bf16[401408, 128]" = torch.ops.aten.view.default(view_28, _shape_param_36);  view_28 = _shape_param_36 = None
        permute_12: "bf16[128, 384]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_4: "bf16[401408, 384]" = torch.ops.aten.addmm.default(arg20_1, view_29, permute_12);  arg20_1 = view_29 = permute_12 = None
        view_30: "bf16[8192, 49, 384]" = torch.ops.aten.view.default(addmm_4, _shape_param_37);  addmm_4 = _shape_param_37 = None
        view_31: "bf16[8192, 49, 3, 4, 32]" = torch.ops.aten.view.default(view_30, _shape_param_38);  view_30 = _shape_param_38 = None
        permute_13: "bf16[3, 8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_31, [2, 0, 3, 1, 4]);  view_31 = None
        unbind_1 = torch.ops.aten.unbind.int(permute_13);  permute_13 = None
        getitem_11: "bf16[8192, 4, 49, 32]" = unbind_1[0]
        getitem_12: "bf16[8192, 4, 49, 32]" = unbind_1[1]
        getitem_13: "bf16[8192, 4, 49, 32]" = unbind_1[2];  unbind_1 = None
        mul_12: "bf16[8192, 4, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_11, 0.1767766952966369);  getitem_11 = None
        expand_4: "bf16[8192, 4, 49, 32]" = torch.ops.aten.expand.default(mul_12, _shape_param_39);  mul_12 = _shape_param_39 = None
        clone_8: "bf16[8192, 4, 49, 32]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_32: "bf16[32768, 49, 32]" = torch.ops.aten.view.default(clone_8, _shape_param_40);  clone_8 = _shape_param_40 = None
        constant_pad_nd_4: "bf16[32768, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_32, _shape_param_41);  view_32 = _shape_param_41 = None
        permute_14: "bf16[8192, 4, 32, 49]" = torch.ops.aten.permute.default(getitem_12, [0, 1, 3, 2]);  getitem_12 = None
        expand_5: "bf16[8192, 4, 32, 49]" = torch.ops.aten.expand.default(permute_14, _shape_param_42);  permute_14 = _shape_param_42 = None
        clone_9: "bf16[8192, 4, 32, 49]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_33: "bf16[32768, 32, 49]" = torch.ops.aten.view.default(clone_9, _shape_param_43);  clone_9 = _shape_param_43 = None
        constant_pad_nd_5: "bf16[32768, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_33, _shape_param_44);  view_33 = _shape_param_44 = None
        bmm_2: "bf16[32768, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_4, constant_pad_nd_5);  constant_pad_nd_4 = constant_pad_nd_5 = None
        slice_4: "bf16[32768, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_2, 1, 0, -7);  bmm_2 = None
        slice_5: "bf16[32768, 49, 49]" = torch.ops.aten.slice.Tensor(slice_4, 2, 0, -7);  slice_4 = None
        view_34: "bf16[8192, 4, 49, 49]" = torch.ops.aten.view.default(slice_5, _shape_param_45);  slice_5 = _shape_param_45 = None
        view_35: "i64[2401]" = torch.ops.aten.view.default(arg21_1, [-1]);  arg21_1 = None
        index_3: "bf16[2401, 4]" = torch.ops.aten.index.Tensor(arg22_1, [view_35]);  arg22_1 = view_35 = None
        view_36: "bf16[49, 49, 4]" = torch.ops.aten.view.default(index_3, _shape_param_46);  index_3 = _shape_param_46 = None
        permute_15: "bf16[4, 49, 49]" = torch.ops.aten.permute.default(view_36, [2, 0, 1]);  view_36 = None
        clone_10: "bf16[4, 49, 49]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        unsqueeze_1: "bf16[1, 4, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_10, 0);  clone_10 = None
        add_14: "bf16[8192, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_34, unsqueeze_1);  view_34 = unsqueeze_1 = None
        view_37: "bf16[128, 64, 4, 49, 49]" = torch.ops.aten.view.default(add_14, _shape_param_47);  add_14 = _shape_param_47 = None
        unsqueeze_2: "bf16[64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg23_1, 1);  arg23_1 = None
        unsqueeze_3: "bf16[1, 64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 0);  unsqueeze_2 = None
        add_15: "bf16[128, 64, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_37, unsqueeze_3);  view_37 = unsqueeze_3 = None
        view_38: "bf16[8192, 4, 49, 49]" = torch.ops.aten.view.default(add_15, _shape_param_48);  add_15 = _shape_param_48 = None
        convert_element_type_12: "f32[8192, 4, 49, 49]" = torch.ops.prims.convert_element_type.default(view_38, torch.float32);  view_38 = None
        amax_1: "f32[8192, 4, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_12, [-1], True)
        sub_5: "f32[8192, 4, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_12, amax_1);  convert_element_type_12 = amax_1 = None
        exp_1: "f32[8192, 4, 49, 49]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_2: "f32[8192, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[8192, 4, 49, 49]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_13: "bf16[8192, 4, 49, 49]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        expand_6: "bf16[8192, 4, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_13, _shape_param_49);  convert_element_type_13 = _shape_param_49 = None
        view_39: "bf16[32768, 49, 49]" = torch.ops.aten.view.default(expand_6, _shape_param_50);  expand_6 = _shape_param_50 = None
        constant_pad_nd_6: "bf16[32768, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_39, _shape_param_51);  view_39 = _shape_param_51 = None
        expand_7: "bf16[8192, 4, 49, 32]" = torch.ops.aten.expand.default(getitem_13, _shape_param_52);  getitem_13 = _shape_param_52 = None
        clone_11: "bf16[8192, 4, 49, 32]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_40: "bf16[32768, 49, 32]" = torch.ops.aten.view.default(clone_11, _shape_param_53);  clone_11 = _shape_param_53 = None
        constant_pad_nd_7: "bf16[32768, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_40, _shape_param_54);  view_40 = _shape_param_54 = None
        bmm_3: "bf16[32768, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_6, constant_pad_nd_7);  constant_pad_nd_6 = constant_pad_nd_7 = None
        slice_6: "bf16[32768, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_3, 1, 0, -7);  bmm_3 = None
        view_41: "bf16[8192, 4, 49, 32]" = torch.ops.aten.view.default(slice_6, _shape_param_55);  slice_6 = _shape_param_55 = None
        permute_16: "bf16[8192, 49, 4, 32]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        clone_12: "bf16[8192, 49, 4, 32]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_42: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(clone_12, _shape_param_56);  clone_12 = _shape_param_56 = None
        view_43: "bf16[401408, 128]" = torch.ops.aten.view.default(view_42, _shape_param_57);  view_42 = _shape_param_57 = None
        permute_17: "bf16[128, 128]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_5: "bf16[401408, 128]" = torch.ops.aten.addmm.default(arg25_1, view_43, permute_17);  arg25_1 = view_43 = permute_17 = None
        view_44: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(addmm_5, _shape_param_58);  addmm_5 = _shape_param_58 = None
        view_45: "bf16[8192, 7, 7, 128]" = torch.ops.aten.view.default(view_44, _shape_param_59);  view_44 = _shape_param_59 = None
        view_46: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_45, _shape_param_60);  view_45 = _shape_param_60 = None
        permute_18: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_46, [0, 1, 3, 2, 4, 5]);  view_46 = None
        clone_13: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_47: "bf16[128, 56, 56, 128]" = torch.ops.aten.view.default(clone_13, _shape_param_61);  clone_13 = _shape_param_61 = None
        iota_2: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_16: "i64[56]" = torch.ops.aten.add.Tensor(iota_2, 53);  iota_2 = None
        fmod_2: "i64[56]" = torch.ops.aten.fmod.Scalar(add_16, 56);  add_16 = None
        index_4: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(view_47, [None, fmod_2]);  view_47 = fmod_2 = None
        iota_3: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_17: "i64[56]" = torch.ops.aten.add.Tensor(iota_3, 53);  iota_3 = None
        fmod_3: "i64[56]" = torch.ops.aten.fmod.Scalar(add_17, 56);  add_17 = None
        index_5: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index_4, [None, None, fmod_3]);  index_4 = fmod_3 = None
        add_18: "bf16[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_25, index_5);  view_25 = index_5 = None
        view_48: "bf16[128, 3136, 128]" = torch.ops.aten.view.default(add_18, _shape_param_62);  add_18 = _shape_param_62 = None
        convert_element_type_14: "f32[128, 3136, 128]" = torch.ops.prims.convert_element_type.default(view_48, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_14, [2], correction = 0, keepdim = True)
        getitem_14: "f32[128, 3136, 1]" = var_mean_4[0]
        getitem_15: "f32[128, 3136, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_6: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_14, getitem_15);  convert_element_type_14 = getitem_15 = None
        add_19: "f32[128, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_4: "f32[128, 3136, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_13: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_14: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_13, arg26_1);  mul_13 = arg26_1 = None
        add_20: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(mul_14, arg27_1);  mul_14 = arg27_1 = None
        convert_element_type_15: "bf16[128, 3136, 128]" = torch.ops.prims.convert_element_type.default(add_20, torch.bfloat16);  add_20 = None
        view_49: "bf16[401408, 128]" = torch.ops.aten.view.default(convert_element_type_15, _shape_param_63);  convert_element_type_15 = _shape_param_63 = None
        permute_19: "bf16[128, 512]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_6: "bf16[401408, 512]" = torch.ops.aten.addmm.default(arg29_1, view_49, permute_19);  arg29_1 = view_49 = permute_19 = None
        view_50: "bf16[128, 3136, 512]" = torch.ops.aten.view.default(addmm_6, _shape_param_64);  addmm_6 = _shape_param_64 = None
        convert_element_type_16: "f32[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(view_50, torch.float32);  view_50 = None
        mul_15: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 0.5)
        mul_16: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 0.7071067811865476);  convert_element_type_16 = None
        erf_1: "f32[128, 3136, 512]" = torch.ops.aten.erf.default(mul_16);  mul_16 = None
        add_21: "f32[128, 3136, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_17: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_15, add_21);  mul_15 = add_21 = None
        convert_element_type_17: "bf16[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        view_51: "bf16[401408, 512]" = torch.ops.aten.view.default(convert_element_type_17, _shape_param_65);  convert_element_type_17 = _shape_param_65 = None
        permute_20: "bf16[512, 128]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
        addmm_7: "bf16[401408, 128]" = torch.ops.aten.addmm.default(arg31_1, view_51, permute_20);  arg31_1 = view_51 = permute_20 = None
        view_52: "bf16[128, 3136, 128]" = torch.ops.aten.view.default(addmm_7, _shape_param_66);  addmm_7 = _shape_param_66 = None
        add_22: "bf16[128, 3136, 128]" = torch.ops.aten.add.Tensor(view_48, view_52);  view_48 = view_52 = None
        view_53: "bf16[128, 56, 56, 128]" = torch.ops.aten.view.default(add_22, _shape_param_67);  add_22 = _shape_param_67 = None
        view_54: "bf16[128, 28, 2, 28, 2, 128]" = torch.ops.aten.view.default(view_53, _shape_param_68);  view_53 = _shape_param_68 = None
        permute_21: "bf16[128, 28, 28, 2, 2, 128]" = torch.ops.aten.permute.default(view_54, [0, 1, 3, 4, 2, 5]);  view_54 = None
        clone_14: "bf16[128, 28, 28, 2, 2, 128]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
        view_55: "bf16[128, 28, 28, 512]" = torch.ops.aten.view.default(clone_14, _shape_param_69);  clone_14 = _shape_param_69 = None
        convert_element_type_18: "f32[128, 28, 28, 512]" = torch.ops.prims.convert_element_type.default(view_55, torch.float32);  view_55 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_18, [3], correction = 0, keepdim = True)
        getitem_16: "f32[128, 28, 28, 1]" = var_mean_5[0]
        getitem_17: "f32[128, 28, 28, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_7: "f32[128, 28, 28, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_18, getitem_17);  convert_element_type_18 = getitem_17 = None
        add_23: "f32[128, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_5: "f32[128, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_18: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = rsqrt_5 = None
        mul_19: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_18, arg32_1);  mul_18 = arg32_1 = None
        add_24: "f32[128, 28, 28, 512]" = torch.ops.aten.add.Tensor(mul_19, arg33_1);  mul_19 = arg33_1 = None
        convert_element_type_19: "bf16[128, 28, 28, 512]" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None
        view_56: "bf16[100352, 512]" = torch.ops.aten.view.default(convert_element_type_19, _shape_param_70);  convert_element_type_19 = _shape_param_70 = None
        permute_22: "bf16[512, 256]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        mm: "bf16[100352, 256]" = torch.ops.aten.mm.default(view_56, permute_22);  view_56 = permute_22 = None
        view_57: "bf16[128, 28, 28, 256]" = torch.ops.aten.view.default(mm, _shape_param_71);  mm = _shape_param_71 = None
        convert_element_type_20: "f32[128, 28, 28, 256]" = torch.ops.prims.convert_element_type.default(view_57, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_20, [3], correction = 0, keepdim = True)
        getitem_18: "f32[128, 28, 28, 1]" = var_mean_6[0]
        getitem_19: "f32[128, 28, 28, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_8: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_20, getitem_19);  convert_element_type_20 = getitem_19 = None
        add_25: "f32[128, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_6: "f32[128, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_20: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = rsqrt_6 = None
        mul_21: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_20, arg35_1);  mul_20 = arg35_1 = None
        add_26: "f32[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_21, arg36_1);  mul_21 = arg36_1 = None
        convert_element_type_21: "bf16[128, 28, 28, 256]" = torch.ops.prims.convert_element_type.default(add_26, torch.bfloat16);  add_26 = None
        view_58: "bf16[128, 4, 7, 4, 7, 256]" = torch.ops.aten.view.default(convert_element_type_21, _shape_param_72);  convert_element_type_21 = _shape_param_72 = None
        permute_23: "bf16[128, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_58, [0, 1, 3, 2, 4, 5]);  view_58 = None
        clone_15: "bf16[128, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_59: "bf16[2048, 7, 7, 256]" = torch.ops.aten.view.default(clone_15, _shape_param_73);  clone_15 = _shape_param_73 = None
        view_60: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(view_59, _shape_param_74);  view_59 = _shape_param_74 = None
        view_61: "bf16[100352, 256]" = torch.ops.aten.view.default(view_60, _shape_param_75);  view_60 = _shape_param_75 = None
        permute_24: "bf16[256, 768]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_8: "bf16[100352, 768]" = torch.ops.aten.addmm.default(arg38_1, view_61, permute_24);  arg38_1 = view_61 = permute_24 = None
        view_62: "bf16[2048, 49, 768]" = torch.ops.aten.view.default(addmm_8, _shape_param_76);  addmm_8 = _shape_param_76 = None
        view_63: "bf16[2048, 49, 3, 8, 32]" = torch.ops.aten.view.default(view_62, _shape_param_77);  view_62 = _shape_param_77 = None
        permute_25: "bf16[3, 2048, 8, 49, 32]" = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None
        unbind_2 = torch.ops.aten.unbind.int(permute_25);  permute_25 = None
        getitem_20: "bf16[2048, 8, 49, 32]" = unbind_2[0]
        getitem_21: "bf16[2048, 8, 49, 32]" = unbind_2[1]
        getitem_22: "bf16[2048, 8, 49, 32]" = unbind_2[2];  unbind_2 = None
        mul_22: "bf16[2048, 8, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_20, 0.1767766952966369);  getitem_20 = None
        expand_8: "bf16[2048, 8, 49, 32]" = torch.ops.aten.expand.default(mul_22, _shape_param_78);  mul_22 = _shape_param_78 = None
        clone_16: "bf16[2048, 8, 49, 32]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_64: "bf16[16384, 49, 32]" = torch.ops.aten.view.default(clone_16, _shape_param_79);  clone_16 = _shape_param_79 = None
        constant_pad_nd_8: "bf16[16384, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_64, _shape_param_80);  view_64 = _shape_param_80 = None
        permute_26: "bf16[2048, 8, 32, 49]" = torch.ops.aten.permute.default(getitem_21, [0, 1, 3, 2]);  getitem_21 = None
        expand_9: "bf16[2048, 8, 32, 49]" = torch.ops.aten.expand.default(permute_26, _shape_param_81);  permute_26 = _shape_param_81 = None
        clone_17: "bf16[2048, 8, 32, 49]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_65: "bf16[16384, 32, 49]" = torch.ops.aten.view.default(clone_17, _shape_param_82);  clone_17 = _shape_param_82 = None
        constant_pad_nd_9: "bf16[16384, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_65, _shape_param_83);  view_65 = _shape_param_83 = None
        bmm_4: "bf16[16384, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_8, constant_pad_nd_9);  constant_pad_nd_8 = constant_pad_nd_9 = None
        slice_7: "bf16[16384, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_4, 1, 0, -7);  bmm_4 = None
        slice_8: "bf16[16384, 49, 49]" = torch.ops.aten.slice.Tensor(slice_7, 2, 0, -7);  slice_7 = None
        view_66: "bf16[2048, 8, 49, 49]" = torch.ops.aten.view.default(slice_8, _shape_param_84);  slice_8 = _shape_param_84 = None
        view_67: "i64[2401]" = torch.ops.aten.view.default(arg39_1, [-1]);  arg39_1 = None
        index_6: "bf16[2401, 8]" = torch.ops.aten.index.Tensor(arg40_1, [view_67]);  arg40_1 = view_67 = None
        view_68: "bf16[49, 49, 8]" = torch.ops.aten.view.default(index_6, _shape_param_85);  index_6 = _shape_param_85 = None
        permute_27: "bf16[8, 49, 49]" = torch.ops.aten.permute.default(view_68, [2, 0, 1]);  view_68 = None
        clone_18: "bf16[8, 49, 49]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        unsqueeze_4: "bf16[1, 8, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_18, 0);  clone_18 = None
        add_27: "bf16[2048, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_66, unsqueeze_4);  view_66 = unsqueeze_4 = None
        convert_element_type_22: "f32[2048, 8, 49, 49]" = torch.ops.prims.convert_element_type.default(add_27, torch.float32);  add_27 = None
        amax_2: "f32[2048, 8, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_22, [-1], True)
        sub_9: "f32[2048, 8, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_22, amax_2);  convert_element_type_22 = amax_2 = None
        exp_2: "f32[2048, 8, 49, 49]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_3: "f32[2048, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[2048, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_23: "bf16[2048, 8, 49, 49]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        expand_10: "bf16[2048, 8, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_23, _shape_param_86);  convert_element_type_23 = _shape_param_86 = None
        view_69: "bf16[16384, 49, 49]" = torch.ops.aten.view.default(expand_10, _shape_param_87);  expand_10 = _shape_param_87 = None
        constant_pad_nd_10: "bf16[16384, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_69, _shape_param_88);  view_69 = _shape_param_88 = None
        expand_11: "bf16[2048, 8, 49, 32]" = torch.ops.aten.expand.default(getitem_22, _shape_param_89);  getitem_22 = _shape_param_89 = None
        clone_19: "bf16[2048, 8, 49, 32]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_70: "bf16[16384, 49, 32]" = torch.ops.aten.view.default(clone_19, _shape_param_90);  clone_19 = _shape_param_90 = None
        constant_pad_nd_11: "bf16[16384, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_70, _shape_param_91);  view_70 = _shape_param_91 = None
        bmm_5: "bf16[16384, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_10, constant_pad_nd_11);  constant_pad_nd_10 = constant_pad_nd_11 = None
        slice_9: "bf16[16384, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_5, 1, 0, -7);  bmm_5 = None
        view_71: "bf16[2048, 8, 49, 32]" = torch.ops.aten.view.default(slice_9, _shape_param_92);  slice_9 = _shape_param_92 = None
        permute_28: "bf16[2048, 49, 8, 32]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        clone_20: "bf16[2048, 49, 8, 32]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
        view_72: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(clone_20, _shape_param_93);  clone_20 = _shape_param_93 = None
        view_73: "bf16[100352, 256]" = torch.ops.aten.view.default(view_72, _shape_param_94);  view_72 = _shape_param_94 = None
        permute_29: "bf16[256, 256]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_9: "bf16[100352, 256]" = torch.ops.aten.addmm.default(arg42_1, view_73, permute_29);  arg42_1 = view_73 = permute_29 = None
        view_74: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(addmm_9, _shape_param_95);  addmm_9 = _shape_param_95 = None
        view_75: "bf16[2048, 7, 7, 256]" = torch.ops.aten.view.default(view_74, _shape_param_96);  view_74 = _shape_param_96 = None
        view_76: "bf16[128, 4, 4, 7, 7, 256]" = torch.ops.aten.view.default(view_75, _shape_param_97);  view_75 = _shape_param_97 = None
        permute_30: "bf16[128, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_76, [0, 1, 3, 2, 4, 5]);  view_76 = None
        clone_21: "bf16[128, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_77: "bf16[128, 28, 28, 256]" = torch.ops.aten.view.default(clone_21, _shape_param_98);  clone_21 = _shape_param_98 = None
        add_28: "bf16[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_57, view_77);  view_57 = view_77 = None
        view_78: "bf16[128, 784, 256]" = torch.ops.aten.view.default(add_28, _shape_param_99);  add_28 = _shape_param_99 = None
        convert_element_type_24: "f32[128, 784, 256]" = torch.ops.prims.convert_element_type.default(view_78, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_24, [2], correction = 0, keepdim = True)
        getitem_23: "f32[128, 784, 1]" = var_mean_7[0]
        getitem_24: "f32[128, 784, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_10: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_24, getitem_24);  convert_element_type_24 = getitem_24 = None
        add_29: "f32[128, 784, 1]" = torch.ops.aten.add.Tensor(getitem_23, 1e-05);  getitem_23 = None
        rsqrt_7: "f32[128, 784, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        mul_23: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_7);  sub_10 = rsqrt_7 = None
        mul_24: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_23, arg43_1);  mul_23 = arg43_1 = None
        add_30: "f32[128, 784, 256]" = torch.ops.aten.add.Tensor(mul_24, arg44_1);  mul_24 = arg44_1 = None
        convert_element_type_25: "bf16[128, 784, 256]" = torch.ops.prims.convert_element_type.default(add_30, torch.bfloat16);  add_30 = None
        view_79: "bf16[100352, 256]" = torch.ops.aten.view.default(convert_element_type_25, _shape_param_100);  convert_element_type_25 = _shape_param_100 = None
        permute_31: "bf16[256, 1024]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_10: "bf16[100352, 1024]" = torch.ops.aten.addmm.default(arg46_1, view_79, permute_31);  arg46_1 = view_79 = permute_31 = None
        view_80: "bf16[128, 784, 1024]" = torch.ops.aten.view.default(addmm_10, _shape_param_101);  addmm_10 = _shape_param_101 = None
        convert_element_type_26: "f32[128, 784, 1024]" = torch.ops.prims.convert_element_type.default(view_80, torch.float32);  view_80 = None
        mul_25: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.5)
        mul_26: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.7071067811865476);  convert_element_type_26 = None
        erf_2: "f32[128, 784, 1024]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_31: "f32[128, 784, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_27: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_25, add_31);  mul_25 = add_31 = None
        convert_element_type_27: "bf16[128, 784, 1024]" = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16);  mul_27 = None
        view_81: "bf16[100352, 1024]" = torch.ops.aten.view.default(convert_element_type_27, _shape_param_102);  convert_element_type_27 = _shape_param_102 = None
        permute_32: "bf16[1024, 256]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_11: "bf16[100352, 256]" = torch.ops.aten.addmm.default(arg48_1, view_81, permute_32);  arg48_1 = view_81 = permute_32 = None
        view_82: "bf16[128, 784, 256]" = torch.ops.aten.view.default(addmm_11, _shape_param_103);  addmm_11 = _shape_param_103 = None
        add_32: "bf16[128, 784, 256]" = torch.ops.aten.add.Tensor(view_78, view_82);  view_78 = view_82 = None
        view_83: "bf16[128, 28, 28, 256]" = torch.ops.aten.view.default(add_32, _shape_param_104);  add_32 = _shape_param_104 = None
        convert_element_type_28: "f32[128, 28, 28, 256]" = torch.ops.prims.convert_element_type.default(view_83, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_28, [3], correction = 0, keepdim = True)
        getitem_25: "f32[128, 28, 28, 1]" = var_mean_8[0]
        getitem_26: "f32[128, 28, 28, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_11: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_28, getitem_26);  convert_element_type_28 = getitem_26 = None
        add_33: "f32[128, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_25, 1e-05);  getitem_25 = None
        rsqrt_8: "f32[128, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        mul_28: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_8);  sub_11 = rsqrt_8 = None
        mul_29: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_28, arg49_1);  mul_28 = arg49_1 = None
        add_34: "f32[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_29, arg50_1);  mul_29 = arg50_1 = None
        convert_element_type_29: "bf16[128, 28, 28, 256]" = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16);  add_34 = None
        iota_4: "i64[28]" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_35: "i64[28]" = torch.ops.aten.add.Tensor(iota_4, 3);  iota_4 = None
        fmod_4: "i64[28]" = torch.ops.aten.fmod.Scalar(add_35, 28);  add_35 = None
        index_7: "bf16[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(convert_element_type_29, [None, fmod_4]);  convert_element_type_29 = fmod_4 = None
        iota_5: "i64[28]" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_36: "i64[28]" = torch.ops.aten.add.Tensor(iota_5, 3);  iota_5 = None
        fmod_5: "i64[28]" = torch.ops.aten.fmod.Scalar(add_36, 28);  add_36 = None
        index_8: "bf16[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(index_7, [None, None, fmod_5]);  index_7 = fmod_5 = None
        view_84: "bf16[128, 4, 7, 4, 7, 256]" = torch.ops.aten.view.default(index_8, _shape_param_105);  index_8 = _shape_param_105 = None
        permute_33: "bf16[128, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_84, [0, 1, 3, 2, 4, 5]);  view_84 = None
        clone_22: "bf16[128, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        view_85: "bf16[2048, 7, 7, 256]" = torch.ops.aten.view.default(clone_22, _shape_param_106);  clone_22 = _shape_param_106 = None
        view_86: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(view_85, _shape_param_107);  view_85 = _shape_param_107 = None
        view_87: "bf16[100352, 256]" = torch.ops.aten.view.default(view_86, _shape_param_108);  view_86 = _shape_param_108 = None
        permute_34: "bf16[256, 768]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_12: "bf16[100352, 768]" = torch.ops.aten.addmm.default(arg52_1, view_87, permute_34);  arg52_1 = view_87 = permute_34 = None
        view_88: "bf16[2048, 49, 768]" = torch.ops.aten.view.default(addmm_12, _shape_param_109);  addmm_12 = _shape_param_109 = None
        view_89: "bf16[2048, 49, 3, 8, 32]" = torch.ops.aten.view.default(view_88, _shape_param_110);  view_88 = _shape_param_110 = None
        permute_35: "bf16[3, 2048, 8, 49, 32]" = torch.ops.aten.permute.default(view_89, [2, 0, 3, 1, 4]);  view_89 = None
        unbind_3 = torch.ops.aten.unbind.int(permute_35);  permute_35 = None
        getitem_27: "bf16[2048, 8, 49, 32]" = unbind_3[0]
        getitem_28: "bf16[2048, 8, 49, 32]" = unbind_3[1]
        getitem_29: "bf16[2048, 8, 49, 32]" = unbind_3[2];  unbind_3 = None
        mul_30: "bf16[2048, 8, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_27, 0.1767766952966369);  getitem_27 = None
        expand_12: "bf16[2048, 8, 49, 32]" = torch.ops.aten.expand.default(mul_30, _shape_param_111);  mul_30 = _shape_param_111 = None
        clone_23: "bf16[2048, 8, 49, 32]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_90: "bf16[16384, 49, 32]" = torch.ops.aten.view.default(clone_23, _shape_param_112);  clone_23 = _shape_param_112 = None
        constant_pad_nd_12: "bf16[16384, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_90, _shape_param_113);  view_90 = _shape_param_113 = None
        permute_36: "bf16[2048, 8, 32, 49]" = torch.ops.aten.permute.default(getitem_28, [0, 1, 3, 2]);  getitem_28 = None
        expand_13: "bf16[2048, 8, 32, 49]" = torch.ops.aten.expand.default(permute_36, _shape_param_114);  permute_36 = _shape_param_114 = None
        clone_24: "bf16[2048, 8, 32, 49]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_91: "bf16[16384, 32, 49]" = torch.ops.aten.view.default(clone_24, _shape_param_115);  clone_24 = _shape_param_115 = None
        constant_pad_nd_13: "bf16[16384, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_91, _shape_param_116);  view_91 = _shape_param_116 = None
        bmm_6: "bf16[16384, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_12, constant_pad_nd_13);  constant_pad_nd_12 = constant_pad_nd_13 = None
        slice_10: "bf16[16384, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_6, 1, 0, -7);  bmm_6 = None
        slice_11: "bf16[16384, 49, 49]" = torch.ops.aten.slice.Tensor(slice_10, 2, 0, -7);  slice_10 = None
        view_92: "bf16[2048, 8, 49, 49]" = torch.ops.aten.view.default(slice_11, _shape_param_117);  slice_11 = _shape_param_117 = None
        view_93: "i64[2401]" = torch.ops.aten.view.default(arg53_1, [-1]);  arg53_1 = None
        index_9: "bf16[2401, 8]" = torch.ops.aten.index.Tensor(arg54_1, [view_93]);  arg54_1 = view_93 = None
        view_94: "bf16[49, 49, 8]" = torch.ops.aten.view.default(index_9, _shape_param_118);  index_9 = _shape_param_118 = None
        permute_37: "bf16[8, 49, 49]" = torch.ops.aten.permute.default(view_94, [2, 0, 1]);  view_94 = None
        clone_25: "bf16[8, 49, 49]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        unsqueeze_5: "bf16[1, 8, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_25, 0);  clone_25 = None
        add_37: "bf16[2048, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_92, unsqueeze_5);  view_92 = unsqueeze_5 = None
        view_95: "bf16[128, 16, 8, 49, 49]" = torch.ops.aten.view.default(add_37, _shape_param_119);  add_37 = _shape_param_119 = None
        unsqueeze_6: "bf16[16, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg55_1, 1);  arg55_1 = None
        unsqueeze_7: "bf16[1, 16, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 0);  unsqueeze_6 = None
        add_38: "bf16[128, 16, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_95, unsqueeze_7);  view_95 = unsqueeze_7 = None
        view_96: "bf16[2048, 8, 49, 49]" = torch.ops.aten.view.default(add_38, _shape_param_120);  add_38 = _shape_param_120 = None
        convert_element_type_30: "f32[2048, 8, 49, 49]" = torch.ops.prims.convert_element_type.default(view_96, torch.float32);  view_96 = None
        amax_3: "f32[2048, 8, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_30, [-1], True)
        sub_12: "f32[2048, 8, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_30, amax_3);  convert_element_type_30 = amax_3 = None
        exp_3: "f32[2048, 8, 49, 49]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_4: "f32[2048, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[2048, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_31: "bf16[2048, 8, 49, 49]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        expand_14: "bf16[2048, 8, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_31, _shape_param_121);  convert_element_type_31 = _shape_param_121 = None
        view_97: "bf16[16384, 49, 49]" = torch.ops.aten.view.default(expand_14, _shape_param_122);  expand_14 = _shape_param_122 = None
        constant_pad_nd_14: "bf16[16384, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_97, _shape_param_123);  view_97 = _shape_param_123 = None
        expand_15: "bf16[2048, 8, 49, 32]" = torch.ops.aten.expand.default(getitem_29, _shape_param_124);  getitem_29 = _shape_param_124 = None
        clone_26: "bf16[2048, 8, 49, 32]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_98: "bf16[16384, 49, 32]" = torch.ops.aten.view.default(clone_26, _shape_param_125);  clone_26 = _shape_param_125 = None
        constant_pad_nd_15: "bf16[16384, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_98, _shape_param_126);  view_98 = _shape_param_126 = None
        bmm_7: "bf16[16384, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_14, constant_pad_nd_15);  constant_pad_nd_14 = constant_pad_nd_15 = None
        slice_12: "bf16[16384, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_7, 1, 0, -7);  bmm_7 = None
        view_99: "bf16[2048, 8, 49, 32]" = torch.ops.aten.view.default(slice_12, _shape_param_127);  slice_12 = _shape_param_127 = None
        permute_38: "bf16[2048, 49, 8, 32]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        clone_27: "bf16[2048, 49, 8, 32]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_100: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(clone_27, _shape_param_128);  clone_27 = _shape_param_128 = None
        view_101: "bf16[100352, 256]" = torch.ops.aten.view.default(view_100, _shape_param_129);  view_100 = _shape_param_129 = None
        permute_39: "bf16[256, 256]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_13: "bf16[100352, 256]" = torch.ops.aten.addmm.default(arg57_1, view_101, permute_39);  arg57_1 = view_101 = permute_39 = None
        view_102: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(addmm_13, _shape_param_130);  addmm_13 = _shape_param_130 = None
        view_103: "bf16[2048, 7, 7, 256]" = torch.ops.aten.view.default(view_102, _shape_param_131);  view_102 = _shape_param_131 = None
        view_104: "bf16[128, 4, 4, 7, 7, 256]" = torch.ops.aten.view.default(view_103, _shape_param_132);  view_103 = _shape_param_132 = None
        permute_40: "bf16[128, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_104, [0, 1, 3, 2, 4, 5]);  view_104 = None
        clone_28: "bf16[128, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_105: "bf16[128, 28, 28, 256]" = torch.ops.aten.view.default(clone_28, _shape_param_133);  clone_28 = _shape_param_133 = None
        iota_6: "i64[28]" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_39: "i64[28]" = torch.ops.aten.add.Tensor(iota_6, 25);  iota_6 = None
        fmod_6: "i64[28]" = torch.ops.aten.fmod.Scalar(add_39, 28);  add_39 = None
        index_10: "bf16[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(view_105, [None, fmod_6]);  view_105 = fmod_6 = None
        iota_7: "i64[28]" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_40: "i64[28]" = torch.ops.aten.add.Tensor(iota_7, 25);  iota_7 = None
        fmod_7: "i64[28]" = torch.ops.aten.fmod.Scalar(add_40, 28);  add_40 = None
        index_11: "bf16[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(index_10, [None, None, fmod_7]);  index_10 = fmod_7 = None
        add_41: "bf16[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_83, index_11);  view_83 = index_11 = None
        view_106: "bf16[128, 784, 256]" = torch.ops.aten.view.default(add_41, _shape_param_134);  add_41 = _shape_param_134 = None
        convert_element_type_32: "f32[128, 784, 256]" = torch.ops.prims.convert_element_type.default(view_106, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_32, [2], correction = 0, keepdim = True)
        getitem_30: "f32[128, 784, 1]" = var_mean_9[0]
        getitem_31: "f32[128, 784, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_13: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_32, getitem_31);  convert_element_type_32 = getitem_31 = None
        add_42: "f32[128, 784, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_9: "f32[128, 784, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_31: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_9);  sub_13 = rsqrt_9 = None
        mul_32: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_31, arg58_1);  mul_31 = arg58_1 = None
        add_43: "f32[128, 784, 256]" = torch.ops.aten.add.Tensor(mul_32, arg59_1);  mul_32 = arg59_1 = None
        convert_element_type_33: "bf16[128, 784, 256]" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None
        view_107: "bf16[100352, 256]" = torch.ops.aten.view.default(convert_element_type_33, _shape_param_135);  convert_element_type_33 = _shape_param_135 = None
        permute_41: "bf16[256, 1024]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
        addmm_14: "bf16[100352, 1024]" = torch.ops.aten.addmm.default(arg61_1, view_107, permute_41);  arg61_1 = view_107 = permute_41 = None
        view_108: "bf16[128, 784, 1024]" = torch.ops.aten.view.default(addmm_14, _shape_param_136);  addmm_14 = _shape_param_136 = None
        convert_element_type_34: "f32[128, 784, 1024]" = torch.ops.prims.convert_element_type.default(view_108, torch.float32);  view_108 = None
        mul_33: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.5)
        mul_34: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.7071067811865476);  convert_element_type_34 = None
        erf_3: "f32[128, 784, 1024]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
        add_44: "f32[128, 784, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_35: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_33, add_44);  mul_33 = add_44 = None
        convert_element_type_35: "bf16[128, 784, 1024]" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None
        view_109: "bf16[100352, 1024]" = torch.ops.aten.view.default(convert_element_type_35, _shape_param_137);  convert_element_type_35 = _shape_param_137 = None
        permute_42: "bf16[1024, 256]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
        addmm_15: "bf16[100352, 256]" = torch.ops.aten.addmm.default(arg63_1, view_109, permute_42);  arg63_1 = view_109 = permute_42 = None
        view_110: "bf16[128, 784, 256]" = torch.ops.aten.view.default(addmm_15, _shape_param_138);  addmm_15 = _shape_param_138 = None
        add_45: "bf16[128, 784, 256]" = torch.ops.aten.add.Tensor(view_106, view_110);  view_106 = view_110 = None
        view_111: "bf16[128, 28, 28, 256]" = torch.ops.aten.view.default(add_45, _shape_param_139);  add_45 = _shape_param_139 = None
        view_112: "bf16[128, 14, 2, 14, 2, 256]" = torch.ops.aten.view.default(view_111, _shape_param_140);  view_111 = _shape_param_140 = None
        permute_43: "bf16[128, 14, 14, 2, 2, 256]" = torch.ops.aten.permute.default(view_112, [0, 1, 3, 4, 2, 5]);  view_112 = None
        clone_29: "bf16[128, 14, 14, 2, 2, 256]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
        view_113: "bf16[128, 14, 14, 1024]" = torch.ops.aten.view.default(clone_29, _shape_param_141);  clone_29 = _shape_param_141 = None
        convert_element_type_36: "f32[128, 14, 14, 1024]" = torch.ops.prims.convert_element_type.default(view_113, torch.float32);  view_113 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_36, [3], correction = 0, keepdim = True)
        getitem_32: "f32[128, 14, 14, 1]" = var_mean_10[0]
        getitem_33: "f32[128, 14, 14, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_14: "f32[128, 14, 14, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_36, getitem_33);  convert_element_type_36 = getitem_33 = None
        add_46: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_10: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_36: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_10);  sub_14 = rsqrt_10 = None
        mul_37: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(mul_36, arg64_1);  mul_36 = arg64_1 = None
        add_47: "f32[128, 14, 14, 1024]" = torch.ops.aten.add.Tensor(mul_37, arg65_1);  mul_37 = arg65_1 = None
        convert_element_type_37: "bf16[128, 14, 14, 1024]" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None
        view_114: "bf16[25088, 1024]" = torch.ops.aten.view.default(convert_element_type_37, _shape_param_142);  convert_element_type_37 = _shape_param_142 = None
        permute_44: "bf16[1024, 512]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
        mm_1: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_114, permute_44);  view_114 = permute_44 = None
        view_115: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(mm_1, _shape_param_143);  mm_1 = _shape_param_143 = None
        convert_element_type_38: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_115, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_38, [3], correction = 0, keepdim = True)
        getitem_34: "f32[128, 14, 14, 1]" = var_mean_11[0]
        getitem_35: "f32[128, 14, 14, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_15: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_38, getitem_35);  convert_element_type_38 = getitem_35 = None
        add_48: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_11: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_38: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = rsqrt_11 = None
        mul_39: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_38, arg67_1);  mul_38 = arg67_1 = None
        add_49: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_39, arg68_1);  mul_39 = arg68_1 = None
        convert_element_type_39: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16);  add_49 = None
        view_116: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(convert_element_type_39, _shape_param_144);  convert_element_type_39 = _shape_param_144 = None
        permute_45: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_116, [0, 1, 3, 2, 4, 5]);  view_116 = None
        clone_30: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_117: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_30, _shape_param_145);  clone_30 = _shape_param_145 = None
        view_118: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_117, _shape_param_146);  view_117 = _shape_param_146 = None
        view_119: "bf16[25088, 512]" = torch.ops.aten.view.default(view_118, _shape_param_147);  view_118 = _shape_param_147 = None
        permute_46: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_16: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg70_1, view_119, permute_46);  arg70_1 = view_119 = permute_46 = None
        view_120: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_16, _shape_param_148);  addmm_16 = _shape_param_148 = None
        view_121: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_120, _shape_param_149);  view_120 = _shape_param_149 = None
        permute_47: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_121, [2, 0, 3, 1, 4]);  view_121 = None
        unbind_4 = torch.ops.aten.unbind.int(permute_47);  permute_47 = None
        getitem_36: "bf16[512, 16, 49, 32]" = unbind_4[0]
        getitem_37: "bf16[512, 16, 49, 32]" = unbind_4[1]
        getitem_38: "bf16[512, 16, 49, 32]" = unbind_4[2];  unbind_4 = None
        mul_40: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_36, 0.1767766952966369);  getitem_36 = None
        expand_16: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_40, _shape_param_150);  mul_40 = _shape_param_150 = None
        clone_31: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_122: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_31, _shape_param_151);  clone_31 = _shape_param_151 = None
        constant_pad_nd_16: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_122, _shape_param_152);  view_122 = _shape_param_152 = None
        permute_48: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_37, [0, 1, 3, 2]);  getitem_37 = None
        expand_17: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_48, _shape_param_153);  permute_48 = _shape_param_153 = None
        clone_32: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_123: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_32, _shape_param_154);  clone_32 = _shape_param_154 = None
        constant_pad_nd_17: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_123, _shape_param_155);  view_123 = _shape_param_155 = None
        bmm_8: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_16, constant_pad_nd_17);  constant_pad_nd_16 = constant_pad_nd_17 = None
        slice_13: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_8, 1, 0, -7);  bmm_8 = None
        slice_14: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_13, 2, 0, -7);  slice_13 = None
        view_124: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_14, _shape_param_156);  slice_14 = _shape_param_156 = None
        view_125: "i64[2401]" = torch.ops.aten.view.default(arg71_1, [-1]);  arg71_1 = None
        index_12: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg72_1, [view_125]);  arg72_1 = view_125 = None
        view_126: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_12, _shape_param_157);  index_12 = _shape_param_157 = None
        permute_49: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_126, [2, 0, 1]);  view_126 = None
        clone_33: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        unsqueeze_8: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_33, 0);  clone_33 = None
        add_50: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_124, unsqueeze_8);  view_124 = unsqueeze_8 = None
        convert_element_type_40: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(add_50, torch.float32);  add_50 = None
        amax_4: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_40, [-1], True)
        sub_16: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_40, amax_4);  convert_element_type_40 = amax_4 = None
        exp_4: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_5: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_41: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        expand_18: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_41, _shape_param_158);  convert_element_type_41 = _shape_param_158 = None
        view_127: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_18, _shape_param_159);  expand_18 = _shape_param_159 = None
        constant_pad_nd_18: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_127, _shape_param_160);  view_127 = _shape_param_160 = None
        expand_19: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_38, _shape_param_161);  getitem_38 = _shape_param_161 = None
        clone_34: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_128: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_34, _shape_param_162);  clone_34 = _shape_param_162 = None
        constant_pad_nd_19: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_128, _shape_param_163);  view_128 = _shape_param_163 = None
        bmm_9: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_18, constant_pad_nd_19);  constant_pad_nd_18 = constant_pad_nd_19 = None
        slice_15: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_9, 1, 0, -7);  bmm_9 = None
        view_129: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_15, _shape_param_164);  slice_15 = _shape_param_164 = None
        permute_50: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        clone_35: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
        view_130: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_35, _shape_param_165);  clone_35 = _shape_param_165 = None
        view_131: "bf16[25088, 512]" = torch.ops.aten.view.default(view_130, _shape_param_166);  view_130 = _shape_param_166 = None
        permute_51: "bf16[512, 512]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_17: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg74_1, view_131, permute_51);  arg74_1 = view_131 = permute_51 = None
        view_132: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_17, _shape_param_167);  addmm_17 = _shape_param_167 = None
        view_133: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_132, _shape_param_168);  view_132 = _shape_param_168 = None
        view_134: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_133, _shape_param_169);  view_133 = _shape_param_169 = None
        permute_52: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_134, [0, 1, 3, 2, 4, 5]);  view_134 = None
        clone_36: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
        view_135: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_36, _shape_param_170);  clone_36 = _shape_param_170 = None
        add_51: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_115, view_135);  view_115 = view_135 = None
        view_136: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_51, _shape_param_171);  add_51 = _shape_param_171 = None
        convert_element_type_42: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_136, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_42, [2], correction = 0, keepdim = True)
        getitem_39: "f32[128, 196, 1]" = var_mean_12[0]
        getitem_40: "f32[128, 196, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_17: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_42, getitem_40);  convert_element_type_42 = getitem_40 = None
        add_52: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_39, 1e-05);  getitem_39 = None
        rsqrt_12: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_41: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_12);  sub_17 = rsqrt_12 = None
        mul_42: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_41, arg75_1);  mul_41 = arg75_1 = None
        add_53: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_42, arg76_1);  mul_42 = arg76_1 = None
        convert_element_type_43: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16);  add_53 = None
        view_137: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_43, _shape_param_172);  convert_element_type_43 = _shape_param_172 = None
        permute_53: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_18: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg78_1, view_137, permute_53);  arg78_1 = view_137 = permute_53 = None
        view_138: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_18, _shape_param_173);  addmm_18 = _shape_param_173 = None
        convert_element_type_44: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_138, torch.float32);  view_138 = None
        mul_43: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 0.5)
        mul_44: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 0.7071067811865476);  convert_element_type_44 = None
        erf_4: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_44);  mul_44 = None
        add_54: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_45: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_43, add_54);  mul_43 = add_54 = None
        convert_element_type_45: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        view_139: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_45, _shape_param_174);  convert_element_type_45 = _shape_param_174 = None
        permute_54: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_19: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg80_1, view_139, permute_54);  arg80_1 = view_139 = permute_54 = None
        view_140: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_19, _shape_param_175);  addmm_19 = _shape_param_175 = None
        add_55: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_136, view_140);  view_136 = view_140 = None
        view_141: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_55, _shape_param_176);  add_55 = _shape_param_176 = None
        convert_element_type_46: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_141, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_46, [3], correction = 0, keepdim = True)
        getitem_41: "f32[128, 14, 14, 1]" = var_mean_13[0]
        getitem_42: "f32[128, 14, 14, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_18: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_46, getitem_42);  convert_element_type_46 = getitem_42 = None
        add_56: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-05);  getitem_41 = None
        rsqrt_13: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_46: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_13);  sub_18 = rsqrt_13 = None
        mul_47: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_46, arg81_1);  mul_46 = arg81_1 = None
        add_57: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_47, arg82_1);  mul_47 = arg82_1 = None
        convert_element_type_47: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16);  add_57 = None
        iota_8: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_58: "i64[14]" = torch.ops.aten.add.Tensor(iota_8, 3);  iota_8 = None
        fmod_8: "i64[14]" = torch.ops.aten.fmod.Scalar(add_58, 14);  add_58 = None
        index_13: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(convert_element_type_47, [None, fmod_8]);  convert_element_type_47 = fmod_8 = None
        iota_9: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_59: "i64[14]" = torch.ops.aten.add.Tensor(iota_9, 3);  iota_9 = None
        fmod_9: "i64[14]" = torch.ops.aten.fmod.Scalar(add_59, 14);  add_59 = None
        index_14: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_13, [None, None, fmod_9]);  index_13 = fmod_9 = None
        view_142: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_14, _shape_param_177);  index_14 = _shape_param_177 = None
        permute_55: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_142, [0, 1, 3, 2, 4, 5]);  view_142 = None
        clone_37: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_143: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_37, _shape_param_178);  clone_37 = _shape_param_178 = None
        view_144: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_143, _shape_param_179);  view_143 = _shape_param_179 = None
        view_145: "bf16[25088, 512]" = torch.ops.aten.view.default(view_144, _shape_param_180);  view_144 = _shape_param_180 = None
        permute_56: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_20: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg84_1, view_145, permute_56);  arg84_1 = view_145 = permute_56 = None
        view_146: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_20, _shape_param_181);  addmm_20 = _shape_param_181 = None
        view_147: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_146, _shape_param_182);  view_146 = _shape_param_182 = None
        permute_57: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_147, [2, 0, 3, 1, 4]);  view_147 = None
        unbind_5 = torch.ops.aten.unbind.int(permute_57);  permute_57 = None
        getitem_43: "bf16[512, 16, 49, 32]" = unbind_5[0]
        getitem_44: "bf16[512, 16, 49, 32]" = unbind_5[1]
        getitem_45: "bf16[512, 16, 49, 32]" = unbind_5[2];  unbind_5 = None
        mul_48: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_43, 0.1767766952966369);  getitem_43 = None
        expand_20: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_48, _shape_param_183);  mul_48 = _shape_param_183 = None
        clone_38: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_148: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_38, _shape_param_184);  clone_38 = _shape_param_184 = None
        constant_pad_nd_20: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_148, _shape_param_185);  view_148 = _shape_param_185 = None
        permute_58: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_44, [0, 1, 3, 2]);  getitem_44 = None
        expand_21: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_58, _shape_param_186);  permute_58 = _shape_param_186 = None
        clone_39: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_149: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_39, _shape_param_187);  clone_39 = _shape_param_187 = None
        constant_pad_nd_21: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_149, _shape_param_188);  view_149 = _shape_param_188 = None
        bmm_10: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_20, constant_pad_nd_21);  constant_pad_nd_20 = constant_pad_nd_21 = None
        slice_16: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_10, 1, 0, -7);  bmm_10 = None
        slice_17: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_16, 2, 0, -7);  slice_16 = None
        view_150: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_17, _shape_param_189);  slice_17 = _shape_param_189 = None
        view_151: "i64[2401]" = torch.ops.aten.view.default(arg85_1, [-1]);  arg85_1 = None
        index_15: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg86_1, [view_151]);  arg86_1 = view_151 = None
        view_152: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_15, _shape_param_190);  index_15 = _shape_param_190 = None
        permute_59: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_152, [2, 0, 1]);  view_152 = None
        clone_40: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        unsqueeze_9: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_40, 0);  clone_40 = None
        add_60: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_150, unsqueeze_9);  view_150 = unsqueeze_9 = None
        view_153: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_60, _shape_param_191);  add_60 = _shape_param_191 = None
        unsqueeze_10: "bf16[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg87_1, 1);  arg87_1 = None
        unsqueeze_11: "bf16[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 0);  unsqueeze_10 = None
        add_61: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_153, unsqueeze_11);  view_153 = unsqueeze_11 = None
        view_154: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(add_61, _shape_param_192);  add_61 = _shape_param_192 = None
        convert_element_type_48: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_154, torch.float32);  view_154 = None
        amax_5: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_48, [-1], True)
        sub_19: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_48, amax_5);  convert_element_type_48 = amax_5 = None
        exp_5: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_6: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_49: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        expand_22: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_49, _shape_param_193);  convert_element_type_49 = _shape_param_193 = None
        view_155: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_22, _shape_param_194);  expand_22 = _shape_param_194 = None
        constant_pad_nd_22: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_155, _shape_param_195);  view_155 = _shape_param_195 = None
        expand_23: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_45, _shape_param_196);  getitem_45 = _shape_param_196 = None
        clone_41: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_156: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_41, _shape_param_197);  clone_41 = _shape_param_197 = None
        constant_pad_nd_23: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_156, _shape_param_198);  view_156 = _shape_param_198 = None
        bmm_11: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_22, constant_pad_nd_23);  constant_pad_nd_22 = constant_pad_nd_23 = None
        slice_18: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_11, 1, 0, -7);  bmm_11 = None
        view_157: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_18, _shape_param_199);  slice_18 = _shape_param_199 = None
        permute_60: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
        clone_42: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_158: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_42, _shape_param_200);  clone_42 = _shape_param_200 = None
        view_159: "bf16[25088, 512]" = torch.ops.aten.view.default(view_158, _shape_param_201);  view_158 = _shape_param_201 = None
        permute_61: "bf16[512, 512]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_21: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg89_1, view_159, permute_61);  arg89_1 = view_159 = permute_61 = None
        view_160: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_21, _shape_param_202);  addmm_21 = _shape_param_202 = None
        view_161: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_160, _shape_param_203);  view_160 = _shape_param_203 = None
        view_162: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_161, _shape_param_204);  view_161 = _shape_param_204 = None
        permute_62: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_162, [0, 1, 3, 2, 4, 5]);  view_162 = None
        clone_43: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_163: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_43, _shape_param_205);  clone_43 = _shape_param_205 = None
        iota_10: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_62: "i64[14]" = torch.ops.aten.add.Tensor(iota_10, 11);  iota_10 = None
        fmod_10: "i64[14]" = torch.ops.aten.fmod.Scalar(add_62, 14);  add_62 = None
        index_16: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_163, [None, fmod_10]);  view_163 = fmod_10 = None
        iota_11: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_63: "i64[14]" = torch.ops.aten.add.Tensor(iota_11, 11);  iota_11 = None
        fmod_11: "i64[14]" = torch.ops.aten.fmod.Scalar(add_63, 14);  add_63 = None
        index_17: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_16, [None, None, fmod_11]);  index_16 = fmod_11 = None
        add_64: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_141, index_17);  view_141 = index_17 = None
        view_164: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_64, _shape_param_206);  add_64 = _shape_param_206 = None
        convert_element_type_50: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_164, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_50, [2], correction = 0, keepdim = True)
        getitem_46: "f32[128, 196, 1]" = var_mean_14[0]
        getitem_47: "f32[128, 196, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_20: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_50, getitem_47);  convert_element_type_50 = getitem_47 = None
        add_65: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_14: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        mul_49: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_14);  sub_20 = rsqrt_14 = None
        mul_50: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_49, arg90_1);  mul_49 = arg90_1 = None
        add_66: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_50, arg91_1);  mul_50 = arg91_1 = None
        convert_element_type_51: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_66, torch.bfloat16);  add_66 = None
        view_165: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_51, _shape_param_207);  convert_element_type_51 = _shape_param_207 = None
        permute_63: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        addmm_22: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg93_1, view_165, permute_63);  arg93_1 = view_165 = permute_63 = None
        view_166: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_22, _shape_param_208);  addmm_22 = _shape_param_208 = None
        convert_element_type_52: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_166, torch.float32);  view_166 = None
        mul_51: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 0.5)
        mul_52: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 0.7071067811865476);  convert_element_type_52 = None
        erf_5: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
        add_67: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_53: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_51, add_67);  mul_51 = add_67 = None
        convert_element_type_53: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16);  mul_53 = None
        view_167: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_53, _shape_param_209);  convert_element_type_53 = _shape_param_209 = None
        permute_64: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
        addmm_23: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg95_1, view_167, permute_64);  arg95_1 = view_167 = permute_64 = None
        view_168: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_23, _shape_param_210);  addmm_23 = _shape_param_210 = None
        add_68: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_164, view_168);  view_164 = view_168 = None
        view_169: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_68, _shape_param_211);  add_68 = _shape_param_211 = None
        convert_element_type_54: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_169, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_54, [3], correction = 0, keepdim = True)
        getitem_48: "f32[128, 14, 14, 1]" = var_mean_15[0]
        getitem_49: "f32[128, 14, 14, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_21: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_54, getitem_49);  convert_element_type_54 = getitem_49 = None
        add_69: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_15: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        mul_54: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_15);  sub_21 = rsqrt_15 = None
        mul_55: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_54, arg96_1);  mul_54 = arg96_1 = None
        add_70: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_55, arg97_1);  mul_55 = arg97_1 = None
        convert_element_type_55: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_70, torch.bfloat16);  add_70 = None
        view_170: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(convert_element_type_55, _shape_param_212);  convert_element_type_55 = _shape_param_212 = None
        permute_65: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_170, [0, 1, 3, 2, 4, 5]);  view_170 = None
        clone_44: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
        view_171: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_44, _shape_param_213);  clone_44 = _shape_param_213 = None
        view_172: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_171, _shape_param_214);  view_171 = _shape_param_214 = None
        view_173: "bf16[25088, 512]" = torch.ops.aten.view.default(view_172, _shape_param_215);  view_172 = _shape_param_215 = None
        permute_66: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
        addmm_24: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg99_1, view_173, permute_66);  arg99_1 = view_173 = permute_66 = None
        view_174: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_24, _shape_param_216);  addmm_24 = _shape_param_216 = None
        view_175: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_174, _shape_param_217);  view_174 = _shape_param_217 = None
        permute_67: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_175, [2, 0, 3, 1, 4]);  view_175 = None
        unbind_6 = torch.ops.aten.unbind.int(permute_67);  permute_67 = None
        getitem_50: "bf16[512, 16, 49, 32]" = unbind_6[0]
        getitem_51: "bf16[512, 16, 49, 32]" = unbind_6[1]
        getitem_52: "bf16[512, 16, 49, 32]" = unbind_6[2];  unbind_6 = None
        mul_56: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_50, 0.1767766952966369);  getitem_50 = None
        expand_24: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_56, _shape_param_218);  mul_56 = _shape_param_218 = None
        clone_45: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_176: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_45, _shape_param_219);  clone_45 = _shape_param_219 = None
        constant_pad_nd_24: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_176, _shape_param_220);  view_176 = _shape_param_220 = None
        permute_68: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_51, [0, 1, 3, 2]);  getitem_51 = None
        expand_25: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_68, _shape_param_221);  permute_68 = _shape_param_221 = None
        clone_46: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_177: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_46, _shape_param_222);  clone_46 = _shape_param_222 = None
        constant_pad_nd_25: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_177, _shape_param_223);  view_177 = _shape_param_223 = None
        bmm_12: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_24, constant_pad_nd_25);  constant_pad_nd_24 = constant_pad_nd_25 = None
        slice_19: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_12, 1, 0, -7);  bmm_12 = None
        slice_20: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_19, 2, 0, -7);  slice_19 = None
        view_178: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_20, _shape_param_224);  slice_20 = _shape_param_224 = None
        view_179: "i64[2401]" = torch.ops.aten.view.default(arg100_1, [-1]);  arg100_1 = None
        index_18: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg101_1, [view_179]);  arg101_1 = view_179 = None
        view_180: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_18, _shape_param_225);  index_18 = _shape_param_225 = None
        permute_69: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_180, [2, 0, 1]);  view_180 = None
        clone_47: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
        unsqueeze_12: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_47, 0);  clone_47 = None
        add_71: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_178, unsqueeze_12);  view_178 = unsqueeze_12 = None
        convert_element_type_56: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(add_71, torch.float32);  add_71 = None
        amax_6: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_56, [-1], True)
        sub_22: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_56, amax_6);  convert_element_type_56 = amax_6 = None
        exp_6: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_7: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_57: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        expand_26: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_57, _shape_param_226);  convert_element_type_57 = _shape_param_226 = None
        view_181: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_26, _shape_param_227);  expand_26 = _shape_param_227 = None
        constant_pad_nd_26: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_181, _shape_param_228);  view_181 = _shape_param_228 = None
        expand_27: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_52, _shape_param_229);  getitem_52 = _shape_param_229 = None
        clone_48: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_182: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_48, _shape_param_230);  clone_48 = _shape_param_230 = None
        constant_pad_nd_27: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_182, _shape_param_231);  view_182 = _shape_param_231 = None
        bmm_13: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_26, constant_pad_nd_27);  constant_pad_nd_26 = constant_pad_nd_27 = None
        slice_21: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_13, 1, 0, -7);  bmm_13 = None
        view_183: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_21, _shape_param_232);  slice_21 = _shape_param_232 = None
        permute_70: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        clone_49: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_184: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_49, _shape_param_233);  clone_49 = _shape_param_233 = None
        view_185: "bf16[25088, 512]" = torch.ops.aten.view.default(view_184, _shape_param_234);  view_184 = _shape_param_234 = None
        permute_71: "bf16[512, 512]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
        addmm_25: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg103_1, view_185, permute_71);  arg103_1 = view_185 = permute_71 = None
        view_186: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_25, _shape_param_235);  addmm_25 = _shape_param_235 = None
        view_187: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_186, _shape_param_236);  view_186 = _shape_param_236 = None
        view_188: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_187, _shape_param_237);  view_187 = _shape_param_237 = None
        permute_72: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_188, [0, 1, 3, 2, 4, 5]);  view_188 = None
        clone_50: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
        view_189: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_50, _shape_param_238);  clone_50 = _shape_param_238 = None
        add_72: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_169, view_189);  view_169 = view_189 = None
        view_190: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_72, _shape_param_239);  add_72 = _shape_param_239 = None
        convert_element_type_58: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_190, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_58, [2], correction = 0, keepdim = True)
        getitem_53: "f32[128, 196, 1]" = var_mean_16[0]
        getitem_54: "f32[128, 196, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_23: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_58, getitem_54);  convert_element_type_58 = getitem_54 = None
        add_73: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_16: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        mul_57: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_16);  sub_23 = rsqrt_16 = None
        mul_58: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_57, arg104_1);  mul_57 = arg104_1 = None
        add_74: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_58, arg105_1);  mul_58 = arg105_1 = None
        convert_element_type_59: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None
        view_191: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_59, _shape_param_240);  convert_element_type_59 = _shape_param_240 = None
        permute_73: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
        addmm_26: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg107_1, view_191, permute_73);  arg107_1 = view_191 = permute_73 = None
        view_192: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_26, _shape_param_241);  addmm_26 = _shape_param_241 = None
        convert_element_type_60: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_192, torch.float32);  view_192 = None
        mul_59: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 0.5)
        mul_60: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 0.7071067811865476);  convert_element_type_60 = None
        erf_6: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
        add_75: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_61: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_59, add_75);  mul_59 = add_75 = None
        convert_element_type_61: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None
        view_193: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_61, _shape_param_242);  convert_element_type_61 = _shape_param_242 = None
        permute_74: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        addmm_27: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg109_1, view_193, permute_74);  arg109_1 = view_193 = permute_74 = None
        view_194: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_27, _shape_param_243);  addmm_27 = _shape_param_243 = None
        add_76: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_190, view_194);  view_190 = view_194 = None
        view_195: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_76, _shape_param_244);  add_76 = _shape_param_244 = None
        convert_element_type_62: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_195, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_62, [3], correction = 0, keepdim = True)
        getitem_55: "f32[128, 14, 14, 1]" = var_mean_17[0]
        getitem_56: "f32[128, 14, 14, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_24: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_62, getitem_56);  convert_element_type_62 = getitem_56 = None
        add_77: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_17: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        mul_62: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_17);  sub_24 = rsqrt_17 = None
        mul_63: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_62, arg110_1);  mul_62 = arg110_1 = None
        add_78: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_63, arg111_1);  mul_63 = arg111_1 = None
        convert_element_type_63: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16);  add_78 = None
        iota_12: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_79: "i64[14]" = torch.ops.aten.add.Tensor(iota_12, 3);  iota_12 = None
        fmod_12: "i64[14]" = torch.ops.aten.fmod.Scalar(add_79, 14);  add_79 = None
        index_19: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(convert_element_type_63, [None, fmod_12]);  convert_element_type_63 = fmod_12 = None
        iota_13: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_80: "i64[14]" = torch.ops.aten.add.Tensor(iota_13, 3);  iota_13 = None
        fmod_13: "i64[14]" = torch.ops.aten.fmod.Scalar(add_80, 14);  add_80 = None
        index_20: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_19, [None, None, fmod_13]);  index_19 = fmod_13 = None
        view_196: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_20, _shape_param_245);  index_20 = _shape_param_245 = None
        permute_75: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_196, [0, 1, 3, 2, 4, 5]);  view_196 = None
        clone_51: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
        view_197: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_51, _shape_param_246);  clone_51 = _shape_param_246 = None
        view_198: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_197, _shape_param_247);  view_197 = _shape_param_247 = None
        view_199: "bf16[25088, 512]" = torch.ops.aten.view.default(view_198, _shape_param_248);  view_198 = _shape_param_248 = None
        permute_76: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
        addmm_28: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg113_1, view_199, permute_76);  arg113_1 = view_199 = permute_76 = None
        view_200: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_28, _shape_param_249);  addmm_28 = _shape_param_249 = None
        view_201: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_200, _shape_param_250);  view_200 = _shape_param_250 = None
        permute_77: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_201, [2, 0, 3, 1, 4]);  view_201 = None
        unbind_7 = torch.ops.aten.unbind.int(permute_77);  permute_77 = None
        getitem_57: "bf16[512, 16, 49, 32]" = unbind_7[0]
        getitem_58: "bf16[512, 16, 49, 32]" = unbind_7[1]
        getitem_59: "bf16[512, 16, 49, 32]" = unbind_7[2];  unbind_7 = None
        mul_64: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_57, 0.1767766952966369);  getitem_57 = None
        expand_28: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_64, _shape_param_251);  mul_64 = _shape_param_251 = None
        clone_52: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_202: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_52, _shape_param_252);  clone_52 = _shape_param_252 = None
        constant_pad_nd_28: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_202, _shape_param_253);  view_202 = _shape_param_253 = None
        permute_78: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_58, [0, 1, 3, 2]);  getitem_58 = None
        expand_29: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_78, _shape_param_254);  permute_78 = _shape_param_254 = None
        clone_53: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_203: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_53, _shape_param_255);  clone_53 = _shape_param_255 = None
        constant_pad_nd_29: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_203, _shape_param_256);  view_203 = _shape_param_256 = None
        bmm_14: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_28, constant_pad_nd_29);  constant_pad_nd_28 = constant_pad_nd_29 = None
        slice_22: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_14, 1, 0, -7);  bmm_14 = None
        slice_23: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_22, 2, 0, -7);  slice_22 = None
        view_204: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_23, _shape_param_257);  slice_23 = _shape_param_257 = None
        view_205: "i64[2401]" = torch.ops.aten.view.default(arg114_1, [-1]);  arg114_1 = None
        index_21: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg115_1, [view_205]);  arg115_1 = view_205 = None
        view_206: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_21, _shape_param_258);  index_21 = _shape_param_258 = None
        permute_79: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_206, [2, 0, 1]);  view_206 = None
        clone_54: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
        unsqueeze_13: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_54, 0);  clone_54 = None
        add_81: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_204, unsqueeze_13);  view_204 = unsqueeze_13 = None
        view_207: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_81, _shape_param_259);  add_81 = _shape_param_259 = None
        unsqueeze_14: "bf16[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg116_1, 1);  arg116_1 = None
        unsqueeze_15: "bf16[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 0);  unsqueeze_14 = None
        add_82: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_207, unsqueeze_15);  view_207 = unsqueeze_15 = None
        view_208: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(add_82, _shape_param_260);  add_82 = _shape_param_260 = None
        convert_element_type_64: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_208, torch.float32);  view_208 = None
        amax_7: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_64, [-1], True)
        sub_25: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_64, amax_7);  convert_element_type_64 = amax_7 = None
        exp_7: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_8: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_65: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        expand_30: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_65, _shape_param_261);  convert_element_type_65 = _shape_param_261 = None
        view_209: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_30, _shape_param_262);  expand_30 = _shape_param_262 = None
        constant_pad_nd_30: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_209, _shape_param_263);  view_209 = _shape_param_263 = None
        expand_31: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_59, _shape_param_264);  getitem_59 = _shape_param_264 = None
        clone_55: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_210: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_55, _shape_param_265);  clone_55 = _shape_param_265 = None
        constant_pad_nd_31: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_210, _shape_param_266);  view_210 = _shape_param_266 = None
        bmm_15: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_30, constant_pad_nd_31);  constant_pad_nd_30 = constant_pad_nd_31 = None
        slice_24: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_15, 1, 0, -7);  bmm_15 = None
        view_211: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_24, _shape_param_267);  slice_24 = _shape_param_267 = None
        permute_80: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None
        clone_56: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_212: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_56, _shape_param_268);  clone_56 = _shape_param_268 = None
        view_213: "bf16[25088, 512]" = torch.ops.aten.view.default(view_212, _shape_param_269);  view_212 = _shape_param_269 = None
        permute_81: "bf16[512, 512]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_29: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg118_1, view_213, permute_81);  arg118_1 = view_213 = permute_81 = None
        view_214: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_29, _shape_param_270);  addmm_29 = _shape_param_270 = None
        view_215: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_214, _shape_param_271);  view_214 = _shape_param_271 = None
        view_216: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_215, _shape_param_272);  view_215 = _shape_param_272 = None
        permute_82: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_216, [0, 1, 3, 2, 4, 5]);  view_216 = None
        clone_57: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_217: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_57, _shape_param_273);  clone_57 = _shape_param_273 = None
        iota_14: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_83: "i64[14]" = torch.ops.aten.add.Tensor(iota_14, 11);  iota_14 = None
        fmod_14: "i64[14]" = torch.ops.aten.fmod.Scalar(add_83, 14);  add_83 = None
        index_22: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_217, [None, fmod_14]);  view_217 = fmod_14 = None
        iota_15: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_84: "i64[14]" = torch.ops.aten.add.Tensor(iota_15, 11);  iota_15 = None
        fmod_15: "i64[14]" = torch.ops.aten.fmod.Scalar(add_84, 14);  add_84 = None
        index_23: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_22, [None, None, fmod_15]);  index_22 = fmod_15 = None
        add_85: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_195, index_23);  view_195 = index_23 = None
        view_218: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_85, _shape_param_274);  add_85 = _shape_param_274 = None
        convert_element_type_66: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_218, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_66, [2], correction = 0, keepdim = True)
        getitem_60: "f32[128, 196, 1]" = var_mean_18[0]
        getitem_61: "f32[128, 196, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_26: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_66, getitem_61);  convert_element_type_66 = getitem_61 = None
        add_86: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_18: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_65: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_18);  sub_26 = rsqrt_18 = None
        mul_66: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_65, arg119_1);  mul_65 = arg119_1 = None
        add_87: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_66, arg120_1);  mul_66 = arg120_1 = None
        convert_element_type_67: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16);  add_87 = None
        view_219: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_67, _shape_param_275);  convert_element_type_67 = _shape_param_275 = None
        permute_83: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_30: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg122_1, view_219, permute_83);  arg122_1 = view_219 = permute_83 = None
        view_220: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_30, _shape_param_276);  addmm_30 = _shape_param_276 = None
        convert_element_type_68: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_220, torch.float32);  view_220 = None
        mul_67: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 0.5)
        mul_68: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 0.7071067811865476);  convert_element_type_68 = None
        erf_7: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_88: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_69: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_67, add_88);  mul_67 = add_88 = None
        convert_element_type_69: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        view_221: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_69, _shape_param_277);  convert_element_type_69 = _shape_param_277 = None
        permute_84: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_31: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg124_1, view_221, permute_84);  arg124_1 = view_221 = permute_84 = None
        view_222: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_31, _shape_param_278);  addmm_31 = _shape_param_278 = None
        add_89: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_218, view_222);  view_218 = view_222 = None
        view_223: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_89, _shape_param_279);  add_89 = _shape_param_279 = None
        convert_element_type_70: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_223, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_70, [3], correction = 0, keepdim = True)
        getitem_62: "f32[128, 14, 14, 1]" = var_mean_19[0]
        getitem_63: "f32[128, 14, 14, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_27: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_70, getitem_63);  convert_element_type_70 = getitem_63 = None
        add_90: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_19: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_70: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_19);  sub_27 = rsqrt_19 = None
        mul_71: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_70, arg125_1);  mul_70 = arg125_1 = None
        add_91: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_71, arg126_1);  mul_71 = arg126_1 = None
        convert_element_type_71: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16);  add_91 = None
        view_224: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(convert_element_type_71, _shape_param_280);  convert_element_type_71 = _shape_param_280 = None
        permute_85: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_224, [0, 1, 3, 2, 4, 5]);  view_224 = None
        clone_58: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
        view_225: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_58, _shape_param_281);  clone_58 = _shape_param_281 = None
        view_226: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_225, _shape_param_282);  view_225 = _shape_param_282 = None
        view_227: "bf16[25088, 512]" = torch.ops.aten.view.default(view_226, _shape_param_283);  view_226 = _shape_param_283 = None
        permute_86: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_32: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg128_1, view_227, permute_86);  arg128_1 = view_227 = permute_86 = None
        view_228: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_32, _shape_param_284);  addmm_32 = _shape_param_284 = None
        view_229: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_228, _shape_param_285);  view_228 = _shape_param_285 = None
        permute_87: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_229, [2, 0, 3, 1, 4]);  view_229 = None
        unbind_8 = torch.ops.aten.unbind.int(permute_87);  permute_87 = None
        getitem_64: "bf16[512, 16, 49, 32]" = unbind_8[0]
        getitem_65: "bf16[512, 16, 49, 32]" = unbind_8[1]
        getitem_66: "bf16[512, 16, 49, 32]" = unbind_8[2];  unbind_8 = None
        mul_72: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_64, 0.1767766952966369);  getitem_64 = None
        expand_32: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_72, _shape_param_286);  mul_72 = _shape_param_286 = None
        clone_59: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_230: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_59, _shape_param_287);  clone_59 = _shape_param_287 = None
        constant_pad_nd_32: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_230, _shape_param_288);  view_230 = _shape_param_288 = None
        permute_88: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_65, [0, 1, 3, 2]);  getitem_65 = None
        expand_33: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_88, _shape_param_289);  permute_88 = _shape_param_289 = None
        clone_60: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_231: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_60, _shape_param_290);  clone_60 = _shape_param_290 = None
        constant_pad_nd_33: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_231, _shape_param_291);  view_231 = _shape_param_291 = None
        bmm_16: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_32, constant_pad_nd_33);  constant_pad_nd_32 = constant_pad_nd_33 = None
        slice_25: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_16, 1, 0, -7);  bmm_16 = None
        slice_26: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_25, 2, 0, -7);  slice_25 = None
        view_232: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_26, _shape_param_292);  slice_26 = _shape_param_292 = None
        view_233: "i64[2401]" = torch.ops.aten.view.default(arg129_1, [-1]);  arg129_1 = None
        index_24: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg130_1, [view_233]);  arg130_1 = view_233 = None
        view_234: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_24, _shape_param_293);  index_24 = _shape_param_293 = None
        permute_89: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_234, [2, 0, 1]);  view_234 = None
        clone_61: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
        unsqueeze_16: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_61, 0);  clone_61 = None
        add_92: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_232, unsqueeze_16);  view_232 = unsqueeze_16 = None
        convert_element_type_72: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(add_92, torch.float32);  add_92 = None
        amax_8: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_72, [-1], True)
        sub_28: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_72, amax_8);  convert_element_type_72 = amax_8 = None
        exp_8: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_9: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_73: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        expand_34: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_73, _shape_param_294);  convert_element_type_73 = _shape_param_294 = None
        view_235: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_34, _shape_param_295);  expand_34 = _shape_param_295 = None
        constant_pad_nd_34: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_235, _shape_param_296);  view_235 = _shape_param_296 = None
        expand_35: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_66, _shape_param_297);  getitem_66 = _shape_param_297 = None
        clone_62: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_236: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_62, _shape_param_298);  clone_62 = _shape_param_298 = None
        constant_pad_nd_35: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_236, _shape_param_299);  view_236 = _shape_param_299 = None
        bmm_17: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_34, constant_pad_nd_35);  constant_pad_nd_34 = constant_pad_nd_35 = None
        slice_27: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_17, 1, 0, -7);  bmm_17 = None
        view_237: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_27, _shape_param_300);  slice_27 = _shape_param_300 = None
        permute_90: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None
        clone_63: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        view_238: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_63, _shape_param_301);  clone_63 = _shape_param_301 = None
        view_239: "bf16[25088, 512]" = torch.ops.aten.view.default(view_238, _shape_param_302);  view_238 = _shape_param_302 = None
        permute_91: "bf16[512, 512]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_33: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg132_1, view_239, permute_91);  arg132_1 = view_239 = permute_91 = None
        view_240: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_33, _shape_param_303);  addmm_33 = _shape_param_303 = None
        view_241: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_240, _shape_param_304);  view_240 = _shape_param_304 = None
        view_242: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_241, _shape_param_305);  view_241 = _shape_param_305 = None
        permute_92: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_242, [0, 1, 3, 2, 4, 5]);  view_242 = None
        clone_64: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_243: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_64, _shape_param_306);  clone_64 = _shape_param_306 = None
        add_93: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_223, view_243);  view_223 = view_243 = None
        view_244: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_93, _shape_param_307);  add_93 = _shape_param_307 = None
        convert_element_type_74: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_244, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_74, [2], correction = 0, keepdim = True)
        getitem_67: "f32[128, 196, 1]" = var_mean_20[0]
        getitem_68: "f32[128, 196, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_29: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_74, getitem_68);  convert_element_type_74 = getitem_68 = None
        add_94: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_67, 1e-05);  getitem_67 = None
        rsqrt_20: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_73: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_20);  sub_29 = rsqrt_20 = None
        mul_74: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_73, arg133_1);  mul_73 = arg133_1 = None
        add_95: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_74, arg134_1);  mul_74 = arg134_1 = None
        convert_element_type_75: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_95, torch.bfloat16);  add_95 = None
        view_245: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_75, _shape_param_308);  convert_element_type_75 = _shape_param_308 = None
        permute_93: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_34: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg136_1, view_245, permute_93);  arg136_1 = view_245 = permute_93 = None
        view_246: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_34, _shape_param_309);  addmm_34 = _shape_param_309 = None
        convert_element_type_76: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_246, torch.float32);  view_246 = None
        mul_75: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_76, 0.5)
        mul_76: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_76, 0.7071067811865476);  convert_element_type_76 = None
        erf_8: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_96: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_77: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_75, add_96);  mul_75 = add_96 = None
        convert_element_type_77: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None
        view_247: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_77, _shape_param_310);  convert_element_type_77 = _shape_param_310 = None
        permute_94: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_35: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg138_1, view_247, permute_94);  arg138_1 = view_247 = permute_94 = None
        view_248: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_35, _shape_param_311);  addmm_35 = _shape_param_311 = None
        add_97: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_244, view_248);  view_244 = view_248 = None
        view_249: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_97, _shape_param_312);  add_97 = _shape_param_312 = None
        convert_element_type_78: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_249, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_78, [3], correction = 0, keepdim = True)
        getitem_69: "f32[128, 14, 14, 1]" = var_mean_21[0]
        getitem_70: "f32[128, 14, 14, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_30: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_78, getitem_70);  convert_element_type_78 = getitem_70 = None
        add_98: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_69, 1e-05);  getitem_69 = None
        rsqrt_21: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        mul_78: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_21);  sub_30 = rsqrt_21 = None
        mul_79: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_78, arg139_1);  mul_78 = arg139_1 = None
        add_99: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_79, arg140_1);  mul_79 = arg140_1 = None
        convert_element_type_79: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None
        iota_16: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_100: "i64[14]" = torch.ops.aten.add.Tensor(iota_16, 3);  iota_16 = None
        fmod_16: "i64[14]" = torch.ops.aten.fmod.Scalar(add_100, 14);  add_100 = None
        index_25: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(convert_element_type_79, [None, fmod_16]);  convert_element_type_79 = fmod_16 = None
        iota_17: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_101: "i64[14]" = torch.ops.aten.add.Tensor(iota_17, 3);  iota_17 = None
        fmod_17: "i64[14]" = torch.ops.aten.fmod.Scalar(add_101, 14);  add_101 = None
        index_26: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_25, [None, None, fmod_17]);  index_25 = fmod_17 = None
        view_250: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_26, _shape_param_313);  index_26 = _shape_param_313 = None
        permute_95: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_250, [0, 1, 3, 2, 4, 5]);  view_250 = None
        clone_65: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_251: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_65, _shape_param_314);  clone_65 = _shape_param_314 = None
        view_252: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_251, _shape_param_315);  view_251 = _shape_param_315 = None
        view_253: "bf16[25088, 512]" = torch.ops.aten.view.default(view_252, _shape_param_316);  view_252 = _shape_param_316 = None
        permute_96: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_36: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg142_1, view_253, permute_96);  arg142_1 = view_253 = permute_96 = None
        view_254: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_36, _shape_param_317);  addmm_36 = _shape_param_317 = None
        view_255: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_254, _shape_param_318);  view_254 = _shape_param_318 = None
        permute_97: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_255, [2, 0, 3, 1, 4]);  view_255 = None
        unbind_9 = torch.ops.aten.unbind.int(permute_97);  permute_97 = None
        getitem_71: "bf16[512, 16, 49, 32]" = unbind_9[0]
        getitem_72: "bf16[512, 16, 49, 32]" = unbind_9[1]
        getitem_73: "bf16[512, 16, 49, 32]" = unbind_9[2];  unbind_9 = None
        mul_80: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_71, 0.1767766952966369);  getitem_71 = None
        expand_36: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_80, _shape_param_319);  mul_80 = _shape_param_319 = None
        clone_66: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_256: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_66, _shape_param_320);  clone_66 = _shape_param_320 = None
        constant_pad_nd_36: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_256, _shape_param_321);  view_256 = _shape_param_321 = None
        permute_98: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_72, [0, 1, 3, 2]);  getitem_72 = None
        expand_37: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_98, _shape_param_322);  permute_98 = _shape_param_322 = None
        clone_67: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_257: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_67, _shape_param_323);  clone_67 = _shape_param_323 = None
        constant_pad_nd_37: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_257, _shape_param_324);  view_257 = _shape_param_324 = None
        bmm_18: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_36, constant_pad_nd_37);  constant_pad_nd_36 = constant_pad_nd_37 = None
        slice_28: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_18, 1, 0, -7);  bmm_18 = None
        slice_29: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_28, 2, 0, -7);  slice_28 = None
        view_258: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_29, _shape_param_325);  slice_29 = _shape_param_325 = None
        view_259: "i64[2401]" = torch.ops.aten.view.default(arg143_1, [-1]);  arg143_1 = None
        index_27: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg144_1, [view_259]);  arg144_1 = view_259 = None
        view_260: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_27, _shape_param_326);  index_27 = _shape_param_326 = None
        permute_99: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_260, [2, 0, 1]);  view_260 = None
        clone_68: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
        unsqueeze_17: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_68, 0);  clone_68 = None
        add_102: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_258, unsqueeze_17);  view_258 = unsqueeze_17 = None
        view_261: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_102, _shape_param_327);  add_102 = _shape_param_327 = None
        unsqueeze_18: "bf16[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg145_1, 1);  arg145_1 = None
        unsqueeze_19: "bf16[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 0);  unsqueeze_18 = None
        add_103: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_261, unsqueeze_19);  view_261 = unsqueeze_19 = None
        view_262: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(add_103, _shape_param_328);  add_103 = _shape_param_328 = None
        convert_element_type_80: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_262, torch.float32);  view_262 = None
        amax_9: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_80, [-1], True)
        sub_31: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_80, amax_9);  convert_element_type_80 = amax_9 = None
        exp_9: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_10: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_81: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        expand_38: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_81, _shape_param_329);  convert_element_type_81 = _shape_param_329 = None
        view_263: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_38, _shape_param_330);  expand_38 = _shape_param_330 = None
        constant_pad_nd_38: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_263, _shape_param_331);  view_263 = _shape_param_331 = None
        expand_39: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_73, _shape_param_332);  getitem_73 = _shape_param_332 = None
        clone_69: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_264: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_69, _shape_param_333);  clone_69 = _shape_param_333 = None
        constant_pad_nd_39: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_264, _shape_param_334);  view_264 = _shape_param_334 = None
        bmm_19: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_38, constant_pad_nd_39);  constant_pad_nd_38 = constant_pad_nd_39 = None
        slice_30: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_19, 1, 0, -7);  bmm_19 = None
        view_265: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_30, _shape_param_335);  slice_30 = _shape_param_335 = None
        permute_100: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
        clone_70: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
        view_266: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_70, _shape_param_336);  clone_70 = _shape_param_336 = None
        view_267: "bf16[25088, 512]" = torch.ops.aten.view.default(view_266, _shape_param_337);  view_266 = _shape_param_337 = None
        permute_101: "bf16[512, 512]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_37: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg147_1, view_267, permute_101);  arg147_1 = view_267 = permute_101 = None
        view_268: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_37, _shape_param_338);  addmm_37 = _shape_param_338 = None
        view_269: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_268, _shape_param_339);  view_268 = _shape_param_339 = None
        view_270: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_269, _shape_param_340);  view_269 = _shape_param_340 = None
        permute_102: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_270, [0, 1, 3, 2, 4, 5]);  view_270 = None
        clone_71: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        view_271: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_71, _shape_param_341);  clone_71 = _shape_param_341 = None
        iota_18: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_104: "i64[14]" = torch.ops.aten.add.Tensor(iota_18, 11);  iota_18 = None
        fmod_18: "i64[14]" = torch.ops.aten.fmod.Scalar(add_104, 14);  add_104 = None
        index_28: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_271, [None, fmod_18]);  view_271 = fmod_18 = None
        iota_19: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_105: "i64[14]" = torch.ops.aten.add.Tensor(iota_19, 11);  iota_19 = None
        fmod_19: "i64[14]" = torch.ops.aten.fmod.Scalar(add_105, 14);  add_105 = None
        index_29: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_28, [None, None, fmod_19]);  index_28 = fmod_19 = None
        add_106: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_249, index_29);  view_249 = index_29 = None
        view_272: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_106, _shape_param_342);  add_106 = _shape_param_342 = None
        convert_element_type_82: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_272, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_82, [2], correction = 0, keepdim = True)
        getitem_74: "f32[128, 196, 1]" = var_mean_22[0]
        getitem_75: "f32[128, 196, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_32: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_82, getitem_75);  convert_element_type_82 = getitem_75 = None
        add_107: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_22: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        mul_81: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_22);  sub_32 = rsqrt_22 = None
        mul_82: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_81, arg148_1);  mul_81 = arg148_1 = None
        add_108: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_82, arg149_1);  mul_82 = arg149_1 = None
        convert_element_type_83: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_108, torch.bfloat16);  add_108 = None
        view_273: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_83, _shape_param_343);  convert_element_type_83 = _shape_param_343 = None
        permute_103: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
        addmm_38: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg151_1, view_273, permute_103);  arg151_1 = view_273 = permute_103 = None
        view_274: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_38, _shape_param_344);  addmm_38 = _shape_param_344 = None
        convert_element_type_84: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_274, torch.float32);  view_274 = None
        mul_83: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_84, 0.5)
        mul_84: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_84, 0.7071067811865476);  convert_element_type_84 = None
        erf_9: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
        add_109: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_85: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_83, add_109);  mul_83 = add_109 = None
        convert_element_type_85: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_85, torch.bfloat16);  mul_85 = None
        view_275: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_85, _shape_param_345);  convert_element_type_85 = _shape_param_345 = None
        permute_104: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_39: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg153_1, view_275, permute_104);  arg153_1 = view_275 = permute_104 = None
        view_276: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_39, _shape_param_346);  addmm_39 = _shape_param_346 = None
        add_110: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_272, view_276);  view_272 = view_276 = None
        view_277: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_110, _shape_param_347);  add_110 = _shape_param_347 = None
        convert_element_type_86: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_277, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_86, [3], correction = 0, keepdim = True)
        getitem_76: "f32[128, 14, 14, 1]" = var_mean_23[0]
        getitem_77: "f32[128, 14, 14, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_33: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_86, getitem_77);  convert_element_type_86 = getitem_77 = None
        add_111: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_23: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        mul_86: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_23);  sub_33 = rsqrt_23 = None
        mul_87: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_86, arg154_1);  mul_86 = arg154_1 = None
        add_112: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_87, arg155_1);  mul_87 = arg155_1 = None
        convert_element_type_87: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16);  add_112 = None
        view_278: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(convert_element_type_87, _shape_param_348);  convert_element_type_87 = _shape_param_348 = None
        permute_105: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_278, [0, 1, 3, 2, 4, 5]);  view_278 = None
        clone_72: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        view_279: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_72, _shape_param_349);  clone_72 = _shape_param_349 = None
        view_280: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_279, _shape_param_350);  view_279 = _shape_param_350 = None
        view_281: "bf16[25088, 512]" = torch.ops.aten.view.default(view_280, _shape_param_351);  view_280 = _shape_param_351 = None
        permute_106: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
        addmm_40: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg157_1, view_281, permute_106);  arg157_1 = view_281 = permute_106 = None
        view_282: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_40, _shape_param_352);  addmm_40 = _shape_param_352 = None
        view_283: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_282, _shape_param_353);  view_282 = _shape_param_353 = None
        permute_107: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_283, [2, 0, 3, 1, 4]);  view_283 = None
        unbind_10 = torch.ops.aten.unbind.int(permute_107);  permute_107 = None
        getitem_78: "bf16[512, 16, 49, 32]" = unbind_10[0]
        getitem_79: "bf16[512, 16, 49, 32]" = unbind_10[1]
        getitem_80: "bf16[512, 16, 49, 32]" = unbind_10[2];  unbind_10 = None
        mul_88: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_78, 0.1767766952966369);  getitem_78 = None
        expand_40: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_88, _shape_param_354);  mul_88 = _shape_param_354 = None
        clone_73: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_284: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_73, _shape_param_355);  clone_73 = _shape_param_355 = None
        constant_pad_nd_40: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_284, _shape_param_356);  view_284 = _shape_param_356 = None
        permute_108: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_79, [0, 1, 3, 2]);  getitem_79 = None
        expand_41: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_108, _shape_param_357);  permute_108 = _shape_param_357 = None
        clone_74: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_285: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_74, _shape_param_358);  clone_74 = _shape_param_358 = None
        constant_pad_nd_41: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_285, _shape_param_359);  view_285 = _shape_param_359 = None
        bmm_20: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_40, constant_pad_nd_41);  constant_pad_nd_40 = constant_pad_nd_41 = None
        slice_31: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_20, 1, 0, -7);  bmm_20 = None
        slice_32: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_31, 2, 0, -7);  slice_31 = None
        view_286: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_32, _shape_param_360);  slice_32 = _shape_param_360 = None
        view_287: "i64[2401]" = torch.ops.aten.view.default(arg158_1, [-1]);  arg158_1 = None
        index_30: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg159_1, [view_287]);  arg159_1 = view_287 = None
        view_288: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_30, _shape_param_361);  index_30 = _shape_param_361 = None
        permute_109: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_288, [2, 0, 1]);  view_288 = None
        clone_75: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None
        unsqueeze_20: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_75, 0);  clone_75 = None
        add_113: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_286, unsqueeze_20);  view_286 = unsqueeze_20 = None
        convert_element_type_88: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(add_113, torch.float32);  add_113 = None
        amax_10: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_88, [-1], True)
        sub_34: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_88, amax_10);  convert_element_type_88 = amax_10 = None
        exp_10: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_11: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_89: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        expand_42: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_89, _shape_param_362);  convert_element_type_89 = _shape_param_362 = None
        view_289: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_42, _shape_param_363);  expand_42 = _shape_param_363 = None
        constant_pad_nd_42: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_289, _shape_param_364);  view_289 = _shape_param_364 = None
        expand_43: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_80, _shape_param_365);  getitem_80 = _shape_param_365 = None
        clone_76: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_290: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_76, _shape_param_366);  clone_76 = _shape_param_366 = None
        constant_pad_nd_43: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_290, _shape_param_367);  view_290 = _shape_param_367 = None
        bmm_21: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_42, constant_pad_nd_43);  constant_pad_nd_42 = constant_pad_nd_43 = None
        slice_33: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_21, 1, 0, -7);  bmm_21 = None
        view_291: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_33, _shape_param_368);  slice_33 = _shape_param_368 = None
        permute_110: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        clone_77: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_292: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_77, _shape_param_369);  clone_77 = _shape_param_369 = None
        view_293: "bf16[25088, 512]" = torch.ops.aten.view.default(view_292, _shape_param_370);  view_292 = _shape_param_370 = None
        permute_111: "bf16[512, 512]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        addmm_41: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg161_1, view_293, permute_111);  arg161_1 = view_293 = permute_111 = None
        view_294: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_41, _shape_param_371);  addmm_41 = _shape_param_371 = None
        view_295: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_294, _shape_param_372);  view_294 = _shape_param_372 = None
        view_296: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_295, _shape_param_373);  view_295 = _shape_param_373 = None
        permute_112: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_296, [0, 1, 3, 2, 4, 5]);  view_296 = None
        clone_78: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        view_297: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_78, _shape_param_374);  clone_78 = _shape_param_374 = None
        add_114: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_277, view_297);  view_277 = view_297 = None
        view_298: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_114, _shape_param_375);  add_114 = _shape_param_375 = None
        convert_element_type_90: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_298, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_90, [2], correction = 0, keepdim = True)
        getitem_81: "f32[128, 196, 1]" = var_mean_24[0]
        getitem_82: "f32[128, 196, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_35: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_90, getitem_82);  convert_element_type_90 = getitem_82 = None
        add_115: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_81, 1e-05);  getitem_81 = None
        rsqrt_24: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_89: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_24);  sub_35 = rsqrt_24 = None
        mul_90: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_89, arg162_1);  mul_89 = arg162_1 = None
        add_116: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_90, arg163_1);  mul_90 = arg163_1 = None
        convert_element_type_91: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_116, torch.bfloat16);  add_116 = None
        view_299: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_91, _shape_param_376);  convert_element_type_91 = _shape_param_376 = None
        permute_113: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_42: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg165_1, view_299, permute_113);  arg165_1 = view_299 = permute_113 = None
        view_300: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_42, _shape_param_377);  addmm_42 = _shape_param_377 = None
        convert_element_type_92: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_300, torch.float32);  view_300 = None
        mul_91: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_92, 0.5)
        mul_92: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_92, 0.7071067811865476);  convert_element_type_92 = None
        erf_10: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
        add_117: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_93: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_91, add_117);  mul_91 = add_117 = None
        convert_element_type_93: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16);  mul_93 = None
        view_301: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_93, _shape_param_378);  convert_element_type_93 = _shape_param_378 = None
        permute_114: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_43: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg167_1, view_301, permute_114);  arg167_1 = view_301 = permute_114 = None
        view_302: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_43, _shape_param_379);  addmm_43 = _shape_param_379 = None
        add_118: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_298, view_302);  view_298 = view_302 = None
        view_303: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_118, _shape_param_380);  add_118 = _shape_param_380 = None
        convert_element_type_94: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_303, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_94, [3], correction = 0, keepdim = True)
        getitem_83: "f32[128, 14, 14, 1]" = var_mean_25[0]
        getitem_84: "f32[128, 14, 14, 1]" = var_mean_25[1];  var_mean_25 = None
        sub_36: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_94, getitem_84);  convert_element_type_94 = getitem_84 = None
        add_119: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_83, 1e-05);  getitem_83 = None
        rsqrt_25: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        mul_94: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_25);  sub_36 = rsqrt_25 = None
        mul_95: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_94, arg168_1);  mul_94 = arg168_1 = None
        add_120: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_95, arg169_1);  mul_95 = arg169_1 = None
        convert_element_type_95: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_120, torch.bfloat16);  add_120 = None
        iota_20: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_121: "i64[14]" = torch.ops.aten.add.Tensor(iota_20, 3);  iota_20 = None
        fmod_20: "i64[14]" = torch.ops.aten.fmod.Scalar(add_121, 14);  add_121 = None
        index_31: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(convert_element_type_95, [None, fmod_20]);  convert_element_type_95 = fmod_20 = None
        iota_21: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_122: "i64[14]" = torch.ops.aten.add.Tensor(iota_21, 3);  iota_21 = None
        fmod_21: "i64[14]" = torch.ops.aten.fmod.Scalar(add_122, 14);  add_122 = None
        index_32: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_31, [None, None, fmod_21]);  index_31 = fmod_21 = None
        view_304: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_32, _shape_param_381);  index_32 = _shape_param_381 = None
        permute_115: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_304, [0, 1, 3, 2, 4, 5]);  view_304 = None
        clone_79: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_305: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_79, _shape_param_382);  clone_79 = _shape_param_382 = None
        view_306: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_305, _shape_param_383);  view_305 = _shape_param_383 = None
        view_307: "bf16[25088, 512]" = torch.ops.aten.view.default(view_306, _shape_param_384);  view_306 = _shape_param_384 = None
        permute_116: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_44: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg171_1, view_307, permute_116);  arg171_1 = view_307 = permute_116 = None
        view_308: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_44, _shape_param_385);  addmm_44 = _shape_param_385 = None
        view_309: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_308, _shape_param_386);  view_308 = _shape_param_386 = None
        permute_117: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_309, [2, 0, 3, 1, 4]);  view_309 = None
        unbind_11 = torch.ops.aten.unbind.int(permute_117);  permute_117 = None
        getitem_85: "bf16[512, 16, 49, 32]" = unbind_11[0]
        getitem_86: "bf16[512, 16, 49, 32]" = unbind_11[1]
        getitem_87: "bf16[512, 16, 49, 32]" = unbind_11[2];  unbind_11 = None
        mul_96: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_85, 0.1767766952966369);  getitem_85 = None
        expand_44: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_96, _shape_param_387);  mul_96 = _shape_param_387 = None
        clone_80: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_310: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_80, _shape_param_388);  clone_80 = _shape_param_388 = None
        constant_pad_nd_44: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_310, _shape_param_389);  view_310 = _shape_param_389 = None
        permute_118: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_86, [0, 1, 3, 2]);  getitem_86 = None
        expand_45: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_118, _shape_param_390);  permute_118 = _shape_param_390 = None
        clone_81: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_311: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_81, _shape_param_391);  clone_81 = _shape_param_391 = None
        constant_pad_nd_45: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_311, _shape_param_392);  view_311 = _shape_param_392 = None
        bmm_22: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_44, constant_pad_nd_45);  constant_pad_nd_44 = constant_pad_nd_45 = None
        slice_34: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_22, 1, 0, -7);  bmm_22 = None
        slice_35: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_34, 2, 0, -7);  slice_34 = None
        view_312: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_35, _shape_param_393);  slice_35 = _shape_param_393 = None
        view_313: "i64[2401]" = torch.ops.aten.view.default(arg172_1, [-1]);  arg172_1 = None
        index_33: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg173_1, [view_313]);  arg173_1 = view_313 = None
        view_314: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_33, _shape_param_394);  index_33 = _shape_param_394 = None
        permute_119: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_314, [2, 0, 1]);  view_314 = None
        clone_82: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_119, memory_format = torch.contiguous_format);  permute_119 = None
        unsqueeze_21: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_82, 0);  clone_82 = None
        add_123: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_312, unsqueeze_21);  view_312 = unsqueeze_21 = None
        view_315: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_123, _shape_param_395);  add_123 = _shape_param_395 = None
        unsqueeze_22: "bf16[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg174_1, 1);  arg174_1 = None
        unsqueeze_23: "bf16[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 0);  unsqueeze_22 = None
        add_124: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_315, unsqueeze_23);  view_315 = unsqueeze_23 = None
        view_316: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(add_124, _shape_param_396);  add_124 = _shape_param_396 = None
        convert_element_type_96: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_316, torch.float32);  view_316 = None
        amax_11: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_96, [-1], True)
        sub_37: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_96, amax_11);  convert_element_type_96 = amax_11 = None
        exp_11: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_12: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_97: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        expand_46: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_97, _shape_param_397);  convert_element_type_97 = _shape_param_397 = None
        view_317: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_46, _shape_param_398);  expand_46 = _shape_param_398 = None
        constant_pad_nd_46: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_317, _shape_param_399);  view_317 = _shape_param_399 = None
        expand_47: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_87, _shape_param_400);  getitem_87 = _shape_param_400 = None
        clone_83: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_318: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_83, _shape_param_401);  clone_83 = _shape_param_401 = None
        constant_pad_nd_47: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_318, _shape_param_402);  view_318 = _shape_param_402 = None
        bmm_23: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_46, constant_pad_nd_47);  constant_pad_nd_46 = constant_pad_nd_47 = None
        slice_36: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_23, 1, 0, -7);  bmm_23 = None
        view_319: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_36, _shape_param_403);  slice_36 = _shape_param_403 = None
        permute_120: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None
        clone_84: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
        view_320: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_84, _shape_param_404);  clone_84 = _shape_param_404 = None
        view_321: "bf16[25088, 512]" = torch.ops.aten.view.default(view_320, _shape_param_405);  view_320 = _shape_param_405 = None
        permute_121: "bf16[512, 512]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_45: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg176_1, view_321, permute_121);  arg176_1 = view_321 = permute_121 = None
        view_322: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_45, _shape_param_406);  addmm_45 = _shape_param_406 = None
        view_323: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_322, _shape_param_407);  view_322 = _shape_param_407 = None
        view_324: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_323, _shape_param_408);  view_323 = _shape_param_408 = None
        permute_122: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_324, [0, 1, 3, 2, 4, 5]);  view_324 = None
        clone_85: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_325: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_85, _shape_param_409);  clone_85 = _shape_param_409 = None
        iota_22: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_125: "i64[14]" = torch.ops.aten.add.Tensor(iota_22, 11);  iota_22 = None
        fmod_22: "i64[14]" = torch.ops.aten.fmod.Scalar(add_125, 14);  add_125 = None
        index_34: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_325, [None, fmod_22]);  view_325 = fmod_22 = None
        iota_23: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_126: "i64[14]" = torch.ops.aten.add.Tensor(iota_23, 11);  iota_23 = None
        fmod_23: "i64[14]" = torch.ops.aten.fmod.Scalar(add_126, 14);  add_126 = None
        index_35: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_34, [None, None, fmod_23]);  index_34 = fmod_23 = None
        add_127: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_303, index_35);  view_303 = index_35 = None
        view_326: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_127, _shape_param_410);  add_127 = _shape_param_410 = None
        convert_element_type_98: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_326, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_98, [2], correction = 0, keepdim = True)
        getitem_88: "f32[128, 196, 1]" = var_mean_26[0]
        getitem_89: "f32[128, 196, 1]" = var_mean_26[1];  var_mean_26 = None
        sub_38: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_98, getitem_89);  convert_element_type_98 = getitem_89 = None
        add_128: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_26: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_97: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_26);  sub_38 = rsqrt_26 = None
        mul_98: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_97, arg177_1);  mul_97 = arg177_1 = None
        add_129: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_98, arg178_1);  mul_98 = arg178_1 = None
        convert_element_type_99: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16);  add_129 = None
        view_327: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_99, _shape_param_411);  convert_element_type_99 = _shape_param_411 = None
        permute_123: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_46: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg180_1, view_327, permute_123);  arg180_1 = view_327 = permute_123 = None
        view_328: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_46, _shape_param_412);  addmm_46 = _shape_param_412 = None
        convert_element_type_100: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_328, torch.float32);  view_328 = None
        mul_99: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_100, 0.5)
        mul_100: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_100, 0.7071067811865476);  convert_element_type_100 = None
        erf_11: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
        add_130: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_101: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_99, add_130);  mul_99 = add_130 = None
        convert_element_type_101: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_101, torch.bfloat16);  mul_101 = None
        view_329: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_101, _shape_param_413);  convert_element_type_101 = _shape_param_413 = None
        permute_124: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_47: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg182_1, view_329, permute_124);  arg182_1 = view_329 = permute_124 = None
        view_330: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_47, _shape_param_414);  addmm_47 = _shape_param_414 = None
        add_131: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_326, view_330);  view_326 = view_330 = None
        view_331: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_131, _shape_param_415);  add_131 = _shape_param_415 = None
        convert_element_type_102: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_331, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_102, [3], correction = 0, keepdim = True)
        getitem_90: "f32[128, 14, 14, 1]" = var_mean_27[0]
        getitem_91: "f32[128, 14, 14, 1]" = var_mean_27[1];  var_mean_27 = None
        sub_39: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_102, getitem_91);  convert_element_type_102 = getitem_91 = None
        add_132: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_27: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        mul_102: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_27);  sub_39 = rsqrt_27 = None
        mul_103: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_102, arg183_1);  mul_102 = arg183_1 = None
        add_133: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_103, arg184_1);  mul_103 = arg184_1 = None
        convert_element_type_103: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_133, torch.bfloat16);  add_133 = None
        view_332: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(convert_element_type_103, _shape_param_416);  convert_element_type_103 = _shape_param_416 = None
        permute_125: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_332, [0, 1, 3, 2, 4, 5]);  view_332 = None
        clone_86: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_333: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_86, _shape_param_417);  clone_86 = _shape_param_417 = None
        view_334: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_333, _shape_param_418);  view_333 = _shape_param_418 = None
        view_335: "bf16[25088, 512]" = torch.ops.aten.view.default(view_334, _shape_param_419);  view_334 = _shape_param_419 = None
        permute_126: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_48: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg186_1, view_335, permute_126);  arg186_1 = view_335 = permute_126 = None
        view_336: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_48, _shape_param_420);  addmm_48 = _shape_param_420 = None
        view_337: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_336, _shape_param_421);  view_336 = _shape_param_421 = None
        permute_127: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_337, [2, 0, 3, 1, 4]);  view_337 = None
        unbind_12 = torch.ops.aten.unbind.int(permute_127);  permute_127 = None
        getitem_92: "bf16[512, 16, 49, 32]" = unbind_12[0]
        getitem_93: "bf16[512, 16, 49, 32]" = unbind_12[1]
        getitem_94: "bf16[512, 16, 49, 32]" = unbind_12[2];  unbind_12 = None
        mul_104: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_92, 0.1767766952966369);  getitem_92 = None
        expand_48: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_104, _shape_param_422);  mul_104 = _shape_param_422 = None
        clone_87: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_338: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_87, _shape_param_423);  clone_87 = _shape_param_423 = None
        constant_pad_nd_48: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_338, _shape_param_424);  view_338 = _shape_param_424 = None
        permute_128: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_93, [0, 1, 3, 2]);  getitem_93 = None
        expand_49: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_128, _shape_param_425);  permute_128 = _shape_param_425 = None
        clone_88: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_339: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_88, _shape_param_426);  clone_88 = _shape_param_426 = None
        constant_pad_nd_49: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_339, _shape_param_427);  view_339 = _shape_param_427 = None
        bmm_24: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_48, constant_pad_nd_49);  constant_pad_nd_48 = constant_pad_nd_49 = None
        slice_37: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_24, 1, 0, -7);  bmm_24 = None
        slice_38: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_37, 2, 0, -7);  slice_37 = None
        view_340: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_38, _shape_param_428);  slice_38 = _shape_param_428 = None
        view_341: "i64[2401]" = torch.ops.aten.view.default(arg187_1, [-1]);  arg187_1 = None
        index_36: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg188_1, [view_341]);  arg188_1 = view_341 = None
        view_342: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_36, _shape_param_429);  index_36 = _shape_param_429 = None
        permute_129: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_342, [2, 0, 1]);  view_342 = None
        clone_89: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        unsqueeze_24: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_89, 0);  clone_89 = None
        add_134: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_340, unsqueeze_24);  view_340 = unsqueeze_24 = None
        convert_element_type_104: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(add_134, torch.float32);  add_134 = None
        amax_12: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_104, [-1], True)
        sub_40: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_104, amax_12);  convert_element_type_104 = amax_12 = None
        exp_12: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        sum_13: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_105: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None
        expand_50: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_105, _shape_param_430);  convert_element_type_105 = _shape_param_430 = None
        view_343: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_50, _shape_param_431);  expand_50 = _shape_param_431 = None
        constant_pad_nd_50: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_343, _shape_param_432);  view_343 = _shape_param_432 = None
        expand_51: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_94, _shape_param_433);  getitem_94 = _shape_param_433 = None
        clone_90: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_344: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_90, _shape_param_434);  clone_90 = _shape_param_434 = None
        constant_pad_nd_51: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_344, _shape_param_435);  view_344 = _shape_param_435 = None
        bmm_25: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_50, constant_pad_nd_51);  constant_pad_nd_50 = constant_pad_nd_51 = None
        slice_39: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_25, 1, 0, -7);  bmm_25 = None
        view_345: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_39, _shape_param_436);  slice_39 = _shape_param_436 = None
        permute_130: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
        clone_91: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
        view_346: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_91, _shape_param_437);  clone_91 = _shape_param_437 = None
        view_347: "bf16[25088, 512]" = torch.ops.aten.view.default(view_346, _shape_param_438);  view_346 = _shape_param_438 = None
        permute_131: "bf16[512, 512]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_49: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg190_1, view_347, permute_131);  arg190_1 = view_347 = permute_131 = None
        view_348: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_49, _shape_param_439);  addmm_49 = _shape_param_439 = None
        view_349: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_348, _shape_param_440);  view_348 = _shape_param_440 = None
        view_350: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_349, _shape_param_441);  view_349 = _shape_param_441 = None
        permute_132: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_350, [0, 1, 3, 2, 4, 5]);  view_350 = None
        clone_92: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
        view_351: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_92, _shape_param_442);  clone_92 = _shape_param_442 = None
        add_135: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_331, view_351);  view_331 = view_351 = None
        view_352: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_135, _shape_param_443);  add_135 = _shape_param_443 = None
        convert_element_type_106: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_352, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_106, [2], correction = 0, keepdim = True)
        getitem_95: "f32[128, 196, 1]" = var_mean_28[0]
        getitem_96: "f32[128, 196, 1]" = var_mean_28[1];  var_mean_28 = None
        sub_41: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_106, getitem_96);  convert_element_type_106 = getitem_96 = None
        add_136: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_95, 1e-05);  getitem_95 = None
        rsqrt_28: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        mul_105: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_28);  sub_41 = rsqrt_28 = None
        mul_106: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_105, arg191_1);  mul_105 = arg191_1 = None
        add_137: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_106, arg192_1);  mul_106 = arg192_1 = None
        convert_element_type_107: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_137, torch.bfloat16);  add_137 = None
        view_353: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_107, _shape_param_444);  convert_element_type_107 = _shape_param_444 = None
        permute_133: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_50: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg194_1, view_353, permute_133);  arg194_1 = view_353 = permute_133 = None
        view_354: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_50, _shape_param_445);  addmm_50 = _shape_param_445 = None
        convert_element_type_108: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_354, torch.float32);  view_354 = None
        mul_107: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_108, 0.5)
        mul_108: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_108, 0.7071067811865476);  convert_element_type_108 = None
        erf_12: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_108);  mul_108 = None
        add_138: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_109: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_107, add_138);  mul_107 = add_138 = None
        convert_element_type_109: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_109, torch.bfloat16);  mul_109 = None
        view_355: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_109, _shape_param_446);  convert_element_type_109 = _shape_param_446 = None
        permute_134: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_51: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg196_1, view_355, permute_134);  arg196_1 = view_355 = permute_134 = None
        view_356: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_51, _shape_param_447);  addmm_51 = _shape_param_447 = None
        add_139: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_352, view_356);  view_352 = view_356 = None
        view_357: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_139, _shape_param_448);  add_139 = _shape_param_448 = None
        convert_element_type_110: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_357, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_110, [3], correction = 0, keepdim = True)
        getitem_97: "f32[128, 14, 14, 1]" = var_mean_29[0]
        getitem_98: "f32[128, 14, 14, 1]" = var_mean_29[1];  var_mean_29 = None
        sub_42: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_110, getitem_98);  convert_element_type_110 = getitem_98 = None
        add_140: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_29: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        mul_110: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_29);  sub_42 = rsqrt_29 = None
        mul_111: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_110, arg197_1);  mul_110 = arg197_1 = None
        add_141: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_111, arg198_1);  mul_111 = arg198_1 = None
        convert_element_type_111: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_141, torch.bfloat16);  add_141 = None
        iota_24: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_142: "i64[14]" = torch.ops.aten.add.Tensor(iota_24, 3);  iota_24 = None
        fmod_24: "i64[14]" = torch.ops.aten.fmod.Scalar(add_142, 14);  add_142 = None
        index_37: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(convert_element_type_111, [None, fmod_24]);  convert_element_type_111 = fmod_24 = None
        iota_25: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_143: "i64[14]" = torch.ops.aten.add.Tensor(iota_25, 3);  iota_25 = None
        fmod_25: "i64[14]" = torch.ops.aten.fmod.Scalar(add_143, 14);  add_143 = None
        index_38: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_37, [None, None, fmod_25]);  index_37 = fmod_25 = None
        view_358: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_38, _shape_param_449);  index_38 = _shape_param_449 = None
        permute_135: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_358, [0, 1, 3, 2, 4, 5]);  view_358 = None
        clone_93: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_359: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_93, _shape_param_450);  clone_93 = _shape_param_450 = None
        view_360: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_359, _shape_param_451);  view_359 = _shape_param_451 = None
        view_361: "bf16[25088, 512]" = torch.ops.aten.view.default(view_360, _shape_param_452);  view_360 = _shape_param_452 = None
        permute_136: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_52: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg200_1, view_361, permute_136);  arg200_1 = view_361 = permute_136 = None
        view_362: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_52, _shape_param_453);  addmm_52 = _shape_param_453 = None
        view_363: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_362, _shape_param_454);  view_362 = _shape_param_454 = None
        permute_137: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_363, [2, 0, 3, 1, 4]);  view_363 = None
        unbind_13 = torch.ops.aten.unbind.int(permute_137);  permute_137 = None
        getitem_99: "bf16[512, 16, 49, 32]" = unbind_13[0]
        getitem_100: "bf16[512, 16, 49, 32]" = unbind_13[1]
        getitem_101: "bf16[512, 16, 49, 32]" = unbind_13[2];  unbind_13 = None
        mul_112: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_99, 0.1767766952966369);  getitem_99 = None
        expand_52: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_112, _shape_param_455);  mul_112 = _shape_param_455 = None
        clone_94: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_364: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_94, _shape_param_456);  clone_94 = _shape_param_456 = None
        constant_pad_nd_52: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_364, _shape_param_457);  view_364 = _shape_param_457 = None
        permute_138: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_100, [0, 1, 3, 2]);  getitem_100 = None
        expand_53: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_138, _shape_param_458);  permute_138 = _shape_param_458 = None
        clone_95: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_365: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_95, _shape_param_459);  clone_95 = _shape_param_459 = None
        constant_pad_nd_53: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_365, _shape_param_460);  view_365 = _shape_param_460 = None
        bmm_26: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_52, constant_pad_nd_53);  constant_pad_nd_52 = constant_pad_nd_53 = None
        slice_40: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_26, 1, 0, -7);  bmm_26 = None
        slice_41: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_40, 2, 0, -7);  slice_40 = None
        view_366: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_41, _shape_param_461);  slice_41 = _shape_param_461 = None
        view_367: "i64[2401]" = torch.ops.aten.view.default(arg201_1, [-1]);  arg201_1 = None
        index_39: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg202_1, [view_367]);  arg202_1 = view_367 = None
        view_368: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_39, _shape_param_462);  index_39 = _shape_param_462 = None
        permute_139: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_368, [2, 0, 1]);  view_368 = None
        clone_96: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        unsqueeze_25: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_96, 0);  clone_96 = None
        add_144: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_366, unsqueeze_25);  view_366 = unsqueeze_25 = None
        view_369: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_144, _shape_param_463);  add_144 = _shape_param_463 = None
        unsqueeze_26: "bf16[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg203_1, 1);  arg203_1 = None
        unsqueeze_27: "bf16[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 0);  unsqueeze_26 = None
        add_145: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_369, unsqueeze_27);  view_369 = unsqueeze_27 = None
        view_370: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(add_145, _shape_param_464);  add_145 = _shape_param_464 = None
        convert_element_type_112: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_370, torch.float32);  view_370 = None
        amax_13: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_112, [-1], True)
        sub_43: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_112, amax_13);  convert_element_type_112 = amax_13 = None
        exp_13: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_14: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_113: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None
        expand_54: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_113, _shape_param_465);  convert_element_type_113 = _shape_param_465 = None
        view_371: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_54, _shape_param_466);  expand_54 = _shape_param_466 = None
        constant_pad_nd_54: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_371, _shape_param_467);  view_371 = _shape_param_467 = None
        expand_55: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_101, _shape_param_468);  getitem_101 = _shape_param_468 = None
        clone_97: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_372: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_97, _shape_param_469);  clone_97 = _shape_param_469 = None
        constant_pad_nd_55: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_372, _shape_param_470);  view_372 = _shape_param_470 = None
        bmm_27: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_54, constant_pad_nd_55);  constant_pad_nd_54 = constant_pad_nd_55 = None
        slice_42: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_27, 1, 0, -7);  bmm_27 = None
        view_373: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_42, _shape_param_471);  slice_42 = _shape_param_471 = None
        permute_140: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
        clone_98: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_374: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_98, _shape_param_472);  clone_98 = _shape_param_472 = None
        view_375: "bf16[25088, 512]" = torch.ops.aten.view.default(view_374, _shape_param_473);  view_374 = _shape_param_473 = None
        permute_141: "bf16[512, 512]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
        addmm_53: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg205_1, view_375, permute_141);  arg205_1 = view_375 = permute_141 = None
        view_376: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_53, _shape_param_474);  addmm_53 = _shape_param_474 = None
        view_377: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_376, _shape_param_475);  view_376 = _shape_param_475 = None
        view_378: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_377, _shape_param_476);  view_377 = _shape_param_476 = None
        permute_142: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_378, [0, 1, 3, 2, 4, 5]);  view_378 = None
        clone_99: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
        view_379: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_99, _shape_param_477);  clone_99 = _shape_param_477 = None
        iota_26: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_146: "i64[14]" = torch.ops.aten.add.Tensor(iota_26, 11);  iota_26 = None
        fmod_26: "i64[14]" = torch.ops.aten.fmod.Scalar(add_146, 14);  add_146 = None
        index_40: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_379, [None, fmod_26]);  view_379 = fmod_26 = None
        iota_27: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_147: "i64[14]" = torch.ops.aten.add.Tensor(iota_27, 11);  iota_27 = None
        fmod_27: "i64[14]" = torch.ops.aten.fmod.Scalar(add_147, 14);  add_147 = None
        index_41: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_40, [None, None, fmod_27]);  index_40 = fmod_27 = None
        add_148: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_357, index_41);  view_357 = index_41 = None
        view_380: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_148, _shape_param_478);  add_148 = _shape_param_478 = None
        convert_element_type_114: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_380, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_114, [2], correction = 0, keepdim = True)
        getitem_102: "f32[128, 196, 1]" = var_mean_30[0]
        getitem_103: "f32[128, 196, 1]" = var_mean_30[1];  var_mean_30 = None
        sub_44: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_114, getitem_103);  convert_element_type_114 = getitem_103 = None
        add_149: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_30: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        mul_113: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_30);  sub_44 = rsqrt_30 = None
        mul_114: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_113, arg206_1);  mul_113 = arg206_1 = None
        add_150: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_114, arg207_1);  mul_114 = arg207_1 = None
        convert_element_type_115: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_150, torch.bfloat16);  add_150 = None
        view_381: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_115, _shape_param_479);  convert_element_type_115 = _shape_param_479 = None
        permute_143: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        addmm_54: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg209_1, view_381, permute_143);  arg209_1 = view_381 = permute_143 = None
        view_382: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_54, _shape_param_480);  addmm_54 = _shape_param_480 = None
        convert_element_type_116: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_382, torch.float32);  view_382 = None
        mul_115: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_116, 0.5)
        mul_116: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_116, 0.7071067811865476);  convert_element_type_116 = None
        erf_13: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_116);  mul_116 = None
        add_151: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_117: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_115, add_151);  mul_115 = add_151 = None
        convert_element_type_117: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_117, torch.bfloat16);  mul_117 = None
        view_383: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_117, _shape_param_481);  convert_element_type_117 = _shape_param_481 = None
        permute_144: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
        addmm_55: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg211_1, view_383, permute_144);  arg211_1 = view_383 = permute_144 = None
        view_384: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_55, _shape_param_482);  addmm_55 = _shape_param_482 = None
        add_152: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_380, view_384);  view_380 = view_384 = None
        view_385: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_152, _shape_param_483);  add_152 = _shape_param_483 = None
        convert_element_type_118: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_385, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_118, [3], correction = 0, keepdim = True)
        getitem_104: "f32[128, 14, 14, 1]" = var_mean_31[0]
        getitem_105: "f32[128, 14, 14, 1]" = var_mean_31[1];  var_mean_31 = None
        sub_45: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_118, getitem_105);  convert_element_type_118 = getitem_105 = None
        add_153: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_31: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        mul_118: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_31);  sub_45 = rsqrt_31 = None
        mul_119: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_118, arg212_1);  mul_118 = arg212_1 = None
        add_154: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_119, arg213_1);  mul_119 = arg213_1 = None
        convert_element_type_119: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_154, torch.bfloat16);  add_154 = None
        view_386: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(convert_element_type_119, _shape_param_484);  convert_element_type_119 = _shape_param_484 = None
        permute_145: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_386, [0, 1, 3, 2, 4, 5]);  view_386 = None
        clone_100: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
        view_387: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_100, _shape_param_485);  clone_100 = _shape_param_485 = None
        view_388: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_387, _shape_param_486);  view_387 = _shape_param_486 = None
        view_389: "bf16[25088, 512]" = torch.ops.aten.view.default(view_388, _shape_param_487);  view_388 = _shape_param_487 = None
        permute_146: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
        addmm_56: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg215_1, view_389, permute_146);  arg215_1 = view_389 = permute_146 = None
        view_390: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_56, _shape_param_488);  addmm_56 = _shape_param_488 = None
        view_391: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_390, _shape_param_489);  view_390 = _shape_param_489 = None
        permute_147: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_391, [2, 0, 3, 1, 4]);  view_391 = None
        unbind_14 = torch.ops.aten.unbind.int(permute_147);  permute_147 = None
        getitem_106: "bf16[512, 16, 49, 32]" = unbind_14[0]
        getitem_107: "bf16[512, 16, 49, 32]" = unbind_14[1]
        getitem_108: "bf16[512, 16, 49, 32]" = unbind_14[2];  unbind_14 = None
        mul_120: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_106, 0.1767766952966369);  getitem_106 = None
        expand_56: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_120, _shape_param_490);  mul_120 = _shape_param_490 = None
        clone_101: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_392: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_101, _shape_param_491);  clone_101 = _shape_param_491 = None
        constant_pad_nd_56: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_392, _shape_param_492);  view_392 = _shape_param_492 = None
        permute_148: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_107, [0, 1, 3, 2]);  getitem_107 = None
        expand_57: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_148, _shape_param_493);  permute_148 = _shape_param_493 = None
        clone_102: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_393: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_102, _shape_param_494);  clone_102 = _shape_param_494 = None
        constant_pad_nd_57: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_393, _shape_param_495);  view_393 = _shape_param_495 = None
        bmm_28: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_56, constant_pad_nd_57);  constant_pad_nd_56 = constant_pad_nd_57 = None
        slice_43: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_28, 1, 0, -7);  bmm_28 = None
        slice_44: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_43, 2, 0, -7);  slice_43 = None
        view_394: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_44, _shape_param_496);  slice_44 = _shape_param_496 = None
        view_395: "i64[2401]" = torch.ops.aten.view.default(arg216_1, [-1]);  arg216_1 = None
        index_42: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg217_1, [view_395]);  arg217_1 = view_395 = None
        view_396: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_42, _shape_param_497);  index_42 = _shape_param_497 = None
        permute_149: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_396, [2, 0, 1]);  view_396 = None
        clone_103: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
        unsqueeze_28: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_103, 0);  clone_103 = None
        add_155: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_394, unsqueeze_28);  view_394 = unsqueeze_28 = None
        convert_element_type_120: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(add_155, torch.float32);  add_155 = None
        amax_14: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_120, [-1], True)
        sub_46: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_120, amax_14);  convert_element_type_120 = amax_14 = None
        exp_14: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_15: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_121: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None
        expand_58: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_121, _shape_param_498);  convert_element_type_121 = _shape_param_498 = None
        view_397: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_58, _shape_param_499);  expand_58 = _shape_param_499 = None
        constant_pad_nd_58: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_397, _shape_param_500);  view_397 = _shape_param_500 = None
        expand_59: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_108, _shape_param_501);  getitem_108 = _shape_param_501 = None
        clone_104: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_398: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_104, _shape_param_502);  clone_104 = _shape_param_502 = None
        constant_pad_nd_59: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_398, _shape_param_503);  view_398 = _shape_param_503 = None
        bmm_29: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_58, constant_pad_nd_59);  constant_pad_nd_58 = constant_pad_nd_59 = None
        slice_45: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_29, 1, 0, -7);  bmm_29 = None
        view_399: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_45, _shape_param_504);  slice_45 = _shape_param_504 = None
        permute_150: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
        clone_105: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_400: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_105, _shape_param_505);  clone_105 = _shape_param_505 = None
        view_401: "bf16[25088, 512]" = torch.ops.aten.view.default(view_400, _shape_param_506);  view_400 = _shape_param_506 = None
        permute_151: "bf16[512, 512]" = torch.ops.aten.permute.default(arg218_1, [1, 0]);  arg218_1 = None
        addmm_57: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg219_1, view_401, permute_151);  arg219_1 = view_401 = permute_151 = None
        view_402: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_57, _shape_param_507);  addmm_57 = _shape_param_507 = None
        view_403: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_402, _shape_param_508);  view_402 = _shape_param_508 = None
        view_404: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_403, _shape_param_509);  view_403 = _shape_param_509 = None
        permute_152: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_404, [0, 1, 3, 2, 4, 5]);  view_404 = None
        clone_106: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format);  permute_152 = None
        view_405: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_106, _shape_param_510);  clone_106 = _shape_param_510 = None
        add_156: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_385, view_405);  view_385 = view_405 = None
        view_406: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_156, _shape_param_511);  add_156 = _shape_param_511 = None
        convert_element_type_122: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_406, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_122, [2], correction = 0, keepdim = True)
        getitem_109: "f32[128, 196, 1]" = var_mean_32[0]
        getitem_110: "f32[128, 196, 1]" = var_mean_32[1];  var_mean_32 = None
        sub_47: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_122, getitem_110);  convert_element_type_122 = getitem_110 = None
        add_157: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_109, 1e-05);  getitem_109 = None
        rsqrt_32: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
        mul_121: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_32);  sub_47 = rsqrt_32 = None
        mul_122: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_121, arg220_1);  mul_121 = arg220_1 = None
        add_158: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_122, arg221_1);  mul_122 = arg221_1 = None
        convert_element_type_123: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_158, torch.bfloat16);  add_158 = None
        view_407: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_123, _shape_param_512);  convert_element_type_123 = _shape_param_512 = None
        permute_153: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
        addmm_58: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg223_1, view_407, permute_153);  arg223_1 = view_407 = permute_153 = None
        view_408: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_58, _shape_param_513);  addmm_58 = _shape_param_513 = None
        convert_element_type_124: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_408, torch.float32);  view_408 = None
        mul_123: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_124, 0.5)
        mul_124: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_124, 0.7071067811865476);  convert_element_type_124 = None
        erf_14: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_124);  mul_124 = None
        add_159: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_125: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_123, add_159);  mul_123 = add_159 = None
        convert_element_type_125: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_125, torch.bfloat16);  mul_125 = None
        view_409: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_125, _shape_param_514);  convert_element_type_125 = _shape_param_514 = None
        permute_154: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
        addmm_59: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg225_1, view_409, permute_154);  arg225_1 = view_409 = permute_154 = None
        view_410: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_59, _shape_param_515);  addmm_59 = _shape_param_515 = None
        add_160: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_406, view_410);  view_406 = view_410 = None
        view_411: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_160, _shape_param_516);  add_160 = _shape_param_516 = None
        convert_element_type_126: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_411, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_126, [3], correction = 0, keepdim = True)
        getitem_111: "f32[128, 14, 14, 1]" = var_mean_33[0]
        getitem_112: "f32[128, 14, 14, 1]" = var_mean_33[1];  var_mean_33 = None
        sub_48: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_126, getitem_112);  convert_element_type_126 = getitem_112 = None
        add_161: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_111, 1e-05);  getitem_111 = None
        rsqrt_33: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        mul_126: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_33);  sub_48 = rsqrt_33 = None
        mul_127: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_126, arg226_1);  mul_126 = arg226_1 = None
        add_162: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_127, arg227_1);  mul_127 = arg227_1 = None
        convert_element_type_127: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_162, torch.bfloat16);  add_162 = None
        iota_28: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_163: "i64[14]" = torch.ops.aten.add.Tensor(iota_28, 3);  iota_28 = None
        fmod_28: "i64[14]" = torch.ops.aten.fmod.Scalar(add_163, 14);  add_163 = None
        index_43: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(convert_element_type_127, [None, fmod_28]);  convert_element_type_127 = fmod_28 = None
        iota_29: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_164: "i64[14]" = torch.ops.aten.add.Tensor(iota_29, 3);  iota_29 = None
        fmod_29: "i64[14]" = torch.ops.aten.fmod.Scalar(add_164, 14);  add_164 = None
        index_44: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_43, [None, None, fmod_29]);  index_43 = fmod_29 = None
        view_412: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_44, _shape_param_517);  index_44 = _shape_param_517 = None
        permute_155: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_412, [0, 1, 3, 2, 4, 5]);  view_412 = None
        clone_107: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_413: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_107, _shape_param_518);  clone_107 = _shape_param_518 = None
        view_414: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_413, _shape_param_519);  view_413 = _shape_param_519 = None
        view_415: "bf16[25088, 512]" = torch.ops.aten.view.default(view_414, _shape_param_520);  view_414 = _shape_param_520 = None
        permute_156: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
        addmm_60: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg229_1, view_415, permute_156);  arg229_1 = view_415 = permute_156 = None
        view_416: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_60, _shape_param_521);  addmm_60 = _shape_param_521 = None
        view_417: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_416, _shape_param_522);  view_416 = _shape_param_522 = None
        permute_157: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_417, [2, 0, 3, 1, 4]);  view_417 = None
        unbind_15 = torch.ops.aten.unbind.int(permute_157);  permute_157 = None
        getitem_113: "bf16[512, 16, 49, 32]" = unbind_15[0]
        getitem_114: "bf16[512, 16, 49, 32]" = unbind_15[1]
        getitem_115: "bf16[512, 16, 49, 32]" = unbind_15[2];  unbind_15 = None
        mul_128: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_113, 0.1767766952966369);  getitem_113 = None
        expand_60: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_128, _shape_param_523);  mul_128 = _shape_param_523 = None
        clone_108: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_418: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_108, _shape_param_524);  clone_108 = _shape_param_524 = None
        constant_pad_nd_60: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_418, _shape_param_525);  view_418 = _shape_param_525 = None
        permute_158: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_114, [0, 1, 3, 2]);  getitem_114 = None
        expand_61: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_158, _shape_param_526);  permute_158 = _shape_param_526 = None
        clone_109: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_419: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_109, _shape_param_527);  clone_109 = _shape_param_527 = None
        constant_pad_nd_61: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_419, _shape_param_528);  view_419 = _shape_param_528 = None
        bmm_30: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_60, constant_pad_nd_61);  constant_pad_nd_60 = constant_pad_nd_61 = None
        slice_46: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_30, 1, 0, -7);  bmm_30 = None
        slice_47: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_46, 2, 0, -7);  slice_46 = None
        view_420: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_47, _shape_param_529);  slice_47 = _shape_param_529 = None
        view_421: "i64[2401]" = torch.ops.aten.view.default(arg230_1, [-1]);  arg230_1 = None
        index_45: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg231_1, [view_421]);  arg231_1 = view_421 = None
        view_422: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_45, _shape_param_530);  index_45 = _shape_param_530 = None
        permute_159: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_422, [2, 0, 1]);  view_422 = None
        clone_110: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        unsqueeze_29: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_110, 0);  clone_110 = None
        add_165: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_420, unsqueeze_29);  view_420 = unsqueeze_29 = None
        view_423: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_165, _shape_param_531);  add_165 = _shape_param_531 = None
        unsqueeze_30: "bf16[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg232_1, 1);  arg232_1 = None
        unsqueeze_31: "bf16[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 0);  unsqueeze_30 = None
        add_166: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_423, unsqueeze_31);  view_423 = unsqueeze_31 = None
        view_424: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(add_166, _shape_param_532);  add_166 = _shape_param_532 = None
        convert_element_type_128: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_424, torch.float32);  view_424 = None
        amax_15: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_128, [-1], True)
        sub_49: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_128, amax_15);  convert_element_type_128 = amax_15 = None
        exp_15: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_16: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_129: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None
        expand_62: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_129, _shape_param_533);  convert_element_type_129 = _shape_param_533 = None
        view_425: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_62, _shape_param_534);  expand_62 = _shape_param_534 = None
        constant_pad_nd_62: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_425, _shape_param_535);  view_425 = _shape_param_535 = None
        expand_63: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_115, _shape_param_536);  getitem_115 = _shape_param_536 = None
        clone_111: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_426: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_111, _shape_param_537);  clone_111 = _shape_param_537 = None
        constant_pad_nd_63: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_426, _shape_param_538);  view_426 = _shape_param_538 = None
        bmm_31: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_62, constant_pad_nd_63);  constant_pad_nd_62 = constant_pad_nd_63 = None
        slice_48: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_31, 1, 0, -7);  bmm_31 = None
        view_427: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_48, _shape_param_539);  slice_48 = _shape_param_539 = None
        permute_160: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
        clone_112: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        view_428: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_112, _shape_param_540);  clone_112 = _shape_param_540 = None
        view_429: "bf16[25088, 512]" = torch.ops.aten.view.default(view_428, _shape_param_541);  view_428 = _shape_param_541 = None
        permute_161: "bf16[512, 512]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_61: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg234_1, view_429, permute_161);  arg234_1 = view_429 = permute_161 = None
        view_430: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_61, _shape_param_542);  addmm_61 = _shape_param_542 = None
        view_431: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_430, _shape_param_543);  view_430 = _shape_param_543 = None
        view_432: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_431, _shape_param_544);  view_431 = _shape_param_544 = None
        permute_162: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_432, [0, 1, 3, 2, 4, 5]);  view_432 = None
        clone_113: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
        view_433: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_113, _shape_param_545);  clone_113 = _shape_param_545 = None
        iota_30: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_167: "i64[14]" = torch.ops.aten.add.Tensor(iota_30, 11);  iota_30 = None
        fmod_30: "i64[14]" = torch.ops.aten.fmod.Scalar(add_167, 14);  add_167 = None
        index_46: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_433, [None, fmod_30]);  view_433 = fmod_30 = None
        iota_31: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_168: "i64[14]" = torch.ops.aten.add.Tensor(iota_31, 11);  iota_31 = None
        fmod_31: "i64[14]" = torch.ops.aten.fmod.Scalar(add_168, 14);  add_168 = None
        index_47: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_46, [None, None, fmod_31]);  index_46 = fmod_31 = None
        add_169: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_411, index_47);  view_411 = index_47 = None
        view_434: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_169, _shape_param_546);  add_169 = _shape_param_546 = None
        convert_element_type_130: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_434, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_130, [2], correction = 0, keepdim = True)
        getitem_116: "f32[128, 196, 1]" = var_mean_34[0]
        getitem_117: "f32[128, 196, 1]" = var_mean_34[1];  var_mean_34 = None
        sub_50: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_130, getitem_117);  convert_element_type_130 = getitem_117 = None
        add_170: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
        rsqrt_34: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        mul_129: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_34);  sub_50 = rsqrt_34 = None
        mul_130: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_129, arg235_1);  mul_129 = arg235_1 = None
        add_171: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_130, arg236_1);  mul_130 = arg236_1 = None
        convert_element_type_131: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_171, torch.bfloat16);  add_171 = None
        view_435: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_131, _shape_param_547);  convert_element_type_131 = _shape_param_547 = None
        permute_163: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        addmm_62: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg238_1, view_435, permute_163);  arg238_1 = view_435 = permute_163 = None
        view_436: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_62, _shape_param_548);  addmm_62 = _shape_param_548 = None
        convert_element_type_132: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_436, torch.float32);  view_436 = None
        mul_131: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_132, 0.5)
        mul_132: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_132, 0.7071067811865476);  convert_element_type_132 = None
        erf_15: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
        add_172: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_133: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_131, add_172);  mul_131 = add_172 = None
        convert_element_type_133: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_133, torch.bfloat16);  mul_133 = None
        view_437: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_133, _shape_param_549);  convert_element_type_133 = _shape_param_549 = None
        permute_164: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_63: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg240_1, view_437, permute_164);  arg240_1 = view_437 = permute_164 = None
        view_438: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_63, _shape_param_550);  addmm_63 = _shape_param_550 = None
        add_173: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_434, view_438);  view_434 = view_438 = None
        view_439: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_173, _shape_param_551);  add_173 = _shape_param_551 = None
        convert_element_type_134: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_439, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_134, [3], correction = 0, keepdim = True)
        getitem_118: "f32[128, 14, 14, 1]" = var_mean_35[0]
        getitem_119: "f32[128, 14, 14, 1]" = var_mean_35[1];  var_mean_35 = None
        sub_51: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_134, getitem_119);  convert_element_type_134 = getitem_119 = None
        add_174: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
        rsqrt_35: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        mul_134: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_35);  sub_51 = rsqrt_35 = None
        mul_135: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_134, arg241_1);  mul_134 = arg241_1 = None
        add_175: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_135, arg242_1);  mul_135 = arg242_1 = None
        convert_element_type_135: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_175, torch.bfloat16);  add_175 = None
        view_440: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(convert_element_type_135, _shape_param_552);  convert_element_type_135 = _shape_param_552 = None
        permute_165: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_440, [0, 1, 3, 2, 4, 5]);  view_440 = None
        clone_114: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
        view_441: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_114, _shape_param_553);  clone_114 = _shape_param_553 = None
        view_442: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_441, _shape_param_554);  view_441 = _shape_param_554 = None
        view_443: "bf16[25088, 512]" = torch.ops.aten.view.default(view_442, _shape_param_555);  view_442 = _shape_param_555 = None
        permute_166: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        addmm_64: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg244_1, view_443, permute_166);  arg244_1 = view_443 = permute_166 = None
        view_444: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_64, _shape_param_556);  addmm_64 = _shape_param_556 = None
        view_445: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_444, _shape_param_557);  view_444 = _shape_param_557 = None
        permute_167: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_445, [2, 0, 3, 1, 4]);  view_445 = None
        unbind_16 = torch.ops.aten.unbind.int(permute_167);  permute_167 = None
        getitem_120: "bf16[512, 16, 49, 32]" = unbind_16[0]
        getitem_121: "bf16[512, 16, 49, 32]" = unbind_16[1]
        getitem_122: "bf16[512, 16, 49, 32]" = unbind_16[2];  unbind_16 = None
        mul_136: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_120, 0.1767766952966369);  getitem_120 = None
        expand_64: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_136, _shape_param_558);  mul_136 = _shape_param_558 = None
        clone_115: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_446: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_115, _shape_param_559);  clone_115 = _shape_param_559 = None
        constant_pad_nd_64: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_446, _shape_param_560);  view_446 = _shape_param_560 = None
        permute_168: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_121, [0, 1, 3, 2]);  getitem_121 = None
        expand_65: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_168, _shape_param_561);  permute_168 = _shape_param_561 = None
        clone_116: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_447: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_116, _shape_param_562);  clone_116 = _shape_param_562 = None
        constant_pad_nd_65: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_447, _shape_param_563);  view_447 = _shape_param_563 = None
        bmm_32: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_64, constant_pad_nd_65);  constant_pad_nd_64 = constant_pad_nd_65 = None
        slice_49: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_32, 1, 0, -7);  bmm_32 = None
        slice_50: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_49, 2, 0, -7);  slice_49 = None
        view_448: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_50, _shape_param_564);  slice_50 = _shape_param_564 = None
        view_449: "i64[2401]" = torch.ops.aten.view.default(arg245_1, [-1]);  arg245_1 = None
        index_48: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg246_1, [view_449]);  arg246_1 = view_449 = None
        view_450: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_48, _shape_param_565);  index_48 = _shape_param_565 = None
        permute_169: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_450, [2, 0, 1]);  view_450 = None
        clone_117: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        unsqueeze_32: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_117, 0);  clone_117 = None
        add_176: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_448, unsqueeze_32);  view_448 = unsqueeze_32 = None
        convert_element_type_136: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(add_176, torch.float32);  add_176 = None
        amax_16: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_136, [-1], True)
        sub_52: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_136, amax_16);  convert_element_type_136 = amax_16 = None
        exp_16: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_17: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_137: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None
        expand_66: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_137, _shape_param_566);  convert_element_type_137 = _shape_param_566 = None
        view_451: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_66, _shape_param_567);  expand_66 = _shape_param_567 = None
        constant_pad_nd_66: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_451, _shape_param_568);  view_451 = _shape_param_568 = None
        expand_67: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_122, _shape_param_569);  getitem_122 = _shape_param_569 = None
        clone_118: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_452: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_118, _shape_param_570);  clone_118 = _shape_param_570 = None
        constant_pad_nd_67: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_452, _shape_param_571);  view_452 = _shape_param_571 = None
        bmm_33: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_66, constant_pad_nd_67);  constant_pad_nd_66 = constant_pad_nd_67 = None
        slice_51: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_33, 1, 0, -7);  bmm_33 = None
        view_453: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_51, _shape_param_572);  slice_51 = _shape_param_572 = None
        permute_170: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
        clone_119: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_454: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_119, _shape_param_573);  clone_119 = _shape_param_573 = None
        view_455: "bf16[25088, 512]" = torch.ops.aten.view.default(view_454, _shape_param_574);  view_454 = _shape_param_574 = None
        permute_171: "bf16[512, 512]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_65: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg248_1, view_455, permute_171);  arg248_1 = view_455 = permute_171 = None
        view_456: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_65, _shape_param_575);  addmm_65 = _shape_param_575 = None
        view_457: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_456, _shape_param_576);  view_456 = _shape_param_576 = None
        view_458: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_457, _shape_param_577);  view_457 = _shape_param_577 = None
        permute_172: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_458, [0, 1, 3, 2, 4, 5]);  view_458 = None
        clone_120: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_459: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_120, _shape_param_578);  clone_120 = _shape_param_578 = None
        add_177: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_439, view_459);  view_439 = view_459 = None
        view_460: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_177, _shape_param_579);  add_177 = _shape_param_579 = None
        convert_element_type_138: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_460, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_138, [2], correction = 0, keepdim = True)
        getitem_123: "f32[128, 196, 1]" = var_mean_36[0]
        getitem_124: "f32[128, 196, 1]" = var_mean_36[1];  var_mean_36 = None
        sub_53: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_138, getitem_124);  convert_element_type_138 = getitem_124 = None
        add_178: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_123, 1e-05);  getitem_123 = None
        rsqrt_36: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        mul_137: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_36);  sub_53 = rsqrt_36 = None
        mul_138: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_137, arg249_1);  mul_137 = arg249_1 = None
        add_179: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_138, arg250_1);  mul_138 = arg250_1 = None
        convert_element_type_139: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_179, torch.bfloat16);  add_179 = None
        view_461: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_139, _shape_param_580);  convert_element_type_139 = _shape_param_580 = None
        permute_173: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_66: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg252_1, view_461, permute_173);  arg252_1 = view_461 = permute_173 = None
        view_462: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_66, _shape_param_581);  addmm_66 = _shape_param_581 = None
        convert_element_type_140: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_462, torch.float32);  view_462 = None
        mul_139: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_140, 0.5)
        mul_140: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_140, 0.7071067811865476);  convert_element_type_140 = None
        erf_16: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_140);  mul_140 = None
        add_180: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_141: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_139, add_180);  mul_139 = add_180 = None
        convert_element_type_141: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_141, torch.bfloat16);  mul_141 = None
        view_463: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_141, _shape_param_582);  convert_element_type_141 = _shape_param_582 = None
        permute_174: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_67: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg254_1, view_463, permute_174);  arg254_1 = view_463 = permute_174 = None
        view_464: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_67, _shape_param_583);  addmm_67 = _shape_param_583 = None
        add_181: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_460, view_464);  view_460 = view_464 = None
        view_465: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_181, _shape_param_584);  add_181 = _shape_param_584 = None
        convert_element_type_142: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_465, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_142, [3], correction = 0, keepdim = True)
        getitem_125: "f32[128, 14, 14, 1]" = var_mean_37[0]
        getitem_126: "f32[128, 14, 14, 1]" = var_mean_37[1];  var_mean_37 = None
        sub_54: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_142, getitem_126);  convert_element_type_142 = getitem_126 = None
        add_182: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_125, 1e-05);  getitem_125 = None
        rsqrt_37: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        mul_142: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_37);  sub_54 = rsqrt_37 = None
        mul_143: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_142, arg255_1);  mul_142 = arg255_1 = None
        add_183: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_143, arg256_1);  mul_143 = arg256_1 = None
        convert_element_type_143: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_183, torch.bfloat16);  add_183 = None
        iota_32: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_184: "i64[14]" = torch.ops.aten.add.Tensor(iota_32, 3);  iota_32 = None
        fmod_32: "i64[14]" = torch.ops.aten.fmod.Scalar(add_184, 14);  add_184 = None
        index_49: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(convert_element_type_143, [None, fmod_32]);  convert_element_type_143 = fmod_32 = None
        iota_33: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_185: "i64[14]" = torch.ops.aten.add.Tensor(iota_33, 3);  iota_33 = None
        fmod_33: "i64[14]" = torch.ops.aten.fmod.Scalar(add_185, 14);  add_185 = None
        index_50: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_49, [None, None, fmod_33]);  index_49 = fmod_33 = None
        view_466: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_50, _shape_param_585);  index_50 = _shape_param_585 = None
        permute_175: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_466, [0, 1, 3, 2, 4, 5]);  view_466 = None
        clone_121: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
        view_467: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_121, _shape_param_586);  clone_121 = _shape_param_586 = None
        view_468: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_467, _shape_param_587);  view_467 = _shape_param_587 = None
        view_469: "bf16[25088, 512]" = torch.ops.aten.view.default(view_468, _shape_param_588);  view_468 = _shape_param_588 = None
        permute_176: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        addmm_68: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg258_1, view_469, permute_176);  arg258_1 = view_469 = permute_176 = None
        view_470: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_68, _shape_param_589);  addmm_68 = _shape_param_589 = None
        view_471: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_470, _shape_param_590);  view_470 = _shape_param_590 = None
        permute_177: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_471, [2, 0, 3, 1, 4]);  view_471 = None
        unbind_17 = torch.ops.aten.unbind.int(permute_177);  permute_177 = None
        getitem_127: "bf16[512, 16, 49, 32]" = unbind_17[0]
        getitem_128: "bf16[512, 16, 49, 32]" = unbind_17[1]
        getitem_129: "bf16[512, 16, 49, 32]" = unbind_17[2];  unbind_17 = None
        mul_144: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_127, 0.1767766952966369);  getitem_127 = None
        expand_68: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_144, _shape_param_591);  mul_144 = _shape_param_591 = None
        clone_122: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_472: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_122, _shape_param_592);  clone_122 = _shape_param_592 = None
        constant_pad_nd_68: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_472, _shape_param_593);  view_472 = _shape_param_593 = None
        permute_178: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_128, [0, 1, 3, 2]);  getitem_128 = None
        expand_69: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_178, _shape_param_594);  permute_178 = _shape_param_594 = None
        clone_123: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_473: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_123, _shape_param_595);  clone_123 = _shape_param_595 = None
        constant_pad_nd_69: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_473, _shape_param_596);  view_473 = _shape_param_596 = None
        bmm_34: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_68, constant_pad_nd_69);  constant_pad_nd_68 = constant_pad_nd_69 = None
        slice_52: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_34, 1, 0, -7);  bmm_34 = None
        slice_53: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_52, 2, 0, -7);  slice_52 = None
        view_474: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_53, _shape_param_597);  slice_53 = _shape_param_597 = None
        view_475: "i64[2401]" = torch.ops.aten.view.default(arg259_1, [-1]);  arg259_1 = None
        index_51: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg260_1, [view_475]);  arg260_1 = view_475 = None
        view_476: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_51, _shape_param_598);  index_51 = _shape_param_598 = None
        permute_179: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_476, [2, 0, 1]);  view_476 = None
        clone_124: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
        unsqueeze_33: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_124, 0);  clone_124 = None
        add_186: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_474, unsqueeze_33);  view_474 = unsqueeze_33 = None
        view_477: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_186, _shape_param_599);  add_186 = _shape_param_599 = None
        unsqueeze_34: "bf16[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg261_1, 1);  arg261_1 = None
        unsqueeze_35: "bf16[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 0);  unsqueeze_34 = None
        add_187: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_477, unsqueeze_35);  view_477 = unsqueeze_35 = None
        view_478: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(add_187, _shape_param_600);  add_187 = _shape_param_600 = None
        convert_element_type_144: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_478, torch.float32);  view_478 = None
        amax_17: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_144, [-1], True)
        sub_55: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_144, amax_17);  convert_element_type_144 = amax_17 = None
        exp_17: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_18: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_17: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_145: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None
        expand_70: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_145, _shape_param_601);  convert_element_type_145 = _shape_param_601 = None
        view_479: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_70, _shape_param_602);  expand_70 = _shape_param_602 = None
        constant_pad_nd_70: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_479, _shape_param_603);  view_479 = _shape_param_603 = None
        expand_71: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_129, _shape_param_604);  getitem_129 = _shape_param_604 = None
        clone_125: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_480: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_125, _shape_param_605);  clone_125 = _shape_param_605 = None
        constant_pad_nd_71: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_480, _shape_param_606);  view_480 = _shape_param_606 = None
        bmm_35: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_70, constant_pad_nd_71);  constant_pad_nd_70 = constant_pad_nd_71 = None
        slice_54: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_35, 1, 0, -7);  bmm_35 = None
        view_481: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_54, _shape_param_607);  slice_54 = _shape_param_607 = None
        permute_180: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_481, [0, 2, 1, 3]);  view_481 = None
        clone_126: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
        view_482: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_126, _shape_param_608);  clone_126 = _shape_param_608 = None
        view_483: "bf16[25088, 512]" = torch.ops.aten.view.default(view_482, _shape_param_609);  view_482 = _shape_param_609 = None
        permute_181: "bf16[512, 512]" = torch.ops.aten.permute.default(arg262_1, [1, 0]);  arg262_1 = None
        addmm_69: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg263_1, view_483, permute_181);  arg263_1 = view_483 = permute_181 = None
        view_484: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_69, _shape_param_610);  addmm_69 = _shape_param_610 = None
        view_485: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_484, _shape_param_611);  view_484 = _shape_param_611 = None
        view_486: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_485, _shape_param_612);  view_485 = _shape_param_612 = None
        permute_182: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_486, [0, 1, 3, 2, 4, 5]);  view_486 = None
        clone_127: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        view_487: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_127, _shape_param_613);  clone_127 = _shape_param_613 = None
        iota_34: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_188: "i64[14]" = torch.ops.aten.add.Tensor(iota_34, 11);  iota_34 = None
        fmod_34: "i64[14]" = torch.ops.aten.fmod.Scalar(add_188, 14);  add_188 = None
        index_52: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_487, [None, fmod_34]);  view_487 = fmod_34 = None
        iota_35: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_189: "i64[14]" = torch.ops.aten.add.Tensor(iota_35, 11);  iota_35 = None
        fmod_35: "i64[14]" = torch.ops.aten.fmod.Scalar(add_189, 14);  add_189 = None
        index_53: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_52, [None, None, fmod_35]);  index_52 = fmod_35 = None
        add_190: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_465, index_53);  view_465 = index_53 = None
        view_488: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_190, _shape_param_614);  add_190 = _shape_param_614 = None
        convert_element_type_146: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_488, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_146, [2], correction = 0, keepdim = True)
        getitem_130: "f32[128, 196, 1]" = var_mean_38[0]
        getitem_131: "f32[128, 196, 1]" = var_mean_38[1];  var_mean_38 = None
        sub_56: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_146, getitem_131);  convert_element_type_146 = getitem_131 = None
        add_191: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_38: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        mul_145: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_38);  sub_56 = rsqrt_38 = None
        mul_146: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_145, arg264_1);  mul_145 = arg264_1 = None
        add_192: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_146, arg265_1);  mul_146 = arg265_1 = None
        convert_element_type_147: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_192, torch.bfloat16);  add_192 = None
        view_489: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_147, _shape_param_615);  convert_element_type_147 = _shape_param_615 = None
        permute_183: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        addmm_70: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg267_1, view_489, permute_183);  arg267_1 = view_489 = permute_183 = None
        view_490: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_70, _shape_param_616);  addmm_70 = _shape_param_616 = None
        convert_element_type_148: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_490, torch.float32);  view_490 = None
        mul_147: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_148, 0.5)
        mul_148: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_148, 0.7071067811865476);  convert_element_type_148 = None
        erf_17: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
        add_193: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_149: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_147, add_193);  mul_147 = add_193 = None
        convert_element_type_149: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_149, torch.bfloat16);  mul_149 = None
        view_491: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_149, _shape_param_617);  convert_element_type_149 = _shape_param_617 = None
        permute_184: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        addmm_71: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg269_1, view_491, permute_184);  arg269_1 = view_491 = permute_184 = None
        view_492: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_71, _shape_param_618);  addmm_71 = _shape_param_618 = None
        add_194: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_488, view_492);  view_488 = view_492 = None
        view_493: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_194, _shape_param_619);  add_194 = _shape_param_619 = None
        convert_element_type_150: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_493, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_150, [3], correction = 0, keepdim = True)
        getitem_132: "f32[128, 14, 14, 1]" = var_mean_39[0]
        getitem_133: "f32[128, 14, 14, 1]" = var_mean_39[1];  var_mean_39 = None
        sub_57: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_150, getitem_133);  convert_element_type_150 = getitem_133 = None
        add_195: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_39: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
        mul_150: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_39);  sub_57 = rsqrt_39 = None
        mul_151: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_150, arg270_1);  mul_150 = arg270_1 = None
        add_196: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_151, arg271_1);  mul_151 = arg271_1 = None
        convert_element_type_151: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_196, torch.bfloat16);  add_196 = None
        view_494: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(convert_element_type_151, _shape_param_620);  convert_element_type_151 = _shape_param_620 = None
        permute_185: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_494, [0, 1, 3, 2, 4, 5]);  view_494 = None
        clone_128: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
        view_495: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_128, _shape_param_621);  clone_128 = _shape_param_621 = None
        view_496: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_495, _shape_param_622);  view_495 = _shape_param_622 = None
        view_497: "bf16[25088, 512]" = torch.ops.aten.view.default(view_496, _shape_param_623);  view_496 = _shape_param_623 = None
        permute_186: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg272_1, [1, 0]);  arg272_1 = None
        addmm_72: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg273_1, view_497, permute_186);  arg273_1 = view_497 = permute_186 = None
        view_498: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_72, _shape_param_624);  addmm_72 = _shape_param_624 = None
        view_499: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_498, _shape_param_625);  view_498 = _shape_param_625 = None
        permute_187: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_499, [2, 0, 3, 1, 4]);  view_499 = None
        unbind_18 = torch.ops.aten.unbind.int(permute_187);  permute_187 = None
        getitem_134: "bf16[512, 16, 49, 32]" = unbind_18[0]
        getitem_135: "bf16[512, 16, 49, 32]" = unbind_18[1]
        getitem_136: "bf16[512, 16, 49, 32]" = unbind_18[2];  unbind_18 = None
        mul_152: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_134, 0.1767766952966369);  getitem_134 = None
        expand_72: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_152, _shape_param_626);  mul_152 = _shape_param_626 = None
        clone_129: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_500: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_129, _shape_param_627);  clone_129 = _shape_param_627 = None
        constant_pad_nd_72: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_500, _shape_param_628);  view_500 = _shape_param_628 = None
        permute_188: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_135, [0, 1, 3, 2]);  getitem_135 = None
        expand_73: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_188, _shape_param_629);  permute_188 = _shape_param_629 = None
        clone_130: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_501: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_130, _shape_param_630);  clone_130 = _shape_param_630 = None
        constant_pad_nd_73: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_501, _shape_param_631);  view_501 = _shape_param_631 = None
        bmm_36: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_72, constant_pad_nd_73);  constant_pad_nd_72 = constant_pad_nd_73 = None
        slice_55: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_36, 1, 0, -7);  bmm_36 = None
        slice_56: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_55, 2, 0, -7);  slice_55 = None
        view_502: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_56, _shape_param_632);  slice_56 = _shape_param_632 = None
        view_503: "i64[2401]" = torch.ops.aten.view.default(arg274_1, [-1]);  arg274_1 = None
        index_54: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg275_1, [view_503]);  arg275_1 = view_503 = None
        view_504: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_54, _shape_param_633);  index_54 = _shape_param_633 = None
        permute_189: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_504, [2, 0, 1]);  view_504 = None
        clone_131: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        unsqueeze_36: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_131, 0);  clone_131 = None
        add_197: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_502, unsqueeze_36);  view_502 = unsqueeze_36 = None
        convert_element_type_152: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(add_197, torch.float32);  add_197 = None
        amax_18: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_152, [-1], True)
        sub_58: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_152, amax_18);  convert_element_type_152 = amax_18 = None
        exp_18: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        sum_19: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_153: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None
        expand_74: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_153, _shape_param_634);  convert_element_type_153 = _shape_param_634 = None
        view_505: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_74, _shape_param_635);  expand_74 = _shape_param_635 = None
        constant_pad_nd_74: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_505, _shape_param_636);  view_505 = _shape_param_636 = None
        expand_75: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_136, _shape_param_637);  getitem_136 = _shape_param_637 = None
        clone_132: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_506: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_132, _shape_param_638);  clone_132 = _shape_param_638 = None
        constant_pad_nd_75: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_506, _shape_param_639);  view_506 = _shape_param_639 = None
        bmm_37: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_74, constant_pad_nd_75);  constant_pad_nd_74 = constant_pad_nd_75 = None
        slice_57: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_37, 1, 0, -7);  bmm_37 = None
        view_507: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_57, _shape_param_640);  slice_57 = _shape_param_640 = None
        permute_190: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_507, [0, 2, 1, 3]);  view_507 = None
        clone_133: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
        view_508: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_133, _shape_param_641);  clone_133 = _shape_param_641 = None
        view_509: "bf16[25088, 512]" = torch.ops.aten.view.default(view_508, _shape_param_642);  view_508 = _shape_param_642 = None
        permute_191: "bf16[512, 512]" = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
        addmm_73: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg277_1, view_509, permute_191);  arg277_1 = view_509 = permute_191 = None
        view_510: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_73, _shape_param_643);  addmm_73 = _shape_param_643 = None
        view_511: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_510, _shape_param_644);  view_510 = _shape_param_644 = None
        view_512: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_511, _shape_param_645);  view_511 = _shape_param_645 = None
        permute_192: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_512, [0, 1, 3, 2, 4, 5]);  view_512 = None
        clone_134: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_513: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_134, _shape_param_646);  clone_134 = _shape_param_646 = None
        add_198: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_493, view_513);  view_493 = view_513 = None
        view_514: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_198, _shape_param_647);  add_198 = _shape_param_647 = None
        convert_element_type_154: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_514, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_154, [2], correction = 0, keepdim = True)
        getitem_137: "f32[128, 196, 1]" = var_mean_40[0]
        getitem_138: "f32[128, 196, 1]" = var_mean_40[1];  var_mean_40 = None
        sub_59: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_154, getitem_138);  convert_element_type_154 = getitem_138 = None
        add_199: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_137, 1e-05);  getitem_137 = None
        rsqrt_40: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        mul_153: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_40);  sub_59 = rsqrt_40 = None
        mul_154: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_153, arg278_1);  mul_153 = arg278_1 = None
        add_200: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_154, arg279_1);  mul_154 = arg279_1 = None
        convert_element_type_155: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_200, torch.bfloat16);  add_200 = None
        view_515: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_155, _shape_param_648);  convert_element_type_155 = _shape_param_648 = None
        permute_193: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
        addmm_74: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg281_1, view_515, permute_193);  arg281_1 = view_515 = permute_193 = None
        view_516: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_74, _shape_param_649);  addmm_74 = _shape_param_649 = None
        convert_element_type_156: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_516, torch.float32);  view_516 = None
        mul_155: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_156, 0.5)
        mul_156: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_156, 0.7071067811865476);  convert_element_type_156 = None
        erf_18: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_156);  mul_156 = None
        add_201: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_157: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_155, add_201);  mul_155 = add_201 = None
        convert_element_type_157: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_157, torch.bfloat16);  mul_157 = None
        view_517: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_157, _shape_param_650);  convert_element_type_157 = _shape_param_650 = None
        permute_194: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
        addmm_75: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg283_1, view_517, permute_194);  arg283_1 = view_517 = permute_194 = None
        view_518: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_75, _shape_param_651);  addmm_75 = _shape_param_651 = None
        add_202: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_514, view_518);  view_514 = view_518 = None
        view_519: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_202, _shape_param_652);  add_202 = _shape_param_652 = None
        convert_element_type_158: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_519, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_158, [3], correction = 0, keepdim = True)
        getitem_139: "f32[128, 14, 14, 1]" = var_mean_41[0]
        getitem_140: "f32[128, 14, 14, 1]" = var_mean_41[1];  var_mean_41 = None
        sub_60: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_158, getitem_140);  convert_element_type_158 = getitem_140 = None
        add_203: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_139, 1e-05);  getitem_139 = None
        rsqrt_41: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        mul_158: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_41);  sub_60 = rsqrt_41 = None
        mul_159: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_158, arg284_1);  mul_158 = arg284_1 = None
        add_204: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_159, arg285_1);  mul_159 = arg285_1 = None
        convert_element_type_159: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_204, torch.bfloat16);  add_204 = None
        iota_36: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_205: "i64[14]" = torch.ops.aten.add.Tensor(iota_36, 3);  iota_36 = None
        fmod_36: "i64[14]" = torch.ops.aten.fmod.Scalar(add_205, 14);  add_205 = None
        index_55: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(convert_element_type_159, [None, fmod_36]);  convert_element_type_159 = fmod_36 = None
        iota_37: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_206: "i64[14]" = torch.ops.aten.add.Tensor(iota_37, 3);  iota_37 = None
        fmod_37: "i64[14]" = torch.ops.aten.fmod.Scalar(add_206, 14);  add_206 = None
        index_56: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_55, [None, None, fmod_37]);  index_55 = fmod_37 = None
        view_520: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_56, _shape_param_653);  index_56 = _shape_param_653 = None
        permute_195: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_520, [0, 1, 3, 2, 4, 5]);  view_520 = None
        clone_135: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
        view_521: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_135, _shape_param_654);  clone_135 = _shape_param_654 = None
        view_522: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_521, _shape_param_655);  view_521 = _shape_param_655 = None
        view_523: "bf16[25088, 512]" = torch.ops.aten.view.default(view_522, _shape_param_656);  view_522 = _shape_param_656 = None
        permute_196: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
        addmm_76: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg287_1, view_523, permute_196);  arg287_1 = view_523 = permute_196 = None
        view_524: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_76, _shape_param_657);  addmm_76 = _shape_param_657 = None
        view_525: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_524, _shape_param_658);  view_524 = _shape_param_658 = None
        permute_197: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_525, [2, 0, 3, 1, 4]);  view_525 = None
        unbind_19 = torch.ops.aten.unbind.int(permute_197);  permute_197 = None
        getitem_141: "bf16[512, 16, 49, 32]" = unbind_19[0]
        getitem_142: "bf16[512, 16, 49, 32]" = unbind_19[1]
        getitem_143: "bf16[512, 16, 49, 32]" = unbind_19[2];  unbind_19 = None
        mul_160: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_141, 0.1767766952966369);  getitem_141 = None
        expand_76: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_160, _shape_param_659);  mul_160 = _shape_param_659 = None
        clone_136: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_526: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_136, _shape_param_660);  clone_136 = _shape_param_660 = None
        constant_pad_nd_76: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_526, _shape_param_661);  view_526 = _shape_param_661 = None
        permute_198: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_142, [0, 1, 3, 2]);  getitem_142 = None
        expand_77: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_198, _shape_param_662);  permute_198 = _shape_param_662 = None
        clone_137: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_527: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_137, _shape_param_663);  clone_137 = _shape_param_663 = None
        constant_pad_nd_77: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_527, _shape_param_664);  view_527 = _shape_param_664 = None
        bmm_38: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_76, constant_pad_nd_77);  constant_pad_nd_76 = constant_pad_nd_77 = None
        slice_58: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_38, 1, 0, -7);  bmm_38 = None
        slice_59: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_58, 2, 0, -7);  slice_58 = None
        view_528: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_59, _shape_param_665);  slice_59 = _shape_param_665 = None
        view_529: "i64[2401]" = torch.ops.aten.view.default(arg288_1, [-1]);  arg288_1 = None
        index_57: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg289_1, [view_529]);  arg289_1 = view_529 = None
        view_530: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_57, _shape_param_666);  index_57 = _shape_param_666 = None
        permute_199: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_530, [2, 0, 1]);  view_530 = None
        clone_138: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
        unsqueeze_37: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_138, 0);  clone_138 = None
        add_207: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_528, unsqueeze_37);  view_528 = unsqueeze_37 = None
        view_531: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_207, _shape_param_667);  add_207 = _shape_param_667 = None
        unsqueeze_38: "bf16[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg290_1, 1);  arg290_1 = None
        unsqueeze_39: "bf16[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 0);  unsqueeze_38 = None
        add_208: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_531, unsqueeze_39);  view_531 = unsqueeze_39 = None
        view_532: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(add_208, _shape_param_668);  add_208 = _shape_param_668 = None
        convert_element_type_160: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_532, torch.float32);  view_532 = None
        amax_19: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_160, [-1], True)
        sub_61: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_160, amax_19);  convert_element_type_160 = amax_19 = None
        exp_19: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_20: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_19: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        convert_element_type_161: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None
        expand_78: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_161, _shape_param_669);  convert_element_type_161 = _shape_param_669 = None
        view_533: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_78, _shape_param_670);  expand_78 = _shape_param_670 = None
        constant_pad_nd_78: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_533, _shape_param_671);  view_533 = _shape_param_671 = None
        expand_79: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_143, _shape_param_672);  getitem_143 = _shape_param_672 = None
        clone_139: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_534: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_139, _shape_param_673);  clone_139 = _shape_param_673 = None
        constant_pad_nd_79: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_534, _shape_param_674);  view_534 = _shape_param_674 = None
        bmm_39: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_78, constant_pad_nd_79);  constant_pad_nd_78 = constant_pad_nd_79 = None
        slice_60: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_39, 1, 0, -7);  bmm_39 = None
        view_535: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_60, _shape_param_675);  slice_60 = _shape_param_675 = None
        permute_200: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None
        clone_140: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
        view_536: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_140, _shape_param_676);  clone_140 = _shape_param_676 = None
        view_537: "bf16[25088, 512]" = torch.ops.aten.view.default(view_536, _shape_param_677);  view_536 = _shape_param_677 = None
        permute_201: "bf16[512, 512]" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
        addmm_77: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg292_1, view_537, permute_201);  arg292_1 = view_537 = permute_201 = None
        view_538: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_77, _shape_param_678);  addmm_77 = _shape_param_678 = None
        view_539: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_538, _shape_param_679);  view_538 = _shape_param_679 = None
        view_540: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_539, _shape_param_680);  view_539 = _shape_param_680 = None
        permute_202: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_540, [0, 1, 3, 2, 4, 5]);  view_540 = None
        clone_141: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
        view_541: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_141, _shape_param_681);  clone_141 = _shape_param_681 = None
        iota_38: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_209: "i64[14]" = torch.ops.aten.add.Tensor(iota_38, 11);  iota_38 = None
        fmod_38: "i64[14]" = torch.ops.aten.fmod.Scalar(add_209, 14);  add_209 = None
        index_58: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_541, [None, fmod_38]);  view_541 = fmod_38 = None
        iota_39: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_210: "i64[14]" = torch.ops.aten.add.Tensor(iota_39, 11);  iota_39 = None
        fmod_39: "i64[14]" = torch.ops.aten.fmod.Scalar(add_210, 14);  add_210 = None
        index_59: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_58, [None, None, fmod_39]);  index_58 = fmod_39 = None
        add_211: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_519, index_59);  view_519 = index_59 = None
        view_542: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_211, _shape_param_682);  add_211 = _shape_param_682 = None
        convert_element_type_162: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_542, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_162, [2], correction = 0, keepdim = True)
        getitem_144: "f32[128, 196, 1]" = var_mean_42[0]
        getitem_145: "f32[128, 196, 1]" = var_mean_42[1];  var_mean_42 = None
        sub_62: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_162, getitem_145);  convert_element_type_162 = getitem_145 = None
        add_212: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
        rsqrt_42: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        mul_161: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_42);  sub_62 = rsqrt_42 = None
        mul_162: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_161, arg293_1);  mul_161 = arg293_1 = None
        add_213: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_162, arg294_1);  mul_162 = arg294_1 = None
        convert_element_type_163: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_213, torch.bfloat16);  add_213 = None
        view_543: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_163, _shape_param_683);  convert_element_type_163 = _shape_param_683 = None
        permute_203: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_78: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg296_1, view_543, permute_203);  arg296_1 = view_543 = permute_203 = None
        view_544: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_78, _shape_param_684);  addmm_78 = _shape_param_684 = None
        convert_element_type_164: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_544, torch.float32);  view_544 = None
        mul_163: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_164, 0.5)
        mul_164: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_164, 0.7071067811865476);  convert_element_type_164 = None
        erf_19: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_164);  mul_164 = None
        add_214: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_165: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_163, add_214);  mul_163 = add_214 = None
        convert_element_type_165: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_165, torch.bfloat16);  mul_165 = None
        view_545: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_165, _shape_param_685);  convert_element_type_165 = _shape_param_685 = None
        permute_204: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_79: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg298_1, view_545, permute_204);  arg298_1 = view_545 = permute_204 = None
        view_546: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_79, _shape_param_686);  addmm_79 = _shape_param_686 = None
        add_215: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_542, view_546);  view_542 = view_546 = None
        view_547: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_215, _shape_param_687);  add_215 = _shape_param_687 = None
        convert_element_type_166: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_547, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_166, [3], correction = 0, keepdim = True)
        getitem_146: "f32[128, 14, 14, 1]" = var_mean_43[0]
        getitem_147: "f32[128, 14, 14, 1]" = var_mean_43[1];  var_mean_43 = None
        sub_63: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_166, getitem_147);  convert_element_type_166 = getitem_147 = None
        add_216: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05);  getitem_146 = None
        rsqrt_43: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
        mul_166: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_43);  sub_63 = rsqrt_43 = None
        mul_167: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_166, arg299_1);  mul_166 = arg299_1 = None
        add_217: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_167, arg300_1);  mul_167 = arg300_1 = None
        convert_element_type_167: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_217, torch.bfloat16);  add_217 = None
        view_548: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(convert_element_type_167, _shape_param_688);  convert_element_type_167 = _shape_param_688 = None
        permute_205: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_548, [0, 1, 3, 2, 4, 5]);  view_548 = None
        clone_142: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_549: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_142, _shape_param_689);  clone_142 = _shape_param_689 = None
        view_550: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_549, _shape_param_690);  view_549 = _shape_param_690 = None
        view_551: "bf16[25088, 512]" = torch.ops.aten.view.default(view_550, _shape_param_691);  view_550 = _shape_param_691 = None
        permute_206: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
        addmm_80: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg302_1, view_551, permute_206);  arg302_1 = view_551 = permute_206 = None
        view_552: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_80, _shape_param_692);  addmm_80 = _shape_param_692 = None
        view_553: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_552, _shape_param_693);  view_552 = _shape_param_693 = None
        permute_207: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_553, [2, 0, 3, 1, 4]);  view_553 = None
        unbind_20 = torch.ops.aten.unbind.int(permute_207);  permute_207 = None
        getitem_148: "bf16[512, 16, 49, 32]" = unbind_20[0]
        getitem_149: "bf16[512, 16, 49, 32]" = unbind_20[1]
        getitem_150: "bf16[512, 16, 49, 32]" = unbind_20[2];  unbind_20 = None
        mul_168: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_148, 0.1767766952966369);  getitem_148 = None
        expand_80: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_168, _shape_param_694);  mul_168 = _shape_param_694 = None
        clone_143: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_554: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_143, _shape_param_695);  clone_143 = _shape_param_695 = None
        constant_pad_nd_80: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_554, _shape_param_696);  view_554 = _shape_param_696 = None
        permute_208: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_149, [0, 1, 3, 2]);  getitem_149 = None
        expand_81: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_208, _shape_param_697);  permute_208 = _shape_param_697 = None
        clone_144: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_555: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_144, _shape_param_698);  clone_144 = _shape_param_698 = None
        constant_pad_nd_81: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_555, _shape_param_699);  view_555 = _shape_param_699 = None
        bmm_40: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_80, constant_pad_nd_81);  constant_pad_nd_80 = constant_pad_nd_81 = None
        slice_61: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_40, 1, 0, -7);  bmm_40 = None
        slice_62: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_61, 2, 0, -7);  slice_61 = None
        view_556: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_62, _shape_param_700);  slice_62 = _shape_param_700 = None
        view_557: "i64[2401]" = torch.ops.aten.view.default(arg303_1, [-1]);  arg303_1 = None
        index_60: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg304_1, [view_557]);  arg304_1 = view_557 = None
        view_558: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_60, _shape_param_701);  index_60 = _shape_param_701 = None
        permute_209: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_558, [2, 0, 1]);  view_558 = None
        clone_145: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
        unsqueeze_40: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_145, 0);  clone_145 = None
        add_218: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_556, unsqueeze_40);  view_556 = unsqueeze_40 = None
        convert_element_type_168: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(add_218, torch.float32);  add_218 = None
        amax_20: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_168, [-1], True)
        sub_64: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_168, amax_20);  convert_element_type_168 = amax_20 = None
        exp_20: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        sum_21: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_169: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None
        expand_82: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_169, _shape_param_702);  convert_element_type_169 = _shape_param_702 = None
        view_559: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_82, _shape_param_703);  expand_82 = _shape_param_703 = None
        constant_pad_nd_82: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_559, _shape_param_704);  view_559 = _shape_param_704 = None
        expand_83: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_150, _shape_param_705);  getitem_150 = _shape_param_705 = None
        clone_146: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_560: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_146, _shape_param_706);  clone_146 = _shape_param_706 = None
        constant_pad_nd_83: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_560, _shape_param_707);  view_560 = _shape_param_707 = None
        bmm_41: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_82, constant_pad_nd_83);  constant_pad_nd_82 = constant_pad_nd_83 = None
        slice_63: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_41, 1, 0, -7);  bmm_41 = None
        view_561: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_63, _shape_param_708);  slice_63 = _shape_param_708 = None
        permute_210: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_561, [0, 2, 1, 3]);  view_561 = None
        clone_147: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_562: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_147, _shape_param_709);  clone_147 = _shape_param_709 = None
        view_563: "bf16[25088, 512]" = torch.ops.aten.view.default(view_562, _shape_param_710);  view_562 = _shape_param_710 = None
        permute_211: "bf16[512, 512]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_81: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg306_1, view_563, permute_211);  arg306_1 = view_563 = permute_211 = None
        view_564: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_81, _shape_param_711);  addmm_81 = _shape_param_711 = None
        view_565: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_564, _shape_param_712);  view_564 = _shape_param_712 = None
        view_566: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_565, _shape_param_713);  view_565 = _shape_param_713 = None
        permute_212: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_566, [0, 1, 3, 2, 4, 5]);  view_566 = None
        clone_148: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_567: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_148, _shape_param_714);  clone_148 = _shape_param_714 = None
        add_219: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_547, view_567);  view_547 = view_567 = None
        view_568: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_219, _shape_param_715);  add_219 = _shape_param_715 = None
        convert_element_type_170: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_568, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_170, [2], correction = 0, keepdim = True)
        getitem_151: "f32[128, 196, 1]" = var_mean_44[0]
        getitem_152: "f32[128, 196, 1]" = var_mean_44[1];  var_mean_44 = None
        sub_65: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_170, getitem_152);  convert_element_type_170 = getitem_152 = None
        add_220: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_151, 1e-05);  getitem_151 = None
        rsqrt_44: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
        mul_169: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_44);  sub_65 = rsqrt_44 = None
        mul_170: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_169, arg307_1);  mul_169 = arg307_1 = None
        add_221: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_170, arg308_1);  mul_170 = arg308_1 = None
        convert_element_type_171: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_221, torch.bfloat16);  add_221 = None
        view_569: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_171, _shape_param_716);  convert_element_type_171 = _shape_param_716 = None
        permute_213: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        addmm_82: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg310_1, view_569, permute_213);  arg310_1 = view_569 = permute_213 = None
        view_570: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_82, _shape_param_717);  addmm_82 = _shape_param_717 = None
        convert_element_type_172: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_570, torch.float32);  view_570 = None
        mul_171: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_172, 0.5)
        mul_172: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_172, 0.7071067811865476);  convert_element_type_172 = None
        erf_20: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_172);  mul_172 = None
        add_222: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_173: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_171, add_222);  mul_171 = add_222 = None
        convert_element_type_173: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_173, torch.bfloat16);  mul_173 = None
        view_571: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_173, _shape_param_718);  convert_element_type_173 = _shape_param_718 = None
        permute_214: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_83: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg312_1, view_571, permute_214);  arg312_1 = view_571 = permute_214 = None
        view_572: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_83, _shape_param_719);  addmm_83 = _shape_param_719 = None
        add_223: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_568, view_572);  view_568 = view_572 = None
        view_573: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_223, _shape_param_720);  add_223 = _shape_param_720 = None
        convert_element_type_174: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_573, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_174, [3], correction = 0, keepdim = True)
        getitem_153: "f32[128, 14, 14, 1]" = var_mean_45[0]
        getitem_154: "f32[128, 14, 14, 1]" = var_mean_45[1];  var_mean_45 = None
        sub_66: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_174, getitem_154);  convert_element_type_174 = getitem_154 = None
        add_224: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_153, 1e-05);  getitem_153 = None
        rsqrt_45: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
        mul_174: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_45);  sub_66 = rsqrt_45 = None
        mul_175: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_174, arg313_1);  mul_174 = arg313_1 = None
        add_225: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_175, arg314_1);  mul_175 = arg314_1 = None
        convert_element_type_175: "bf16[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(add_225, torch.bfloat16);  add_225 = None
        iota_40: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_226: "i64[14]" = torch.ops.aten.add.Tensor(iota_40, 3);  iota_40 = None
        fmod_40: "i64[14]" = torch.ops.aten.fmod.Scalar(add_226, 14);  add_226 = None
        index_61: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(convert_element_type_175, [None, fmod_40]);  convert_element_type_175 = fmod_40 = None
        iota_41: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_227: "i64[14]" = torch.ops.aten.add.Tensor(iota_41, 3);  iota_41 = None
        fmod_41: "i64[14]" = torch.ops.aten.fmod.Scalar(add_227, 14);  add_227 = None
        index_62: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_61, [None, None, fmod_41]);  index_61 = fmod_41 = None
        view_574: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_62, _shape_param_721);  index_62 = _shape_param_721 = None
        permute_215: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_574, [0, 1, 3, 2, 4, 5]);  view_574 = None
        clone_149: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_575: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_149, _shape_param_722);  clone_149 = _shape_param_722 = None
        view_576: "bf16[512, 49, 512]" = torch.ops.aten.view.default(view_575, _shape_param_723);  view_575 = _shape_param_723 = None
        view_577: "bf16[25088, 512]" = torch.ops.aten.view.default(view_576, _shape_param_724);  view_576 = _shape_param_724 = None
        permute_216: "bf16[512, 1536]" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_84: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(arg316_1, view_577, permute_216);  arg316_1 = view_577 = permute_216 = None
        view_578: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_84, _shape_param_725);  addmm_84 = _shape_param_725 = None
        view_579: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_578, _shape_param_726);  view_578 = _shape_param_726 = None
        permute_217: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_579, [2, 0, 3, 1, 4]);  view_579 = None
        unbind_21 = torch.ops.aten.unbind.int(permute_217);  permute_217 = None
        getitem_155: "bf16[512, 16, 49, 32]" = unbind_21[0]
        getitem_156: "bf16[512, 16, 49, 32]" = unbind_21[1]
        getitem_157: "bf16[512, 16, 49, 32]" = unbind_21[2];  unbind_21 = None
        mul_176: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_155, 0.1767766952966369);  getitem_155 = None
        expand_84: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_176, _shape_param_727);  mul_176 = _shape_param_727 = None
        clone_150: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_580: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_150, _shape_param_728);  clone_150 = _shape_param_728 = None
        constant_pad_nd_84: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_580, _shape_param_729);  view_580 = _shape_param_729 = None
        permute_218: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_156, [0, 1, 3, 2]);  getitem_156 = None
        expand_85: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_218, _shape_param_730);  permute_218 = _shape_param_730 = None
        clone_151: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_581: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_151, _shape_param_731);  clone_151 = _shape_param_731 = None
        constant_pad_nd_85: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_581, _shape_param_732);  view_581 = _shape_param_732 = None
        bmm_42: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_84, constant_pad_nd_85);  constant_pad_nd_84 = constant_pad_nd_85 = None
        slice_64: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_42, 1, 0, -7);  bmm_42 = None
        slice_65: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_64, 2, 0, -7);  slice_64 = None
        view_582: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_65, _shape_param_733);  slice_65 = _shape_param_733 = None
        view_583: "i64[2401]" = torch.ops.aten.view.default(arg317_1, [-1]);  arg317_1 = None
        index_63: "bf16[2401, 16]" = torch.ops.aten.index.Tensor(arg318_1, [view_583]);  arg318_1 = view_583 = None
        view_584: "bf16[49, 49, 16]" = torch.ops.aten.view.default(index_63, _shape_param_734);  index_63 = _shape_param_734 = None
        permute_219: "bf16[16, 49, 49]" = torch.ops.aten.permute.default(view_584, [2, 0, 1]);  view_584 = None
        clone_152: "bf16[16, 49, 49]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
        unsqueeze_41: "bf16[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_152, 0);  clone_152 = None
        add_228: "bf16[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_582, unsqueeze_41);  view_582 = unsqueeze_41 = None
        view_585: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_228, _shape_param_735);  add_228 = _shape_param_735 = None
        unsqueeze_42: "bf16[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg319_1, 1);  arg319_1 = None
        unsqueeze_43: "bf16[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 0);  unsqueeze_42 = None
        add_229: "bf16[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_585, unsqueeze_43);  view_585 = unsqueeze_43 = None
        view_586: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(add_229, _shape_param_736);  add_229 = _shape_param_736 = None
        convert_element_type_176: "f32[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(view_586, torch.float32);  view_586 = None
        amax_21: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_176, [-1], True)
        sub_67: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_176, amax_21);  convert_element_type_176 = amax_21 = None
        exp_21: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_22: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_21: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        convert_element_type_177: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None
        expand_86: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_177, _shape_param_737);  convert_element_type_177 = _shape_param_737 = None
        view_587: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_86, _shape_param_738);  expand_86 = _shape_param_738 = None
        constant_pad_nd_86: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_587, _shape_param_739);  view_587 = _shape_param_739 = None
        expand_87: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_157, _shape_param_740);  getitem_157 = _shape_param_740 = None
        clone_153: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_588: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_153, _shape_param_741);  clone_153 = _shape_param_741 = None
        constant_pad_nd_87: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_588, _shape_param_742);  view_588 = _shape_param_742 = None
        bmm_43: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_86, constant_pad_nd_87);  constant_pad_nd_86 = constant_pad_nd_87 = None
        slice_66: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_43, 1, 0, -7);  bmm_43 = None
        view_589: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_66, _shape_param_743);  slice_66 = _shape_param_743 = None
        permute_220: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_589, [0, 2, 1, 3]);  view_589 = None
        clone_154: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        view_590: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_154, _shape_param_744);  clone_154 = _shape_param_744 = None
        view_591: "bf16[25088, 512]" = torch.ops.aten.view.default(view_590, _shape_param_745);  view_590 = _shape_param_745 = None
        permute_221: "bf16[512, 512]" = torch.ops.aten.permute.default(arg320_1, [1, 0]);  arg320_1 = None
        addmm_85: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg321_1, view_591, permute_221);  arg321_1 = view_591 = permute_221 = None
        view_592: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_85, _shape_param_746);  addmm_85 = _shape_param_746 = None
        view_593: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_592, _shape_param_747);  view_592 = _shape_param_747 = None
        view_594: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_593, _shape_param_748);  view_593 = _shape_param_748 = None
        permute_222: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_594, [0, 1, 3, 2, 4, 5]);  view_594 = None
        clone_155: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
        view_595: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_155, _shape_param_749);  clone_155 = _shape_param_749 = None
        iota_42: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_230: "i64[14]" = torch.ops.aten.add.Tensor(iota_42, 11);  iota_42 = None
        fmod_42: "i64[14]" = torch.ops.aten.fmod.Scalar(add_230, 14);  add_230 = None
        index_64: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_595, [None, fmod_42]);  view_595 = fmod_42 = None
        iota_43: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_231: "i64[14]" = torch.ops.aten.add.Tensor(iota_43, 11);  iota_43 = None
        fmod_43: "i64[14]" = torch.ops.aten.fmod.Scalar(add_231, 14);  add_231 = None
        index_65: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_64, [None, None, fmod_43]);  index_64 = fmod_43 = None
        add_232: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_573, index_65);  view_573 = index_65 = None
        view_596: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_232, _shape_param_750);  add_232 = _shape_param_750 = None
        convert_element_type_178: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_596, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_178, [2], correction = 0, keepdim = True)
        getitem_158: "f32[128, 196, 1]" = var_mean_46[0]
        getitem_159: "f32[128, 196, 1]" = var_mean_46[1];  var_mean_46 = None
        sub_68: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_178, getitem_159);  convert_element_type_178 = getitem_159 = None
        add_233: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_46: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
        mul_177: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_46);  sub_68 = rsqrt_46 = None
        mul_178: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_177, arg322_1);  mul_177 = arg322_1 = None
        add_234: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_178, arg323_1);  mul_178 = arg323_1 = None
        convert_element_type_179: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_234, torch.bfloat16);  add_234 = None
        view_597: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_179, _shape_param_751);  convert_element_type_179 = _shape_param_751 = None
        permute_223: "bf16[512, 2048]" = torch.ops.aten.permute.default(arg324_1, [1, 0]);  arg324_1 = None
        addmm_86: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(arg325_1, view_597, permute_223);  arg325_1 = view_597 = permute_223 = None
        view_598: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_86, _shape_param_752);  addmm_86 = _shape_param_752 = None
        convert_element_type_180: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_598, torch.float32);  view_598 = None
        mul_179: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_180, 0.5)
        mul_180: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_180, 0.7071067811865476);  convert_element_type_180 = None
        erf_21: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_180);  mul_180 = None
        add_235: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_181: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_179, add_235);  mul_179 = add_235 = None
        convert_element_type_181: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_181, torch.bfloat16);  mul_181 = None
        view_599: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_181, _shape_param_753);  convert_element_type_181 = _shape_param_753 = None
        permute_224: "bf16[2048, 512]" = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
        addmm_87: "bf16[25088, 512]" = torch.ops.aten.addmm.default(arg327_1, view_599, permute_224);  arg327_1 = view_599 = permute_224 = None
        view_600: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_87, _shape_param_754);  addmm_87 = _shape_param_754 = None
        add_236: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_596, view_600);  view_596 = view_600 = None
        view_601: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_236, _shape_param_755);  add_236 = _shape_param_755 = None
        view_602: "bf16[128, 7, 2, 7, 2, 512]" = torch.ops.aten.view.default(view_601, _shape_param_756);  view_601 = _shape_param_756 = None
        permute_225: "bf16[128, 7, 7, 2, 2, 512]" = torch.ops.aten.permute.default(view_602, [0, 1, 3, 4, 2, 5]);  view_602 = None
        clone_156: "bf16[128, 7, 7, 2, 2, 512]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_603: "bf16[128, 7, 7, 2048]" = torch.ops.aten.view.default(clone_156, _shape_param_757);  clone_156 = _shape_param_757 = None
        convert_element_type_182: "f32[128, 7, 7, 2048]" = torch.ops.prims.convert_element_type.default(view_603, torch.float32);  view_603 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_182, [3], correction = 0, keepdim = True)
        getitem_160: "f32[128, 7, 7, 1]" = var_mean_47[0]
        getitem_161: "f32[128, 7, 7, 1]" = var_mean_47[1];  var_mean_47 = None
        sub_69: "f32[128, 7, 7, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_182, getitem_161);  convert_element_type_182 = getitem_161 = None
        add_237: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_47: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_237);  add_237 = None
        mul_182: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_47);  sub_69 = rsqrt_47 = None
        mul_183: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_182, arg328_1);  mul_182 = arg328_1 = None
        add_238: "f32[128, 7, 7, 2048]" = torch.ops.aten.add.Tensor(mul_183, arg329_1);  mul_183 = arg329_1 = None
        convert_element_type_183: "bf16[128, 7, 7, 2048]" = torch.ops.prims.convert_element_type.default(add_238, torch.bfloat16);  add_238 = None
        view_604: "bf16[6272, 2048]" = torch.ops.aten.view.default(convert_element_type_183, _shape_param_758);  convert_element_type_183 = _shape_param_758 = None
        permute_226: "bf16[2048, 1024]" = torch.ops.aten.permute.default(arg330_1, [1, 0]);  arg330_1 = None
        mm_2: "bf16[6272, 1024]" = torch.ops.aten.mm.default(view_604, permute_226);  view_604 = permute_226 = None
        view_605: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(mm_2, _shape_param_759);  mm_2 = _shape_param_759 = None
        convert_element_type_184: "f32[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(view_605, torch.float32)
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_184, [3], correction = 0, keepdim = True)
        getitem_162: "f32[128, 7, 7, 1]" = var_mean_48[0]
        getitem_163: "f32[128, 7, 7, 1]" = var_mean_48[1];  var_mean_48 = None
        sub_70: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_184, getitem_163);  convert_element_type_184 = getitem_163 = None
        add_239: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05);  getitem_162 = None
        rsqrt_48: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        mul_184: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_48);  sub_70 = rsqrt_48 = None
        mul_185: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_184, arg331_1);  mul_184 = arg331_1 = None
        add_240: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_185, arg332_1);  mul_185 = arg332_1 = None
        convert_element_type_185: "bf16[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(add_240, torch.bfloat16);  add_240 = None
        view_606: "bf16[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.view.default(convert_element_type_185, _shape_param_760);  convert_element_type_185 = _shape_param_760 = None
        permute_227: "bf16[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_606, [0, 1, 3, 2, 4, 5]);  view_606 = None
        view_607: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(permute_227, _shape_param_761);  permute_227 = _shape_param_761 = None
        view_608: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(view_607, _shape_param_762);  view_607 = _shape_param_762 = None
        view_609: "bf16[6272, 1024]" = torch.ops.aten.view.default(view_608, _shape_param_763);  view_608 = _shape_param_763 = None
        permute_228: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
        addmm_88: "bf16[6272, 3072]" = torch.ops.aten.addmm.default(arg334_1, view_609, permute_228);  arg334_1 = view_609 = permute_228 = None
        view_610: "bf16[128, 49, 3072]" = torch.ops.aten.view.default(addmm_88, _shape_param_764);  addmm_88 = _shape_param_764 = None
        view_611: "bf16[128, 49, 3, 32, 32]" = torch.ops.aten.view.default(view_610, _shape_param_765);  view_610 = _shape_param_765 = None
        permute_229: "bf16[3, 128, 32, 49, 32]" = torch.ops.aten.permute.default(view_611, [2, 0, 3, 1, 4]);  view_611 = None
        unbind_22 = torch.ops.aten.unbind.int(permute_229);  permute_229 = None
        getitem_164: "bf16[128, 32, 49, 32]" = unbind_22[0]
        getitem_165: "bf16[128, 32, 49, 32]" = unbind_22[1]
        getitem_166: "bf16[128, 32, 49, 32]" = unbind_22[2];  unbind_22 = None
        mul_186: "bf16[128, 32, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_164, 0.1767766952966369);  getitem_164 = None
        expand_88: "bf16[128, 32, 49, 32]" = torch.ops.aten.expand.default(mul_186, _shape_param_766);  mul_186 = _shape_param_766 = None
        clone_157: "bf16[128, 32, 49, 32]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_612: "bf16[4096, 49, 32]" = torch.ops.aten.view.default(clone_157, _shape_param_767);  clone_157 = _shape_param_767 = None
        constant_pad_nd_88: "bf16[4096, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_612, _shape_param_768);  view_612 = _shape_param_768 = None
        permute_230: "bf16[128, 32, 32, 49]" = torch.ops.aten.permute.default(getitem_165, [0, 1, 3, 2]);  getitem_165 = None
        expand_89: "bf16[128, 32, 32, 49]" = torch.ops.aten.expand.default(permute_230, _shape_param_769);  permute_230 = _shape_param_769 = None
        clone_158: "bf16[128, 32, 32, 49]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_613: "bf16[4096, 32, 49]" = torch.ops.aten.view.default(clone_158, _shape_param_770);  clone_158 = _shape_param_770 = None
        constant_pad_nd_89: "bf16[4096, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_613, _shape_param_771);  view_613 = _shape_param_771 = None
        bmm_44: "bf16[4096, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_88, constant_pad_nd_89);  constant_pad_nd_88 = constant_pad_nd_89 = None
        slice_67: "bf16[4096, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_44, 1, 0, -7);  bmm_44 = None
        slice_68: "bf16[4096, 49, 49]" = torch.ops.aten.slice.Tensor(slice_67, 2, 0, -7);  slice_67 = None
        view_614: "bf16[128, 32, 49, 49]" = torch.ops.aten.view.default(slice_68, _shape_param_772);  slice_68 = _shape_param_772 = None
        view_615: "i64[2401]" = torch.ops.aten.view.default(arg335_1, [-1]);  arg335_1 = None
        index_66: "bf16[2401, 32]" = torch.ops.aten.index.Tensor(arg336_1, [view_615]);  arg336_1 = view_615 = None
        view_616: "bf16[49, 49, 32]" = torch.ops.aten.view.default(index_66, _shape_param_773);  index_66 = _shape_param_773 = None
        permute_231: "bf16[32, 49, 49]" = torch.ops.aten.permute.default(view_616, [2, 0, 1]);  view_616 = None
        clone_159: "bf16[32, 49, 49]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
        unsqueeze_44: "bf16[1, 32, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_159, 0);  clone_159 = None
        add_241: "bf16[128, 32, 49, 49]" = torch.ops.aten.add.Tensor(view_614, unsqueeze_44);  view_614 = unsqueeze_44 = None
        convert_element_type_186: "f32[128, 32, 49, 49]" = torch.ops.prims.convert_element_type.default(add_241, torch.float32);  add_241 = None
        amax_22: "f32[128, 32, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_186, [-1], True)
        sub_71: "f32[128, 32, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_186, amax_22);  convert_element_type_186 = amax_22 = None
        exp_22: "f32[128, 32, 49, 49]" = torch.ops.aten.exp.default(sub_71);  sub_71 = None
        sum_23: "f32[128, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22: "f32[128, 32, 49, 49]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_187: "bf16[128, 32, 49, 49]" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None
        expand_90: "bf16[128, 32, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_187, _shape_param_774);  convert_element_type_187 = _shape_param_774 = None
        view_617: "bf16[4096, 49, 49]" = torch.ops.aten.view.default(expand_90, _shape_param_775);  expand_90 = _shape_param_775 = None
        constant_pad_nd_90: "bf16[4096, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_617, _shape_param_776);  view_617 = _shape_param_776 = None
        expand_91: "bf16[128, 32, 49, 32]" = torch.ops.aten.expand.default(getitem_166, _shape_param_777);  getitem_166 = _shape_param_777 = None
        clone_160: "bf16[128, 32, 49, 32]" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_618: "bf16[4096, 49, 32]" = torch.ops.aten.view.default(clone_160, _shape_param_778);  clone_160 = _shape_param_778 = None
        constant_pad_nd_91: "bf16[4096, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_618, _shape_param_779);  view_618 = _shape_param_779 = None
        bmm_45: "bf16[4096, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_90, constant_pad_nd_91);  constant_pad_nd_90 = constant_pad_nd_91 = None
        slice_69: "bf16[4096, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_45, 1, 0, -7);  bmm_45 = None
        view_619: "bf16[128, 32, 49, 32]" = torch.ops.aten.view.default(slice_69, _shape_param_780);  slice_69 = _shape_param_780 = None
        permute_232: "bf16[128, 49, 32, 32]" = torch.ops.aten.permute.default(view_619, [0, 2, 1, 3]);  view_619 = None
        clone_161: "bf16[128, 49, 32, 32]" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
        view_620: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(clone_161, _shape_param_781);  clone_161 = _shape_param_781 = None
        view_621: "bf16[6272, 1024]" = torch.ops.aten.view.default(view_620, _shape_param_782);  view_620 = _shape_param_782 = None
        permute_233: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        addmm_89: "bf16[6272, 1024]" = torch.ops.aten.addmm.default(arg338_1, view_621, permute_233);  arg338_1 = view_621 = permute_233 = None
        view_622: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(addmm_89, _shape_param_783);  addmm_89 = _shape_param_783 = None
        view_623: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(view_622, _shape_param_784);  view_622 = _shape_param_784 = None
        view_624: "bf16[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.view.default(view_623, _shape_param_785);  view_623 = _shape_param_785 = None
        permute_234: "bf16[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_624, [0, 1, 3, 2, 4, 5]);  view_624 = None
        view_625: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(permute_234, _shape_param_786);  permute_234 = _shape_param_786 = None
        add_242: "bf16[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_605, view_625);  view_605 = view_625 = None
        view_626: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(add_242, _shape_param_787);  add_242 = _shape_param_787 = None
        convert_element_type_188: "f32[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(view_626, torch.float32)
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_188, [2], correction = 0, keepdim = True)
        getitem_167: "f32[128, 49, 1]" = var_mean_49[0]
        getitem_168: "f32[128, 49, 1]" = var_mean_49[1];  var_mean_49 = None
        sub_72: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_188, getitem_168);  convert_element_type_188 = getitem_168 = None
        add_243: "f32[128, 49, 1]" = torch.ops.aten.add.Tensor(getitem_167, 1e-05);  getitem_167 = None
        rsqrt_49: "f32[128, 49, 1]" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
        mul_187: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_49);  sub_72 = rsqrt_49 = None
        mul_188: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_187, arg339_1);  mul_187 = arg339_1 = None
        add_244: "f32[128, 49, 1024]" = torch.ops.aten.add.Tensor(mul_188, arg340_1);  mul_188 = arg340_1 = None
        convert_element_type_189: "bf16[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(add_244, torch.bfloat16);  add_244 = None
        view_627: "bf16[6272, 1024]" = torch.ops.aten.view.default(convert_element_type_189, _shape_param_788);  convert_element_type_189 = _shape_param_788 = None
        permute_235: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_90: "bf16[6272, 4096]" = torch.ops.aten.addmm.default(arg342_1, view_627, permute_235);  arg342_1 = view_627 = permute_235 = None
        view_628: "bf16[128, 49, 4096]" = torch.ops.aten.view.default(addmm_90, _shape_param_789);  addmm_90 = _shape_param_789 = None
        convert_element_type_190: "f32[128, 49, 4096]" = torch.ops.prims.convert_element_type.default(view_628, torch.float32);  view_628 = None
        mul_189: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_190, 0.5)
        mul_190: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_190, 0.7071067811865476);  convert_element_type_190 = None
        erf_22: "f32[128, 49, 4096]" = torch.ops.aten.erf.default(mul_190);  mul_190 = None
        add_245: "f32[128, 49, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_191: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(mul_189, add_245);  mul_189 = add_245 = None
        convert_element_type_191: "bf16[128, 49, 4096]" = torch.ops.prims.convert_element_type.default(mul_191, torch.bfloat16);  mul_191 = None
        view_629: "bf16[6272, 4096]" = torch.ops.aten.view.default(convert_element_type_191, _shape_param_790);  convert_element_type_191 = _shape_param_790 = None
        permute_236: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_91: "bf16[6272, 1024]" = torch.ops.aten.addmm.default(arg344_1, view_629, permute_236);  arg344_1 = view_629 = permute_236 = None
        view_630: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(addmm_91, _shape_param_791);  addmm_91 = _shape_param_791 = None
        add_246: "bf16[128, 49, 1024]" = torch.ops.aten.add.Tensor(view_626, view_630);  view_626 = view_630 = None
        view_631: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(add_246, _shape_param_792);  add_246 = _shape_param_792 = None
        convert_element_type_192: "f32[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(view_631, torch.float32)
        var_mean_50 = torch.ops.aten.var_mean.correction(convert_element_type_192, [3], correction = 0, keepdim = True)
        getitem_169: "f32[128, 7, 7, 1]" = var_mean_50[0]
        getitem_170: "f32[128, 7, 7, 1]" = var_mean_50[1];  var_mean_50 = None
        sub_73: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_192, getitem_170);  convert_element_type_192 = getitem_170 = None
        add_247: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_169, 1e-05);  getitem_169 = None
        rsqrt_50: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        mul_192: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_50);  sub_73 = rsqrt_50 = None
        mul_193: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_192, arg345_1);  mul_192 = arg345_1 = None
        add_248: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_193, arg346_1);  mul_193 = arg346_1 = None
        convert_element_type_193: "bf16[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(add_248, torch.bfloat16);  add_248 = None
        view_632: "bf16[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.view.default(convert_element_type_193, _shape_param_793);  convert_element_type_193 = _shape_param_793 = None
        permute_237: "bf16[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_632, [0, 1, 3, 2, 4, 5]);  view_632 = None
        view_633: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(permute_237, _shape_param_794);  permute_237 = _shape_param_794 = None
        view_634: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(view_633, _shape_param_795);  view_633 = _shape_param_795 = None
        view_635: "bf16[6272, 1024]" = torch.ops.aten.view.default(view_634, _shape_param_796);  view_634 = _shape_param_796 = None
        permute_238: "bf16[1024, 3072]" = torch.ops.aten.permute.default(arg347_1, [1, 0]);  arg347_1 = None
        addmm_92: "bf16[6272, 3072]" = torch.ops.aten.addmm.default(arg348_1, view_635, permute_238);  arg348_1 = view_635 = permute_238 = None
        view_636: "bf16[128, 49, 3072]" = torch.ops.aten.view.default(addmm_92, _shape_param_797);  addmm_92 = _shape_param_797 = None
        view_637: "bf16[128, 49, 3, 32, 32]" = torch.ops.aten.view.default(view_636, _shape_param_798);  view_636 = _shape_param_798 = None
        permute_239: "bf16[3, 128, 32, 49, 32]" = torch.ops.aten.permute.default(view_637, [2, 0, 3, 1, 4]);  view_637 = None
        unbind_23 = torch.ops.aten.unbind.int(permute_239);  permute_239 = None
        getitem_171: "bf16[128, 32, 49, 32]" = unbind_23[0]
        getitem_172: "bf16[128, 32, 49, 32]" = unbind_23[1]
        getitem_173: "bf16[128, 32, 49, 32]" = unbind_23[2];  unbind_23 = None
        mul_194: "bf16[128, 32, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_171, 0.1767766952966369);  getitem_171 = None
        expand_92: "bf16[128, 32, 49, 32]" = torch.ops.aten.expand.default(mul_194, _shape_param_799);  mul_194 = _shape_param_799 = None
        clone_162: "bf16[128, 32, 49, 32]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_638: "bf16[4096, 49, 32]" = torch.ops.aten.view.default(clone_162, _shape_param_800);  clone_162 = _shape_param_800 = None
        constant_pad_nd_92: "bf16[4096, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_638, _shape_param_801);  view_638 = _shape_param_801 = None
        permute_240: "bf16[128, 32, 32, 49]" = torch.ops.aten.permute.default(getitem_172, [0, 1, 3, 2]);  getitem_172 = None
        expand_93: "bf16[128, 32, 32, 49]" = torch.ops.aten.expand.default(permute_240, _shape_param_802);  permute_240 = _shape_param_802 = None
        clone_163: "bf16[128, 32, 32, 49]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_639: "bf16[4096, 32, 49]" = torch.ops.aten.view.default(clone_163, _shape_param_803);  clone_163 = _shape_param_803 = None
        constant_pad_nd_93: "bf16[4096, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_639, _shape_param_804);  view_639 = _shape_param_804 = None
        bmm_46: "bf16[4096, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_92, constant_pad_nd_93);  constant_pad_nd_92 = constant_pad_nd_93 = None
        slice_70: "bf16[4096, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_46, 1, 0, -7);  bmm_46 = None
        slice_71: "bf16[4096, 49, 49]" = torch.ops.aten.slice.Tensor(slice_70, 2, 0, -7);  slice_70 = None
        view_640: "bf16[128, 32, 49, 49]" = torch.ops.aten.view.default(slice_71, _shape_param_805);  slice_71 = _shape_param_805 = None
        view_641: "i64[2401]" = torch.ops.aten.view.default(arg349_1, [-1]);  arg349_1 = None
        index_67: "bf16[2401, 32]" = torch.ops.aten.index.Tensor(arg350_1, [view_641]);  arg350_1 = view_641 = None
        view_642: "bf16[49, 49, 32]" = torch.ops.aten.view.default(index_67, _shape_param_806);  index_67 = _shape_param_806 = None
        permute_241: "bf16[32, 49, 49]" = torch.ops.aten.permute.default(view_642, [2, 0, 1]);  view_642 = None
        clone_164: "bf16[32, 49, 49]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
        unsqueeze_45: "bf16[1, 32, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_164, 0);  clone_164 = None
        add_249: "bf16[128, 32, 49, 49]" = torch.ops.aten.add.Tensor(view_640, unsqueeze_45);  view_640 = unsqueeze_45 = None
        convert_element_type_194: "f32[128, 32, 49, 49]" = torch.ops.prims.convert_element_type.default(add_249, torch.float32);  add_249 = None
        amax_23: "f32[128, 32, 49, 1]" = torch.ops.aten.amax.default(convert_element_type_194, [-1], True)
        sub_74: "f32[128, 32, 49, 49]" = torch.ops.aten.sub.Tensor(convert_element_type_194, amax_23);  convert_element_type_194 = amax_23 = None
        exp_23: "f32[128, 32, 49, 49]" = torch.ops.aten.exp.default(sub_74);  sub_74 = None
        sum_24: "f32[128, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_23: "f32[128, 32, 49, 49]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        convert_element_type_195: "bf16[128, 32, 49, 49]" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None
        expand_94: "bf16[128, 32, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_195, _shape_param_807);  convert_element_type_195 = _shape_param_807 = None
        view_643: "bf16[4096, 49, 49]" = torch.ops.aten.view.default(expand_94, _shape_param_808);  expand_94 = _shape_param_808 = None
        constant_pad_nd_94: "bf16[4096, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_643, _shape_param_809);  view_643 = _shape_param_809 = None
        expand_95: "bf16[128, 32, 49, 32]" = torch.ops.aten.expand.default(getitem_173, _shape_param_810);  getitem_173 = _shape_param_810 = None
        clone_165: "bf16[128, 32, 49, 32]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_644: "bf16[4096, 49, 32]" = torch.ops.aten.view.default(clone_165, _shape_param_811);  clone_165 = _shape_param_811 = None
        constant_pad_nd_95: "bf16[4096, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_644, _shape_param_812);  view_644 = _shape_param_812 = None
        bmm_47: "bf16[4096, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_94, constant_pad_nd_95);  constant_pad_nd_94 = constant_pad_nd_95 = None
        slice_72: "bf16[4096, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_47, 1, 0, -7);  bmm_47 = None
        view_645: "bf16[128, 32, 49, 32]" = torch.ops.aten.view.default(slice_72, _shape_param_813);  slice_72 = _shape_param_813 = None
        permute_242: "bf16[128, 49, 32, 32]" = torch.ops.aten.permute.default(view_645, [0, 2, 1, 3]);  view_645 = None
        clone_166: "bf16[128, 49, 32, 32]" = torch.ops.aten.clone.default(permute_242, memory_format = torch.contiguous_format);  permute_242 = None
        view_646: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(clone_166, _shape_param_814);  clone_166 = _shape_param_814 = None
        view_647: "bf16[6272, 1024]" = torch.ops.aten.view.default(view_646, _shape_param_815);  view_646 = _shape_param_815 = None
        permute_243: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
        addmm_93: "bf16[6272, 1024]" = torch.ops.aten.addmm.default(arg352_1, view_647, permute_243);  arg352_1 = view_647 = permute_243 = None
        view_648: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(addmm_93, _shape_param_816);  addmm_93 = _shape_param_816 = None
        view_649: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(view_648, _shape_param_817);  view_648 = _shape_param_817 = None
        view_650: "bf16[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.view.default(view_649, _shape_param_818);  view_649 = _shape_param_818 = None
        permute_244: "bf16[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_650, [0, 1, 3, 2, 4, 5]);  view_650 = None
        view_651: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(permute_244, _shape_param_819);  permute_244 = _shape_param_819 = None
        add_250: "bf16[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_631, view_651);  view_631 = view_651 = None
        view_652: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(add_250, _shape_param_820);  add_250 = _shape_param_820 = None
        convert_element_type_196: "f32[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(view_652, torch.float32)
        var_mean_51 = torch.ops.aten.var_mean.correction(convert_element_type_196, [2], correction = 0, keepdim = True)
        getitem_174: "f32[128, 49, 1]" = var_mean_51[0]
        getitem_175: "f32[128, 49, 1]" = var_mean_51[1];  var_mean_51 = None
        sub_75: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_196, getitem_175);  convert_element_type_196 = getitem_175 = None
        add_251: "f32[128, 49, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_51: "f32[128, 49, 1]" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        mul_195: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_51);  sub_75 = rsqrt_51 = None
        mul_196: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_195, arg353_1);  mul_195 = arg353_1 = None
        add_252: "f32[128, 49, 1024]" = torch.ops.aten.add.Tensor(mul_196, arg354_1);  mul_196 = arg354_1 = None
        convert_element_type_197: "bf16[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(add_252, torch.bfloat16);  add_252 = None
        view_653: "bf16[6272, 1024]" = torch.ops.aten.view.default(convert_element_type_197, _shape_param_821);  convert_element_type_197 = _shape_param_821 = None
        permute_245: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg355_1, [1, 0]);  arg355_1 = None
        addmm_94: "bf16[6272, 4096]" = torch.ops.aten.addmm.default(arg356_1, view_653, permute_245);  arg356_1 = view_653 = permute_245 = None
        view_654: "bf16[128, 49, 4096]" = torch.ops.aten.view.default(addmm_94, _shape_param_822);  addmm_94 = _shape_param_822 = None
        convert_element_type_198: "f32[128, 49, 4096]" = torch.ops.prims.convert_element_type.default(view_654, torch.float32);  view_654 = None
        mul_197: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_198, 0.5)
        mul_198: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_198, 0.7071067811865476);  convert_element_type_198 = None
        erf_23: "f32[128, 49, 4096]" = torch.ops.aten.erf.default(mul_198);  mul_198 = None
        add_253: "f32[128, 49, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_199: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(mul_197, add_253);  mul_197 = add_253 = None
        convert_element_type_199: "bf16[128, 49, 4096]" = torch.ops.prims.convert_element_type.default(mul_199, torch.bfloat16);  mul_199 = None
        view_655: "bf16[6272, 4096]" = torch.ops.aten.view.default(convert_element_type_199, _shape_param_823);  convert_element_type_199 = _shape_param_823 = None
        permute_246: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        addmm_95: "bf16[6272, 1024]" = torch.ops.aten.addmm.default(arg358_1, view_655, permute_246);  arg358_1 = view_655 = permute_246 = None
        view_656: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(addmm_95, _shape_param_824);  addmm_95 = _shape_param_824 = None
        add_254: "bf16[128, 49, 1024]" = torch.ops.aten.add.Tensor(view_652, view_656);  view_652 = view_656 = None
        view_657: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(add_254, _shape_param_825);  add_254 = _shape_param_825 = None
        convert_element_type_200: "f32[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(view_657, torch.float32);  view_657 = None
        var_mean_52 = torch.ops.aten.var_mean.correction(convert_element_type_200, [3], correction = 0, keepdim = True)
        getitem_176: "f32[128, 7, 7, 1]" = var_mean_52[0]
        getitem_177: "f32[128, 7, 7, 1]" = var_mean_52[1];  var_mean_52 = None
        sub_76: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_200, getitem_177);  convert_element_type_200 = getitem_177 = None
        add_255: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_52: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        mul_200: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_52);  sub_76 = rsqrt_52 = None
        mul_201: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_200, arg359_1);  mul_200 = arg359_1 = None
        add_256: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_201, arg360_1);  mul_201 = arg360_1 = None
        convert_element_type_201: "bf16[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(add_256, torch.bfloat16);  add_256 = None
        mean: "bf16[128, 1024]" = torch.ops.aten.mean.dim(convert_element_type_201, [1, 2]);  convert_element_type_201 = None
        permute_247: "bf16[1024, 1000]" = torch.ops.aten.permute.default(arg361_1, [1, 0]);  arg361_1 = None
        addmm_96: "bf16[128, 1000]" = torch.ops.aten.addmm.default(arg362_1, mean, permute_247);  arg362_1 = mean = permute_247 = None
        return addmm_96



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
