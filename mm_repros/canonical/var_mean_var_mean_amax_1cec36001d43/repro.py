"""
Standalone repro captured via capture_hook.
Label: swin_base_patch4_window7_224
Pattern hash: 1cec36001d43
Shape hash: bb6b3d94
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
    def forward(self, arg0_1: "bf16[128, 128, 56, 56]", arg1_1: "f32[128]", arg2_1: "f32[128]", arg3_1: "f32[128]", arg4_1: "f32[128]", arg5_1: "f32[384]", arg6_1: "f32[384, 128]", arg7_1: "i64[49, 49]", arg8_1: "f32[169, 4]", arg9_1: "f32[128]", arg10_1: "f32[128, 128]", arg11_1: "f32[128]", arg12_1: "f32[128]", arg13_1: "f32[512]", arg14_1: "f32[512, 128]", arg15_1: "f32[128]", arg16_1: "f32[128, 512]", arg17_1: "f32[128]", arg18_1: "f32[128]", arg19_1: "f32[384]", arg20_1: "f32[384, 128]", arg21_1: "i64[49, 49]", arg22_1: "f32[169, 4]", arg23_1: "f32[64, 49, 49]", arg24_1: "f32[128]", arg25_1: "f32[128, 128]", arg26_1: "f32[128]", arg27_1: "f32[128]", arg28_1: "f32[512]", arg29_1: "f32[512, 128]", arg30_1: "f32[128]", arg31_1: "f32[128, 512]", arg32_1: "f32[512]", arg33_1: "f32[512]", arg34_1: "f32[256, 512]", arg35_1: "f32[256]", arg36_1: "f32[256]", arg37_1: "f32[768]", arg38_1: "f32[768, 256]", arg39_1: "i64[49, 49]", arg40_1: "f32[169, 8]", arg41_1: "f32[256]", arg42_1: "f32[256, 256]", arg43_1: "f32[256]", arg44_1: "f32[256]", arg45_1: "f32[1024]", arg46_1: "f32[1024, 256]", arg47_1: "f32[256]", arg48_1: "f32[256, 1024]", arg49_1: "f32[256]", arg50_1: "f32[256]", arg51_1: "f32[768]", arg52_1: "f32[768, 256]", arg53_1: "i64[49, 49]", arg54_1: "f32[169, 8]", arg55_1: "f32[16, 49, 49]", arg56_1: "f32[256]", arg57_1: "f32[256, 256]", arg58_1: "f32[256]", arg59_1: "f32[256]", arg60_1: "f32[1024]", arg61_1: "f32[1024, 256]", arg62_1: "f32[256]", arg63_1: "f32[256, 1024]", arg64_1: "f32[1024]", arg65_1: "f32[1024]", arg66_1: "f32[512, 1024]", arg67_1: "f32[512]", arg68_1: "f32[512]", arg69_1: "f32[1536]", arg70_1: "f32[1536, 512]", arg71_1: "i64[49, 49]", arg72_1: "f32[169, 16]", arg73_1: "f32[512]", arg74_1: "f32[512, 512]", arg75_1: "f32[512]", arg76_1: "f32[512]", arg77_1: "f32[2048]", arg78_1: "f32[2048, 512]", arg79_1: "f32[512]", arg80_1: "f32[512, 2048]", arg81_1: "f32[512]", arg82_1: "f32[512]", arg83_1: "f32[1536]", arg84_1: "f32[1536, 512]", arg85_1: "i64[49, 49]", arg86_1: "f32[169, 16]", arg87_1: "f32[4, 49, 49]", arg88_1: "f32[512]", arg89_1: "f32[512, 512]", arg90_1: "f32[512]", arg91_1: "f32[512]", arg92_1: "f32[2048]", arg93_1: "f32[2048, 512]", arg94_1: "f32[512]", arg95_1: "f32[512, 2048]", arg96_1: "f32[512]", arg97_1: "f32[512]", arg98_1: "f32[1536]", arg99_1: "f32[1536, 512]", arg100_1: "i64[49, 49]", arg101_1: "f32[169, 16]", arg102_1: "f32[512]", arg103_1: "f32[512, 512]", arg104_1: "f32[512]", arg105_1: "f32[512]", arg106_1: "f32[2048]", arg107_1: "f32[2048, 512]", arg108_1: "f32[512]", arg109_1: "f32[512, 2048]", arg110_1: "f32[512]", arg111_1: "f32[512]", arg112_1: "f32[1536]", arg113_1: "f32[1536, 512]", arg114_1: "i64[49, 49]", arg115_1: "f32[169, 16]", arg116_1: "f32[4, 49, 49]", arg117_1: "f32[512]", arg118_1: "f32[512, 512]", arg119_1: "f32[512]", arg120_1: "f32[512]", arg121_1: "f32[2048]", arg122_1: "f32[2048, 512]", arg123_1: "f32[512]", arg124_1: "f32[512, 2048]", arg125_1: "f32[512]", arg126_1: "f32[512]", arg127_1: "f32[1536]", arg128_1: "f32[1536, 512]", arg129_1: "i64[49, 49]", arg130_1: "f32[169, 16]", arg131_1: "f32[512]", arg132_1: "f32[512, 512]", arg133_1: "f32[512]", arg134_1: "f32[512]", arg135_1: "f32[2048]", arg136_1: "f32[2048, 512]", arg137_1: "f32[512]", arg138_1: "f32[512, 2048]", arg139_1: "f32[512]", arg140_1: "f32[512]", arg141_1: "f32[1536]", arg142_1: "f32[1536, 512]", arg143_1: "i64[49, 49]", arg144_1: "f32[169, 16]", arg145_1: "f32[4, 49, 49]", arg146_1: "f32[512]", arg147_1: "f32[512, 512]", arg148_1: "f32[512]", arg149_1: "f32[512]", arg150_1: "f32[2048]", arg151_1: "f32[2048, 512]", arg152_1: "f32[512]", arg153_1: "f32[512, 2048]", arg154_1: "f32[512]", arg155_1: "f32[512]", arg156_1: "f32[1536]", arg157_1: "f32[1536, 512]", arg158_1: "i64[49, 49]", arg159_1: "f32[169, 16]", arg160_1: "f32[512]", arg161_1: "f32[512, 512]", arg162_1: "f32[512]", arg163_1: "f32[512]", arg164_1: "f32[2048]", arg165_1: "f32[2048, 512]", arg166_1: "f32[512]", arg167_1: "f32[512, 2048]", arg168_1: "f32[512]", arg169_1: "f32[512]", arg170_1: "f32[1536]", arg171_1: "f32[1536, 512]", arg172_1: "i64[49, 49]", arg173_1: "f32[169, 16]", arg174_1: "f32[4, 49, 49]", arg175_1: "f32[512]", arg176_1: "f32[512, 512]", arg177_1: "f32[512]", arg178_1: "f32[512]", arg179_1: "f32[2048]", arg180_1: "f32[2048, 512]", arg181_1: "f32[512]", arg182_1: "f32[512, 2048]", arg183_1: "f32[512]", arg184_1: "f32[512]", arg185_1: "f32[1536]", arg186_1: "f32[1536, 512]", arg187_1: "i64[49, 49]", arg188_1: "f32[169, 16]", arg189_1: "f32[512]", arg190_1: "f32[512, 512]", arg191_1: "f32[512]", arg192_1: "f32[512]", arg193_1: "f32[2048]", arg194_1: "f32[2048, 512]", arg195_1: "f32[512]", arg196_1: "f32[512, 2048]", arg197_1: "f32[512]", arg198_1: "f32[512]", arg199_1: "f32[1536]", arg200_1: "f32[1536, 512]", arg201_1: "i64[49, 49]", arg202_1: "f32[169, 16]", arg203_1: "f32[4, 49, 49]", arg204_1: "f32[512]", arg205_1: "f32[512, 512]", arg206_1: "f32[512]", arg207_1: "f32[512]", arg208_1: "f32[2048]", arg209_1: "f32[2048, 512]", arg210_1: "f32[512]", arg211_1: "f32[512, 2048]", arg212_1: "f32[512]", arg213_1: "f32[512]", arg214_1: "f32[1536]", arg215_1: "f32[1536, 512]", arg216_1: "i64[49, 49]", arg217_1: "f32[169, 16]", arg218_1: "f32[512]", arg219_1: "f32[512, 512]", arg220_1: "f32[512]", arg221_1: "f32[512]", arg222_1: "f32[2048]", arg223_1: "f32[2048, 512]", arg224_1: "f32[512]", arg225_1: "f32[512, 2048]", arg226_1: "f32[512]", arg227_1: "f32[512]", arg228_1: "f32[1536]", arg229_1: "f32[1536, 512]", arg230_1: "i64[49, 49]", arg231_1: "f32[169, 16]", arg232_1: "f32[4, 49, 49]", arg233_1: "f32[512]", arg234_1: "f32[512, 512]", arg235_1: "f32[512]", arg236_1: "f32[512]", arg237_1: "f32[2048]", arg238_1: "f32[2048, 512]", arg239_1: "f32[512]", arg240_1: "f32[512, 2048]", arg241_1: "f32[512]", arg242_1: "f32[512]", arg243_1: "f32[1536]", arg244_1: "f32[1536, 512]", arg245_1: "i64[49, 49]", arg246_1: "f32[169, 16]", arg247_1: "f32[512]", arg248_1: "f32[512, 512]", arg249_1: "f32[512]", arg250_1: "f32[512]", arg251_1: "f32[2048]", arg252_1: "f32[2048, 512]", arg253_1: "f32[512]", arg254_1: "f32[512, 2048]", arg255_1: "f32[512]", arg256_1: "f32[512]", arg257_1: "f32[1536]", arg258_1: "f32[1536, 512]", arg259_1: "i64[49, 49]", arg260_1: "f32[169, 16]", arg261_1: "f32[4, 49, 49]", arg262_1: "f32[512]", arg263_1: "f32[512, 512]", arg264_1: "f32[512]", arg265_1: "f32[512]", arg266_1: "f32[2048]", arg267_1: "f32[2048, 512]", arg268_1: "f32[512]", arg269_1: "f32[512, 2048]", arg270_1: "f32[512]", arg271_1: "f32[512]", arg272_1: "f32[1536]", arg273_1: "f32[1536, 512]", arg274_1: "i64[49, 49]", arg275_1: "f32[169, 16]", arg276_1: "f32[512]", arg277_1: "f32[512, 512]", arg278_1: "f32[512]", arg279_1: "f32[512]", arg280_1: "f32[2048]", arg281_1: "f32[2048, 512]", arg282_1: "f32[512]", arg283_1: "f32[512, 2048]", arg284_1: "f32[512]", arg285_1: "f32[512]", arg286_1: "f32[1536]", arg287_1: "f32[1536, 512]", arg288_1: "i64[49, 49]", arg289_1: "f32[169, 16]", arg290_1: "f32[4, 49, 49]", arg291_1: "f32[512]", arg292_1: "f32[512, 512]", arg293_1: "f32[512]", arg294_1: "f32[512]", arg295_1: "f32[2048]", arg296_1: "f32[2048, 512]", arg297_1: "f32[512]", arg298_1: "f32[512, 2048]", arg299_1: "f32[512]", arg300_1: "f32[512]", arg301_1: "f32[1536]", arg302_1: "f32[1536, 512]", arg303_1: "i64[49, 49]", arg304_1: "f32[169, 16]", arg305_1: "f32[512]", arg306_1: "f32[512, 512]", arg307_1: "f32[512]", arg308_1: "f32[512]", arg309_1: "f32[2048]", arg310_1: "f32[2048, 512]", arg311_1: "f32[512]", arg312_1: "f32[512, 2048]", arg313_1: "f32[512]", arg314_1: "f32[512]", arg315_1: "f32[1536]", arg316_1: "f32[1536, 512]", arg317_1: "i64[49, 49]", arg318_1: "f32[169, 16]", arg319_1: "f32[4, 49, 49]", arg320_1: "f32[512]", arg321_1: "f32[512, 512]", arg322_1: "f32[512]", arg323_1: "f32[512]", arg324_1: "f32[2048]", arg325_1: "f32[2048, 512]", arg326_1: "f32[512]", arg327_1: "f32[512, 2048]", arg328_1: "f32[2048]", arg329_1: "f32[2048]", arg330_1: "f32[1024, 2048]", arg331_1: "f32[1024]", arg332_1: "f32[1024]", arg333_1: "f32[3072]", arg334_1: "f32[3072, 1024]", arg335_1: "i64[49, 49]", arg336_1: "f32[169, 32]", arg337_1: "f32[1024]", arg338_1: "f32[1024, 1024]", arg339_1: "f32[1024]", arg340_1: "f32[1024]", arg341_1: "f32[4096]", arg342_1: "f32[4096, 1024]", arg343_1: "f32[1024]", arg344_1: "f32[1024, 4096]", arg345_1: "f32[1024]", arg346_1: "f32[1024]", arg347_1: "f32[3072]", arg348_1: "f32[3072, 1024]", arg349_1: "i64[49, 49]", arg350_1: "f32[169, 32]", arg351_1: "f32[1024]", arg352_1: "f32[1024, 1024]", arg353_1: "f32[1024]", arg354_1: "f32[1024]", arg355_1: "f32[4096]", arg356_1: "f32[4096, 1024]", arg357_1: "f32[1024]", arg358_1: "f32[1024, 4096]", arg359_1: "f32[1024]", arg360_1: "f32[1024]", arg361_1: "f32[1000]", arg362_1: "f32[1000, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631, _shape_param_632, _shape_param_633, _shape_param_634, _shape_param_635, _shape_param_636, _shape_param_637, _shape_param_638, _shape_param_639, _shape_param_640, _shape_param_641, _shape_param_642, _shape_param_643, _shape_param_644, _shape_param_645, _shape_param_646, _shape_param_647, _shape_param_648, _shape_param_649, _shape_param_650, _shape_param_651, _shape_param_652, _shape_param_653, _shape_param_654, _shape_param_655, _shape_param_656, _shape_param_657, _shape_param_658, _shape_param_659, _shape_param_660, _shape_param_661, _shape_param_662, _shape_param_663, _shape_param_664, _shape_param_665, _shape_param_666, _shape_param_667, _shape_param_668, _shape_param_669, _shape_param_670, _shape_param_671, _shape_param_672, _shape_param_673, _shape_param_674, _shape_param_675, _shape_param_676, _shape_param_677, _shape_param_678, _shape_param_679, _shape_param_680, _shape_param_681, _shape_param_682, _shape_param_683, _shape_param_684, _shape_param_685, _shape_param_686, _shape_param_687, _shape_param_688, _shape_param_689, _shape_param_690, _shape_param_691, _shape_param_692, _shape_param_693, _shape_param_694, _shape_param_695, _shape_param_696, _shape_param_697, _shape_param_698, _shape_param_699, _shape_param_700, _shape_param_701, _shape_param_702, _shape_param_703, _shape_param_704, _shape_param_705, _shape_param_706, _shape_param_707, _shape_param_708, _shape_param_709, _shape_param_710, _shape_param_711, _shape_param_712, _shape_param_713, _shape_param_714, _shape_param_715, _shape_param_716, _shape_param_717, _shape_param_718, _shape_param_719, _shape_param_720, _shape_param_721, _shape_param_722, _shape_param_723, _shape_param_724, _shape_param_725, _shape_param_726, _shape_param_727, _shape_param_728, _shape_param_729, _shape_param_730, _shape_param_731, _shape_param_732, _shape_param_733, _shape_param_734, _shape_param_735, _shape_param_736, _shape_param_737, _shape_param_738, _shape_param_739, _shape_param_740, _shape_param_741, _shape_param_742, _shape_param_743, _shape_param_744, _shape_param_745, _shape_param_746, _shape_param_747, _shape_param_748, _shape_param_749, _shape_param_750, _shape_param_751, _shape_param_752, _shape_param_753, _shape_param_754, _shape_param_755, _shape_param_756, _shape_param_757, _shape_param_758, _shape_param_759, _shape_param_760, _shape_param_761, _shape_param_762, _shape_param_763, _shape_param_764, _shape_param_765, _shape_param_766, _shape_param_767, _shape_param_768, _shape_param_769, _shape_param_770, _shape_param_771, _shape_param_772, _shape_param_773, _shape_param_774, _shape_param_775, _shape_param_776, _shape_param_777, _shape_param_778, _shape_param_779, _shape_param_780, _shape_param_781, _shape_param_782, _shape_param_783, _shape_param_784, _shape_param_785, _shape_param_786, _shape_param_787, _shape_param_788, _shape_param_789, _shape_param_790, _shape_param_791, _shape_param_792, _shape_param_793, _shape_param_794, _shape_param_795, _shape_param_796, _shape_param_797, _shape_param_798, _shape_param_799, _shape_param_800, _shape_param_801, _shape_param_802, _shape_param_803, _shape_param_804, _shape_param_805, _shape_param_806, _shape_param_807, _shape_param_808, _shape_param_809, _shape_param_810, _shape_param_811, _shape_param_812, _shape_param_813, _shape_param_814, _shape_param_815, _shape_param_816, _shape_param_817, _shape_param_818, _shape_param_819, _shape_param_820, _shape_param_821, _shape_param_822, _shape_param_823, _shape_param_824, _shape_param_825, _shape_param_826, _shape_param_827, _shape_param_828, _shape_param_829, _shape_param_830, _shape_param_831, _shape_param_832, _shape_param_833, _shape_param_834, _shape_param_835, _shape_param_836, _shape_param_837, _shape_param_838, _shape_param_839, _shape_param_840, _shape_param_841, _shape_param_842, _shape_param_843, _shape_param_844, _shape_param_845, _shape_param_846, _shape_param_847, _shape_param_848, _shape_param_849, _shape_param_850, _shape_param_851, _shape_param_852, _shape_param_853, _shape_param_854, _shape_param_855, _shape_param_856, _shape_param_857, _shape_param_858, _shape_param_859, _shape_param_860, _shape_param_861, _shape_param_862, _shape_param_863, _shape_param_864, _shape_param_865, _shape_param_866, _shape_param_867):
        # No stacktrace found for following nodes
        permute: "bf16[128, 56, 56, 128]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 3, 1]);  arg0_1 = None
        convert_element_type: "f32[128, 56, 56, 128]" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 56, 56, 1]" = var_mean[0]
        getitem_1: "f32[128, 56, 56, 1]" = var_mean[1];  var_mean = None
        add: "f32[128, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[128, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = None
        mul: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        add_1: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_1, [3], correction = 0, keepdim = True)
        getitem_2: "f32[128, 56, 56, 1]" = var_mean_1[0]
        getitem_3: "f32[128, 56, 56, 1]" = var_mean_1[1];  var_mean_1 = None
        add_2: "f32[128, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[128, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_3)
        mul_2: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_3: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_2, arg3_1);  mul_2 = arg3_1 = None
        add_3: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_3, arg4_1);  mul_3 = arg4_1 = None
        view: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(add_3, _shape_param_0);  add_3 = _shape_param_0 = None
        permute_1: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view, [0, 1, 3, 2, 4, 5]);  view = None
        clone: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_1: "f32[8192, 7, 7, 128]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        view_2: "f32[8192, 49, 128]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        convert_element_type_1: "bf16[384]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        convert_element_type_2: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        convert_element_type_3: "bf16[8192, 49, 128]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        view_3: "bf16[401408, 128]" = torch.ops.aten.view.default(convert_element_type_3, _shape_param_3);  convert_element_type_3 = _shape_param_3 = None
        permute_2: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_2, [1, 0]);  convert_element_type_2 = None
        addmm: "bf16[401408, 384]" = torch.ops.aten.addmm.default(convert_element_type_1, view_3, permute_2);  convert_element_type_1 = None
        view_4: "bf16[8192, 49, 384]" = torch.ops.aten.view.default(addmm, _shape_param_4);  addmm = _shape_param_4 = None
        view_5: "bf16[8192, 49, 3, 4, 32]" = torch.ops.aten.view.default(view_4, _shape_param_5);  view_4 = _shape_param_5 = None
        permute_3: "bf16[3, 8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_5, [2, 0, 3, 1, 4]);  view_5 = None
        unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
        getitem_4: "bf16[8192, 4, 49, 32]" = unbind[0]
        getitem_5: "bf16[8192, 4, 49, 32]" = unbind[1]
        getitem_6: "bf16[8192, 4, 49, 32]" = unbind[2];  unbind = None
        mul_4: "bf16[8192, 4, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_4, 0.1767766952966369);  getitem_4 = None
        permute_4: "bf16[8192, 4, 32, 49]" = torch.ops.aten.permute.default(getitem_5, [0, 1, 3, 2]);  getitem_5 = None
        expand: "bf16[8192, 4, 49, 32]" = torch.ops.aten.expand.default(mul_4, _shape_param_6);  mul_4 = _shape_param_6 = None
        clone_1: "bf16[8192, 4, 49, 32]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_6: "bf16[32768, 49, 32]" = torch.ops.aten.view.default(clone_1, _shape_param_7);  clone_1 = _shape_param_7 = None
        expand_1: "bf16[8192, 4, 32, 49]" = torch.ops.aten.expand.default(permute_4, _shape_param_8);  permute_4 = _shape_param_8 = None
        clone_2: "bf16[8192, 4, 32, 49]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_7: "bf16[32768, 32, 49]" = torch.ops.aten.view.default(clone_2, _shape_param_9);  clone_2 = _shape_param_9 = None
        constant_pad_nd: "bf16[32768, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_6, _shape_param_10);  _shape_param_10 = None
        constant_pad_nd_1: "bf16[32768, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_7, _shape_param_11);  _shape_param_11 = None
        bmm: "bf16[32768, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        slice_1: "bf16[32768, 49, 56]" = torch.ops.aten.slice.Tensor(bmm, 1, 0, -7)
        slice_2: "bf16[32768, 49, 49]" = torch.ops.aten.slice.Tensor(slice_1, 2, 0, -7);  slice_1 = None
        view_8: "bf16[8192, 4, 49, 49]" = torch.ops.aten.view.default(slice_2, _shape_param_12);  slice_2 = _shape_param_12 = None
        view_9: "i64[2401]" = torch.ops.aten.view.default(arg7_1, [-1]);  arg7_1 = None
        index: "f32[2401, 4]" = torch.ops.aten.index.Tensor(arg8_1, [view_9]);  arg8_1 = view_9 = None
        view_10: "f32[49, 49, 4]" = torch.ops.aten.view.default(index, _shape_param_13);  index = _shape_param_13 = None
        permute_5: "f32[4, 49, 49]" = torch.ops.aten.permute.default(view_10, [2, 0, 1]);  view_10 = None
        clone_3: "f32[4, 49, 49]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        unsqueeze: "f32[1, 4, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_3, 0);  clone_3 = None
        add_4: "f32[8192, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_8, unsqueeze);  view_8 = unsqueeze = None
        amax: "f32[8192, 4, 49, 1]" = torch.ops.aten.amax.default(add_4, [-1], True)
        sub_2: "f32[8192, 4, 49, 49]" = torch.ops.aten.sub.Tensor(add_4, amax);  add_4 = None
        exp: "f32[8192, 4, 49, 49]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1: "f32[8192, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[8192, 4, 49, 49]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_4: "bf16[8192, 4, 49, 49]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand_2: "bf16[8192, 4, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_4, _shape_param_14);  convert_element_type_4 = _shape_param_14 = None
        view_11: "bf16[32768, 49, 49]" = torch.ops.aten.view.default(expand_2, _shape_param_15);  expand_2 = _shape_param_15 = None
        expand_3: "bf16[8192, 4, 49, 32]" = torch.ops.aten.expand.default(getitem_6, _shape_param_16);  getitem_6 = _shape_param_16 = None
        clone_4: "bf16[8192, 4, 49, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_12: "bf16[32768, 49, 32]" = torch.ops.aten.view.default(clone_4, _shape_param_17);  clone_4 = _shape_param_17 = None
        constant_pad_nd_2: "bf16[32768, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_11, _shape_param_18);  _shape_param_18 = None
        constant_pad_nd_3: "bf16[32768, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_12, _shape_param_19);  _shape_param_19 = None
        bmm_1: "bf16[32768, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_2, constant_pad_nd_3);  constant_pad_nd_2 = constant_pad_nd_3 = None
        slice_3: "bf16[32768, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_1, 1, 0, -7);  bmm_1 = None
        view_13: "bf16[8192, 4, 49, 32]" = torch.ops.aten.view.default(slice_3, _shape_param_20);  slice_3 = _shape_param_20 = None
        permute_6: "bf16[8192, 49, 4, 32]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
        clone_5: "bf16[8192, 49, 4, 32]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_14: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(clone_5, _shape_param_21);  clone_5 = _shape_param_21 = None
        convert_element_type_5: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        convert_element_type_6: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        view_15: "bf16[401408, 128]" = torch.ops.aten.view.default(view_14, _shape_param_22);  view_14 = _shape_param_22 = None
        permute_7: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_6, [1, 0]);  convert_element_type_6 = None
        addmm_1: "bf16[401408, 128]" = torch.ops.aten.addmm.default(convert_element_type_5, view_15, permute_7);  convert_element_type_5 = None
        view_16: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(addmm_1, _shape_param_23);  addmm_1 = _shape_param_23 = None
        view_17: "bf16[8192, 7, 7, 128]" = torch.ops.aten.view.default(view_16, _shape_param_24);  view_16 = _shape_param_24 = None
        view_18: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_17, _shape_param_25);  view_17 = _shape_param_25 = None
        permute_8: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_18, [0, 1, 3, 2, 4, 5]);  view_18 = None
        clone_6: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_19: "bf16[128, 56, 56, 128]" = torch.ops.aten.view.default(clone_6, _shape_param_26);  clone_6 = _shape_param_26 = None
        add_5: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(add_1, view_19);  add_1 = view_19 = None
        view_20: "f32[128, 3136, 128]" = torch.ops.aten.view.default(add_5, _shape_param_27);  add_5 = _shape_param_27 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(view_20, [2], correction = 0, keepdim = True)
        getitem_7: "f32[128, 3136, 1]" = var_mean_2[0]
        getitem_8: "f32[128, 3136, 1]" = var_mean_2[1];  var_mean_2 = None
        add_6: "f32[128, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_7, 1e-05);  getitem_7 = None
        rsqrt_2: "f32[128, 3136, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_3: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(view_20, getitem_8);  getitem_8 = None
        mul_5: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_6: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_5, arg11_1);  arg11_1 = None
        add_7: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(mul_6, arg12_1);  mul_6 = arg12_1 = None
        convert_element_type_7: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        convert_element_type_8: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.bfloat16);  arg14_1 = None
        convert_element_type_9: "bf16[128, 3136, 128]" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None
        view_21: "bf16[401408, 128]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_28);  convert_element_type_9 = _shape_param_28 = None
        permute_9: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_8, [1, 0]);  convert_element_type_8 = None
        addmm_2: "bf16[401408, 512]" = torch.ops.aten.addmm.default(convert_element_type_7, view_21, permute_9);  convert_element_type_7 = None
        view_22: "bf16[128, 3136, 512]" = torch.ops.aten.view.default(addmm_2, _shape_param_29);  _shape_param_29 = None
        convert_element_type_10: "f32[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(view_22, torch.float32);  view_22 = None
        mul_7: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.5)
        mul_8: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.7071067811865476);  convert_element_type_10 = None
        erf: "f32[128, 3136, 512]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_8: "f32[128, 3136, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_7, add_8);  mul_7 = add_8 = None
        convert_element_type_11: "bf16[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(mul_9, torch.bfloat16);  mul_9 = None
        convert_element_type_12: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg15_1, torch.bfloat16);  arg15_1 = None
        convert_element_type_13: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.bfloat16);  arg16_1 = None
        view_23: "bf16[401408, 512]" = torch.ops.aten.view.default(convert_element_type_11, _shape_param_30);  convert_element_type_11 = _shape_param_30 = None
        permute_10: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        addmm_3: "bf16[401408, 128]" = torch.ops.aten.addmm.default(convert_element_type_12, view_23, permute_10);  convert_element_type_12 = None
        view_24: "bf16[128, 3136, 128]" = torch.ops.aten.view.default(addmm_3, _shape_param_31);  addmm_3 = _shape_param_31 = None
        add_9: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(view_20, view_24);  view_20 = view_24 = None
        view_25: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(add_9, _shape_param_32);  add_9 = _shape_param_32 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(view_25, [3], correction = 0, keepdim = True)
        getitem_9: "f32[128, 56, 56, 1]" = var_mean_3[0]
        getitem_10: "f32[128, 56, 56, 1]" = var_mean_3[1];  var_mean_3 = None
        add_10: "f32[128, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-05);  getitem_9 = None
        rsqrt_3: "f32[128, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(view_25, getitem_10);  getitem_10 = None
        mul_10: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_11: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_10, arg17_1);  arg17_1 = None
        add_11: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_11, arg18_1);  mul_11 = arg18_1 = None
        iota: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_12: "i64[56]" = torch.ops.aten.add.Tensor(iota, 3)
        fmod: "i64[56]" = torch.ops.aten.fmod.Scalar(add_12, 56);  add_12 = None
        index_1: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(add_11, [None, fmod]);  add_11 = None
        index_2: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index_1, [None, None, fmod]);  index_1 = fmod = None
        view_26: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(index_2, _shape_param_33);  index_2 = _shape_param_33 = None
        permute_11: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_26, [0, 1, 3, 2, 4, 5]);  view_26 = None
        clone_7: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        view_27: "f32[8192, 7, 7, 128]" = torch.ops.aten.view.default(clone_7, _shape_param_34);  clone_7 = _shape_param_34 = None
        view_28: "f32[8192, 49, 128]" = torch.ops.aten.view.default(view_27, _shape_param_35);  view_27 = _shape_param_35 = None
        convert_element_type_14: "bf16[384]" = torch.ops.prims.convert_element_type.default(arg19_1, torch.bfloat16);  arg19_1 = None
        convert_element_type_15: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.bfloat16);  arg20_1 = None
        convert_element_type_16: "bf16[8192, 49, 128]" = torch.ops.prims.convert_element_type.default(view_28, torch.bfloat16);  view_28 = None
        view_29: "bf16[401408, 128]" = torch.ops.aten.view.default(convert_element_type_16, _shape_param_36);  convert_element_type_16 = _shape_param_36 = None
        permute_12: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_15, [1, 0]);  convert_element_type_15 = None
        addmm_4: "bf16[401408, 384]" = torch.ops.aten.addmm.default(convert_element_type_14, view_29, permute_12);  convert_element_type_14 = None
        view_30: "bf16[8192, 49, 384]" = torch.ops.aten.view.default(addmm_4, _shape_param_37);  addmm_4 = _shape_param_37 = None
        view_31: "bf16[8192, 49, 3, 4, 32]" = torch.ops.aten.view.default(view_30, _shape_param_38);  view_30 = _shape_param_38 = None
        permute_13: "bf16[3, 8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_31, [2, 0, 3, 1, 4]);  view_31 = None
        unbind_1 = torch.ops.aten.unbind.int(permute_13);  permute_13 = None
        getitem_11: "bf16[8192, 4, 49, 32]" = unbind_1[0]
        getitem_12: "bf16[8192, 4, 49, 32]" = unbind_1[1]
        getitem_13: "bf16[8192, 4, 49, 32]" = unbind_1[2];  unbind_1 = None
        mul_12: "bf16[8192, 4, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_11, 0.1767766952966369);  getitem_11 = None
        permute_14: "bf16[8192, 4, 32, 49]" = torch.ops.aten.permute.default(getitem_12, [0, 1, 3, 2]);  getitem_12 = None
        expand_4: "bf16[8192, 4, 49, 32]" = torch.ops.aten.expand.default(mul_12, _shape_param_39);  mul_12 = _shape_param_39 = None
        clone_8: "bf16[8192, 4, 49, 32]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_32: "bf16[32768, 49, 32]" = torch.ops.aten.view.default(clone_8, _shape_param_40);  clone_8 = _shape_param_40 = None
        expand_5: "bf16[8192, 4, 32, 49]" = torch.ops.aten.expand.default(permute_14, _shape_param_41);  permute_14 = _shape_param_41 = None
        clone_9: "bf16[8192, 4, 32, 49]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_33: "bf16[32768, 32, 49]" = torch.ops.aten.view.default(clone_9, _shape_param_42);  clone_9 = _shape_param_42 = None
        constant_pad_nd_4: "bf16[32768, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_32, _shape_param_43);  _shape_param_43 = None
        constant_pad_nd_5: "bf16[32768, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_33, _shape_param_44);  _shape_param_44 = None
        bmm_2: "bf16[32768, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_4, constant_pad_nd_5);  constant_pad_nd_4 = constant_pad_nd_5 = None
        slice_4: "bf16[32768, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_2, 1, 0, -7)
        slice_5: "bf16[32768, 49, 49]" = torch.ops.aten.slice.Tensor(slice_4, 2, 0, -7);  slice_4 = None
        view_34: "bf16[8192, 4, 49, 49]" = torch.ops.aten.view.default(slice_5, _shape_param_45);  slice_5 = _shape_param_45 = None
        view_35: "i64[2401]" = torch.ops.aten.view.default(arg21_1, [-1]);  arg21_1 = None
        index_3: "f32[2401, 4]" = torch.ops.aten.index.Tensor(arg22_1, [view_35]);  arg22_1 = view_35 = None
        view_36: "f32[49, 49, 4]" = torch.ops.aten.view.default(index_3, _shape_param_46);  index_3 = _shape_param_46 = None
        permute_15: "f32[4, 49, 49]" = torch.ops.aten.permute.default(view_36, [2, 0, 1]);  view_36 = None
        clone_10: "f32[4, 49, 49]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        unsqueeze_1: "f32[1, 4, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_10, 0);  clone_10 = None
        add_13: "f32[8192, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_34, unsqueeze_1);  view_34 = unsqueeze_1 = None
        view_37: "f32[128, 64, 4, 49, 49]" = torch.ops.aten.view.default(add_13, _shape_param_47);  add_13 = _shape_param_47 = None
        unsqueeze_2: "f32[64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg23_1, 1);  arg23_1 = None
        unsqueeze_3: "f32[1, 64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 0);  unsqueeze_2 = None
        add_14: "f32[128, 64, 4, 49, 49]" = torch.ops.aten.add.Tensor(view_37, unsqueeze_3);  view_37 = unsqueeze_3 = None
        view_38: "f32[8192, 4, 49, 49]" = torch.ops.aten.view.default(add_14, _shape_param_48);  add_14 = _shape_param_48 = None
        amax_1: "f32[8192, 4, 49, 1]" = torch.ops.aten.amax.default(view_38, [-1], True)
        sub_5: "f32[8192, 4, 49, 49]" = torch.ops.aten.sub.Tensor(view_38, amax_1);  view_38 = None
        exp_1: "f32[8192, 4, 49, 49]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_2: "f32[8192, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[8192, 4, 49, 49]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None
        convert_element_type_17: "bf16[8192, 4, 49, 49]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        expand_6: "bf16[8192, 4, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_17, _shape_param_49);  convert_element_type_17 = _shape_param_49 = None
        view_39: "bf16[32768, 49, 49]" = torch.ops.aten.view.default(expand_6, _shape_param_50);  expand_6 = _shape_param_50 = None
        expand_7: "bf16[8192, 4, 49, 32]" = torch.ops.aten.expand.default(getitem_13, _shape_param_51);  getitem_13 = _shape_param_51 = None
        clone_11: "bf16[8192, 4, 49, 32]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_40: "bf16[32768, 49, 32]" = torch.ops.aten.view.default(clone_11, _shape_param_52);  clone_11 = _shape_param_52 = None
        constant_pad_nd_6: "bf16[32768, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_39, _shape_param_53);  _shape_param_53 = None
        constant_pad_nd_7: "bf16[32768, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_40, _shape_param_54);  _shape_param_54 = None
        bmm_3: "bf16[32768, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_6, constant_pad_nd_7);  constant_pad_nd_6 = constant_pad_nd_7 = None
        slice_6: "bf16[32768, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_3, 1, 0, -7);  bmm_3 = None
        view_41: "bf16[8192, 4, 49, 32]" = torch.ops.aten.view.default(slice_6, _shape_param_55);  slice_6 = _shape_param_55 = None
        permute_16: "bf16[8192, 49, 4, 32]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3]);  view_41 = None
        clone_12: "bf16[8192, 49, 4, 32]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_42: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(clone_12, _shape_param_56);  clone_12 = _shape_param_56 = None
        convert_element_type_18: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg24_1, torch.bfloat16);  arg24_1 = None
        convert_element_type_19: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg25_1, torch.bfloat16);  arg25_1 = None
        view_43: "bf16[401408, 128]" = torch.ops.aten.view.default(view_42, _shape_param_57);  view_42 = _shape_param_57 = None
        permute_17: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_19, [1, 0]);  convert_element_type_19 = None
        addmm_5: "bf16[401408, 128]" = torch.ops.aten.addmm.default(convert_element_type_18, view_43, permute_17);  convert_element_type_18 = None
        view_44: "bf16[8192, 49, 128]" = torch.ops.aten.view.default(addmm_5, _shape_param_58);  addmm_5 = _shape_param_58 = None
        view_45: "bf16[8192, 7, 7, 128]" = torch.ops.aten.view.default(view_44, _shape_param_59);  view_44 = _shape_param_59 = None
        view_46: "bf16[128, 8, 8, 7, 7, 128]" = torch.ops.aten.view.default(view_45, _shape_param_60);  view_45 = _shape_param_60 = None
        permute_18: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_46, [0, 1, 3, 2, 4, 5]);  view_46 = None
        clone_13: "bf16[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_47: "bf16[128, 56, 56, 128]" = torch.ops.aten.view.default(clone_13, _shape_param_61);  clone_13 = _shape_param_61 = None
        add_15: "i64[56]" = torch.ops.aten.add.Tensor(iota, 53);  iota = None
        fmod_1: "i64[56]" = torch.ops.aten.fmod.Scalar(add_15, 56);  add_15 = None
        index_4: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(view_47, [None, fmod_1]);  view_47 = None
        index_5: "bf16[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index_4, [None, None, fmod_1]);  index_4 = None
        inductor_seeds: "i64[46]" = torch.ops.prims.inductor_seeds.default(46, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_62, inductor_lookup_seed, 'rand');  _shape_param_62 = inductor_lookup_seed = None
        lt: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random, 0.9956521736457944);  inductor_random = None
        convert_element_type_20: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt, torch.bfloat16)
        div_2: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_20, 0.9956521736457944);  convert_element_type_20 = None
        mul_13: "bf16[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(index_5, div_2);  index_5 = div_2 = None
        add_16: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_25, mul_13);  view_25 = mul_13 = None
        view_48: "f32[128, 3136, 128]" = torch.ops.aten.view.default(add_16, _shape_param_63);  add_16 = _shape_param_63 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(view_48, [2], correction = 0, keepdim = True)
        getitem_14: "f32[128, 3136, 1]" = var_mean_4[0]
        getitem_15: "f32[128, 3136, 1]" = var_mean_4[1];  var_mean_4 = None
        add_17: "f32[128, 3136, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_4: "f32[128, 3136, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_6: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(view_48, getitem_15);  getitem_15 = None
        mul_14: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_15: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_14, arg26_1);  arg26_1 = None
        add_18: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(mul_15, arg27_1);  mul_15 = arg27_1 = None
        convert_element_type_21: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg28_1, torch.bfloat16);  arg28_1 = None
        convert_element_type_22: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg29_1, torch.bfloat16);  arg29_1 = None
        convert_element_type_23: "bf16[128, 3136, 128]" = torch.ops.prims.convert_element_type.default(add_18, torch.bfloat16);  add_18 = None
        view_49: "bf16[401408, 128]" = torch.ops.aten.view.default(convert_element_type_23, _shape_param_64);  convert_element_type_23 = _shape_param_64 = None
        permute_19: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_22, [1, 0]);  convert_element_type_22 = None
        addmm_6: "bf16[401408, 512]" = torch.ops.aten.addmm.default(convert_element_type_21, view_49, permute_19);  convert_element_type_21 = None
        view_50: "bf16[128, 3136, 512]" = torch.ops.aten.view.default(addmm_6, _shape_param_65);  _shape_param_65 = None
        convert_element_type_24: "f32[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(view_50, torch.float32);  view_50 = None
        mul_16: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.5)
        mul_17: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.7071067811865476);  convert_element_type_24 = None
        erf_1: "f32[128, 3136, 512]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
        add_19: "f32[128, 3136, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_18: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_16, add_19);  mul_16 = add_19 = None
        convert_element_type_25: "bf16[128, 3136, 512]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        convert_element_type_26: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg30_1, torch.bfloat16);  arg30_1 = None
        convert_element_type_27: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg31_1, torch.bfloat16);  arg31_1 = None
        view_51: "bf16[401408, 512]" = torch.ops.aten.view.default(convert_element_type_25, _shape_param_66);  convert_element_type_25 = _shape_param_66 = None
        permute_20: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_27, [1, 0]);  convert_element_type_27 = None
        addmm_7: "bf16[401408, 128]" = torch.ops.aten.addmm.default(convert_element_type_26, view_51, permute_20);  convert_element_type_26 = None
        view_52: "bf16[128, 3136, 128]" = torch.ops.aten.view.default(addmm_7, _shape_param_67);  addmm_7 = _shape_param_67 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1)
        inductor_random_1: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_68, inductor_lookup_seed_1, 'rand');  _shape_param_68 = inductor_lookup_seed_1 = None
        lt_1: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_1, 0.9956521736457944);  inductor_random_1 = None
        convert_element_type_28: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_1, torch.bfloat16)
        div_3: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_28, 0.9956521736457944);  convert_element_type_28 = None
        mul_19: "bf16[128, 3136, 128]" = torch.ops.aten.mul.Tensor(view_52, div_3);  view_52 = div_3 = None
        add_20: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(view_48, mul_19);  view_48 = mul_19 = None
        view_53: "f32[128, 56, 56, 128]" = torch.ops.aten.view.default(add_20, _shape_param_69);  add_20 = _shape_param_69 = None
        view_54: "f32[128, 28, 2, 28, 2, 128]" = torch.ops.aten.view.default(view_53, _shape_param_70);  view_53 = _shape_param_70 = None
        permute_21: "f32[128, 28, 28, 2, 2, 128]" = torch.ops.aten.permute.default(view_54, [0, 1, 3, 4, 2, 5]);  view_54 = None
        clone_14: "f32[128, 28, 28, 2, 2, 128]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
        view_55: "f32[128, 28, 28, 512]" = torch.ops.aten.view.default(clone_14, _shape_param_71);  clone_14 = _shape_param_71 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(view_55, [3], correction = 0, keepdim = True)
        getitem_16: "f32[128, 28, 28, 1]" = var_mean_5[0]
        getitem_17: "f32[128, 28, 28, 1]" = var_mean_5[1];  var_mean_5 = None
        add_21: "f32[128, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_5: "f32[128, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_7: "f32[128, 28, 28, 512]" = torch.ops.aten.sub.Tensor(view_55, getitem_17);  view_55 = getitem_17 = None
        mul_20: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = None
        mul_21: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_20, arg32_1);  arg32_1 = None
        add_22: "f32[128, 28, 28, 512]" = torch.ops.aten.add.Tensor(mul_21, arg33_1);  mul_21 = arg33_1 = None
        convert_element_type_29: "bf16[256, 512]" = torch.ops.prims.convert_element_type.default(arg34_1, torch.bfloat16);  arg34_1 = None
        convert_element_type_30: "bf16[128, 28, 28, 512]" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16);  add_22 = None
        permute_22: "bf16[512, 256]" = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        view_56: "bf16[100352, 512]" = torch.ops.aten.view.default(convert_element_type_30, _shape_param_72);  convert_element_type_30 = _shape_param_72 = None
        mm: "bf16[100352, 256]" = torch.ops.aten.mm.default(view_56, permute_22)
        view_57: "bf16[128, 28, 28, 256]" = torch.ops.aten.view.default(mm, _shape_param_73);  _shape_param_73 = None
        convert_element_type_31: "f32[128, 28, 28, 256]" = torch.ops.prims.convert_element_type.default(view_57, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_31, [3], correction = 0, keepdim = True)
        getitem_18: "f32[128, 28, 28, 1]" = var_mean_6[0]
        getitem_19: "f32[128, 28, 28, 1]" = var_mean_6[1];  var_mean_6 = None
        add_23: "f32[128, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_6: "f32[128, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_8: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_31, getitem_19);  convert_element_type_31 = None
        mul_22: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
        mul_23: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_22, arg35_1);  mul_22 = arg35_1 = None
        add_24: "f32[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_23, arg36_1);  mul_23 = arg36_1 = None
        view_58: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.view.default(add_24, _shape_param_74);  add_24 = _shape_param_74 = None
        permute_23: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_58, [0, 1, 3, 2, 4, 5]);  view_58 = None
        clone_15: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_59: "f32[2048, 7, 7, 256]" = torch.ops.aten.view.default(clone_15, _shape_param_75);  clone_15 = _shape_param_75 = None
        view_60: "f32[2048, 49, 256]" = torch.ops.aten.view.default(view_59, _shape_param_76);  view_59 = _shape_param_76 = None
        convert_element_type_32: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg37_1, torch.bfloat16);  arg37_1 = None
        convert_element_type_33: "bf16[768, 256]" = torch.ops.prims.convert_element_type.default(arg38_1, torch.bfloat16);  arg38_1 = None
        convert_element_type_34: "bf16[2048, 49, 256]" = torch.ops.prims.convert_element_type.default(view_60, torch.bfloat16);  view_60 = None
        view_61: "bf16[100352, 256]" = torch.ops.aten.view.default(convert_element_type_34, _shape_param_77);  convert_element_type_34 = _shape_param_77 = None
        permute_24: "bf16[256, 768]" = torch.ops.aten.permute.default(convert_element_type_33, [1, 0]);  convert_element_type_33 = None
        addmm_8: "bf16[100352, 768]" = torch.ops.aten.addmm.default(convert_element_type_32, view_61, permute_24);  convert_element_type_32 = None
        view_62: "bf16[2048, 49, 768]" = torch.ops.aten.view.default(addmm_8, _shape_param_78);  addmm_8 = _shape_param_78 = None
        view_63: "bf16[2048, 49, 3, 8, 32]" = torch.ops.aten.view.default(view_62, _shape_param_79);  view_62 = _shape_param_79 = None
        permute_25: "bf16[3, 2048, 8, 49, 32]" = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None
        unbind_2 = torch.ops.aten.unbind.int(permute_25);  permute_25 = None
        getitem_20: "bf16[2048, 8, 49, 32]" = unbind_2[0]
        getitem_21: "bf16[2048, 8, 49, 32]" = unbind_2[1]
        getitem_22: "bf16[2048, 8, 49, 32]" = unbind_2[2];  unbind_2 = None
        mul_24: "bf16[2048, 8, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_20, 0.1767766952966369);  getitem_20 = None
        permute_26: "bf16[2048, 8, 32, 49]" = torch.ops.aten.permute.default(getitem_21, [0, 1, 3, 2]);  getitem_21 = None
        expand_8: "bf16[2048, 8, 49, 32]" = torch.ops.aten.expand.default(mul_24, _shape_param_80);  mul_24 = _shape_param_80 = None
        clone_16: "bf16[2048, 8, 49, 32]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_64: "bf16[16384, 49, 32]" = torch.ops.aten.view.default(clone_16, _shape_param_81);  clone_16 = _shape_param_81 = None
        expand_9: "bf16[2048, 8, 32, 49]" = torch.ops.aten.expand.default(permute_26, _shape_param_82);  permute_26 = _shape_param_82 = None
        clone_17: "bf16[2048, 8, 32, 49]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_65: "bf16[16384, 32, 49]" = torch.ops.aten.view.default(clone_17, _shape_param_83);  clone_17 = _shape_param_83 = None
        constant_pad_nd_8: "bf16[16384, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_64, _shape_param_84);  _shape_param_84 = None
        constant_pad_nd_9: "bf16[16384, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_65, _shape_param_85);  _shape_param_85 = None
        bmm_4: "bf16[16384, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_8, constant_pad_nd_9);  constant_pad_nd_8 = constant_pad_nd_9 = None
        slice_7: "bf16[16384, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_4, 1, 0, -7)
        slice_8: "bf16[16384, 49, 49]" = torch.ops.aten.slice.Tensor(slice_7, 2, 0, -7);  slice_7 = None
        view_66: "bf16[2048, 8, 49, 49]" = torch.ops.aten.view.default(slice_8, _shape_param_86);  slice_8 = _shape_param_86 = None
        view_67: "i64[2401]" = torch.ops.aten.view.default(arg39_1, [-1]);  arg39_1 = None
        index_6: "f32[2401, 8]" = torch.ops.aten.index.Tensor(arg40_1, [view_67]);  arg40_1 = view_67 = None
        view_68: "f32[49, 49, 8]" = torch.ops.aten.view.default(index_6, _shape_param_87);  index_6 = _shape_param_87 = None
        permute_27: "f32[8, 49, 49]" = torch.ops.aten.permute.default(view_68, [2, 0, 1]);  view_68 = None
        clone_18: "f32[8, 49, 49]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        unsqueeze_4: "f32[1, 8, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_18, 0);  clone_18 = None
        add_25: "f32[2048, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_66, unsqueeze_4);  view_66 = unsqueeze_4 = None
        amax_2: "f32[2048, 8, 49, 1]" = torch.ops.aten.amax.default(add_25, [-1], True)
        sub_9: "f32[2048, 8, 49, 49]" = torch.ops.aten.sub.Tensor(add_25, amax_2);  add_25 = None
        exp_2: "f32[2048, 8, 49, 49]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_3: "f32[2048, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_4: "f32[2048, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = None
        convert_element_type_35: "bf16[2048, 8, 49, 49]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        expand_10: "bf16[2048, 8, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_35, _shape_param_88);  convert_element_type_35 = _shape_param_88 = None
        view_69: "bf16[16384, 49, 49]" = torch.ops.aten.view.default(expand_10, _shape_param_89);  expand_10 = _shape_param_89 = None
        expand_11: "bf16[2048, 8, 49, 32]" = torch.ops.aten.expand.default(getitem_22, _shape_param_90);  getitem_22 = _shape_param_90 = None
        clone_19: "bf16[2048, 8, 49, 32]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_70: "bf16[16384, 49, 32]" = torch.ops.aten.view.default(clone_19, _shape_param_91);  clone_19 = _shape_param_91 = None
        constant_pad_nd_10: "bf16[16384, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_69, _shape_param_92);  _shape_param_92 = None
        constant_pad_nd_11: "bf16[16384, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_70, _shape_param_93);  _shape_param_93 = None
        bmm_5: "bf16[16384, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_10, constant_pad_nd_11);  constant_pad_nd_10 = constant_pad_nd_11 = None
        slice_9: "bf16[16384, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_5, 1, 0, -7);  bmm_5 = None
        view_71: "bf16[2048, 8, 49, 32]" = torch.ops.aten.view.default(slice_9, _shape_param_94);  slice_9 = _shape_param_94 = None
        permute_28: "bf16[2048, 49, 8, 32]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        clone_20: "bf16[2048, 49, 8, 32]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
        view_72: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(clone_20, _shape_param_95);  clone_20 = _shape_param_95 = None
        convert_element_type_36: "bf16[256]" = torch.ops.prims.convert_element_type.default(arg41_1, torch.bfloat16);  arg41_1 = None
        convert_element_type_37: "bf16[256, 256]" = torch.ops.prims.convert_element_type.default(arg42_1, torch.bfloat16);  arg42_1 = None
        view_73: "bf16[100352, 256]" = torch.ops.aten.view.default(view_72, _shape_param_96);  view_72 = _shape_param_96 = None
        permute_29: "bf16[256, 256]" = torch.ops.aten.permute.default(convert_element_type_37, [1, 0]);  convert_element_type_37 = None
        addmm_9: "bf16[100352, 256]" = torch.ops.aten.addmm.default(convert_element_type_36, view_73, permute_29);  convert_element_type_36 = None
        view_74: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(addmm_9, _shape_param_97);  addmm_9 = _shape_param_97 = None
        view_75: "bf16[2048, 7, 7, 256]" = torch.ops.aten.view.default(view_74, _shape_param_98);  view_74 = _shape_param_98 = None
        view_76: "bf16[128, 4, 4, 7, 7, 256]" = torch.ops.aten.view.default(view_75, _shape_param_99);  view_75 = _shape_param_99 = None
        permute_30: "bf16[128, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_76, [0, 1, 3, 2, 4, 5]);  view_76 = None
        clone_21: "bf16[128, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_77: "bf16[128, 28, 28, 256]" = torch.ops.aten.view.default(clone_21, _shape_param_100);  clone_21 = _shape_param_100 = None
        inductor_lookup_seed_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 2)
        inductor_random_2: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_101, inductor_lookup_seed_2, 'rand');  _shape_param_101 = inductor_lookup_seed_2 = None
        lt_2: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_2, 0.9913043472915888);  inductor_random_2 = None
        convert_element_type_38: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_2, torch.bfloat16)
        div_5: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_38, 0.9913043472915888);  convert_element_type_38 = None
        mul_25: "bf16[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_77, div_5);  view_77 = div_5 = None
        add_26: "bf16[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_57, mul_25);  view_57 = mul_25 = None
        view_78: "bf16[128, 784, 256]" = torch.ops.aten.view.default(add_26, _shape_param_102);  add_26 = _shape_param_102 = None
        convert_element_type_39: "f32[128, 784, 256]" = torch.ops.prims.convert_element_type.default(view_78, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_39, [2], correction = 0, keepdim = True)
        getitem_23: "f32[128, 784, 1]" = var_mean_7[0]
        getitem_24: "f32[128, 784, 1]" = var_mean_7[1];  var_mean_7 = None
        add_27: "f32[128, 784, 1]" = torch.ops.aten.add.Tensor(getitem_23, 1e-05);  getitem_23 = None
        rsqrt_7: "f32[128, 784, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_10: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_39, getitem_24);  convert_element_type_39 = None
        mul_26: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_7);  sub_10 = None
        mul_27: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_26, arg43_1);  mul_26 = arg43_1 = None
        add_28: "f32[128, 784, 256]" = torch.ops.aten.add.Tensor(mul_27, arg44_1);  mul_27 = arg44_1 = None
        convert_element_type_40: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg45_1, torch.bfloat16);  arg45_1 = None
        convert_element_type_41: "bf16[1024, 256]" = torch.ops.prims.convert_element_type.default(arg46_1, torch.bfloat16);  arg46_1 = None
        convert_element_type_42: "bf16[128, 784, 256]" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None
        view_79: "bf16[100352, 256]" = torch.ops.aten.view.default(convert_element_type_42, _shape_param_103);  convert_element_type_42 = _shape_param_103 = None
        permute_31: "bf16[256, 1024]" = torch.ops.aten.permute.default(convert_element_type_41, [1, 0]);  convert_element_type_41 = None
        addmm_10: "bf16[100352, 1024]" = torch.ops.aten.addmm.default(convert_element_type_40, view_79, permute_31);  convert_element_type_40 = None
        view_80: "bf16[128, 784, 1024]" = torch.ops.aten.view.default(addmm_10, _shape_param_104);  _shape_param_104 = None
        convert_element_type_43: "f32[128, 784, 1024]" = torch.ops.prims.convert_element_type.default(view_80, torch.float32);  view_80 = None
        mul_28: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 0.5)
        mul_29: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 0.7071067811865476);  convert_element_type_43 = None
        erf_2: "f32[128, 784, 1024]" = torch.ops.aten.erf.default(mul_29);  mul_29 = None
        add_29: "f32[128, 784, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_30: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_28, add_29);  mul_28 = add_29 = None
        convert_element_type_44: "bf16[128, 784, 1024]" = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        convert_element_type_45: "bf16[256]" = torch.ops.prims.convert_element_type.default(arg47_1, torch.bfloat16);  arg47_1 = None
        convert_element_type_46: "bf16[256, 1024]" = torch.ops.prims.convert_element_type.default(arg48_1, torch.bfloat16);  arg48_1 = None
        view_81: "bf16[100352, 1024]" = torch.ops.aten.view.default(convert_element_type_44, _shape_param_105);  convert_element_type_44 = _shape_param_105 = None
        permute_32: "bf16[1024, 256]" = torch.ops.aten.permute.default(convert_element_type_46, [1, 0]);  convert_element_type_46 = None
        addmm_11: "bf16[100352, 256]" = torch.ops.aten.addmm.default(convert_element_type_45, view_81, permute_32);  convert_element_type_45 = None
        view_82: "bf16[128, 784, 256]" = torch.ops.aten.view.default(addmm_11, _shape_param_106);  addmm_11 = _shape_param_106 = None
        inductor_lookup_seed_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 3)
        inductor_random_3: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_107, inductor_lookup_seed_3, 'rand');  _shape_param_107 = inductor_lookup_seed_3 = None
        lt_3: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_3, 0.9913043472915888);  inductor_random_3 = None
        convert_element_type_47: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_3, torch.bfloat16)
        div_6: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_47, 0.9913043472915888);  convert_element_type_47 = None
        mul_31: "bf16[128, 784, 256]" = torch.ops.aten.mul.Tensor(view_82, div_6);  view_82 = div_6 = None
        add_30: "bf16[128, 784, 256]" = torch.ops.aten.add.Tensor(view_78, mul_31);  mul_31 = None
        view_83: "bf16[128, 28, 28, 256]" = torch.ops.aten.view.default(add_30, _shape_param_108);  add_30 = _shape_param_108 = None
        convert_element_type_48: "f32[128, 28, 28, 256]" = torch.ops.prims.convert_element_type.default(view_83, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_48, [3], correction = 0, keepdim = True)
        getitem_25: "f32[128, 28, 28, 1]" = var_mean_8[0]
        getitem_26: "f32[128, 28, 28, 1]" = var_mean_8[1];  var_mean_8 = None
        add_31: "f32[128, 28, 28, 1]" = torch.ops.aten.add.Tensor(getitem_25, 1e-05);  getitem_25 = None
        rsqrt_8: "f32[128, 28, 28, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_11: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_48, getitem_26);  convert_element_type_48 = None
        mul_32: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_8);  sub_11 = None
        mul_33: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_32, arg49_1);  mul_32 = arg49_1 = None
        add_32: "f32[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(mul_33, arg50_1);  mul_33 = arg50_1 = None
        iota_1: "i64[28]" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_33: "i64[28]" = torch.ops.aten.add.Tensor(iota_1, 3)
        fmod_2: "i64[28]" = torch.ops.aten.fmod.Scalar(add_33, 28);  add_33 = None
        index_7: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(add_32, [None, fmod_2]);  add_32 = None
        index_8: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(index_7, [None, None, fmod_2]);  index_7 = fmod_2 = None
        view_84: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.view.default(index_8, _shape_param_109);  index_8 = _shape_param_109 = None
        permute_33: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_84, [0, 1, 3, 2, 4, 5]);  view_84 = None
        clone_22: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        view_85: "f32[2048, 7, 7, 256]" = torch.ops.aten.view.default(clone_22, _shape_param_110);  clone_22 = _shape_param_110 = None
        view_86: "f32[2048, 49, 256]" = torch.ops.aten.view.default(view_85, _shape_param_111);  view_85 = _shape_param_111 = None
        convert_element_type_49: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg51_1, torch.bfloat16);  arg51_1 = None
        convert_element_type_50: "bf16[768, 256]" = torch.ops.prims.convert_element_type.default(arg52_1, torch.bfloat16);  arg52_1 = None
        convert_element_type_51: "bf16[2048, 49, 256]" = torch.ops.prims.convert_element_type.default(view_86, torch.bfloat16);  view_86 = None
        view_87: "bf16[100352, 256]" = torch.ops.aten.view.default(convert_element_type_51, _shape_param_112);  convert_element_type_51 = _shape_param_112 = None
        permute_34: "bf16[256, 768]" = torch.ops.aten.permute.default(convert_element_type_50, [1, 0]);  convert_element_type_50 = None
        addmm_12: "bf16[100352, 768]" = torch.ops.aten.addmm.default(convert_element_type_49, view_87, permute_34);  convert_element_type_49 = None
        view_88: "bf16[2048, 49, 768]" = torch.ops.aten.view.default(addmm_12, _shape_param_113);  addmm_12 = _shape_param_113 = None
        view_89: "bf16[2048, 49, 3, 8, 32]" = torch.ops.aten.view.default(view_88, _shape_param_114);  view_88 = _shape_param_114 = None
        permute_35: "bf16[3, 2048, 8, 49, 32]" = torch.ops.aten.permute.default(view_89, [2, 0, 3, 1, 4]);  view_89 = None
        unbind_3 = torch.ops.aten.unbind.int(permute_35);  permute_35 = None
        getitem_27: "bf16[2048, 8, 49, 32]" = unbind_3[0]
        getitem_28: "bf16[2048, 8, 49, 32]" = unbind_3[1]
        getitem_29: "bf16[2048, 8, 49, 32]" = unbind_3[2];  unbind_3 = None
        mul_34: "bf16[2048, 8, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_27, 0.1767766952966369);  getitem_27 = None
        permute_36: "bf16[2048, 8, 32, 49]" = torch.ops.aten.permute.default(getitem_28, [0, 1, 3, 2]);  getitem_28 = None
        expand_12: "bf16[2048, 8, 49, 32]" = torch.ops.aten.expand.default(mul_34, _shape_param_115);  mul_34 = _shape_param_115 = None
        clone_23: "bf16[2048, 8, 49, 32]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_90: "bf16[16384, 49, 32]" = torch.ops.aten.view.default(clone_23, _shape_param_116);  clone_23 = _shape_param_116 = None
        expand_13: "bf16[2048, 8, 32, 49]" = torch.ops.aten.expand.default(permute_36, _shape_param_117);  permute_36 = _shape_param_117 = None
        clone_24: "bf16[2048, 8, 32, 49]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_91: "bf16[16384, 32, 49]" = torch.ops.aten.view.default(clone_24, _shape_param_118);  clone_24 = _shape_param_118 = None
        constant_pad_nd_12: "bf16[16384, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_90, _shape_param_119);  _shape_param_119 = None
        constant_pad_nd_13: "bf16[16384, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_91, _shape_param_120);  _shape_param_120 = None
        bmm_6: "bf16[16384, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_12, constant_pad_nd_13);  constant_pad_nd_12 = constant_pad_nd_13 = None
        slice_10: "bf16[16384, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_6, 1, 0, -7)
        slice_11: "bf16[16384, 49, 49]" = torch.ops.aten.slice.Tensor(slice_10, 2, 0, -7);  slice_10 = None
        view_92: "bf16[2048, 8, 49, 49]" = torch.ops.aten.view.default(slice_11, _shape_param_121);  slice_11 = _shape_param_121 = None
        view_93: "i64[2401]" = torch.ops.aten.view.default(arg53_1, [-1]);  arg53_1 = None
        index_9: "f32[2401, 8]" = torch.ops.aten.index.Tensor(arg54_1, [view_93]);  arg54_1 = view_93 = None
        view_94: "f32[49, 49, 8]" = torch.ops.aten.view.default(index_9, _shape_param_122);  index_9 = _shape_param_122 = None
        permute_37: "f32[8, 49, 49]" = torch.ops.aten.permute.default(view_94, [2, 0, 1]);  view_94 = None
        clone_25: "f32[8, 49, 49]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        unsqueeze_5: "f32[1, 8, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_25, 0);  clone_25 = None
        add_34: "f32[2048, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_92, unsqueeze_5);  view_92 = unsqueeze_5 = None
        view_95: "f32[128, 16, 8, 49, 49]" = torch.ops.aten.view.default(add_34, _shape_param_123);  add_34 = _shape_param_123 = None
        unsqueeze_6: "f32[16, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg55_1, 1);  arg55_1 = None
        unsqueeze_7: "f32[1, 16, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 0);  unsqueeze_6 = None
        add_35: "f32[128, 16, 8, 49, 49]" = torch.ops.aten.add.Tensor(view_95, unsqueeze_7);  view_95 = unsqueeze_7 = None
        view_96: "f32[2048, 8, 49, 49]" = torch.ops.aten.view.default(add_35, _shape_param_124);  add_35 = _shape_param_124 = None
        amax_3: "f32[2048, 8, 49, 1]" = torch.ops.aten.amax.default(view_96, [-1], True)
        sub_12: "f32[2048, 8, 49, 49]" = torch.ops.aten.sub.Tensor(view_96, amax_3);  view_96 = None
        exp_3: "f32[2048, 8, 49, 49]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_4: "f32[2048, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7: "f32[2048, 8, 49, 49]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = None
        convert_element_type_52: "bf16[2048, 8, 49, 49]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        expand_14: "bf16[2048, 8, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_52, _shape_param_125);  convert_element_type_52 = _shape_param_125 = None
        view_97: "bf16[16384, 49, 49]" = torch.ops.aten.view.default(expand_14, _shape_param_126);  expand_14 = _shape_param_126 = None
        expand_15: "bf16[2048, 8, 49, 32]" = torch.ops.aten.expand.default(getitem_29, _shape_param_127);  getitem_29 = _shape_param_127 = None
        clone_26: "bf16[2048, 8, 49, 32]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_98: "bf16[16384, 49, 32]" = torch.ops.aten.view.default(clone_26, _shape_param_128);  clone_26 = _shape_param_128 = None
        constant_pad_nd_14: "bf16[16384, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_97, _shape_param_129);  _shape_param_129 = None
        constant_pad_nd_15: "bf16[16384, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_98, _shape_param_130);  _shape_param_130 = None
        bmm_7: "bf16[16384, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_14, constant_pad_nd_15);  constant_pad_nd_14 = constant_pad_nd_15 = None
        slice_12: "bf16[16384, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_7, 1, 0, -7);  bmm_7 = None
        view_99: "bf16[2048, 8, 49, 32]" = torch.ops.aten.view.default(slice_12, _shape_param_131);  slice_12 = _shape_param_131 = None
        permute_38: "bf16[2048, 49, 8, 32]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        clone_27: "bf16[2048, 49, 8, 32]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_100: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(clone_27, _shape_param_132);  clone_27 = _shape_param_132 = None
        convert_element_type_53: "bf16[256]" = torch.ops.prims.convert_element_type.default(arg56_1, torch.bfloat16);  arg56_1 = None
        convert_element_type_54: "bf16[256, 256]" = torch.ops.prims.convert_element_type.default(arg57_1, torch.bfloat16);  arg57_1 = None
        view_101: "bf16[100352, 256]" = torch.ops.aten.view.default(view_100, _shape_param_133);  view_100 = _shape_param_133 = None
        permute_39: "bf16[256, 256]" = torch.ops.aten.permute.default(convert_element_type_54, [1, 0]);  convert_element_type_54 = None
        addmm_13: "bf16[100352, 256]" = torch.ops.aten.addmm.default(convert_element_type_53, view_101, permute_39);  convert_element_type_53 = None
        view_102: "bf16[2048, 49, 256]" = torch.ops.aten.view.default(addmm_13, _shape_param_134);  addmm_13 = _shape_param_134 = None
        view_103: "bf16[2048, 7, 7, 256]" = torch.ops.aten.view.default(view_102, _shape_param_135);  view_102 = _shape_param_135 = None
        view_104: "bf16[128, 4, 4, 7, 7, 256]" = torch.ops.aten.view.default(view_103, _shape_param_136);  view_103 = _shape_param_136 = None
        permute_40: "bf16[128, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_104, [0, 1, 3, 2, 4, 5]);  view_104 = None
        clone_28: "bf16[128, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_105: "bf16[128, 28, 28, 256]" = torch.ops.aten.view.default(clone_28, _shape_param_137);  clone_28 = _shape_param_137 = None
        add_36: "i64[28]" = torch.ops.aten.add.Tensor(iota_1, 25);  iota_1 = None
        fmod_3: "i64[28]" = torch.ops.aten.fmod.Scalar(add_36, 28);  add_36 = None
        index_10: "bf16[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(view_105, [None, fmod_3]);  view_105 = None
        index_11: "bf16[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(index_10, [None, None, fmod_3]);  index_10 = None
        inductor_lookup_seed_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 4)
        inductor_random_4: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_138, inductor_lookup_seed_4, 'rand');  _shape_param_138 = inductor_lookup_seed_4 = None
        lt_4: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_4, 0.9869565209373832);  inductor_random_4 = None
        convert_element_type_55: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_4, torch.bfloat16)
        div_8: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_55, 0.9869565209373832);  convert_element_type_55 = None
        mul_35: "bf16[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(index_11, div_8);  index_11 = div_8 = None
        add_37: "bf16[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_83, mul_35);  mul_35 = None
        view_106: "bf16[128, 784, 256]" = torch.ops.aten.view.default(add_37, _shape_param_139);  add_37 = _shape_param_139 = None
        convert_element_type_56: "f32[128, 784, 256]" = torch.ops.prims.convert_element_type.default(view_106, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_56, [2], correction = 0, keepdim = True)
        getitem_30: "f32[128, 784, 1]" = var_mean_9[0]
        getitem_31: "f32[128, 784, 1]" = var_mean_9[1];  var_mean_9 = None
        add_38: "f32[128, 784, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_9: "f32[128, 784, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_13: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(convert_element_type_56, getitem_31);  convert_element_type_56 = None
        mul_36: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_9);  sub_13 = None
        mul_37: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_36, arg58_1);  mul_36 = arg58_1 = None
        add_39: "f32[128, 784, 256]" = torch.ops.aten.add.Tensor(mul_37, arg59_1);  mul_37 = arg59_1 = None
        convert_element_type_57: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg60_1, torch.bfloat16);  arg60_1 = None
        convert_element_type_58: "bf16[1024, 256]" = torch.ops.prims.convert_element_type.default(arg61_1, torch.bfloat16);  arg61_1 = None
        convert_element_type_59: "bf16[128, 784, 256]" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None
        view_107: "bf16[100352, 256]" = torch.ops.aten.view.default(convert_element_type_59, _shape_param_140);  convert_element_type_59 = _shape_param_140 = None
        permute_41: "bf16[256, 1024]" = torch.ops.aten.permute.default(convert_element_type_58, [1, 0]);  convert_element_type_58 = None
        addmm_14: "bf16[100352, 1024]" = torch.ops.aten.addmm.default(convert_element_type_57, view_107, permute_41);  convert_element_type_57 = None
        view_108: "bf16[128, 784, 1024]" = torch.ops.aten.view.default(addmm_14, _shape_param_141);  _shape_param_141 = None
        convert_element_type_60: "f32[128, 784, 1024]" = torch.ops.prims.convert_element_type.default(view_108, torch.float32);  view_108 = None
        mul_38: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 0.5)
        mul_39: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 0.7071067811865476);  convert_element_type_60 = None
        erf_3: "f32[128, 784, 1024]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_40: "f32[128, 784, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_40: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_38, add_40);  mul_38 = add_40 = None
        convert_element_type_61: "bf16[128, 784, 1024]" = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16);  mul_40 = None
        convert_element_type_62: "bf16[256]" = torch.ops.prims.convert_element_type.default(arg62_1, torch.bfloat16);  arg62_1 = None
        convert_element_type_63: "bf16[256, 1024]" = torch.ops.prims.convert_element_type.default(arg63_1, torch.bfloat16);  arg63_1 = None
        view_109: "bf16[100352, 1024]" = torch.ops.aten.view.default(convert_element_type_61, _shape_param_142);  convert_element_type_61 = _shape_param_142 = None
        permute_42: "bf16[1024, 256]" = torch.ops.aten.permute.default(convert_element_type_63, [1, 0]);  convert_element_type_63 = None
        addmm_15: "bf16[100352, 256]" = torch.ops.aten.addmm.default(convert_element_type_62, view_109, permute_42);  convert_element_type_62 = None
        view_110: "bf16[128, 784, 256]" = torch.ops.aten.view.default(addmm_15, _shape_param_143);  addmm_15 = _shape_param_143 = None
        inductor_lookup_seed_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 5)
        inductor_random_5: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_144, inductor_lookup_seed_5, 'rand');  _shape_param_144 = inductor_lookup_seed_5 = None
        lt_5: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_5, 0.9869565209373832);  inductor_random_5 = None
        convert_element_type_64: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_5, torch.bfloat16)
        div_9: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_64, 0.9869565209373832);  convert_element_type_64 = None
        mul_41: "bf16[128, 784, 256]" = torch.ops.aten.mul.Tensor(view_110, div_9);  view_110 = div_9 = None
        add_41: "bf16[128, 784, 256]" = torch.ops.aten.add.Tensor(view_106, mul_41);  mul_41 = None
        view_111: "bf16[128, 28, 28, 256]" = torch.ops.aten.view.default(add_41, _shape_param_145);  add_41 = _shape_param_145 = None
        view_112: "bf16[128, 14, 2, 14, 2, 256]" = torch.ops.aten.view.default(view_111, _shape_param_146);  view_111 = _shape_param_146 = None
        permute_43: "bf16[128, 14, 14, 2, 2, 256]" = torch.ops.aten.permute.default(view_112, [0, 1, 3, 4, 2, 5]);  view_112 = None
        clone_29: "bf16[128, 14, 14, 2, 2, 256]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
        view_113: "bf16[128, 14, 14, 1024]" = torch.ops.aten.view.default(clone_29, _shape_param_147);  clone_29 = _shape_param_147 = None
        convert_element_type_65: "f32[128, 14, 14, 1024]" = torch.ops.prims.convert_element_type.default(view_113, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_65, [3], correction = 0, keepdim = True)
        getitem_32: "f32[128, 14, 14, 1]" = var_mean_10[0]
        getitem_33: "f32[128, 14, 14, 1]" = var_mean_10[1];  var_mean_10 = None
        add_42: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_10: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_14: "f32[128, 14, 14, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_65, getitem_33);  convert_element_type_65 = None
        mul_42: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_10);  sub_14 = None
        mul_43: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(mul_42, arg64_1);  mul_42 = arg64_1 = None
        add_43: "f32[128, 14, 14, 1024]" = torch.ops.aten.add.Tensor(mul_43, arg65_1);  mul_43 = arg65_1 = None
        convert_element_type_66: "bf16[512, 1024]" = torch.ops.prims.convert_element_type.default(arg66_1, torch.bfloat16);  arg66_1 = None
        convert_element_type_67: "bf16[128, 14, 14, 1024]" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None
        permute_44: "bf16[1024, 512]" = torch.ops.aten.permute.default(convert_element_type_66, [1, 0]);  convert_element_type_66 = None
        view_114: "bf16[25088, 1024]" = torch.ops.aten.view.default(convert_element_type_67, _shape_param_148);  convert_element_type_67 = _shape_param_148 = None
        mm_1: "bf16[25088, 512]" = torch.ops.aten.mm.default(view_114, permute_44)
        view_115: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(mm_1, _shape_param_149);  _shape_param_149 = None
        convert_element_type_68: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_115, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_68, [3], correction = 0, keepdim = True)
        getitem_34: "f32[128, 14, 14, 1]" = var_mean_11[0]
        getitem_35: "f32[128, 14, 14, 1]" = var_mean_11[1];  var_mean_11 = None
        add_44: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_11: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_15: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_68, getitem_35);  convert_element_type_68 = None
        mul_44: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = None
        mul_45: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_44, arg67_1);  mul_44 = arg67_1 = None
        add_45: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_45, arg68_1);  mul_45 = arg68_1 = None
        view_116: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_45, _shape_param_150);  add_45 = _shape_param_150 = None
        permute_45: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_116, [0, 1, 3, 2, 4, 5]);  view_116 = None
        clone_30: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_117: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_30, _shape_param_151);  clone_30 = _shape_param_151 = None
        view_118: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_117, _shape_param_152);  view_117 = _shape_param_152 = None
        convert_element_type_69: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg69_1, torch.bfloat16);  arg69_1 = None
        convert_element_type_70: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg70_1, torch.bfloat16);  arg70_1 = None
        convert_element_type_71: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_118, torch.bfloat16);  view_118 = None
        view_119: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_71, _shape_param_153);  convert_element_type_71 = _shape_param_153 = None
        permute_46: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_70, [1, 0]);  convert_element_type_70 = None
        addmm_16: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_69, view_119, permute_46);  convert_element_type_69 = None
        view_120: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_16, _shape_param_154);  addmm_16 = _shape_param_154 = None
        view_121: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_120, _shape_param_155);  view_120 = _shape_param_155 = None
        permute_47: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_121, [2, 0, 3, 1, 4]);  view_121 = None
        unbind_4 = torch.ops.aten.unbind.int(permute_47);  permute_47 = None
        getitem_36: "bf16[512, 16, 49, 32]" = unbind_4[0]
        getitem_37: "bf16[512, 16, 49, 32]" = unbind_4[1]
        getitem_38: "bf16[512, 16, 49, 32]" = unbind_4[2];  unbind_4 = None
        mul_46: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_36, 0.1767766952966369);  getitem_36 = None
        permute_48: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_37, [0, 1, 3, 2]);  getitem_37 = None
        expand_16: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_46, _shape_param_156);  mul_46 = _shape_param_156 = None
        clone_31: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_122: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_31, _shape_param_157);  clone_31 = _shape_param_157 = None
        expand_17: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_48, _shape_param_158);  permute_48 = _shape_param_158 = None
        clone_32: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_123: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_32, _shape_param_159);  clone_32 = _shape_param_159 = None
        constant_pad_nd_16: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_122, _shape_param_160);  _shape_param_160 = None
        constant_pad_nd_17: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_123, _shape_param_161);  _shape_param_161 = None
        bmm_8: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_16, constant_pad_nd_17);  constant_pad_nd_16 = constant_pad_nd_17 = None
        slice_13: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_8, 1, 0, -7)
        slice_14: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_13, 2, 0, -7);  slice_13 = None
        view_124: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_14, _shape_param_162);  slice_14 = _shape_param_162 = None
        view_125: "i64[2401]" = torch.ops.aten.view.default(arg71_1, [-1]);  arg71_1 = None
        index_12: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg72_1, [view_125]);  arg72_1 = view_125 = None
        view_126: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_12, _shape_param_163);  index_12 = _shape_param_163 = None
        permute_49: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_126, [2, 0, 1]);  view_126 = None
        clone_33: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        unsqueeze_8: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_33, 0);  clone_33 = None
        add_46: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_124, unsqueeze_8);  view_124 = unsqueeze_8 = None
        amax_4: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_46, [-1], True)
        sub_16: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_46, amax_4);  add_46 = None
        exp_4: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_5: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_10: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = None
        convert_element_type_72: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        expand_18: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_72, _shape_param_164);  convert_element_type_72 = _shape_param_164 = None
        view_127: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_18, _shape_param_165);  expand_18 = _shape_param_165 = None
        expand_19: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_38, _shape_param_166);  getitem_38 = _shape_param_166 = None
        clone_34: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_128: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_34, _shape_param_167);  clone_34 = _shape_param_167 = None
        constant_pad_nd_18: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_127, _shape_param_168);  _shape_param_168 = None
        constant_pad_nd_19: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_128, _shape_param_169);  _shape_param_169 = None
        bmm_9: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_18, constant_pad_nd_19);  constant_pad_nd_18 = constant_pad_nd_19 = None
        slice_15: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_9, 1, 0, -7);  bmm_9 = None
        view_129: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_15, _shape_param_170);  slice_15 = _shape_param_170 = None
        permute_50: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        clone_35: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
        view_130: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_35, _shape_param_171);  clone_35 = _shape_param_171 = None
        convert_element_type_73: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg73_1, torch.bfloat16);  arg73_1 = None
        convert_element_type_74: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg74_1, torch.bfloat16);  arg74_1 = None
        view_131: "bf16[25088, 512]" = torch.ops.aten.view.default(view_130, _shape_param_172);  view_130 = _shape_param_172 = None
        permute_51: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_74, [1, 0]);  convert_element_type_74 = None
        addmm_17: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_73, view_131, permute_51);  convert_element_type_73 = None
        view_132: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_17, _shape_param_173);  addmm_17 = _shape_param_173 = None
        view_133: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_132, _shape_param_174);  view_132 = _shape_param_174 = None
        view_134: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_133, _shape_param_175);  view_133 = _shape_param_175 = None
        permute_52: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_134, [0, 1, 3, 2, 4, 5]);  view_134 = None
        clone_36: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
        view_135: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_36, _shape_param_176);  clone_36 = _shape_param_176 = None
        inductor_lookup_seed_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 6)
        inductor_random_6: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_177, inductor_lookup_seed_6, 'rand');  _shape_param_177 = inductor_lookup_seed_6 = None
        lt_6: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_6, 0.9826086945831776);  inductor_random_6 = None
        convert_element_type_75: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_6, torch.bfloat16)
        div_11: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_75, 0.9826086945831776);  convert_element_type_75 = None
        mul_47: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_135, div_11);  view_135 = div_11 = None
        add_47: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_115, mul_47);  view_115 = mul_47 = None
        view_136: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_47, _shape_param_178);  add_47 = _shape_param_178 = None
        convert_element_type_76: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_136, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_76, [2], correction = 0, keepdim = True)
        getitem_39: "f32[128, 196, 1]" = var_mean_12[0]
        getitem_40: "f32[128, 196, 1]" = var_mean_12[1];  var_mean_12 = None
        add_48: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_39, 1e-05);  getitem_39 = None
        rsqrt_12: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_17: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_76, getitem_40);  convert_element_type_76 = None
        mul_48: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_12);  sub_17 = None
        mul_49: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_48, arg75_1);  mul_48 = arg75_1 = None
        add_49: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_49, arg76_1);  mul_49 = arg76_1 = None
        convert_element_type_77: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg77_1, torch.bfloat16);  arg77_1 = None
        convert_element_type_78: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg78_1, torch.bfloat16);  arg78_1 = None
        convert_element_type_79: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16);  add_49 = None
        view_137: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_79, _shape_param_179);  convert_element_type_79 = _shape_param_179 = None
        permute_53: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_78, [1, 0]);  convert_element_type_78 = None
        addmm_18: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_77, view_137, permute_53);  convert_element_type_77 = None
        view_138: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_18, _shape_param_180);  _shape_param_180 = None
        convert_element_type_80: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_138, torch.float32);  view_138 = None
        mul_50: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_80, 0.5)
        mul_51: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_80, 0.7071067811865476);  convert_element_type_80 = None
        erf_4: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_50: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_52: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_50, add_50);  mul_50 = add_50 = None
        convert_element_type_81: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None
        convert_element_type_82: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg79_1, torch.bfloat16);  arg79_1 = None
        convert_element_type_83: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg80_1, torch.bfloat16);  arg80_1 = None
        view_139: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_81, _shape_param_181);  convert_element_type_81 = _shape_param_181 = None
        permute_54: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_83, [1, 0]);  convert_element_type_83 = None
        addmm_19: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_82, view_139, permute_54);  convert_element_type_82 = None
        view_140: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_19, _shape_param_182);  addmm_19 = _shape_param_182 = None
        inductor_lookup_seed_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 7)
        inductor_random_7: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_183, inductor_lookup_seed_7, 'rand');  _shape_param_183 = inductor_lookup_seed_7 = None
        lt_7: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_7, 0.9826086945831776);  inductor_random_7 = None
        convert_element_type_84: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_7, torch.bfloat16)
        div_12: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_84, 0.9826086945831776);  convert_element_type_84 = None
        mul_53: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_140, div_12);  view_140 = div_12 = None
        add_51: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_136, mul_53);  mul_53 = None
        view_141: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_51, _shape_param_184);  add_51 = _shape_param_184 = None
        convert_element_type_85: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_141, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_85, [3], correction = 0, keepdim = True)
        getitem_41: "f32[128, 14, 14, 1]" = var_mean_13[0]
        getitem_42: "f32[128, 14, 14, 1]" = var_mean_13[1];  var_mean_13 = None
        add_52: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_41, 1e-05);  getitem_41 = None
        rsqrt_13: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_18: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_85, getitem_42);  convert_element_type_85 = None
        mul_54: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_13);  sub_18 = None
        mul_55: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_54, arg81_1);  mul_54 = arg81_1 = None
        add_53: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_55, arg82_1);  mul_55 = arg82_1 = None
        iota_2: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_54: "i64[14]" = torch.ops.aten.add.Tensor(iota_2, 3)
        fmod_4: "i64[14]" = torch.ops.aten.fmod.Scalar(add_54, 14);  add_54 = None
        index_13: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_53, [None, fmod_4]);  add_53 = None
        index_14: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_13, [None, None, fmod_4]);  index_13 = None
        view_142: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_14, _shape_param_185);  index_14 = _shape_param_185 = None
        permute_55: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_142, [0, 1, 3, 2, 4, 5]);  view_142 = None
        clone_37: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_143: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_37, _shape_param_186);  clone_37 = _shape_param_186 = None
        view_144: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_143, _shape_param_187);  view_143 = _shape_param_187 = None
        convert_element_type_86: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg83_1, torch.bfloat16);  arg83_1 = None
        convert_element_type_87: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg84_1, torch.bfloat16);  arg84_1 = None
        convert_element_type_88: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_144, torch.bfloat16);  view_144 = None
        view_145: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_88, _shape_param_188);  convert_element_type_88 = _shape_param_188 = None
        permute_56: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_87, [1, 0]);  convert_element_type_87 = None
        addmm_20: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_86, view_145, permute_56);  convert_element_type_86 = None
        view_146: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_20, _shape_param_189);  addmm_20 = _shape_param_189 = None
        view_147: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_146, _shape_param_190);  view_146 = _shape_param_190 = None
        permute_57: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_147, [2, 0, 3, 1, 4]);  view_147 = None
        unbind_5 = torch.ops.aten.unbind.int(permute_57);  permute_57 = None
        getitem_43: "bf16[512, 16, 49, 32]" = unbind_5[0]
        getitem_44: "bf16[512, 16, 49, 32]" = unbind_5[1]
        getitem_45: "bf16[512, 16, 49, 32]" = unbind_5[2];  unbind_5 = None
        mul_56: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_43, 0.1767766952966369);  getitem_43 = None
        permute_58: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_44, [0, 1, 3, 2]);  getitem_44 = None
        expand_20: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_56, _shape_param_191);  mul_56 = _shape_param_191 = None
        clone_38: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_148: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_38, _shape_param_192);  clone_38 = _shape_param_192 = None
        expand_21: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_58, _shape_param_193);  permute_58 = _shape_param_193 = None
        clone_39: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_149: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_39, _shape_param_194);  clone_39 = _shape_param_194 = None
        constant_pad_nd_20: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_148, _shape_param_195);  _shape_param_195 = None
        constant_pad_nd_21: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_149, _shape_param_196);  _shape_param_196 = None
        bmm_10: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_20, constant_pad_nd_21);  constant_pad_nd_20 = constant_pad_nd_21 = None
        slice_16: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_10, 1, 0, -7)
        slice_17: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_16, 2, 0, -7);  slice_16 = None
        view_150: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_17, _shape_param_197);  slice_17 = _shape_param_197 = None
        view_151: "i64[2401]" = torch.ops.aten.view.default(arg85_1, [-1]);  arg85_1 = None
        index_15: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg86_1, [view_151]);  arg86_1 = view_151 = None
        view_152: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_15, _shape_param_198);  index_15 = _shape_param_198 = None
        permute_59: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_152, [2, 0, 1]);  view_152 = None
        clone_40: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        unsqueeze_9: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_40, 0);  clone_40 = None
        add_55: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_150, unsqueeze_9);  view_150 = unsqueeze_9 = None
        view_153: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_55, _shape_param_199);  add_55 = _shape_param_199 = None
        unsqueeze_10: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg87_1, 1);  arg87_1 = None
        unsqueeze_11: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 0);  unsqueeze_10 = None
        add_56: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_153, unsqueeze_11);  view_153 = unsqueeze_11 = None
        view_154: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_56, _shape_param_200);  add_56 = _shape_param_200 = None
        amax_5: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_154, [-1], True)
        sub_19: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_154, amax_5);  view_154 = None
        exp_5: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_6: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_13: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = None
        convert_element_type_89: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None
        expand_22: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_89, _shape_param_201);  convert_element_type_89 = _shape_param_201 = None
        view_155: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_22, _shape_param_202);  expand_22 = _shape_param_202 = None
        expand_23: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_45, _shape_param_203);  getitem_45 = _shape_param_203 = None
        clone_41: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_156: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_41, _shape_param_204);  clone_41 = _shape_param_204 = None
        constant_pad_nd_22: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_155, _shape_param_205);  _shape_param_205 = None
        constant_pad_nd_23: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_156, _shape_param_206);  _shape_param_206 = None
        bmm_11: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_22, constant_pad_nd_23);  constant_pad_nd_22 = constant_pad_nd_23 = None
        slice_18: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_11, 1, 0, -7);  bmm_11 = None
        view_157: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_18, _shape_param_207);  slice_18 = _shape_param_207 = None
        permute_60: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
        clone_42: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_158: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_42, _shape_param_208);  clone_42 = _shape_param_208 = None
        convert_element_type_90: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg88_1, torch.bfloat16);  arg88_1 = None
        convert_element_type_91: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg89_1, torch.bfloat16);  arg89_1 = None
        view_159: "bf16[25088, 512]" = torch.ops.aten.view.default(view_158, _shape_param_209);  view_158 = _shape_param_209 = None
        permute_61: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_91, [1, 0]);  convert_element_type_91 = None
        addmm_21: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_90, view_159, permute_61);  convert_element_type_90 = None
        view_160: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_21, _shape_param_210);  addmm_21 = _shape_param_210 = None
        view_161: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_160, _shape_param_211);  view_160 = _shape_param_211 = None
        view_162: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_161, _shape_param_212);  view_161 = _shape_param_212 = None
        permute_62: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_162, [0, 1, 3, 2, 4, 5]);  view_162 = None
        clone_43: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_163: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_43, _shape_param_213);  clone_43 = _shape_param_213 = None
        add_57: "i64[14]" = torch.ops.aten.add.Tensor(iota_2, 11);  iota_2 = None
        fmod_5: "i64[14]" = torch.ops.aten.fmod.Scalar(add_57, 14);  add_57 = None
        index_16: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_163, [None, fmod_5]);  view_163 = None
        index_17: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_16, [None, None, fmod_5]);  index_16 = None
        inductor_lookup_seed_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 8)
        inductor_random_8: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_214, inductor_lookup_seed_8, 'rand');  _shape_param_214 = inductor_lookup_seed_8 = None
        lt_8: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_8, 0.9782608672976494);  inductor_random_8 = None
        convert_element_type_92: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_8, torch.bfloat16)
        div_14: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_92, 0.9782608672976494);  convert_element_type_92 = None
        mul_57: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_17, div_14);  index_17 = div_14 = None
        add_58: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_141, mul_57);  mul_57 = None
        view_164: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_58, _shape_param_215);  add_58 = _shape_param_215 = None
        convert_element_type_93: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_164, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_93, [2], correction = 0, keepdim = True)
        getitem_46: "f32[128, 196, 1]" = var_mean_14[0]
        getitem_47: "f32[128, 196, 1]" = var_mean_14[1];  var_mean_14 = None
        add_59: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_14: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_20: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_93, getitem_47);  convert_element_type_93 = None
        mul_58: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_14);  sub_20 = None
        mul_59: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_58, arg90_1);  mul_58 = arg90_1 = None
        add_60: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_59, arg91_1);  mul_59 = arg91_1 = None
        convert_element_type_94: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg92_1, torch.bfloat16);  arg92_1 = None
        convert_element_type_95: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg93_1, torch.bfloat16);  arg93_1 = None
        convert_element_type_96: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_60, torch.bfloat16);  add_60 = None
        view_165: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_96, _shape_param_216);  convert_element_type_96 = _shape_param_216 = None
        permute_63: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_95, [1, 0]);  convert_element_type_95 = None
        addmm_22: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_94, view_165, permute_63);  convert_element_type_94 = None
        view_166: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_22, _shape_param_217);  _shape_param_217 = None
        convert_element_type_97: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_166, torch.float32);  view_166 = None
        mul_60: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_97, 0.5)
        mul_61: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_97, 0.7071067811865476);  convert_element_type_97 = None
        erf_5: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_61: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_62: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_60, add_61);  mul_60 = add_61 = None
        convert_element_type_98: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        convert_element_type_99: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg94_1, torch.bfloat16);  arg94_1 = None
        convert_element_type_100: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg95_1, torch.bfloat16);  arg95_1 = None
        view_167: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_98, _shape_param_218);  convert_element_type_98 = _shape_param_218 = None
        permute_64: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_100, [1, 0]);  convert_element_type_100 = None
        addmm_23: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_99, view_167, permute_64);  convert_element_type_99 = None
        view_168: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_23, _shape_param_219);  addmm_23 = _shape_param_219 = None
        inductor_lookup_seed_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 9)
        inductor_random_9: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_220, inductor_lookup_seed_9, 'rand');  _shape_param_220 = inductor_lookup_seed_9 = None
        lt_9: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_9, 0.9782608672976494);  inductor_random_9 = None
        convert_element_type_101: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_9, torch.bfloat16)
        div_15: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_101, 0.9782608672976494);  convert_element_type_101 = None
        mul_63: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_168, div_15);  view_168 = div_15 = None
        add_62: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_164, mul_63);  mul_63 = None
        view_169: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_62, _shape_param_221);  add_62 = _shape_param_221 = None
        convert_element_type_102: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_169, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_102, [3], correction = 0, keepdim = True)
        getitem_48: "f32[128, 14, 14, 1]" = var_mean_15[0]
        getitem_49: "f32[128, 14, 14, 1]" = var_mean_15[1];  var_mean_15 = None
        add_63: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_15: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        sub_21: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_102, getitem_49);  convert_element_type_102 = None
        mul_64: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_15);  sub_21 = None
        mul_65: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_64, arg96_1);  mul_64 = arg96_1 = None
        add_64: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_65, arg97_1);  mul_65 = arg97_1 = None
        view_170: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_64, _shape_param_222);  add_64 = _shape_param_222 = None
        permute_65: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_170, [0, 1, 3, 2, 4, 5]);  view_170 = None
        clone_44: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
        view_171: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_44, _shape_param_223);  clone_44 = _shape_param_223 = None
        view_172: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_171, _shape_param_224);  view_171 = _shape_param_224 = None
        convert_element_type_103: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg98_1, torch.bfloat16);  arg98_1 = None
        convert_element_type_104: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg99_1, torch.bfloat16);  arg99_1 = None
        convert_element_type_105: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_172, torch.bfloat16);  view_172 = None
        view_173: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_105, _shape_param_225);  convert_element_type_105 = _shape_param_225 = None
        permute_66: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_104, [1, 0]);  convert_element_type_104 = None
        addmm_24: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_103, view_173, permute_66);  convert_element_type_103 = None
        view_174: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_24, _shape_param_226);  addmm_24 = _shape_param_226 = None
        view_175: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_174, _shape_param_227);  view_174 = _shape_param_227 = None
        permute_67: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_175, [2, 0, 3, 1, 4]);  view_175 = None
        unbind_6 = torch.ops.aten.unbind.int(permute_67);  permute_67 = None
        getitem_50: "bf16[512, 16, 49, 32]" = unbind_6[0]
        getitem_51: "bf16[512, 16, 49, 32]" = unbind_6[1]
        getitem_52: "bf16[512, 16, 49, 32]" = unbind_6[2];  unbind_6 = None
        mul_66: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_50, 0.1767766952966369);  getitem_50 = None
        permute_68: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_51, [0, 1, 3, 2]);  getitem_51 = None
        expand_24: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_66, _shape_param_228);  mul_66 = _shape_param_228 = None
        clone_45: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_176: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_45, _shape_param_229);  clone_45 = _shape_param_229 = None
        expand_25: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_68, _shape_param_230);  permute_68 = _shape_param_230 = None
        clone_46: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_177: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_46, _shape_param_231);  clone_46 = _shape_param_231 = None
        constant_pad_nd_24: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_176, _shape_param_232);  _shape_param_232 = None
        constant_pad_nd_25: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_177, _shape_param_233);  _shape_param_233 = None
        bmm_12: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_24, constant_pad_nd_25);  constant_pad_nd_24 = constant_pad_nd_25 = None
        slice_19: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_12, 1, 0, -7)
        slice_20: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_19, 2, 0, -7);  slice_19 = None
        view_178: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_20, _shape_param_234);  slice_20 = _shape_param_234 = None
        view_179: "i64[2401]" = torch.ops.aten.view.default(arg100_1, [-1]);  arg100_1 = None
        index_18: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg101_1, [view_179]);  arg101_1 = view_179 = None
        view_180: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_18, _shape_param_235);  index_18 = _shape_param_235 = None
        permute_69: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_180, [2, 0, 1]);  view_180 = None
        clone_47: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
        unsqueeze_12: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_47, 0);  clone_47 = None
        add_65: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_178, unsqueeze_12);  view_178 = unsqueeze_12 = None
        amax_6: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_65, [-1], True)
        sub_22: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_65, amax_6);  add_65 = None
        exp_6: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_7: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_16: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = None
        convert_element_type_106: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None
        expand_26: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_106, _shape_param_236);  convert_element_type_106 = _shape_param_236 = None
        view_181: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_26, _shape_param_237);  expand_26 = _shape_param_237 = None
        expand_27: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_52, _shape_param_238);  getitem_52 = _shape_param_238 = None
        clone_48: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_182: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_48, _shape_param_239);  clone_48 = _shape_param_239 = None
        constant_pad_nd_26: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_181, _shape_param_240);  _shape_param_240 = None
        constant_pad_nd_27: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_182, _shape_param_241);  _shape_param_241 = None
        bmm_13: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_26, constant_pad_nd_27);  constant_pad_nd_26 = constant_pad_nd_27 = None
        slice_21: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_13, 1, 0, -7);  bmm_13 = None
        view_183: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_21, _shape_param_242);  slice_21 = _shape_param_242 = None
        permute_70: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        clone_49: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_184: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_49, _shape_param_243);  clone_49 = _shape_param_243 = None
        convert_element_type_107: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg102_1, torch.bfloat16);  arg102_1 = None
        convert_element_type_108: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg103_1, torch.bfloat16);  arg103_1 = None
        view_185: "bf16[25088, 512]" = torch.ops.aten.view.default(view_184, _shape_param_244);  view_184 = _shape_param_244 = None
        permute_71: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_108, [1, 0]);  convert_element_type_108 = None
        addmm_25: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_107, view_185, permute_71);  convert_element_type_107 = None
        view_186: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_25, _shape_param_245);  addmm_25 = _shape_param_245 = None
        view_187: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_186, _shape_param_246);  view_186 = _shape_param_246 = None
        view_188: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_187, _shape_param_247);  view_187 = _shape_param_247 = None
        permute_72: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_188, [0, 1, 3, 2, 4, 5]);  view_188 = None
        clone_50: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
        view_189: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_50, _shape_param_248);  clone_50 = _shape_param_248 = None
        inductor_lookup_seed_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 10)
        inductor_random_10: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_249, inductor_lookup_seed_10, 'rand');  _shape_param_249 = inductor_lookup_seed_10 = None
        lt_10: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_10, 0.9739130418747663);  inductor_random_10 = None
        convert_element_type_109: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_10, torch.bfloat16)
        div_17: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_109, 0.9739130418747663);  convert_element_type_109 = None
        mul_67: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_189, div_17);  view_189 = div_17 = None
        add_66: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_169, mul_67);  mul_67 = None
        view_190: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_66, _shape_param_250);  add_66 = _shape_param_250 = None
        convert_element_type_110: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_190, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_110, [2], correction = 0, keepdim = True)
        getitem_53: "f32[128, 196, 1]" = var_mean_16[0]
        getitem_54: "f32[128, 196, 1]" = var_mean_16[1];  var_mean_16 = None
        add_67: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-05);  getitem_53 = None
        rsqrt_16: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        sub_23: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_110, getitem_54);  convert_element_type_110 = None
        mul_68: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_16);  sub_23 = None
        mul_69: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_68, arg104_1);  mul_68 = arg104_1 = None
        add_68: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_69, arg105_1);  mul_69 = arg105_1 = None
        convert_element_type_111: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg106_1, torch.bfloat16);  arg106_1 = None
        convert_element_type_112: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg107_1, torch.bfloat16);  arg107_1 = None
        convert_element_type_113: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16);  add_68 = None
        view_191: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_113, _shape_param_251);  convert_element_type_113 = _shape_param_251 = None
        permute_73: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_112, [1, 0]);  convert_element_type_112 = None
        addmm_26: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_111, view_191, permute_73);  convert_element_type_111 = None
        view_192: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_26, _shape_param_252);  _shape_param_252 = None
        convert_element_type_114: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_192, torch.float32);  view_192 = None
        mul_70: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_114, 0.5)
        mul_71: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_114, 0.7071067811865476);  convert_element_type_114 = None
        erf_6: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_69: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_72: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_70, add_69);  mul_70 = add_69 = None
        convert_element_type_115: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None
        convert_element_type_116: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg108_1, torch.bfloat16);  arg108_1 = None
        convert_element_type_117: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg109_1, torch.bfloat16);  arg109_1 = None
        view_193: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_115, _shape_param_253);  convert_element_type_115 = _shape_param_253 = None
        permute_74: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_117, [1, 0]);  convert_element_type_117 = None
        addmm_27: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_116, view_193, permute_74);  convert_element_type_116 = None
        view_194: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_27, _shape_param_254);  addmm_27 = _shape_param_254 = None
        inductor_lookup_seed_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 11)
        inductor_random_11: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_255, inductor_lookup_seed_11, 'rand');  _shape_param_255 = inductor_lookup_seed_11 = None
        lt_11: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_11, 0.9739130418747663);  inductor_random_11 = None
        convert_element_type_118: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_11, torch.bfloat16)
        div_18: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_118, 0.9739130418747663);  convert_element_type_118 = None
        mul_73: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_194, div_18);  view_194 = div_18 = None
        add_70: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_190, mul_73);  mul_73 = None
        view_195: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_70, _shape_param_256);  add_70 = _shape_param_256 = None
        convert_element_type_119: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_195, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_119, [3], correction = 0, keepdim = True)
        getitem_55: "f32[128, 14, 14, 1]" = var_mean_17[0]
        getitem_56: "f32[128, 14, 14, 1]" = var_mean_17[1];  var_mean_17 = None
        add_71: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-05);  getitem_55 = None
        rsqrt_17: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_24: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_119, getitem_56);  convert_element_type_119 = None
        mul_74: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_17);  sub_24 = None
        mul_75: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_74, arg110_1);  mul_74 = arg110_1 = None
        add_72: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_75, arg111_1);  mul_75 = arg111_1 = None
        index_19: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_72, [None, fmod_4]);  add_72 = None
        index_20: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_19, [None, None, fmod_4]);  index_19 = None
        view_196: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_20, _shape_param_257);  index_20 = _shape_param_257 = None
        permute_75: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_196, [0, 1, 3, 2, 4, 5]);  view_196 = None
        clone_51: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
        view_197: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_51, _shape_param_258);  clone_51 = _shape_param_258 = None
        view_198: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_197, _shape_param_259);  view_197 = _shape_param_259 = None
        convert_element_type_120: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg112_1, torch.bfloat16);  arg112_1 = None
        convert_element_type_121: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg113_1, torch.bfloat16);  arg113_1 = None
        convert_element_type_122: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_198, torch.bfloat16);  view_198 = None
        view_199: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_122, _shape_param_260);  convert_element_type_122 = _shape_param_260 = None
        permute_76: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_121, [1, 0]);  convert_element_type_121 = None
        addmm_28: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_120, view_199, permute_76);  convert_element_type_120 = None
        view_200: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_28, _shape_param_261);  addmm_28 = _shape_param_261 = None
        view_201: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_200, _shape_param_262);  view_200 = _shape_param_262 = None
        permute_77: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_201, [2, 0, 3, 1, 4]);  view_201 = None
        unbind_7 = torch.ops.aten.unbind.int(permute_77);  permute_77 = None
        getitem_57: "bf16[512, 16, 49, 32]" = unbind_7[0]
        getitem_58: "bf16[512, 16, 49, 32]" = unbind_7[1]
        getitem_59: "bf16[512, 16, 49, 32]" = unbind_7[2];  unbind_7 = None
        mul_76: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_57, 0.1767766952966369);  getitem_57 = None
        permute_78: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_58, [0, 1, 3, 2]);  getitem_58 = None
        expand_28: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_76, _shape_param_263);  mul_76 = _shape_param_263 = None
        clone_52: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_202: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_52, _shape_param_264);  clone_52 = _shape_param_264 = None
        expand_29: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_78, _shape_param_265);  permute_78 = _shape_param_265 = None
        clone_53: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_203: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_53, _shape_param_266);  clone_53 = _shape_param_266 = None
        constant_pad_nd_28: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_202, _shape_param_267);  _shape_param_267 = None
        constant_pad_nd_29: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_203, _shape_param_268);  _shape_param_268 = None
        bmm_14: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_28, constant_pad_nd_29);  constant_pad_nd_28 = constant_pad_nd_29 = None
        slice_22: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_14, 1, 0, -7)
        slice_23: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_22, 2, 0, -7);  slice_22 = None
        view_204: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_23, _shape_param_269);  slice_23 = _shape_param_269 = None
        view_205: "i64[2401]" = torch.ops.aten.view.default(arg114_1, [-1]);  arg114_1 = None
        index_21: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg115_1, [view_205]);  arg115_1 = view_205 = None
        view_206: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_21, _shape_param_270);  index_21 = _shape_param_270 = None
        permute_79: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_206, [2, 0, 1]);  view_206 = None
        clone_54: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
        unsqueeze_13: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_54, 0);  clone_54 = None
        add_73: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_204, unsqueeze_13);  view_204 = unsqueeze_13 = None
        view_207: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_73, _shape_param_271);  add_73 = _shape_param_271 = None
        unsqueeze_14: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg116_1, 1);  arg116_1 = None
        unsqueeze_15: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 0);  unsqueeze_14 = None
        add_74: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_207, unsqueeze_15);  view_207 = unsqueeze_15 = None
        view_208: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_74, _shape_param_272);  add_74 = _shape_param_272 = None
        amax_7: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_208, [-1], True)
        sub_25: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_208, amax_7);  view_208 = None
        exp_7: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_8: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_19: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = None
        convert_element_type_123: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None
        expand_30: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_123, _shape_param_273);  convert_element_type_123 = _shape_param_273 = None
        view_209: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_30, _shape_param_274);  expand_30 = _shape_param_274 = None
        expand_31: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_59, _shape_param_275);  getitem_59 = _shape_param_275 = None
        clone_55: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_210: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_55, _shape_param_276);  clone_55 = _shape_param_276 = None
        constant_pad_nd_30: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_209, _shape_param_277);  _shape_param_277 = None
        constant_pad_nd_31: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_210, _shape_param_278);  _shape_param_278 = None
        bmm_15: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_30, constant_pad_nd_31);  constant_pad_nd_30 = constant_pad_nd_31 = None
        slice_24: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_15, 1, 0, -7);  bmm_15 = None
        view_211: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_24, _shape_param_279);  slice_24 = _shape_param_279 = None
        permute_80: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None
        clone_56: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_212: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_56, _shape_param_280);  clone_56 = _shape_param_280 = None
        convert_element_type_124: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg117_1, torch.bfloat16);  arg117_1 = None
        convert_element_type_125: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg118_1, torch.bfloat16);  arg118_1 = None
        view_213: "bf16[25088, 512]" = torch.ops.aten.view.default(view_212, _shape_param_281);  view_212 = _shape_param_281 = None
        permute_81: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_125, [1, 0]);  convert_element_type_125 = None
        addmm_29: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_124, view_213, permute_81);  convert_element_type_124 = None
        view_214: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_29, _shape_param_282);  addmm_29 = _shape_param_282 = None
        view_215: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_214, _shape_param_283);  view_214 = _shape_param_283 = None
        view_216: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_215, _shape_param_284);  view_215 = _shape_param_284 = None
        permute_82: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_216, [0, 1, 3, 2, 4, 5]);  view_216 = None
        clone_57: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_217: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_57, _shape_param_285);  clone_57 = _shape_param_285 = None
        index_22: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_217, [None, fmod_5]);  view_217 = None
        index_23: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_22, [None, None, fmod_5]);  index_22 = None
        inductor_lookup_seed_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 12)
        inductor_random_12: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_286, inductor_lookup_seed_12, 'rand');  _shape_param_286 = inductor_lookup_seed_12 = None
        lt_12: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_12, 0.9695652164518833);  inductor_random_12 = None
        convert_element_type_126: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_12, torch.bfloat16)
        div_20: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_126, 0.9695652164518833);  convert_element_type_126 = None
        mul_77: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_23, div_20);  index_23 = div_20 = None
        add_75: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_195, mul_77);  mul_77 = None
        view_218: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_75, _shape_param_287);  add_75 = _shape_param_287 = None
        convert_element_type_127: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_218, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_127, [2], correction = 0, keepdim = True)
        getitem_60: "f32[128, 196, 1]" = var_mean_18[0]
        getitem_61: "f32[128, 196, 1]" = var_mean_18[1];  var_mean_18 = None
        add_76: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_18: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_26: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_127, getitem_61);  convert_element_type_127 = None
        mul_78: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_18);  sub_26 = None
        mul_79: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_78, arg119_1);  mul_78 = arg119_1 = None
        add_77: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_79, arg120_1);  mul_79 = arg120_1 = None
        convert_element_type_128: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg121_1, torch.bfloat16);  arg121_1 = None
        convert_element_type_129: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg122_1, torch.bfloat16);  arg122_1 = None
        convert_element_type_130: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None
        view_219: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_130, _shape_param_288);  convert_element_type_130 = _shape_param_288 = None
        permute_83: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_129, [1, 0]);  convert_element_type_129 = None
        addmm_30: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_128, view_219, permute_83);  convert_element_type_128 = None
        view_220: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_30, _shape_param_289);  _shape_param_289 = None
        convert_element_type_131: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_220, torch.float32);  view_220 = None
        mul_80: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_131, 0.5)
        mul_81: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_131, 0.7071067811865476);  convert_element_type_131 = None
        erf_7: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_81);  mul_81 = None
        add_78: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_82: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_80, add_78);  mul_80 = add_78 = None
        convert_element_type_132: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_82, torch.bfloat16);  mul_82 = None
        convert_element_type_133: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg123_1, torch.bfloat16);  arg123_1 = None
        convert_element_type_134: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg124_1, torch.bfloat16);  arg124_1 = None
        view_221: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_132, _shape_param_290);  convert_element_type_132 = _shape_param_290 = None
        permute_84: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_134, [1, 0]);  convert_element_type_134 = None
        addmm_31: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_133, view_221, permute_84);  convert_element_type_133 = None
        view_222: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_31, _shape_param_291);  addmm_31 = _shape_param_291 = None
        inductor_lookup_seed_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 13)
        inductor_random_13: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_292, inductor_lookup_seed_13, 'rand');  _shape_param_292 = inductor_lookup_seed_13 = None
        lt_13: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_13, 0.9695652164518833);  inductor_random_13 = None
        convert_element_type_135: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_13, torch.bfloat16)
        div_21: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_135, 0.9695652164518833);  convert_element_type_135 = None
        mul_83: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_222, div_21);  view_222 = div_21 = None
        add_79: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_218, mul_83);  mul_83 = None
        view_223: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_79, _shape_param_293);  add_79 = _shape_param_293 = None
        convert_element_type_136: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_223, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_136, [3], correction = 0, keepdim = True)
        getitem_62: "f32[128, 14, 14, 1]" = var_mean_19[0]
        getitem_63: "f32[128, 14, 14, 1]" = var_mean_19[1];  var_mean_19 = None
        add_80: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_19: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_27: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_136, getitem_63);  convert_element_type_136 = None
        mul_84: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_19);  sub_27 = None
        mul_85: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_84, arg125_1);  mul_84 = arg125_1 = None
        add_81: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_85, arg126_1);  mul_85 = arg126_1 = None
        view_224: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_81, _shape_param_294);  add_81 = _shape_param_294 = None
        permute_85: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_224, [0, 1, 3, 2, 4, 5]);  view_224 = None
        clone_58: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
        view_225: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_58, _shape_param_295);  clone_58 = _shape_param_295 = None
        view_226: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_225, _shape_param_296);  view_225 = _shape_param_296 = None
        convert_element_type_137: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg127_1, torch.bfloat16);  arg127_1 = None
        convert_element_type_138: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg128_1, torch.bfloat16);  arg128_1 = None
        convert_element_type_139: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_226, torch.bfloat16);  view_226 = None
        view_227: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_139, _shape_param_297);  convert_element_type_139 = _shape_param_297 = None
        permute_86: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_138, [1, 0]);  convert_element_type_138 = None
        addmm_32: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_137, view_227, permute_86);  convert_element_type_137 = None
        view_228: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_32, _shape_param_298);  addmm_32 = _shape_param_298 = None
        view_229: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_228, _shape_param_299);  view_228 = _shape_param_299 = None
        permute_87: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_229, [2, 0, 3, 1, 4]);  view_229 = None
        unbind_8 = torch.ops.aten.unbind.int(permute_87);  permute_87 = None
        getitem_64: "bf16[512, 16, 49, 32]" = unbind_8[0]
        getitem_65: "bf16[512, 16, 49, 32]" = unbind_8[1]
        getitem_66: "bf16[512, 16, 49, 32]" = unbind_8[2];  unbind_8 = None
        mul_86: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_64, 0.1767766952966369);  getitem_64 = None
        permute_88: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_65, [0, 1, 3, 2]);  getitem_65 = None
        expand_32: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_86, _shape_param_300);  mul_86 = _shape_param_300 = None
        clone_59: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_230: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_59, _shape_param_301);  clone_59 = _shape_param_301 = None
        expand_33: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_88, _shape_param_302);  permute_88 = _shape_param_302 = None
        clone_60: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_231: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_60, _shape_param_303);  clone_60 = _shape_param_303 = None
        constant_pad_nd_32: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_230, _shape_param_304);  _shape_param_304 = None
        constant_pad_nd_33: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_231, _shape_param_305);  _shape_param_305 = None
        bmm_16: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_32, constant_pad_nd_33);  constant_pad_nd_32 = constant_pad_nd_33 = None
        slice_25: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_16, 1, 0, -7)
        slice_26: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_25, 2, 0, -7);  slice_25 = None
        view_232: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_26, _shape_param_306);  slice_26 = _shape_param_306 = None
        view_233: "i64[2401]" = torch.ops.aten.view.default(arg129_1, [-1]);  arg129_1 = None
        index_24: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg130_1, [view_233]);  arg130_1 = view_233 = None
        view_234: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_24, _shape_param_307);  index_24 = _shape_param_307 = None
        permute_89: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_234, [2, 0, 1]);  view_234 = None
        clone_61: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
        unsqueeze_16: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_61, 0);  clone_61 = None
        add_82: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_232, unsqueeze_16);  view_232 = unsqueeze_16 = None
        amax_8: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_82, [-1], True)
        sub_28: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_82, amax_8);  add_82 = None
        exp_8: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_9: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_22: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = None
        convert_element_type_140: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None
        expand_34: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_140, _shape_param_308);  convert_element_type_140 = _shape_param_308 = None
        view_235: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_34, _shape_param_309);  expand_34 = _shape_param_309 = None
        expand_35: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_66, _shape_param_310);  getitem_66 = _shape_param_310 = None
        clone_62: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_236: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_62, _shape_param_311);  clone_62 = _shape_param_311 = None
        constant_pad_nd_34: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_235, _shape_param_312);  _shape_param_312 = None
        constant_pad_nd_35: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_236, _shape_param_313);  _shape_param_313 = None
        bmm_17: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_34, constant_pad_nd_35);  constant_pad_nd_34 = constant_pad_nd_35 = None
        slice_27: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_17, 1, 0, -7);  bmm_17 = None
        view_237: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_27, _shape_param_314);  slice_27 = _shape_param_314 = None
        permute_90: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None
        clone_63: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        view_238: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_63, _shape_param_315);  clone_63 = _shape_param_315 = None
        convert_element_type_141: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg131_1, torch.bfloat16);  arg131_1 = None
        convert_element_type_142: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg132_1, torch.bfloat16);  arg132_1 = None
        view_239: "bf16[25088, 512]" = torch.ops.aten.view.default(view_238, _shape_param_316);  view_238 = _shape_param_316 = None
        permute_91: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_142, [1, 0]);  convert_element_type_142 = None
        addmm_33: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_141, view_239, permute_91);  convert_element_type_141 = None
        view_240: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_33, _shape_param_317);  addmm_33 = _shape_param_317 = None
        view_241: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_240, _shape_param_318);  view_240 = _shape_param_318 = None
        view_242: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_241, _shape_param_319);  view_241 = _shape_param_319 = None
        permute_92: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_242, [0, 1, 3, 2, 4, 5]);  view_242 = None
        clone_64: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_243: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_64, _shape_param_320);  clone_64 = _shape_param_320 = None
        inductor_lookup_seed_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 14)
        inductor_random_14: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_321, inductor_lookup_seed_14, 'rand');  _shape_param_321 = inductor_lookup_seed_14 = None
        lt_14: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_14, 0.9652173891663551);  inductor_random_14 = None
        convert_element_type_143: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_14, torch.bfloat16)
        div_23: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_143, 0.9652173891663551);  convert_element_type_143 = None
        mul_87: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_243, div_23);  view_243 = div_23 = None
        add_83: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_223, mul_87);  mul_87 = None
        view_244: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_83, _shape_param_322);  add_83 = _shape_param_322 = None
        convert_element_type_144: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_244, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_144, [2], correction = 0, keepdim = True)
        getitem_67: "f32[128, 196, 1]" = var_mean_20[0]
        getitem_68: "f32[128, 196, 1]" = var_mean_20[1];  var_mean_20 = None
        add_84: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_67, 1e-05);  getitem_67 = None
        rsqrt_20: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_29: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_144, getitem_68);  convert_element_type_144 = None
        mul_88: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_20);  sub_29 = None
        mul_89: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_88, arg133_1);  mul_88 = arg133_1 = None
        add_85: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_89, arg134_1);  mul_89 = arg134_1 = None
        convert_element_type_145: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg135_1, torch.bfloat16);  arg135_1 = None
        convert_element_type_146: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg136_1, torch.bfloat16);  arg136_1 = None
        convert_element_type_147: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16);  add_85 = None
        view_245: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_147, _shape_param_323);  convert_element_type_147 = _shape_param_323 = None
        permute_93: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_146, [1, 0]);  convert_element_type_146 = None
        addmm_34: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_145, view_245, permute_93);  convert_element_type_145 = None
        view_246: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_34, _shape_param_324);  _shape_param_324 = None
        convert_element_type_148: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_246, torch.float32);  view_246 = None
        mul_90: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_148, 0.5)
        mul_91: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_148, 0.7071067811865476);  convert_element_type_148 = None
        erf_8: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_91);  mul_91 = None
        add_86: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_92: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_90, add_86);  mul_90 = add_86 = None
        convert_element_type_149: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_92, torch.bfloat16);  mul_92 = None
        convert_element_type_150: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg137_1, torch.bfloat16);  arg137_1 = None
        convert_element_type_151: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg138_1, torch.bfloat16);  arg138_1 = None
        view_247: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_149, _shape_param_325);  convert_element_type_149 = _shape_param_325 = None
        permute_94: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_151, [1, 0]);  convert_element_type_151 = None
        addmm_35: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_150, view_247, permute_94);  convert_element_type_150 = None
        view_248: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_35, _shape_param_326);  addmm_35 = _shape_param_326 = None
        inductor_lookup_seed_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 15)
        inductor_random_15: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_327, inductor_lookup_seed_15, 'rand');  _shape_param_327 = inductor_lookup_seed_15 = None
        lt_15: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_15, 0.9652173891663551);  inductor_random_15 = None
        convert_element_type_152: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_15, torch.bfloat16)
        div_24: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_152, 0.9652173891663551);  convert_element_type_152 = None
        mul_93: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_248, div_24);  view_248 = div_24 = None
        add_87: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_244, mul_93);  mul_93 = None
        view_249: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_87, _shape_param_328);  add_87 = _shape_param_328 = None
        convert_element_type_153: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_249, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_153, [3], correction = 0, keepdim = True)
        getitem_69: "f32[128, 14, 14, 1]" = var_mean_21[0]
        getitem_70: "f32[128, 14, 14, 1]" = var_mean_21[1];  var_mean_21 = None
        add_88: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_69, 1e-05);  getitem_69 = None
        rsqrt_21: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_30: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_153, getitem_70);  convert_element_type_153 = None
        mul_94: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_21);  sub_30 = None
        mul_95: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_94, arg139_1);  mul_94 = arg139_1 = None
        add_89: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_95, arg140_1);  mul_95 = arg140_1 = None
        index_25: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_89, [None, fmod_4]);  add_89 = None
        index_26: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_25, [None, None, fmod_4]);  index_25 = None
        view_250: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_26, _shape_param_329);  index_26 = _shape_param_329 = None
        permute_95: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_250, [0, 1, 3, 2, 4, 5]);  view_250 = None
        clone_65: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_251: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_65, _shape_param_330);  clone_65 = _shape_param_330 = None
        view_252: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_251, _shape_param_331);  view_251 = _shape_param_331 = None
        convert_element_type_154: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg141_1, torch.bfloat16);  arg141_1 = None
        convert_element_type_155: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg142_1, torch.bfloat16);  arg142_1 = None
        convert_element_type_156: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_252, torch.bfloat16);  view_252 = None
        view_253: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_156, _shape_param_332);  convert_element_type_156 = _shape_param_332 = None
        permute_96: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_155, [1, 0]);  convert_element_type_155 = None
        addmm_36: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_154, view_253, permute_96);  convert_element_type_154 = None
        view_254: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_36, _shape_param_333);  addmm_36 = _shape_param_333 = None
        view_255: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_254, _shape_param_334);  view_254 = _shape_param_334 = None
        permute_97: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_255, [2, 0, 3, 1, 4]);  view_255 = None
        unbind_9 = torch.ops.aten.unbind.int(permute_97);  permute_97 = None
        getitem_71: "bf16[512, 16, 49, 32]" = unbind_9[0]
        getitem_72: "bf16[512, 16, 49, 32]" = unbind_9[1]
        getitem_73: "bf16[512, 16, 49, 32]" = unbind_9[2];  unbind_9 = None
        mul_96: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_71, 0.1767766952966369);  getitem_71 = None
        permute_98: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_72, [0, 1, 3, 2]);  getitem_72 = None
        expand_36: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_96, _shape_param_335);  mul_96 = _shape_param_335 = None
        clone_66: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_256: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_66, _shape_param_336);  clone_66 = _shape_param_336 = None
        expand_37: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_98, _shape_param_337);  permute_98 = _shape_param_337 = None
        clone_67: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_257: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_67, _shape_param_338);  clone_67 = _shape_param_338 = None
        constant_pad_nd_36: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_256, _shape_param_339);  _shape_param_339 = None
        constant_pad_nd_37: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_257, _shape_param_340);  _shape_param_340 = None
        bmm_18: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_36, constant_pad_nd_37);  constant_pad_nd_36 = constant_pad_nd_37 = None
        slice_28: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_18, 1, 0, -7)
        slice_29: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_28, 2, 0, -7);  slice_28 = None
        view_258: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_29, _shape_param_341);  slice_29 = _shape_param_341 = None
        view_259: "i64[2401]" = torch.ops.aten.view.default(arg143_1, [-1]);  arg143_1 = None
        index_27: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg144_1, [view_259]);  arg144_1 = view_259 = None
        view_260: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_27, _shape_param_342);  index_27 = _shape_param_342 = None
        permute_99: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_260, [2, 0, 1]);  view_260 = None
        clone_68: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
        unsqueeze_17: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_68, 0);  clone_68 = None
        add_90: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_258, unsqueeze_17);  view_258 = unsqueeze_17 = None
        view_261: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_90, _shape_param_343);  add_90 = _shape_param_343 = None
        unsqueeze_18: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg145_1, 1);  arg145_1 = None
        unsqueeze_19: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 0);  unsqueeze_18 = None
        add_91: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_261, unsqueeze_19);  view_261 = unsqueeze_19 = None
        view_262: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_91, _shape_param_344);  add_91 = _shape_param_344 = None
        amax_9: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_262, [-1], True)
        sub_31: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_262, amax_9);  view_262 = None
        exp_9: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_10: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_25: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = None
        convert_element_type_157: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None
        expand_38: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_157, _shape_param_345);  convert_element_type_157 = _shape_param_345 = None
        view_263: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_38, _shape_param_346);  expand_38 = _shape_param_346 = None
        expand_39: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_73, _shape_param_347);  getitem_73 = _shape_param_347 = None
        clone_69: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_264: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_69, _shape_param_348);  clone_69 = _shape_param_348 = None
        constant_pad_nd_38: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_263, _shape_param_349);  _shape_param_349 = None
        constant_pad_nd_39: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_264, _shape_param_350);  _shape_param_350 = None
        bmm_19: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_38, constant_pad_nd_39);  constant_pad_nd_38 = constant_pad_nd_39 = None
        slice_30: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_19, 1, 0, -7);  bmm_19 = None
        view_265: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_30, _shape_param_351);  slice_30 = _shape_param_351 = None
        permute_100: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
        clone_70: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
        view_266: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_70, _shape_param_352);  clone_70 = _shape_param_352 = None
        convert_element_type_158: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg146_1, torch.bfloat16);  arg146_1 = None
        convert_element_type_159: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg147_1, torch.bfloat16);  arg147_1 = None
        view_267: "bf16[25088, 512]" = torch.ops.aten.view.default(view_266, _shape_param_353);  view_266 = _shape_param_353 = None
        permute_101: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_159, [1, 0]);  convert_element_type_159 = None
        addmm_37: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_158, view_267, permute_101);  convert_element_type_158 = None
        view_268: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_37, _shape_param_354);  addmm_37 = _shape_param_354 = None
        view_269: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_268, _shape_param_355);  view_268 = _shape_param_355 = None
        view_270: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_269, _shape_param_356);  view_269 = _shape_param_356 = None
        permute_102: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_270, [0, 1, 3, 2, 4, 5]);  view_270 = None
        clone_71: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        view_271: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_71, _shape_param_357);  clone_71 = _shape_param_357 = None
        index_28: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_271, [None, fmod_5]);  view_271 = None
        index_29: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_28, [None, None, fmod_5]);  index_28 = None
        inductor_lookup_seed_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 16)
        inductor_random_16: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_358, inductor_lookup_seed_16, 'rand');  _shape_param_358 = inductor_lookup_seed_16 = None
        lt_16: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_16, 0.960869561880827);  inductor_random_16 = None
        convert_element_type_160: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_16, torch.bfloat16)
        div_26: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_160, 0.960869561880827);  convert_element_type_160 = None
        mul_97: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_29, div_26);  index_29 = div_26 = None
        add_92: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_249, mul_97);  mul_97 = None
        view_272: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_92, _shape_param_359);  add_92 = _shape_param_359 = None
        convert_element_type_161: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_272, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_161, [2], correction = 0, keepdim = True)
        getitem_74: "f32[128, 196, 1]" = var_mean_22[0]
        getitem_75: "f32[128, 196, 1]" = var_mean_22[1];  var_mean_22 = None
        add_93: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_22: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_32: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_161, getitem_75);  convert_element_type_161 = None
        mul_98: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_22);  sub_32 = None
        mul_99: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_98, arg148_1);  mul_98 = arg148_1 = None
        add_94: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_99, arg149_1);  mul_99 = arg149_1 = None
        convert_element_type_162: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg150_1, torch.bfloat16);  arg150_1 = None
        convert_element_type_163: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg151_1, torch.bfloat16);  arg151_1 = None
        convert_element_type_164: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16);  add_94 = None
        view_273: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_164, _shape_param_360);  convert_element_type_164 = _shape_param_360 = None
        permute_103: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_163, [1, 0]);  convert_element_type_163 = None
        addmm_38: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_162, view_273, permute_103);  convert_element_type_162 = None
        view_274: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_38, _shape_param_361);  _shape_param_361 = None
        convert_element_type_165: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_274, torch.float32);  view_274 = None
        mul_100: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_165, 0.5)
        mul_101: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_165, 0.7071067811865476);  convert_element_type_165 = None
        erf_9: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_101);  mul_101 = None
        add_95: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_102: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_100, add_95);  mul_100 = add_95 = None
        convert_element_type_166: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_102, torch.bfloat16);  mul_102 = None
        convert_element_type_167: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg152_1, torch.bfloat16);  arg152_1 = None
        convert_element_type_168: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg153_1, torch.bfloat16);  arg153_1 = None
        view_275: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_166, _shape_param_362);  convert_element_type_166 = _shape_param_362 = None
        permute_104: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_168, [1, 0]);  convert_element_type_168 = None
        addmm_39: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_167, view_275, permute_104);  convert_element_type_167 = None
        view_276: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_39, _shape_param_363);  addmm_39 = _shape_param_363 = None
        inductor_lookup_seed_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 17)
        inductor_random_17: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_364, inductor_lookup_seed_17, 'rand');  _shape_param_364 = inductor_lookup_seed_17 = None
        lt_17: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_17, 0.960869561880827);  inductor_random_17 = None
        convert_element_type_169: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_17, torch.bfloat16)
        div_27: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_169, 0.960869561880827);  convert_element_type_169 = None
        mul_103: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_276, div_27);  view_276 = div_27 = None
        add_96: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_272, mul_103);  mul_103 = None
        view_277: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_96, _shape_param_365);  add_96 = _shape_param_365 = None
        convert_element_type_170: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_277, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_170, [3], correction = 0, keepdim = True)
        getitem_76: "f32[128, 14, 14, 1]" = var_mean_23[0]
        getitem_77: "f32[128, 14, 14, 1]" = var_mean_23[1];  var_mean_23 = None
        add_97: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_23: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        sub_33: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_170, getitem_77);  convert_element_type_170 = None
        mul_104: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_23);  sub_33 = None
        mul_105: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_104, arg154_1);  mul_104 = arg154_1 = None
        add_98: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_105, arg155_1);  mul_105 = arg155_1 = None
        view_278: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_98, _shape_param_366);  add_98 = _shape_param_366 = None
        permute_105: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_278, [0, 1, 3, 2, 4, 5]);  view_278 = None
        clone_72: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        view_279: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_72, _shape_param_367);  clone_72 = _shape_param_367 = None
        view_280: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_279, _shape_param_368);  view_279 = _shape_param_368 = None
        convert_element_type_171: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg156_1, torch.bfloat16);  arg156_1 = None
        convert_element_type_172: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg157_1, torch.bfloat16);  arg157_1 = None
        convert_element_type_173: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_280, torch.bfloat16);  view_280 = None
        view_281: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_173, _shape_param_369);  convert_element_type_173 = _shape_param_369 = None
        permute_106: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_172, [1, 0]);  convert_element_type_172 = None
        addmm_40: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_171, view_281, permute_106);  convert_element_type_171 = None
        view_282: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_40, _shape_param_370);  addmm_40 = _shape_param_370 = None
        view_283: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_282, _shape_param_371);  view_282 = _shape_param_371 = None
        permute_107: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_283, [2, 0, 3, 1, 4]);  view_283 = None
        unbind_10 = torch.ops.aten.unbind.int(permute_107);  permute_107 = None
        getitem_78: "bf16[512, 16, 49, 32]" = unbind_10[0]
        getitem_79: "bf16[512, 16, 49, 32]" = unbind_10[1]
        getitem_80: "bf16[512, 16, 49, 32]" = unbind_10[2];  unbind_10 = None
        mul_106: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_78, 0.1767766952966369);  getitem_78 = None
        permute_108: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_79, [0, 1, 3, 2]);  getitem_79 = None
        expand_40: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_106, _shape_param_372);  mul_106 = _shape_param_372 = None
        clone_73: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_284: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_73, _shape_param_373);  clone_73 = _shape_param_373 = None
        expand_41: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_108, _shape_param_374);  permute_108 = _shape_param_374 = None
        clone_74: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_285: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_74, _shape_param_375);  clone_74 = _shape_param_375 = None
        constant_pad_nd_40: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_284, _shape_param_376);  _shape_param_376 = None
        constant_pad_nd_41: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_285, _shape_param_377);  _shape_param_377 = None
        bmm_20: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_40, constant_pad_nd_41);  constant_pad_nd_40 = constant_pad_nd_41 = None
        slice_31: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_20, 1, 0, -7)
        slice_32: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_31, 2, 0, -7);  slice_31 = None
        view_286: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_32, _shape_param_378);  slice_32 = _shape_param_378 = None
        view_287: "i64[2401]" = torch.ops.aten.view.default(arg158_1, [-1]);  arg158_1 = None
        index_30: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg159_1, [view_287]);  arg159_1 = view_287 = None
        view_288: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_30, _shape_param_379);  index_30 = _shape_param_379 = None
        permute_109: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_288, [2, 0, 1]);  view_288 = None
        clone_75: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None
        unsqueeze_20: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_75, 0);  clone_75 = None
        add_99: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_286, unsqueeze_20);  view_286 = unsqueeze_20 = None
        amax_10: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_99, [-1], True)
        sub_34: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_99, amax_10);  add_99 = None
        exp_10: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_11: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_28: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = None
        convert_element_type_174: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_28, torch.bfloat16);  div_28 = None
        expand_42: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_174, _shape_param_380);  convert_element_type_174 = _shape_param_380 = None
        view_289: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_42, _shape_param_381);  expand_42 = _shape_param_381 = None
        expand_43: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_80, _shape_param_382);  getitem_80 = _shape_param_382 = None
        clone_76: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_290: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_76, _shape_param_383);  clone_76 = _shape_param_383 = None
        constant_pad_nd_42: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_289, _shape_param_384);  _shape_param_384 = None
        constant_pad_nd_43: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_290, _shape_param_385);  _shape_param_385 = None
        bmm_21: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_42, constant_pad_nd_43);  constant_pad_nd_42 = constant_pad_nd_43 = None
        slice_33: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_21, 1, 0, -7);  bmm_21 = None
        view_291: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_33, _shape_param_386);  slice_33 = _shape_param_386 = None
        permute_110: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        clone_77: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_292: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_77, _shape_param_387);  clone_77 = _shape_param_387 = None
        convert_element_type_175: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg160_1, torch.bfloat16);  arg160_1 = None
        convert_element_type_176: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg161_1, torch.bfloat16);  arg161_1 = None
        view_293: "bf16[25088, 512]" = torch.ops.aten.view.default(view_292, _shape_param_388);  view_292 = _shape_param_388 = None
        permute_111: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_176, [1, 0]);  convert_element_type_176 = None
        addmm_41: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_175, view_293, permute_111);  convert_element_type_175 = None
        view_294: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_41, _shape_param_389);  addmm_41 = _shape_param_389 = None
        view_295: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_294, _shape_param_390);  view_294 = _shape_param_390 = None
        view_296: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_295, _shape_param_391);  view_295 = _shape_param_391 = None
        permute_112: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_296, [0, 1, 3, 2, 4, 5]);  view_296 = None
        clone_78: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        view_297: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_78, _shape_param_392);  clone_78 = _shape_param_392 = None
        inductor_lookup_seed_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 18)
        inductor_random_18: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_393, inductor_lookup_seed_18, 'rand');  _shape_param_393 = inductor_lookup_seed_18 = None
        lt_18: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_18, 0.9565217345952988);  inductor_random_18 = None
        convert_element_type_177: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_18, torch.bfloat16)
        div_29: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_177, 0.9565217345952988);  convert_element_type_177 = None
        mul_107: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_297, div_29);  view_297 = div_29 = None
        add_100: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_277, mul_107);  mul_107 = None
        view_298: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_100, _shape_param_394);  add_100 = _shape_param_394 = None
        convert_element_type_178: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_298, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_178, [2], correction = 0, keepdim = True)
        getitem_81: "f32[128, 196, 1]" = var_mean_24[0]
        getitem_82: "f32[128, 196, 1]" = var_mean_24[1];  var_mean_24 = None
        add_101: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_81, 1e-05);  getitem_81 = None
        rsqrt_24: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        sub_35: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_178, getitem_82);  convert_element_type_178 = None
        mul_108: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_24);  sub_35 = None
        mul_109: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_108, arg162_1);  mul_108 = arg162_1 = None
        add_102: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_109, arg163_1);  mul_109 = arg163_1 = None
        convert_element_type_179: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg164_1, torch.bfloat16);  arg164_1 = None
        convert_element_type_180: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg165_1, torch.bfloat16);  arg165_1 = None
        convert_element_type_181: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_102, torch.bfloat16);  add_102 = None
        view_299: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_181, _shape_param_395);  convert_element_type_181 = _shape_param_395 = None
        permute_113: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_180, [1, 0]);  convert_element_type_180 = None
        addmm_42: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_179, view_299, permute_113);  convert_element_type_179 = None
        view_300: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_42, _shape_param_396);  _shape_param_396 = None
        convert_element_type_182: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_300, torch.float32);  view_300 = None
        mul_110: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.5)
        mul_111: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.7071067811865476);  convert_element_type_182 = None
        erf_10: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_103: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_112: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_110, add_103);  mul_110 = add_103 = None
        convert_element_type_183: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None
        convert_element_type_184: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg166_1, torch.bfloat16);  arg166_1 = None
        convert_element_type_185: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg167_1, torch.bfloat16);  arg167_1 = None
        view_301: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_183, _shape_param_397);  convert_element_type_183 = _shape_param_397 = None
        permute_114: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_185, [1, 0]);  convert_element_type_185 = None
        addmm_43: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_184, view_301, permute_114);  convert_element_type_184 = None
        view_302: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_43, _shape_param_398);  addmm_43 = _shape_param_398 = None
        inductor_lookup_seed_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 19)
        inductor_random_19: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_399, inductor_lookup_seed_19, 'rand');  _shape_param_399 = inductor_lookup_seed_19 = None
        lt_19: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_19, 0.9565217345952988);  inductor_random_19 = None
        convert_element_type_186: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_19, torch.bfloat16)
        div_30: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_186, 0.9565217345952988);  convert_element_type_186 = None
        mul_113: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_302, div_30);  view_302 = div_30 = None
        add_104: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_298, mul_113);  mul_113 = None
        view_303: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_104, _shape_param_400);  add_104 = _shape_param_400 = None
        convert_element_type_187: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_303, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_187, [3], correction = 0, keepdim = True)
        getitem_83: "f32[128, 14, 14, 1]" = var_mean_25[0]
        getitem_84: "f32[128, 14, 14, 1]" = var_mean_25[1];  var_mean_25 = None
        add_105: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_83, 1e-05);  getitem_83 = None
        rsqrt_25: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
        sub_36: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_187, getitem_84);  convert_element_type_187 = None
        mul_114: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_25);  sub_36 = None
        mul_115: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_114, arg168_1);  mul_114 = arg168_1 = None
        add_106: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_115, arg169_1);  mul_115 = arg169_1 = None
        index_31: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_106, [None, fmod_4]);  add_106 = None
        index_32: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_31, [None, None, fmod_4]);  index_31 = None
        view_304: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_32, _shape_param_401);  index_32 = _shape_param_401 = None
        permute_115: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_304, [0, 1, 3, 2, 4, 5]);  view_304 = None
        clone_79: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_305: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_79, _shape_param_402);  clone_79 = _shape_param_402 = None
        view_306: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_305, _shape_param_403);  view_305 = _shape_param_403 = None
        convert_element_type_188: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg170_1, torch.bfloat16);  arg170_1 = None
        convert_element_type_189: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg171_1, torch.bfloat16);  arg171_1 = None
        convert_element_type_190: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_306, torch.bfloat16);  view_306 = None
        view_307: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_190, _shape_param_404);  convert_element_type_190 = _shape_param_404 = None
        permute_116: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_189, [1, 0]);  convert_element_type_189 = None
        addmm_44: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_188, view_307, permute_116);  convert_element_type_188 = None
        view_308: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_44, _shape_param_405);  addmm_44 = _shape_param_405 = None
        view_309: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_308, _shape_param_406);  view_308 = _shape_param_406 = None
        permute_117: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_309, [2, 0, 3, 1, 4]);  view_309 = None
        unbind_11 = torch.ops.aten.unbind.int(permute_117);  permute_117 = None
        getitem_85: "bf16[512, 16, 49, 32]" = unbind_11[0]
        getitem_86: "bf16[512, 16, 49, 32]" = unbind_11[1]
        getitem_87: "bf16[512, 16, 49, 32]" = unbind_11[2];  unbind_11 = None
        mul_116: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_85, 0.1767766952966369);  getitem_85 = None
        permute_118: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_86, [0, 1, 3, 2]);  getitem_86 = None
        expand_44: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_116, _shape_param_407);  mul_116 = _shape_param_407 = None
        clone_80: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_310: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_80, _shape_param_408);  clone_80 = _shape_param_408 = None
        expand_45: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_118, _shape_param_409);  permute_118 = _shape_param_409 = None
        clone_81: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_311: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_81, _shape_param_410);  clone_81 = _shape_param_410 = None
        constant_pad_nd_44: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_310, _shape_param_411);  _shape_param_411 = None
        constant_pad_nd_45: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_311, _shape_param_412);  _shape_param_412 = None
        bmm_22: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_44, constant_pad_nd_45);  constant_pad_nd_44 = constant_pad_nd_45 = None
        slice_34: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_22, 1, 0, -7)
        slice_35: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_34, 2, 0, -7);  slice_34 = None
        view_312: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_35, _shape_param_413);  slice_35 = _shape_param_413 = None
        view_313: "i64[2401]" = torch.ops.aten.view.default(arg172_1, [-1]);  arg172_1 = None
        index_33: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg173_1, [view_313]);  arg173_1 = view_313 = None
        view_314: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_33, _shape_param_414);  index_33 = _shape_param_414 = None
        permute_119: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_314, [2, 0, 1]);  view_314 = None
        clone_82: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_119, memory_format = torch.contiguous_format);  permute_119 = None
        unsqueeze_21: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_82, 0);  clone_82 = None
        add_107: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_312, unsqueeze_21);  view_312 = unsqueeze_21 = None
        view_315: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_107, _shape_param_415);  add_107 = _shape_param_415 = None
        unsqueeze_22: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg174_1, 1);  arg174_1 = None
        unsqueeze_23: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 0);  unsqueeze_22 = None
        add_108: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_315, unsqueeze_23);  view_315 = unsqueeze_23 = None
        view_316: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_108, _shape_param_416);  add_108 = _shape_param_416 = None
        amax_11: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_316, [-1], True)
        sub_37: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_316, amax_11);  view_316 = None
        exp_11: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_12: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_31: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = None
        convert_element_type_191: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_31, torch.bfloat16);  div_31 = None
        expand_46: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_191, _shape_param_417);  convert_element_type_191 = _shape_param_417 = None
        view_317: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_46, _shape_param_418);  expand_46 = _shape_param_418 = None
        expand_47: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_87, _shape_param_419);  getitem_87 = _shape_param_419 = None
        clone_83: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_318: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_83, _shape_param_420);  clone_83 = _shape_param_420 = None
        constant_pad_nd_46: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_317, _shape_param_421);  _shape_param_421 = None
        constant_pad_nd_47: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_318, _shape_param_422);  _shape_param_422 = None
        bmm_23: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_46, constant_pad_nd_47);  constant_pad_nd_46 = constant_pad_nd_47 = None
        slice_36: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_23, 1, 0, -7);  bmm_23 = None
        view_319: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_36, _shape_param_423);  slice_36 = _shape_param_423 = None
        permute_120: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None
        clone_84: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
        view_320: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_84, _shape_param_424);  clone_84 = _shape_param_424 = None
        convert_element_type_192: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg175_1, torch.bfloat16);  arg175_1 = None
        convert_element_type_193: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg176_1, torch.bfloat16);  arg176_1 = None
        view_321: "bf16[25088, 512]" = torch.ops.aten.view.default(view_320, _shape_param_425);  view_320 = _shape_param_425 = None
        permute_121: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_193, [1, 0]);  convert_element_type_193 = None
        addmm_45: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_192, view_321, permute_121);  convert_element_type_192 = None
        view_322: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_45, _shape_param_426);  addmm_45 = _shape_param_426 = None
        view_323: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_322, _shape_param_427);  view_322 = _shape_param_427 = None
        view_324: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_323, _shape_param_428);  view_323 = _shape_param_428 = None
        permute_122: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_324, [0, 1, 3, 2, 4, 5]);  view_324 = None
        clone_85: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_325: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_85, _shape_param_429);  clone_85 = _shape_param_429 = None
        index_34: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_325, [None, fmod_5]);  view_325 = None
        index_35: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_34, [None, None, fmod_5]);  index_34 = None
        inductor_lookup_seed_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 20)
        inductor_random_20: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_430, inductor_lookup_seed_20, 'rand');  _shape_param_430 = inductor_lookup_seed_20 = None
        lt_20: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_20, 0.9521739110350609);  inductor_random_20 = None
        convert_element_type_194: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_20, torch.bfloat16)
        div_32: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_194, 0.9521739110350609);  convert_element_type_194 = None
        mul_117: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_35, div_32);  index_35 = div_32 = None
        add_109: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_303, mul_117);  mul_117 = None
        view_326: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_109, _shape_param_431);  add_109 = _shape_param_431 = None
        convert_element_type_195: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_326, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_195, [2], correction = 0, keepdim = True)
        getitem_88: "f32[128, 196, 1]" = var_mean_26[0]
        getitem_89: "f32[128, 196, 1]" = var_mean_26[1];  var_mean_26 = None
        add_110: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_26: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        sub_38: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_195, getitem_89);  convert_element_type_195 = None
        mul_118: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_26);  sub_38 = None
        mul_119: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_118, arg177_1);  mul_118 = arg177_1 = None
        add_111: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_119, arg178_1);  mul_119 = arg178_1 = None
        convert_element_type_196: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg179_1, torch.bfloat16);  arg179_1 = None
        convert_element_type_197: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg180_1, torch.bfloat16);  arg180_1 = None
        convert_element_type_198: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_111, torch.bfloat16);  add_111 = None
        view_327: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_198, _shape_param_432);  convert_element_type_198 = _shape_param_432 = None
        permute_123: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_197, [1, 0]);  convert_element_type_197 = None
        addmm_46: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_196, view_327, permute_123);  convert_element_type_196 = None
        view_328: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_46, _shape_param_433);  _shape_param_433 = None
        convert_element_type_199: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_328, torch.float32);  view_328 = None
        mul_120: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_199, 0.5)
        mul_121: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_199, 0.7071067811865476);  convert_element_type_199 = None
        erf_11: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_121);  mul_121 = None
        add_112: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_122: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_120, add_112);  mul_120 = add_112 = None
        convert_element_type_200: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_122, torch.bfloat16);  mul_122 = None
        convert_element_type_201: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg181_1, torch.bfloat16);  arg181_1 = None
        convert_element_type_202: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg182_1, torch.bfloat16);  arg182_1 = None
        view_329: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_200, _shape_param_434);  convert_element_type_200 = _shape_param_434 = None
        permute_124: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_202, [1, 0]);  convert_element_type_202 = None
        addmm_47: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_201, view_329, permute_124);  convert_element_type_201 = None
        view_330: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_47, _shape_param_435);  addmm_47 = _shape_param_435 = None
        inductor_lookup_seed_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 21)
        inductor_random_21: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_436, inductor_lookup_seed_21, 'rand');  _shape_param_436 = inductor_lookup_seed_21 = None
        lt_21: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_21, 0.9521739110350609);  inductor_random_21 = None
        convert_element_type_203: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_21, torch.bfloat16)
        div_33: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_203, 0.9521739110350609);  convert_element_type_203 = None
        mul_123: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_330, div_33);  view_330 = div_33 = None
        add_113: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_326, mul_123);  mul_123 = None
        view_331: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_113, _shape_param_437);  add_113 = _shape_param_437 = None
        convert_element_type_204: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_331, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_204, [3], correction = 0, keepdim = True)
        getitem_90: "f32[128, 14, 14, 1]" = var_mean_27[0]
        getitem_91: "f32[128, 14, 14, 1]" = var_mean_27[1];  var_mean_27 = None
        add_114: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_27: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        sub_39: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_204, getitem_91);  convert_element_type_204 = None
        mul_124: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_27);  sub_39 = None
        mul_125: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_124, arg183_1);  mul_124 = arg183_1 = None
        add_115: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_125, arg184_1);  mul_125 = arg184_1 = None
        view_332: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_115, _shape_param_438);  add_115 = _shape_param_438 = None
        permute_125: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_332, [0, 1, 3, 2, 4, 5]);  view_332 = None
        clone_86: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_333: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_86, _shape_param_439);  clone_86 = _shape_param_439 = None
        view_334: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_333, _shape_param_440);  view_333 = _shape_param_440 = None
        convert_element_type_205: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg185_1, torch.bfloat16);  arg185_1 = None
        convert_element_type_206: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg186_1, torch.bfloat16);  arg186_1 = None
        convert_element_type_207: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_334, torch.bfloat16);  view_334 = None
        view_335: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_207, _shape_param_441);  convert_element_type_207 = _shape_param_441 = None
        permute_126: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_206, [1, 0]);  convert_element_type_206 = None
        addmm_48: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_205, view_335, permute_126);  convert_element_type_205 = None
        view_336: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_48, _shape_param_442);  addmm_48 = _shape_param_442 = None
        view_337: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_336, _shape_param_443);  view_336 = _shape_param_443 = None
        permute_127: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_337, [2, 0, 3, 1, 4]);  view_337 = None
        unbind_12 = torch.ops.aten.unbind.int(permute_127);  permute_127 = None
        getitem_92: "bf16[512, 16, 49, 32]" = unbind_12[0]
        getitem_93: "bf16[512, 16, 49, 32]" = unbind_12[1]
        getitem_94: "bf16[512, 16, 49, 32]" = unbind_12[2];  unbind_12 = None
        mul_126: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_92, 0.1767766952966369);  getitem_92 = None
        permute_128: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_93, [0, 1, 3, 2]);  getitem_93 = None
        expand_48: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_126, _shape_param_444);  mul_126 = _shape_param_444 = None
        clone_87: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_338: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_87, _shape_param_445);  clone_87 = _shape_param_445 = None
        expand_49: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_128, _shape_param_446);  permute_128 = _shape_param_446 = None
        clone_88: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_339: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_88, _shape_param_447);  clone_88 = _shape_param_447 = None
        constant_pad_nd_48: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_338, _shape_param_448);  _shape_param_448 = None
        constant_pad_nd_49: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_339, _shape_param_449);  _shape_param_449 = None
        bmm_24: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_48, constant_pad_nd_49);  constant_pad_nd_48 = constant_pad_nd_49 = None
        slice_37: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_24, 1, 0, -7)
        slice_38: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_37, 2, 0, -7);  slice_37 = None
        view_340: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_38, _shape_param_450);  slice_38 = _shape_param_450 = None
        view_341: "i64[2401]" = torch.ops.aten.view.default(arg187_1, [-1]);  arg187_1 = None
        index_36: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg188_1, [view_341]);  arg188_1 = view_341 = None
        view_342: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_36, _shape_param_451);  index_36 = _shape_param_451 = None
        permute_129: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_342, [2, 0, 1]);  view_342 = None
        clone_89: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        unsqueeze_24: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_89, 0);  clone_89 = None
        add_116: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_340, unsqueeze_24);  view_340 = unsqueeze_24 = None
        amax_12: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_116, [-1], True)
        sub_40: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_116, amax_12);  add_116 = None
        exp_12: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        sum_13: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_34: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = None
        convert_element_type_208: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_34, torch.bfloat16);  div_34 = None
        expand_50: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_208, _shape_param_452);  convert_element_type_208 = _shape_param_452 = None
        view_343: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_50, _shape_param_453);  expand_50 = _shape_param_453 = None
        expand_51: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_94, _shape_param_454);  getitem_94 = _shape_param_454 = None
        clone_90: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_344: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_90, _shape_param_455);  clone_90 = _shape_param_455 = None
        constant_pad_nd_50: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_343, _shape_param_456);  _shape_param_456 = None
        constant_pad_nd_51: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_344, _shape_param_457);  _shape_param_457 = None
        bmm_25: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_50, constant_pad_nd_51);  constant_pad_nd_50 = constant_pad_nd_51 = None
        slice_39: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_25, 1, 0, -7);  bmm_25 = None
        view_345: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_39, _shape_param_458);  slice_39 = _shape_param_458 = None
        permute_130: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
        clone_91: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
        view_346: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_91, _shape_param_459);  clone_91 = _shape_param_459 = None
        convert_element_type_209: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg189_1, torch.bfloat16);  arg189_1 = None
        convert_element_type_210: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg190_1, torch.bfloat16);  arg190_1 = None
        view_347: "bf16[25088, 512]" = torch.ops.aten.view.default(view_346, _shape_param_460);  view_346 = _shape_param_460 = None
        permute_131: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_210, [1, 0]);  convert_element_type_210 = None
        addmm_49: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_209, view_347, permute_131);  convert_element_type_209 = None
        view_348: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_49, _shape_param_461);  addmm_49 = _shape_param_461 = None
        view_349: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_348, _shape_param_462);  view_348 = _shape_param_462 = None
        view_350: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_349, _shape_param_463);  view_349 = _shape_param_463 = None
        permute_132: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_350, [0, 1, 3, 2, 4, 5]);  view_350 = None
        clone_92: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
        view_351: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_92, _shape_param_464);  clone_92 = _shape_param_464 = None
        inductor_lookup_seed_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 22)
        inductor_random_22: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_465, inductor_lookup_seed_22, 'rand');  _shape_param_465 = inductor_lookup_seed_22 = None
        lt_22: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_22, 0.947826087474823);  inductor_random_22 = None
        convert_element_type_211: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_22, torch.bfloat16)
        div_35: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_211, 0.947826087474823);  convert_element_type_211 = None
        mul_127: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_351, div_35);  view_351 = div_35 = None
        add_117: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_331, mul_127);  mul_127 = None
        view_352: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_117, _shape_param_466);  add_117 = _shape_param_466 = None
        convert_element_type_212: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_352, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_212, [2], correction = 0, keepdim = True)
        getitem_95: "f32[128, 196, 1]" = var_mean_28[0]
        getitem_96: "f32[128, 196, 1]" = var_mean_28[1];  var_mean_28 = None
        add_118: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_95, 1e-05);  getitem_95 = None
        rsqrt_28: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_41: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_212, getitem_96);  convert_element_type_212 = None
        mul_128: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_28);  sub_41 = None
        mul_129: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_128, arg191_1);  mul_128 = arg191_1 = None
        add_119: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_129, arg192_1);  mul_129 = arg192_1 = None
        convert_element_type_213: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg193_1, torch.bfloat16);  arg193_1 = None
        convert_element_type_214: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg194_1, torch.bfloat16);  arg194_1 = None
        convert_element_type_215: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None
        view_353: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_215, _shape_param_467);  convert_element_type_215 = _shape_param_467 = None
        permute_133: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_214, [1, 0]);  convert_element_type_214 = None
        addmm_50: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_213, view_353, permute_133);  convert_element_type_213 = None
        view_354: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_50, _shape_param_468);  _shape_param_468 = None
        convert_element_type_216: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_354, torch.float32);  view_354 = None
        mul_130: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_216, 0.5)
        mul_131: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_216, 0.7071067811865476);  convert_element_type_216 = None
        erf_12: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
        add_120: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_132: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_130, add_120);  mul_130 = add_120 = None
        convert_element_type_217: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_132, torch.bfloat16);  mul_132 = None
        convert_element_type_218: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg195_1, torch.bfloat16);  arg195_1 = None
        convert_element_type_219: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg196_1, torch.bfloat16);  arg196_1 = None
        view_355: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_217, _shape_param_469);  convert_element_type_217 = _shape_param_469 = None
        permute_134: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_219, [1, 0]);  convert_element_type_219 = None
        addmm_51: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_218, view_355, permute_134);  convert_element_type_218 = None
        view_356: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_51, _shape_param_470);  addmm_51 = _shape_param_470 = None
        inductor_lookup_seed_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 23)
        inductor_random_23: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_471, inductor_lookup_seed_23, 'rand');  _shape_param_471 = inductor_lookup_seed_23 = None
        lt_23: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_23, 0.947826087474823);  inductor_random_23 = None
        convert_element_type_220: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_23, torch.bfloat16)
        div_36: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_220, 0.947826087474823);  convert_element_type_220 = None
        mul_133: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_356, div_36);  view_356 = div_36 = None
        add_121: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_352, mul_133);  mul_133 = None
        view_357: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_121, _shape_param_472);  add_121 = _shape_param_472 = None
        convert_element_type_221: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_357, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_221, [3], correction = 0, keepdim = True)
        getitem_97: "f32[128, 14, 14, 1]" = var_mean_29[0]
        getitem_98: "f32[128, 14, 14, 1]" = var_mean_29[1];  var_mean_29 = None
        add_122: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-05);  getitem_97 = None
        rsqrt_29: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_42: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_221, getitem_98);  convert_element_type_221 = None
        mul_134: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_29);  sub_42 = None
        mul_135: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_134, arg197_1);  mul_134 = arg197_1 = None
        add_123: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_135, arg198_1);  mul_135 = arg198_1 = None
        index_37: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_123, [None, fmod_4]);  add_123 = None
        index_38: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_37, [None, None, fmod_4]);  index_37 = None
        view_358: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_38, _shape_param_473);  index_38 = _shape_param_473 = None
        permute_135: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_358, [0, 1, 3, 2, 4, 5]);  view_358 = None
        clone_93: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_359: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_93, _shape_param_474);  clone_93 = _shape_param_474 = None
        view_360: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_359, _shape_param_475);  view_359 = _shape_param_475 = None
        convert_element_type_222: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg199_1, torch.bfloat16);  arg199_1 = None
        convert_element_type_223: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg200_1, torch.bfloat16);  arg200_1 = None
        convert_element_type_224: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_360, torch.bfloat16);  view_360 = None
        view_361: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_224, _shape_param_476);  convert_element_type_224 = _shape_param_476 = None
        permute_136: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_223, [1, 0]);  convert_element_type_223 = None
        addmm_52: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_222, view_361, permute_136);  convert_element_type_222 = None
        view_362: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_52, _shape_param_477);  addmm_52 = _shape_param_477 = None
        view_363: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_362, _shape_param_478);  view_362 = _shape_param_478 = None
        permute_137: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_363, [2, 0, 3, 1, 4]);  view_363 = None
        unbind_13 = torch.ops.aten.unbind.int(permute_137);  permute_137 = None
        getitem_99: "bf16[512, 16, 49, 32]" = unbind_13[0]
        getitem_100: "bf16[512, 16, 49, 32]" = unbind_13[1]
        getitem_101: "bf16[512, 16, 49, 32]" = unbind_13[2];  unbind_13 = None
        mul_136: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_99, 0.1767766952966369);  getitem_99 = None
        permute_138: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_100, [0, 1, 3, 2]);  getitem_100 = None
        expand_52: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_136, _shape_param_479);  mul_136 = _shape_param_479 = None
        clone_94: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_364: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_94, _shape_param_480);  clone_94 = _shape_param_480 = None
        expand_53: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_138, _shape_param_481);  permute_138 = _shape_param_481 = None
        clone_95: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_365: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_95, _shape_param_482);  clone_95 = _shape_param_482 = None
        constant_pad_nd_52: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_364, _shape_param_483);  _shape_param_483 = None
        constant_pad_nd_53: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_365, _shape_param_484);  _shape_param_484 = None
        bmm_26: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_52, constant_pad_nd_53);  constant_pad_nd_52 = constant_pad_nd_53 = None
        slice_40: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_26, 1, 0, -7)
        slice_41: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_40, 2, 0, -7);  slice_40 = None
        view_366: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_41, _shape_param_485);  slice_41 = _shape_param_485 = None
        view_367: "i64[2401]" = torch.ops.aten.view.default(arg201_1, [-1]);  arg201_1 = None
        index_39: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg202_1, [view_367]);  arg202_1 = view_367 = None
        view_368: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_39, _shape_param_486);  index_39 = _shape_param_486 = None
        permute_139: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_368, [2, 0, 1]);  view_368 = None
        clone_96: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        unsqueeze_25: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_96, 0);  clone_96 = None
        add_124: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_366, unsqueeze_25);  view_366 = unsqueeze_25 = None
        view_369: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_124, _shape_param_487);  add_124 = _shape_param_487 = None
        unsqueeze_26: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg203_1, 1);  arg203_1 = None
        unsqueeze_27: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 0);  unsqueeze_26 = None
        add_125: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_369, unsqueeze_27);  view_369 = unsqueeze_27 = None
        view_370: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_125, _shape_param_488);  add_125 = _shape_param_488 = None
        amax_13: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_370, [-1], True)
        sub_43: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_370, amax_13);  view_370 = None
        exp_13: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_14: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_37: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = None
        convert_element_type_225: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_37, torch.bfloat16);  div_37 = None
        expand_54: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_225, _shape_param_489);  convert_element_type_225 = _shape_param_489 = None
        view_371: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_54, _shape_param_490);  expand_54 = _shape_param_490 = None
        expand_55: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_101, _shape_param_491);  getitem_101 = _shape_param_491 = None
        clone_97: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_372: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_97, _shape_param_492);  clone_97 = _shape_param_492 = None
        constant_pad_nd_54: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_371, _shape_param_493);  _shape_param_493 = None
        constant_pad_nd_55: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_372, _shape_param_494);  _shape_param_494 = None
        bmm_27: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_54, constant_pad_nd_55);  constant_pad_nd_54 = constant_pad_nd_55 = None
        slice_42: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_27, 1, 0, -7);  bmm_27 = None
        view_373: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_42, _shape_param_495);  slice_42 = _shape_param_495 = None
        permute_140: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
        clone_98: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_374: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_98, _shape_param_496);  clone_98 = _shape_param_496 = None
        convert_element_type_226: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg204_1, torch.bfloat16);  arg204_1 = None
        convert_element_type_227: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg205_1, torch.bfloat16);  arg205_1 = None
        view_375: "bf16[25088, 512]" = torch.ops.aten.view.default(view_374, _shape_param_497);  view_374 = _shape_param_497 = None
        permute_141: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_227, [1, 0]);  convert_element_type_227 = None
        addmm_53: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_226, view_375, permute_141);  convert_element_type_226 = None
        view_376: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_53, _shape_param_498);  addmm_53 = _shape_param_498 = None
        view_377: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_376, _shape_param_499);  view_376 = _shape_param_499 = None
        view_378: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_377, _shape_param_500);  view_377 = _shape_param_500 = None
        permute_142: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_378, [0, 1, 3, 2, 4, 5]);  view_378 = None
        clone_99: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
        view_379: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_99, _shape_param_501);  clone_99 = _shape_param_501 = None
        index_40: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_379, [None, fmod_5]);  view_379 = None
        index_41: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_40, [None, None, fmod_5]);  index_40 = None
        inductor_lookup_seed_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 24)
        inductor_random_24: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_502, inductor_lookup_seed_24, 'rand');  _shape_param_502 = inductor_lookup_seed_24 = None
        lt_24: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_24, 0.9434782639145851);  inductor_random_24 = None
        convert_element_type_228: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_24, torch.bfloat16)
        div_38: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_228, 0.9434782639145851);  convert_element_type_228 = None
        mul_137: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_41, div_38);  index_41 = div_38 = None
        add_126: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_357, mul_137);  mul_137 = None
        view_380: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_126, _shape_param_503);  add_126 = _shape_param_503 = None
        convert_element_type_229: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_380, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_229, [2], correction = 0, keepdim = True)
        getitem_102: "f32[128, 196, 1]" = var_mean_30[0]
        getitem_103: "f32[128, 196, 1]" = var_mean_30[1];  var_mean_30 = None
        add_127: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
        rsqrt_30: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        sub_44: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_229, getitem_103);  convert_element_type_229 = None
        mul_138: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_30);  sub_44 = None
        mul_139: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_138, arg206_1);  mul_138 = arg206_1 = None
        add_128: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_139, arg207_1);  mul_139 = arg207_1 = None
        convert_element_type_230: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg208_1, torch.bfloat16);  arg208_1 = None
        convert_element_type_231: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg209_1, torch.bfloat16);  arg209_1 = None
        convert_element_type_232: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_128, torch.bfloat16);  add_128 = None
        view_381: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_232, _shape_param_504);  convert_element_type_232 = _shape_param_504 = None
        permute_143: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_231, [1, 0]);  convert_element_type_231 = None
        addmm_54: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_230, view_381, permute_143);  convert_element_type_230 = None
        view_382: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_54, _shape_param_505);  _shape_param_505 = None
        convert_element_type_233: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_382, torch.float32);  view_382 = None
        mul_140: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_233, 0.5)
        mul_141: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_233, 0.7071067811865476);  convert_element_type_233 = None
        erf_13: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
        add_129: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_142: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_140, add_129);  mul_140 = add_129 = None
        convert_element_type_234: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_142, torch.bfloat16);  mul_142 = None
        convert_element_type_235: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg210_1, torch.bfloat16);  arg210_1 = None
        convert_element_type_236: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg211_1, torch.bfloat16);  arg211_1 = None
        view_383: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_234, _shape_param_506);  convert_element_type_234 = _shape_param_506 = None
        permute_144: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_236, [1, 0]);  convert_element_type_236 = None
        addmm_55: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_235, view_383, permute_144);  convert_element_type_235 = None
        view_384: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_55, _shape_param_507);  addmm_55 = _shape_param_507 = None
        inductor_lookup_seed_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 25)
        inductor_random_25: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_508, inductor_lookup_seed_25, 'rand');  _shape_param_508 = inductor_lookup_seed_25 = None
        lt_25: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_25, 0.9434782639145851);  inductor_random_25 = None
        convert_element_type_237: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_25, torch.bfloat16)
        div_39: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_237, 0.9434782639145851);  convert_element_type_237 = None
        mul_143: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_384, div_39);  view_384 = div_39 = None
        add_130: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_380, mul_143);  mul_143 = None
        view_385: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_130, _shape_param_509);  add_130 = _shape_param_509 = None
        convert_element_type_238: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_385, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_238, [3], correction = 0, keepdim = True)
        getitem_104: "f32[128, 14, 14, 1]" = var_mean_31[0]
        getitem_105: "f32[128, 14, 14, 1]" = var_mean_31[1];  var_mean_31 = None
        add_131: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
        rsqrt_31: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        sub_45: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_238, getitem_105);  convert_element_type_238 = None
        mul_144: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_31);  sub_45 = None
        mul_145: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_144, arg212_1);  mul_144 = arg212_1 = None
        add_132: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_145, arg213_1);  mul_145 = arg213_1 = None
        view_386: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_132, _shape_param_510);  add_132 = _shape_param_510 = None
        permute_145: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_386, [0, 1, 3, 2, 4, 5]);  view_386 = None
        clone_100: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
        view_387: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_100, _shape_param_511);  clone_100 = _shape_param_511 = None
        view_388: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_387, _shape_param_512);  view_387 = _shape_param_512 = None
        convert_element_type_239: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg214_1, torch.bfloat16);  arg214_1 = None
        convert_element_type_240: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg215_1, torch.bfloat16);  arg215_1 = None
        convert_element_type_241: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_388, torch.bfloat16);  view_388 = None
        view_389: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_241, _shape_param_513);  convert_element_type_241 = _shape_param_513 = None
        permute_146: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_240, [1, 0]);  convert_element_type_240 = None
        addmm_56: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_239, view_389, permute_146);  convert_element_type_239 = None
        view_390: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_56, _shape_param_514);  addmm_56 = _shape_param_514 = None
        view_391: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_390, _shape_param_515);  view_390 = _shape_param_515 = None
        permute_147: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_391, [2, 0, 3, 1, 4]);  view_391 = None
        unbind_14 = torch.ops.aten.unbind.int(permute_147);  permute_147 = None
        getitem_106: "bf16[512, 16, 49, 32]" = unbind_14[0]
        getitem_107: "bf16[512, 16, 49, 32]" = unbind_14[1]
        getitem_108: "bf16[512, 16, 49, 32]" = unbind_14[2];  unbind_14 = None
        mul_146: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_106, 0.1767766952966369);  getitem_106 = None
        permute_148: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_107, [0, 1, 3, 2]);  getitem_107 = None
        expand_56: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_146, _shape_param_516);  mul_146 = _shape_param_516 = None
        clone_101: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_392: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_101, _shape_param_517);  clone_101 = _shape_param_517 = None
        expand_57: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_148, _shape_param_518);  permute_148 = _shape_param_518 = None
        clone_102: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_393: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_102, _shape_param_519);  clone_102 = _shape_param_519 = None
        constant_pad_nd_56: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_392, _shape_param_520);  _shape_param_520 = None
        constant_pad_nd_57: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_393, _shape_param_521);  _shape_param_521 = None
        bmm_28: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_56, constant_pad_nd_57);  constant_pad_nd_56 = constant_pad_nd_57 = None
        slice_43: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_28, 1, 0, -7)
        slice_44: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_43, 2, 0, -7);  slice_43 = None
        view_394: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_44, _shape_param_522);  slice_44 = _shape_param_522 = None
        view_395: "i64[2401]" = torch.ops.aten.view.default(arg216_1, [-1]);  arg216_1 = None
        index_42: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg217_1, [view_395]);  arg217_1 = view_395 = None
        view_396: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_42, _shape_param_523);  index_42 = _shape_param_523 = None
        permute_149: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_396, [2, 0, 1]);  view_396 = None
        clone_103: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
        unsqueeze_28: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_103, 0);  clone_103 = None
        add_133: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_394, unsqueeze_28);  view_394 = unsqueeze_28 = None
        amax_14: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_133, [-1], True)
        sub_46: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_133, amax_14);  add_133 = None
        exp_14: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_15: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_40: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = None
        convert_element_type_242: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_40, torch.bfloat16);  div_40 = None
        expand_58: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_242, _shape_param_524);  convert_element_type_242 = _shape_param_524 = None
        view_397: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_58, _shape_param_525);  expand_58 = _shape_param_525 = None
        expand_59: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_108, _shape_param_526);  getitem_108 = _shape_param_526 = None
        clone_104: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_398: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_104, _shape_param_527);  clone_104 = _shape_param_527 = None
        constant_pad_nd_58: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_397, _shape_param_528);  _shape_param_528 = None
        constant_pad_nd_59: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_398, _shape_param_529);  _shape_param_529 = None
        bmm_29: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_58, constant_pad_nd_59);  constant_pad_nd_58 = constant_pad_nd_59 = None
        slice_45: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_29, 1, 0, -7);  bmm_29 = None
        view_399: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_45, _shape_param_530);  slice_45 = _shape_param_530 = None
        permute_150: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
        clone_105: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_400: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_105, _shape_param_531);  clone_105 = _shape_param_531 = None
        convert_element_type_243: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg218_1, torch.bfloat16);  arg218_1 = None
        convert_element_type_244: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg219_1, torch.bfloat16);  arg219_1 = None
        view_401: "bf16[25088, 512]" = torch.ops.aten.view.default(view_400, _shape_param_532);  view_400 = _shape_param_532 = None
        permute_151: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_244, [1, 0]);  convert_element_type_244 = None
        addmm_57: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_243, view_401, permute_151);  convert_element_type_243 = None
        view_402: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_57, _shape_param_533);  addmm_57 = _shape_param_533 = None
        view_403: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_402, _shape_param_534);  view_402 = _shape_param_534 = None
        view_404: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_403, _shape_param_535);  view_403 = _shape_param_535 = None
        permute_152: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_404, [0, 1, 3, 2, 4, 5]);  view_404 = None
        clone_106: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format);  permute_152 = None
        view_405: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_106, _shape_param_536);  clone_106 = _shape_param_536 = None
        inductor_lookup_seed_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 26)
        inductor_random_26: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_537, inductor_lookup_seed_26, 'rand');  _shape_param_537 = inductor_lookup_seed_26 = None
        lt_26: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_26, 0.9391304366290569);  inductor_random_26 = None
        convert_element_type_245: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_26, torch.bfloat16)
        div_41: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_245, 0.9391304366290569);  convert_element_type_245 = None
        mul_147: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_405, div_41);  view_405 = div_41 = None
        add_134: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_385, mul_147);  mul_147 = None
        view_406: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_134, _shape_param_538);  add_134 = _shape_param_538 = None
        convert_element_type_246: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_406, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_246, [2], correction = 0, keepdim = True)
        getitem_109: "f32[128, 196, 1]" = var_mean_32[0]
        getitem_110: "f32[128, 196, 1]" = var_mean_32[1];  var_mean_32 = None
        add_135: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_109, 1e-05);  getitem_109 = None
        rsqrt_32: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
        sub_47: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_246, getitem_110);  convert_element_type_246 = None
        mul_148: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_32);  sub_47 = None
        mul_149: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_148, arg220_1);  mul_148 = arg220_1 = None
        add_136: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_149, arg221_1);  mul_149 = arg221_1 = None
        convert_element_type_247: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg222_1, torch.bfloat16);  arg222_1 = None
        convert_element_type_248: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg223_1, torch.bfloat16);  arg223_1 = None
        convert_element_type_249: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_136, torch.bfloat16);  add_136 = None
        view_407: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_249, _shape_param_539);  convert_element_type_249 = _shape_param_539 = None
        permute_153: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_248, [1, 0]);  convert_element_type_248 = None
        addmm_58: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_247, view_407, permute_153);  convert_element_type_247 = None
        view_408: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_58, _shape_param_540);  _shape_param_540 = None
        convert_element_type_250: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_408, torch.float32);  view_408 = None
        mul_150: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_250, 0.5)
        mul_151: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_250, 0.7071067811865476);  convert_element_type_250 = None
        erf_14: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_151);  mul_151 = None
        add_137: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_152: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_150, add_137);  mul_150 = add_137 = None
        convert_element_type_251: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None
        convert_element_type_252: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg224_1, torch.bfloat16);  arg224_1 = None
        convert_element_type_253: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg225_1, torch.bfloat16);  arg225_1 = None
        view_409: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_251, _shape_param_541);  convert_element_type_251 = _shape_param_541 = None
        permute_154: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_253, [1, 0]);  convert_element_type_253 = None
        addmm_59: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_252, view_409, permute_154);  convert_element_type_252 = None
        view_410: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_59, _shape_param_542);  addmm_59 = _shape_param_542 = None
        inductor_lookup_seed_27: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 27)
        inductor_random_27: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_543, inductor_lookup_seed_27, 'rand');  _shape_param_543 = inductor_lookup_seed_27 = None
        lt_27: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_27, 0.9391304366290569);  inductor_random_27 = None
        convert_element_type_254: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_27, torch.bfloat16)
        div_42: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_254, 0.9391304366290569);  convert_element_type_254 = None
        mul_153: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_410, div_42);  view_410 = div_42 = None
        add_138: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_406, mul_153);  mul_153 = None
        view_411: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_138, _shape_param_544);  add_138 = _shape_param_544 = None
        convert_element_type_255: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_411, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_255, [3], correction = 0, keepdim = True)
        getitem_111: "f32[128, 14, 14, 1]" = var_mean_33[0]
        getitem_112: "f32[128, 14, 14, 1]" = var_mean_33[1];  var_mean_33 = None
        add_139: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_111, 1e-05);  getitem_111 = None
        rsqrt_33: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        sub_48: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_255, getitem_112);  convert_element_type_255 = None
        mul_154: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_33);  sub_48 = None
        mul_155: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_154, arg226_1);  mul_154 = arg226_1 = None
        add_140: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_155, arg227_1);  mul_155 = arg227_1 = None
        index_43: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_140, [None, fmod_4]);  add_140 = None
        index_44: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_43, [None, None, fmod_4]);  index_43 = None
        view_412: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_44, _shape_param_545);  index_44 = _shape_param_545 = None
        permute_155: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_412, [0, 1, 3, 2, 4, 5]);  view_412 = None
        clone_107: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_413: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_107, _shape_param_546);  clone_107 = _shape_param_546 = None
        view_414: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_413, _shape_param_547);  view_413 = _shape_param_547 = None
        convert_element_type_256: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg228_1, torch.bfloat16);  arg228_1 = None
        convert_element_type_257: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg229_1, torch.bfloat16);  arg229_1 = None
        convert_element_type_258: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_414, torch.bfloat16);  view_414 = None
        view_415: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_258, _shape_param_548);  convert_element_type_258 = _shape_param_548 = None
        permute_156: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_257, [1, 0]);  convert_element_type_257 = None
        addmm_60: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_256, view_415, permute_156);  convert_element_type_256 = None
        view_416: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_60, _shape_param_549);  addmm_60 = _shape_param_549 = None
        view_417: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_416, _shape_param_550);  view_416 = _shape_param_550 = None
        permute_157: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_417, [2, 0, 3, 1, 4]);  view_417 = None
        unbind_15 = torch.ops.aten.unbind.int(permute_157);  permute_157 = None
        getitem_113: "bf16[512, 16, 49, 32]" = unbind_15[0]
        getitem_114: "bf16[512, 16, 49, 32]" = unbind_15[1]
        getitem_115: "bf16[512, 16, 49, 32]" = unbind_15[2];  unbind_15 = None
        mul_156: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_113, 0.1767766952966369);  getitem_113 = None
        permute_158: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_114, [0, 1, 3, 2]);  getitem_114 = None
        expand_60: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_156, _shape_param_551);  mul_156 = _shape_param_551 = None
        clone_108: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_418: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_108, _shape_param_552);  clone_108 = _shape_param_552 = None
        expand_61: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_158, _shape_param_553);  permute_158 = _shape_param_553 = None
        clone_109: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_419: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_109, _shape_param_554);  clone_109 = _shape_param_554 = None
        constant_pad_nd_60: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_418, _shape_param_555);  _shape_param_555 = None
        constant_pad_nd_61: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_419, _shape_param_556);  _shape_param_556 = None
        bmm_30: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_60, constant_pad_nd_61);  constant_pad_nd_60 = constant_pad_nd_61 = None
        slice_46: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_30, 1, 0, -7)
        slice_47: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_46, 2, 0, -7);  slice_46 = None
        view_420: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_47, _shape_param_557);  slice_47 = _shape_param_557 = None
        view_421: "i64[2401]" = torch.ops.aten.view.default(arg230_1, [-1]);  arg230_1 = None
        index_45: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg231_1, [view_421]);  arg231_1 = view_421 = None
        view_422: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_45, _shape_param_558);  index_45 = _shape_param_558 = None
        permute_159: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_422, [2, 0, 1]);  view_422 = None
        clone_110: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        unsqueeze_29: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_110, 0);  clone_110 = None
        add_141: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_420, unsqueeze_29);  view_420 = unsqueeze_29 = None
        view_423: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_141, _shape_param_559);  add_141 = _shape_param_559 = None
        unsqueeze_30: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg232_1, 1);  arg232_1 = None
        unsqueeze_31: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 0);  unsqueeze_30 = None
        add_142: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_423, unsqueeze_31);  view_423 = unsqueeze_31 = None
        view_424: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_142, _shape_param_560);  add_142 = _shape_param_560 = None
        amax_15: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_424, [-1], True)
        sub_49: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_424, amax_15);  view_424 = None
        exp_15: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_16: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_43: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = None
        convert_element_type_259: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_43, torch.bfloat16);  div_43 = None
        expand_62: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_259, _shape_param_561);  convert_element_type_259 = _shape_param_561 = None
        view_425: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_62, _shape_param_562);  expand_62 = _shape_param_562 = None
        expand_63: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_115, _shape_param_563);  getitem_115 = _shape_param_563 = None
        clone_111: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_426: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_111, _shape_param_564);  clone_111 = _shape_param_564 = None
        constant_pad_nd_62: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_425, _shape_param_565);  _shape_param_565 = None
        constant_pad_nd_63: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_426, _shape_param_566);  _shape_param_566 = None
        bmm_31: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_62, constant_pad_nd_63);  constant_pad_nd_62 = constant_pad_nd_63 = None
        slice_48: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_31, 1, 0, -7);  bmm_31 = None
        view_427: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_48, _shape_param_567);  slice_48 = _shape_param_567 = None
        permute_160: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
        clone_112: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        view_428: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_112, _shape_param_568);  clone_112 = _shape_param_568 = None
        convert_element_type_260: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg233_1, torch.bfloat16);  arg233_1 = None
        convert_element_type_261: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg234_1, torch.bfloat16);  arg234_1 = None
        view_429: "bf16[25088, 512]" = torch.ops.aten.view.default(view_428, _shape_param_569);  view_428 = _shape_param_569 = None
        permute_161: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_261, [1, 0]);  convert_element_type_261 = None
        addmm_61: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_260, view_429, permute_161);  convert_element_type_260 = None
        view_430: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_61, _shape_param_570);  addmm_61 = _shape_param_570 = None
        view_431: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_430, _shape_param_571);  view_430 = _shape_param_571 = None
        view_432: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_431, _shape_param_572);  view_431 = _shape_param_572 = None
        permute_162: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_432, [0, 1, 3, 2, 4, 5]);  view_432 = None
        clone_113: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
        view_433: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_113, _shape_param_573);  clone_113 = _shape_param_573 = None
        index_46: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_433, [None, fmod_5]);  view_433 = None
        index_47: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_46, [None, None, fmod_5]);  index_46 = None
        inductor_lookup_seed_28: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 28)
        inductor_random_28: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_574, inductor_lookup_seed_28, 'rand');  _shape_param_574 = inductor_lookup_seed_28 = None
        lt_28: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_28, 0.9347826093435287);  inductor_random_28 = None
        convert_element_type_262: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_28, torch.bfloat16)
        div_44: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_262, 0.9347826093435287);  convert_element_type_262 = None
        mul_157: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_47, div_44);  index_47 = div_44 = None
        add_143: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_411, mul_157);  mul_157 = None
        view_434: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_143, _shape_param_575);  add_143 = _shape_param_575 = None
        convert_element_type_263: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_434, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_263, [2], correction = 0, keepdim = True)
        getitem_116: "f32[128, 196, 1]" = var_mean_34[0]
        getitem_117: "f32[128, 196, 1]" = var_mean_34[1];  var_mean_34 = None
        add_144: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
        rsqrt_34: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        sub_50: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_263, getitem_117);  convert_element_type_263 = None
        mul_158: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_34);  sub_50 = None
        mul_159: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_158, arg235_1);  mul_158 = arg235_1 = None
        add_145: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_159, arg236_1);  mul_159 = arg236_1 = None
        convert_element_type_264: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg237_1, torch.bfloat16);  arg237_1 = None
        convert_element_type_265: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg238_1, torch.bfloat16);  arg238_1 = None
        convert_element_type_266: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_145, torch.bfloat16);  add_145 = None
        view_435: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_266, _shape_param_576);  convert_element_type_266 = _shape_param_576 = None
        permute_163: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_265, [1, 0]);  convert_element_type_265 = None
        addmm_62: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_264, view_435, permute_163);  convert_element_type_264 = None
        view_436: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_62, _shape_param_577);  _shape_param_577 = None
        convert_element_type_267: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_436, torch.float32);  view_436 = None
        mul_160: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_267, 0.5)
        mul_161: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_267, 0.7071067811865476);  convert_element_type_267 = None
        erf_15: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_161);  mul_161 = None
        add_146: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_162: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_160, add_146);  mul_160 = add_146 = None
        convert_element_type_268: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_162, torch.bfloat16);  mul_162 = None
        convert_element_type_269: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg239_1, torch.bfloat16);  arg239_1 = None
        convert_element_type_270: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg240_1, torch.bfloat16);  arg240_1 = None
        view_437: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_268, _shape_param_578);  convert_element_type_268 = _shape_param_578 = None
        permute_164: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_270, [1, 0]);  convert_element_type_270 = None
        addmm_63: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_269, view_437, permute_164);  convert_element_type_269 = None
        view_438: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_63, _shape_param_579);  addmm_63 = _shape_param_579 = None
        inductor_lookup_seed_29: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 29)
        inductor_random_29: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_580, inductor_lookup_seed_29, 'rand');  _shape_param_580 = inductor_lookup_seed_29 = None
        lt_29: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_29, 0.9347826093435287);  inductor_random_29 = None
        convert_element_type_271: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_29, torch.bfloat16)
        div_45: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_271, 0.9347826093435287);  convert_element_type_271 = None
        mul_163: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_438, div_45);  view_438 = div_45 = None
        add_147: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_434, mul_163);  mul_163 = None
        view_439: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_147, _shape_param_581);  add_147 = _shape_param_581 = None
        convert_element_type_272: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_439, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_272, [3], correction = 0, keepdim = True)
        getitem_118: "f32[128, 14, 14, 1]" = var_mean_35[0]
        getitem_119: "f32[128, 14, 14, 1]" = var_mean_35[1];  var_mean_35 = None
        add_148: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
        rsqrt_35: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        sub_51: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_272, getitem_119);  convert_element_type_272 = None
        mul_164: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_35);  sub_51 = None
        mul_165: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_164, arg241_1);  mul_164 = arg241_1 = None
        add_149: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_165, arg242_1);  mul_165 = arg242_1 = None
        view_440: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_149, _shape_param_582);  add_149 = _shape_param_582 = None
        permute_165: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_440, [0, 1, 3, 2, 4, 5]);  view_440 = None
        clone_114: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
        view_441: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_114, _shape_param_583);  clone_114 = _shape_param_583 = None
        view_442: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_441, _shape_param_584);  view_441 = _shape_param_584 = None
        convert_element_type_273: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg243_1, torch.bfloat16);  arg243_1 = None
        convert_element_type_274: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg244_1, torch.bfloat16);  arg244_1 = None
        convert_element_type_275: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_442, torch.bfloat16);  view_442 = None
        view_443: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_275, _shape_param_585);  convert_element_type_275 = _shape_param_585 = None
        permute_166: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_274, [1, 0]);  convert_element_type_274 = None
        addmm_64: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_273, view_443, permute_166);  convert_element_type_273 = None
        view_444: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_64, _shape_param_586);  addmm_64 = _shape_param_586 = None
        view_445: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_444, _shape_param_587);  view_444 = _shape_param_587 = None
        permute_167: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_445, [2, 0, 3, 1, 4]);  view_445 = None
        unbind_16 = torch.ops.aten.unbind.int(permute_167);  permute_167 = None
        getitem_120: "bf16[512, 16, 49, 32]" = unbind_16[0]
        getitem_121: "bf16[512, 16, 49, 32]" = unbind_16[1]
        getitem_122: "bf16[512, 16, 49, 32]" = unbind_16[2];  unbind_16 = None
        mul_166: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_120, 0.1767766952966369);  getitem_120 = None
        permute_168: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_121, [0, 1, 3, 2]);  getitem_121 = None
        expand_64: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_166, _shape_param_588);  mul_166 = _shape_param_588 = None
        clone_115: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_446: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_115, _shape_param_589);  clone_115 = _shape_param_589 = None
        expand_65: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_168, _shape_param_590);  permute_168 = _shape_param_590 = None
        clone_116: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_447: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_116, _shape_param_591);  clone_116 = _shape_param_591 = None
        constant_pad_nd_64: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_446, _shape_param_592);  _shape_param_592 = None
        constant_pad_nd_65: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_447, _shape_param_593);  _shape_param_593 = None
        bmm_32: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_64, constant_pad_nd_65);  constant_pad_nd_64 = constant_pad_nd_65 = None
        slice_49: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_32, 1, 0, -7)
        slice_50: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_49, 2, 0, -7);  slice_49 = None
        view_448: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_50, _shape_param_594);  slice_50 = _shape_param_594 = None
        view_449: "i64[2401]" = torch.ops.aten.view.default(arg245_1, [-1]);  arg245_1 = None
        index_48: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg246_1, [view_449]);  arg246_1 = view_449 = None
        view_450: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_48, _shape_param_595);  index_48 = _shape_param_595 = None
        permute_169: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_450, [2, 0, 1]);  view_450 = None
        clone_117: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        unsqueeze_32: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_117, 0);  clone_117 = None
        add_150: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_448, unsqueeze_32);  view_448 = unsqueeze_32 = None
        amax_16: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_150, [-1], True)
        sub_52: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_150, amax_16);  add_150 = None
        exp_16: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_17: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_46: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = None
        convert_element_type_276: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_46, torch.bfloat16);  div_46 = None
        expand_66: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_276, _shape_param_596);  convert_element_type_276 = _shape_param_596 = None
        view_451: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_66, _shape_param_597);  expand_66 = _shape_param_597 = None
        expand_67: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_122, _shape_param_598);  getitem_122 = _shape_param_598 = None
        clone_118: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_452: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_118, _shape_param_599);  clone_118 = _shape_param_599 = None
        constant_pad_nd_66: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_451, _shape_param_600);  _shape_param_600 = None
        constant_pad_nd_67: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_452, _shape_param_601);  _shape_param_601 = None
        bmm_33: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_66, constant_pad_nd_67);  constant_pad_nd_66 = constant_pad_nd_67 = None
        slice_51: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_33, 1, 0, -7);  bmm_33 = None
        view_453: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_51, _shape_param_602);  slice_51 = _shape_param_602 = None
        permute_170: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
        clone_119: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_454: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_119, _shape_param_603);  clone_119 = _shape_param_603 = None
        convert_element_type_277: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg247_1, torch.bfloat16);  arg247_1 = None
        convert_element_type_278: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg248_1, torch.bfloat16);  arg248_1 = None
        view_455: "bf16[25088, 512]" = torch.ops.aten.view.default(view_454, _shape_param_604);  view_454 = _shape_param_604 = None
        permute_171: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_278, [1, 0]);  convert_element_type_278 = None
        addmm_65: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_277, view_455, permute_171);  convert_element_type_277 = None
        view_456: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_65, _shape_param_605);  addmm_65 = _shape_param_605 = None
        view_457: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_456, _shape_param_606);  view_456 = _shape_param_606 = None
        view_458: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_457, _shape_param_607);  view_457 = _shape_param_607 = None
        permute_172: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_458, [0, 1, 3, 2, 4, 5]);  view_458 = None
        clone_120: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_459: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_120, _shape_param_608);  clone_120 = _shape_param_608 = None
        inductor_lookup_seed_30: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 30)
        inductor_random_30: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_609, inductor_lookup_seed_30, 'rand');  _shape_param_609 = inductor_lookup_seed_30 = None
        lt_30: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_30, 0.9304347857832909);  inductor_random_30 = None
        convert_element_type_279: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_30, torch.bfloat16)
        div_47: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_279, 0.9304347857832909);  convert_element_type_279 = None
        mul_167: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_459, div_47);  view_459 = div_47 = None
        add_151: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_439, mul_167);  mul_167 = None
        view_460: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_151, _shape_param_610);  add_151 = _shape_param_610 = None
        convert_element_type_280: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_460, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_280, [2], correction = 0, keepdim = True)
        getitem_123: "f32[128, 196, 1]" = var_mean_36[0]
        getitem_124: "f32[128, 196, 1]" = var_mean_36[1];  var_mean_36 = None
        add_152: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_123, 1e-05);  getitem_123 = None
        rsqrt_36: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        sub_53: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_280, getitem_124);  convert_element_type_280 = None
        mul_168: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_36);  sub_53 = None
        mul_169: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_168, arg249_1);  mul_168 = arg249_1 = None
        add_153: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_169, arg250_1);  mul_169 = arg250_1 = None
        convert_element_type_281: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg251_1, torch.bfloat16);  arg251_1 = None
        convert_element_type_282: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg252_1, torch.bfloat16);  arg252_1 = None
        convert_element_type_283: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_153, torch.bfloat16);  add_153 = None
        view_461: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_283, _shape_param_611);  convert_element_type_283 = _shape_param_611 = None
        permute_173: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_282, [1, 0]);  convert_element_type_282 = None
        addmm_66: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_281, view_461, permute_173);  convert_element_type_281 = None
        view_462: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_66, _shape_param_612);  _shape_param_612 = None
        convert_element_type_284: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_462, torch.float32);  view_462 = None
        mul_170: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_284, 0.5)
        mul_171: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_284, 0.7071067811865476);  convert_element_type_284 = None
        erf_16: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
        add_154: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_172: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_170, add_154);  mul_170 = add_154 = None
        convert_element_type_285: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_172, torch.bfloat16);  mul_172 = None
        convert_element_type_286: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg253_1, torch.bfloat16);  arg253_1 = None
        convert_element_type_287: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg254_1, torch.bfloat16);  arg254_1 = None
        view_463: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_285, _shape_param_613);  convert_element_type_285 = _shape_param_613 = None
        permute_174: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_287, [1, 0]);  convert_element_type_287 = None
        addmm_67: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_286, view_463, permute_174);  convert_element_type_286 = None
        view_464: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_67, _shape_param_614);  addmm_67 = _shape_param_614 = None
        inductor_lookup_seed_31: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 31)
        inductor_random_31: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_615, inductor_lookup_seed_31, 'rand');  _shape_param_615 = inductor_lookup_seed_31 = None
        lt_31: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_31, 0.9304347857832909);  inductor_random_31 = None
        convert_element_type_288: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_31, torch.bfloat16)
        div_48: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_288, 0.9304347857832909);  convert_element_type_288 = None
        mul_173: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_464, div_48);  view_464 = div_48 = None
        add_155: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_460, mul_173);  mul_173 = None
        view_465: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_155, _shape_param_616);  add_155 = _shape_param_616 = None
        convert_element_type_289: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_465, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_289, [3], correction = 0, keepdim = True)
        getitem_125: "f32[128, 14, 14, 1]" = var_mean_37[0]
        getitem_126: "f32[128, 14, 14, 1]" = var_mean_37[1];  var_mean_37 = None
        add_156: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_125, 1e-05);  getitem_125 = None
        rsqrt_37: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        sub_54: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_289, getitem_126);  convert_element_type_289 = None
        mul_174: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_37);  sub_54 = None
        mul_175: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_174, arg255_1);  mul_174 = arg255_1 = None
        add_157: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_175, arg256_1);  mul_175 = arg256_1 = None
        index_49: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_157, [None, fmod_4]);  add_157 = None
        index_50: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_49, [None, None, fmod_4]);  index_49 = None
        view_466: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_50, _shape_param_617);  index_50 = _shape_param_617 = None
        permute_175: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_466, [0, 1, 3, 2, 4, 5]);  view_466 = None
        clone_121: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
        view_467: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_121, _shape_param_618);  clone_121 = _shape_param_618 = None
        view_468: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_467, _shape_param_619);  view_467 = _shape_param_619 = None
        convert_element_type_290: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg257_1, torch.bfloat16);  arg257_1 = None
        convert_element_type_291: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg258_1, torch.bfloat16);  arg258_1 = None
        convert_element_type_292: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_468, torch.bfloat16);  view_468 = None
        view_469: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_292, _shape_param_620);  convert_element_type_292 = _shape_param_620 = None
        permute_176: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_291, [1, 0]);  convert_element_type_291 = None
        addmm_68: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_290, view_469, permute_176);  convert_element_type_290 = None
        view_470: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_68, _shape_param_621);  addmm_68 = _shape_param_621 = None
        view_471: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_470, _shape_param_622);  view_470 = _shape_param_622 = None
        permute_177: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_471, [2, 0, 3, 1, 4]);  view_471 = None
        unbind_17 = torch.ops.aten.unbind.int(permute_177);  permute_177 = None
        getitem_127: "bf16[512, 16, 49, 32]" = unbind_17[0]
        getitem_128: "bf16[512, 16, 49, 32]" = unbind_17[1]
        getitem_129: "bf16[512, 16, 49, 32]" = unbind_17[2];  unbind_17 = None
        mul_176: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_127, 0.1767766952966369);  getitem_127 = None
        permute_178: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_128, [0, 1, 3, 2]);  getitem_128 = None
        expand_68: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_176, _shape_param_623);  mul_176 = _shape_param_623 = None
        clone_122: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_472: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_122, _shape_param_624);  clone_122 = _shape_param_624 = None
        expand_69: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_178, _shape_param_625);  permute_178 = _shape_param_625 = None
        clone_123: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_473: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_123, _shape_param_626);  clone_123 = _shape_param_626 = None
        constant_pad_nd_68: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_472, _shape_param_627);  _shape_param_627 = None
        constant_pad_nd_69: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_473, _shape_param_628);  _shape_param_628 = None
        bmm_34: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_68, constant_pad_nd_69);  constant_pad_nd_68 = constant_pad_nd_69 = None
        slice_52: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_34, 1, 0, -7)
        slice_53: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_52, 2, 0, -7);  slice_52 = None
        view_474: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_53, _shape_param_629);  slice_53 = _shape_param_629 = None
        view_475: "i64[2401]" = torch.ops.aten.view.default(arg259_1, [-1]);  arg259_1 = None
        index_51: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg260_1, [view_475]);  arg260_1 = view_475 = None
        view_476: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_51, _shape_param_630);  index_51 = _shape_param_630 = None
        permute_179: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_476, [2, 0, 1]);  view_476 = None
        clone_124: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
        unsqueeze_33: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_124, 0);  clone_124 = None
        add_158: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_474, unsqueeze_33);  view_474 = unsqueeze_33 = None
        view_477: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_158, _shape_param_631);  add_158 = _shape_param_631 = None
        unsqueeze_34: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg261_1, 1);  arg261_1 = None
        unsqueeze_35: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 0);  unsqueeze_34 = None
        add_159: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_477, unsqueeze_35);  view_477 = unsqueeze_35 = None
        view_478: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_159, _shape_param_632);  add_159 = _shape_param_632 = None
        amax_17: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_478, [-1], True)
        sub_55: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_478, amax_17);  view_478 = None
        exp_17: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_18: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_49: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = None
        convert_element_type_293: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_49, torch.bfloat16);  div_49 = None
        expand_70: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_293, _shape_param_633);  convert_element_type_293 = _shape_param_633 = None
        view_479: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_70, _shape_param_634);  expand_70 = _shape_param_634 = None
        expand_71: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_129, _shape_param_635);  getitem_129 = _shape_param_635 = None
        clone_125: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_480: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_125, _shape_param_636);  clone_125 = _shape_param_636 = None
        constant_pad_nd_70: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_479, _shape_param_637);  _shape_param_637 = None
        constant_pad_nd_71: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_480, _shape_param_638);  _shape_param_638 = None
        bmm_35: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_70, constant_pad_nd_71);  constant_pad_nd_70 = constant_pad_nd_71 = None
        slice_54: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_35, 1, 0, -7);  bmm_35 = None
        view_481: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_54, _shape_param_639);  slice_54 = _shape_param_639 = None
        permute_180: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_481, [0, 2, 1, 3]);  view_481 = None
        clone_126: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
        view_482: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_126, _shape_param_640);  clone_126 = _shape_param_640 = None
        convert_element_type_294: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg262_1, torch.bfloat16);  arg262_1 = None
        convert_element_type_295: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg263_1, torch.bfloat16);  arg263_1 = None
        view_483: "bf16[25088, 512]" = torch.ops.aten.view.default(view_482, _shape_param_641);  view_482 = _shape_param_641 = None
        permute_181: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_295, [1, 0]);  convert_element_type_295 = None
        addmm_69: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_294, view_483, permute_181);  convert_element_type_294 = None
        view_484: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_69, _shape_param_642);  addmm_69 = _shape_param_642 = None
        view_485: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_484, _shape_param_643);  view_484 = _shape_param_643 = None
        view_486: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_485, _shape_param_644);  view_485 = _shape_param_644 = None
        permute_182: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_486, [0, 1, 3, 2, 4, 5]);  view_486 = None
        clone_127: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        view_487: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_127, _shape_param_645);  clone_127 = _shape_param_645 = None
        index_52: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_487, [None, fmod_5]);  view_487 = None
        index_53: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_52, [None, None, fmod_5]);  index_52 = None
        inductor_lookup_seed_32: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 32)
        inductor_random_32: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_646, inductor_lookup_seed_32, 'rand');  _shape_param_646 = inductor_lookup_seed_32 = None
        lt_32: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_32, 0.9260869547724724);  inductor_random_32 = None
        convert_element_type_296: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_32, torch.bfloat16)
        div_50: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_296, 0.9260869547724724);  convert_element_type_296 = None
        mul_177: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_53, div_50);  index_53 = div_50 = None
        add_160: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_465, mul_177);  mul_177 = None
        view_488: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_160, _shape_param_647);  add_160 = _shape_param_647 = None
        convert_element_type_297: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_488, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_297, [2], correction = 0, keepdim = True)
        getitem_130: "f32[128, 196, 1]" = var_mean_38[0]
        getitem_131: "f32[128, 196, 1]" = var_mean_38[1];  var_mean_38 = None
        add_161: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_38: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        sub_56: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_297, getitem_131);  convert_element_type_297 = None
        mul_178: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_38);  sub_56 = None
        mul_179: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_178, arg264_1);  mul_178 = arg264_1 = None
        add_162: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_179, arg265_1);  mul_179 = arg265_1 = None
        convert_element_type_298: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg266_1, torch.bfloat16);  arg266_1 = None
        convert_element_type_299: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg267_1, torch.bfloat16);  arg267_1 = None
        convert_element_type_300: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_162, torch.bfloat16);  add_162 = None
        view_489: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_300, _shape_param_648);  convert_element_type_300 = _shape_param_648 = None
        permute_183: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_299, [1, 0]);  convert_element_type_299 = None
        addmm_70: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_298, view_489, permute_183);  convert_element_type_298 = None
        view_490: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_70, _shape_param_649);  _shape_param_649 = None
        convert_element_type_301: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_490, torch.float32);  view_490 = None
        mul_180: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_301, 0.5)
        mul_181: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_301, 0.7071067811865476);  convert_element_type_301 = None
        erf_17: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_181);  mul_181 = None
        add_163: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_182: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_180, add_163);  mul_180 = add_163 = None
        convert_element_type_302: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_182, torch.bfloat16);  mul_182 = None
        convert_element_type_303: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg268_1, torch.bfloat16);  arg268_1 = None
        convert_element_type_304: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg269_1, torch.bfloat16);  arg269_1 = None
        view_491: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_302, _shape_param_650);  convert_element_type_302 = _shape_param_650 = None
        permute_184: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_304, [1, 0]);  convert_element_type_304 = None
        addmm_71: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_303, view_491, permute_184);  convert_element_type_303 = None
        view_492: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_71, _shape_param_651);  addmm_71 = _shape_param_651 = None
        inductor_lookup_seed_33: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 33)
        inductor_random_33: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_652, inductor_lookup_seed_33, 'rand');  _shape_param_652 = inductor_lookup_seed_33 = None
        lt_33: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_33, 0.9260869547724724);  inductor_random_33 = None
        convert_element_type_305: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_33, torch.bfloat16)
        div_51: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_305, 0.9260869547724724);  convert_element_type_305 = None
        mul_183: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_492, div_51);  view_492 = div_51 = None
        add_164: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_488, mul_183);  mul_183 = None
        view_493: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_164, _shape_param_653);  add_164 = _shape_param_653 = None
        convert_element_type_306: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_493, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_306, [3], correction = 0, keepdim = True)
        getitem_132: "f32[128, 14, 14, 1]" = var_mean_39[0]
        getitem_133: "f32[128, 14, 14, 1]" = var_mean_39[1];  var_mean_39 = None
        add_165: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_39: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_57: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_306, getitem_133);  convert_element_type_306 = None
        mul_184: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_39);  sub_57 = None
        mul_185: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_184, arg270_1);  mul_184 = arg270_1 = None
        add_166: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_185, arg271_1);  mul_185 = arg271_1 = None
        view_494: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_166, _shape_param_654);  add_166 = _shape_param_654 = None
        permute_185: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_494, [0, 1, 3, 2, 4, 5]);  view_494 = None
        clone_128: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
        view_495: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_128, _shape_param_655);  clone_128 = _shape_param_655 = None
        view_496: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_495, _shape_param_656);  view_495 = _shape_param_656 = None
        convert_element_type_307: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg272_1, torch.bfloat16);  arg272_1 = None
        convert_element_type_308: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg273_1, torch.bfloat16);  arg273_1 = None
        convert_element_type_309: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_496, torch.bfloat16);  view_496 = None
        view_497: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_309, _shape_param_657);  convert_element_type_309 = _shape_param_657 = None
        permute_186: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_308, [1, 0]);  convert_element_type_308 = None
        addmm_72: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_307, view_497, permute_186);  convert_element_type_307 = None
        view_498: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_72, _shape_param_658);  addmm_72 = _shape_param_658 = None
        view_499: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_498, _shape_param_659);  view_498 = _shape_param_659 = None
        permute_187: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_499, [2, 0, 3, 1, 4]);  view_499 = None
        unbind_18 = torch.ops.aten.unbind.int(permute_187);  permute_187 = None
        getitem_134: "bf16[512, 16, 49, 32]" = unbind_18[0]
        getitem_135: "bf16[512, 16, 49, 32]" = unbind_18[1]
        getitem_136: "bf16[512, 16, 49, 32]" = unbind_18[2];  unbind_18 = None
        mul_186: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_134, 0.1767766952966369);  getitem_134 = None
        permute_188: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_135, [0, 1, 3, 2]);  getitem_135 = None
        expand_72: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_186, _shape_param_660);  mul_186 = _shape_param_660 = None
        clone_129: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_500: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_129, _shape_param_661);  clone_129 = _shape_param_661 = None
        expand_73: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_188, _shape_param_662);  permute_188 = _shape_param_662 = None
        clone_130: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_501: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_130, _shape_param_663);  clone_130 = _shape_param_663 = None
        constant_pad_nd_72: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_500, _shape_param_664);  _shape_param_664 = None
        constant_pad_nd_73: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_501, _shape_param_665);  _shape_param_665 = None
        bmm_36: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_72, constant_pad_nd_73);  constant_pad_nd_72 = constant_pad_nd_73 = None
        slice_55: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_36, 1, 0, -7)
        slice_56: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_55, 2, 0, -7);  slice_55 = None
        view_502: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_56, _shape_param_666);  slice_56 = _shape_param_666 = None
        view_503: "i64[2401]" = torch.ops.aten.view.default(arg274_1, [-1]);  arg274_1 = None
        index_54: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg275_1, [view_503]);  arg275_1 = view_503 = None
        view_504: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_54, _shape_param_667);  index_54 = _shape_param_667 = None
        permute_189: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_504, [2, 0, 1]);  view_504 = None
        clone_131: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        unsqueeze_36: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_131, 0);  clone_131 = None
        add_167: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_502, unsqueeze_36);  view_502 = unsqueeze_36 = None
        amax_18: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_167, [-1], True)
        sub_58: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_167, amax_18);  add_167 = None
        exp_18: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        sum_19: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_52: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = None
        convert_element_type_310: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_52, torch.bfloat16);  div_52 = None
        expand_74: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_310, _shape_param_668);  convert_element_type_310 = _shape_param_668 = None
        view_505: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_74, _shape_param_669);  expand_74 = _shape_param_669 = None
        expand_75: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_136, _shape_param_670);  getitem_136 = _shape_param_670 = None
        clone_132: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_506: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_132, _shape_param_671);  clone_132 = _shape_param_671 = None
        constant_pad_nd_74: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_505, _shape_param_672);  _shape_param_672 = None
        constant_pad_nd_75: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_506, _shape_param_673);  _shape_param_673 = None
        bmm_37: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_74, constant_pad_nd_75);  constant_pad_nd_74 = constant_pad_nd_75 = None
        slice_57: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_37, 1, 0, -7);  bmm_37 = None
        view_507: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_57, _shape_param_674);  slice_57 = _shape_param_674 = None
        permute_190: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_507, [0, 2, 1, 3]);  view_507 = None
        clone_133: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
        view_508: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_133, _shape_param_675);  clone_133 = _shape_param_675 = None
        convert_element_type_311: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg276_1, torch.bfloat16);  arg276_1 = None
        convert_element_type_312: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg277_1, torch.bfloat16);  arg277_1 = None
        view_509: "bf16[25088, 512]" = torch.ops.aten.view.default(view_508, _shape_param_676);  view_508 = _shape_param_676 = None
        permute_191: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_312, [1, 0]);  convert_element_type_312 = None
        addmm_73: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_311, view_509, permute_191);  convert_element_type_311 = None
        view_510: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_73, _shape_param_677);  addmm_73 = _shape_param_677 = None
        view_511: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_510, _shape_param_678);  view_510 = _shape_param_678 = None
        view_512: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_511, _shape_param_679);  view_511 = _shape_param_679 = None
        permute_192: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_512, [0, 1, 3, 2, 4, 5]);  view_512 = None
        clone_134: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_513: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_134, _shape_param_680);  clone_134 = _shape_param_680 = None
        inductor_lookup_seed_34: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 34)
        inductor_random_34: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_681, inductor_lookup_seed_34, 'rand');  _shape_param_681 = inductor_lookup_seed_34 = None
        lt_34: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_34, 0.9217391312122345);  inductor_random_34 = None
        convert_element_type_313: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_34, torch.bfloat16)
        div_53: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_313, 0.9217391312122345);  convert_element_type_313 = None
        mul_187: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_513, div_53);  view_513 = div_53 = None
        add_168: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_493, mul_187);  mul_187 = None
        view_514: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_168, _shape_param_682);  add_168 = _shape_param_682 = None
        convert_element_type_314: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_514, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_314, [2], correction = 0, keepdim = True)
        getitem_137: "f32[128, 196, 1]" = var_mean_40[0]
        getitem_138: "f32[128, 196, 1]" = var_mean_40[1];  var_mean_40 = None
        add_169: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_137, 1e-05);  getitem_137 = None
        rsqrt_40: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
        sub_59: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_314, getitem_138);  convert_element_type_314 = None
        mul_188: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_40);  sub_59 = None
        mul_189: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_188, arg278_1);  mul_188 = arg278_1 = None
        add_170: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_189, arg279_1);  mul_189 = arg279_1 = None
        convert_element_type_315: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg280_1, torch.bfloat16);  arg280_1 = None
        convert_element_type_316: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg281_1, torch.bfloat16);  arg281_1 = None
        convert_element_type_317: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_170, torch.bfloat16);  add_170 = None
        view_515: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_317, _shape_param_683);  convert_element_type_317 = _shape_param_683 = None
        permute_193: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_316, [1, 0]);  convert_element_type_316 = None
        addmm_74: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_315, view_515, permute_193);  convert_element_type_315 = None
        view_516: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_74, _shape_param_684);  _shape_param_684 = None
        convert_element_type_318: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_516, torch.float32);  view_516 = None
        mul_190: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_318, 0.5)
        mul_191: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_318, 0.7071067811865476);  convert_element_type_318 = None
        erf_18: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_191);  mul_191 = None
        add_171: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_192: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_190, add_171);  mul_190 = add_171 = None
        convert_element_type_319: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_192, torch.bfloat16);  mul_192 = None
        convert_element_type_320: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg282_1, torch.bfloat16);  arg282_1 = None
        convert_element_type_321: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg283_1, torch.bfloat16);  arg283_1 = None
        view_517: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_319, _shape_param_685);  convert_element_type_319 = _shape_param_685 = None
        permute_194: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_321, [1, 0]);  convert_element_type_321 = None
        addmm_75: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_320, view_517, permute_194);  convert_element_type_320 = None
        view_518: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_75, _shape_param_686);  addmm_75 = _shape_param_686 = None
        inductor_lookup_seed_35: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 35)
        inductor_random_35: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_687, inductor_lookup_seed_35, 'rand');  _shape_param_687 = inductor_lookup_seed_35 = None
        lt_35: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_35, 0.9217391312122345);  inductor_random_35 = None
        convert_element_type_322: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_35, torch.bfloat16)
        div_54: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_322, 0.9217391312122345);  convert_element_type_322 = None
        mul_193: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_518, div_54);  view_518 = div_54 = None
        add_172: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_514, mul_193);  mul_193 = None
        view_519: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_172, _shape_param_688);  add_172 = _shape_param_688 = None
        convert_element_type_323: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_519, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_323, [3], correction = 0, keepdim = True)
        getitem_139: "f32[128, 14, 14, 1]" = var_mean_41[0]
        getitem_140: "f32[128, 14, 14, 1]" = var_mean_41[1];  var_mean_41 = None
        add_173: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_139, 1e-05);  getitem_139 = None
        rsqrt_41: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
        sub_60: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_323, getitem_140);  convert_element_type_323 = None
        mul_194: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_41);  sub_60 = None
        mul_195: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_194, arg284_1);  mul_194 = arg284_1 = None
        add_174: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_195, arg285_1);  mul_195 = arg285_1 = None
        index_55: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_174, [None, fmod_4]);  add_174 = None
        index_56: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_55, [None, None, fmod_4]);  index_55 = None
        view_520: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_56, _shape_param_689);  index_56 = _shape_param_689 = None
        permute_195: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_520, [0, 1, 3, 2, 4, 5]);  view_520 = None
        clone_135: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
        view_521: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_135, _shape_param_690);  clone_135 = _shape_param_690 = None
        view_522: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_521, _shape_param_691);  view_521 = _shape_param_691 = None
        convert_element_type_324: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg286_1, torch.bfloat16);  arg286_1 = None
        convert_element_type_325: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg287_1, torch.bfloat16);  arg287_1 = None
        convert_element_type_326: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_522, torch.bfloat16);  view_522 = None
        view_523: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_326, _shape_param_692);  convert_element_type_326 = _shape_param_692 = None
        permute_196: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_325, [1, 0]);  convert_element_type_325 = None
        addmm_76: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_324, view_523, permute_196);  convert_element_type_324 = None
        view_524: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_76, _shape_param_693);  addmm_76 = _shape_param_693 = None
        view_525: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_524, _shape_param_694);  view_524 = _shape_param_694 = None
        permute_197: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_525, [2, 0, 3, 1, 4]);  view_525 = None
        unbind_19 = torch.ops.aten.unbind.int(permute_197);  permute_197 = None
        getitem_141: "bf16[512, 16, 49, 32]" = unbind_19[0]
        getitem_142: "bf16[512, 16, 49, 32]" = unbind_19[1]
        getitem_143: "bf16[512, 16, 49, 32]" = unbind_19[2];  unbind_19 = None
        mul_196: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_141, 0.1767766952966369);  getitem_141 = None
        permute_198: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_142, [0, 1, 3, 2]);  getitem_142 = None
        expand_76: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_196, _shape_param_695);  mul_196 = _shape_param_695 = None
        clone_136: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_526: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_136, _shape_param_696);  clone_136 = _shape_param_696 = None
        expand_77: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_198, _shape_param_697);  permute_198 = _shape_param_697 = None
        clone_137: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_527: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_137, _shape_param_698);  clone_137 = _shape_param_698 = None
        constant_pad_nd_76: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_526, _shape_param_699);  _shape_param_699 = None
        constant_pad_nd_77: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_527, _shape_param_700);  _shape_param_700 = None
        bmm_38: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_76, constant_pad_nd_77);  constant_pad_nd_76 = constant_pad_nd_77 = None
        slice_58: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_38, 1, 0, -7)
        slice_59: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_58, 2, 0, -7);  slice_58 = None
        view_528: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_59, _shape_param_701);  slice_59 = _shape_param_701 = None
        view_529: "i64[2401]" = torch.ops.aten.view.default(arg288_1, [-1]);  arg288_1 = None
        index_57: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg289_1, [view_529]);  arg289_1 = view_529 = None
        view_530: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_57, _shape_param_702);  index_57 = _shape_param_702 = None
        permute_199: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_530, [2, 0, 1]);  view_530 = None
        clone_138: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
        unsqueeze_37: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_138, 0);  clone_138 = None
        add_175: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_528, unsqueeze_37);  view_528 = unsqueeze_37 = None
        view_531: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_175, _shape_param_703);  add_175 = _shape_param_703 = None
        unsqueeze_38: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg290_1, 1);  arg290_1 = None
        unsqueeze_39: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 0);  unsqueeze_38 = None
        add_176: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_531, unsqueeze_39);  view_531 = unsqueeze_39 = None
        view_532: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_176, _shape_param_704);  add_176 = _shape_param_704 = None
        amax_19: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_532, [-1], True)
        sub_61: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_532, amax_19);  view_532 = None
        exp_19: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_20: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_55: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = None
        convert_element_type_327: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_55, torch.bfloat16);  div_55 = None
        expand_78: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_327, _shape_param_705);  convert_element_type_327 = _shape_param_705 = None
        view_533: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_78, _shape_param_706);  expand_78 = _shape_param_706 = None
        expand_79: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_143, _shape_param_707);  getitem_143 = _shape_param_707 = None
        clone_139: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_534: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_139, _shape_param_708);  clone_139 = _shape_param_708 = None
        constant_pad_nd_78: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_533, _shape_param_709);  _shape_param_709 = None
        constant_pad_nd_79: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_534, _shape_param_710);  _shape_param_710 = None
        bmm_39: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_78, constant_pad_nd_79);  constant_pad_nd_78 = constant_pad_nd_79 = None
        slice_60: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_39, 1, 0, -7);  bmm_39 = None
        view_535: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_60, _shape_param_711);  slice_60 = _shape_param_711 = None
        permute_200: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None
        clone_140: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
        view_536: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_140, _shape_param_712);  clone_140 = _shape_param_712 = None
        convert_element_type_328: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg291_1, torch.bfloat16);  arg291_1 = None
        convert_element_type_329: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg292_1, torch.bfloat16);  arg292_1 = None
        view_537: "bf16[25088, 512]" = torch.ops.aten.view.default(view_536, _shape_param_713);  view_536 = _shape_param_713 = None
        permute_201: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_329, [1, 0]);  convert_element_type_329 = None
        addmm_77: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_328, view_537, permute_201);  convert_element_type_328 = None
        view_538: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_77, _shape_param_714);  addmm_77 = _shape_param_714 = None
        view_539: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_538, _shape_param_715);  view_538 = _shape_param_715 = None
        view_540: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_539, _shape_param_716);  view_539 = _shape_param_716 = None
        permute_202: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_540, [0, 1, 3, 2, 4, 5]);  view_540 = None
        clone_141: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
        view_541: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_141, _shape_param_717);  clone_141 = _shape_param_717 = None
        index_58: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_541, [None, fmod_5]);  view_541 = None
        index_59: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_58, [None, None, fmod_5]);  index_58 = None
        inductor_lookup_seed_36: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 36)
        inductor_random_36: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_718, inductor_lookup_seed_36, 'rand');  _shape_param_718 = inductor_lookup_seed_36 = None
        lt_36: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_36, 0.917391300201416);  inductor_random_36 = None
        convert_element_type_330: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_36, torch.bfloat16)
        div_56: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_330, 0.917391300201416);  convert_element_type_330 = None
        mul_197: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_59, div_56);  index_59 = div_56 = None
        add_177: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_519, mul_197);  mul_197 = None
        view_542: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_177, _shape_param_719);  add_177 = _shape_param_719 = None
        convert_element_type_331: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_542, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_331, [2], correction = 0, keepdim = True)
        getitem_144: "f32[128, 196, 1]" = var_mean_42[0]
        getitem_145: "f32[128, 196, 1]" = var_mean_42[1];  var_mean_42 = None
        add_178: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
        rsqrt_42: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        sub_62: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_331, getitem_145);  convert_element_type_331 = None
        mul_198: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_42);  sub_62 = None
        mul_199: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_198, arg293_1);  mul_198 = arg293_1 = None
        add_179: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_199, arg294_1);  mul_199 = arg294_1 = None
        convert_element_type_332: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg295_1, torch.bfloat16);  arg295_1 = None
        convert_element_type_333: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg296_1, torch.bfloat16);  arg296_1 = None
        convert_element_type_334: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_179, torch.bfloat16);  add_179 = None
        view_543: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_334, _shape_param_720);  convert_element_type_334 = _shape_param_720 = None
        permute_203: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_333, [1, 0]);  convert_element_type_333 = None
        addmm_78: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_332, view_543, permute_203);  convert_element_type_332 = None
        view_544: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_78, _shape_param_721);  _shape_param_721 = None
        convert_element_type_335: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_544, torch.float32);  view_544 = None
        mul_200: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_335, 0.5)
        mul_201: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_335, 0.7071067811865476);  convert_element_type_335 = None
        erf_19: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_201);  mul_201 = None
        add_180: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_202: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_200, add_180);  mul_200 = add_180 = None
        convert_element_type_336: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_202, torch.bfloat16);  mul_202 = None
        convert_element_type_337: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg297_1, torch.bfloat16);  arg297_1 = None
        convert_element_type_338: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg298_1, torch.bfloat16);  arg298_1 = None
        view_545: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_336, _shape_param_722);  convert_element_type_336 = _shape_param_722 = None
        permute_204: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_338, [1, 0]);  convert_element_type_338 = None
        addmm_79: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_337, view_545, permute_204);  convert_element_type_337 = None
        view_546: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_79, _shape_param_723);  addmm_79 = _shape_param_723 = None
        inductor_lookup_seed_37: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 37)
        inductor_random_37: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_724, inductor_lookup_seed_37, 'rand');  _shape_param_724 = inductor_lookup_seed_37 = None
        lt_37: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_37, 0.917391300201416);  inductor_random_37 = None
        convert_element_type_339: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_37, torch.bfloat16)
        div_57: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_339, 0.917391300201416);  convert_element_type_339 = None
        mul_203: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_546, div_57);  view_546 = div_57 = None
        add_181: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_542, mul_203);  mul_203 = None
        view_547: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_181, _shape_param_725);  add_181 = _shape_param_725 = None
        convert_element_type_340: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_547, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_340, [3], correction = 0, keepdim = True)
        getitem_146: "f32[128, 14, 14, 1]" = var_mean_43[0]
        getitem_147: "f32[128, 14, 14, 1]" = var_mean_43[1];  var_mean_43 = None
        add_182: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05);  getitem_146 = None
        rsqrt_43: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        sub_63: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_340, getitem_147);  convert_element_type_340 = None
        mul_204: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_43);  sub_63 = None
        mul_205: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_204, arg299_1);  mul_204 = arg299_1 = None
        add_183: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_205, arg300_1);  mul_205 = arg300_1 = None
        view_548: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(add_183, _shape_param_726);  add_183 = _shape_param_726 = None
        permute_205: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_548, [0, 1, 3, 2, 4, 5]);  view_548 = None
        clone_142: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_549: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_142, _shape_param_727);  clone_142 = _shape_param_727 = None
        view_550: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_549, _shape_param_728);  view_549 = _shape_param_728 = None
        convert_element_type_341: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg301_1, torch.bfloat16);  arg301_1 = None
        convert_element_type_342: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg302_1, torch.bfloat16);  arg302_1 = None
        convert_element_type_343: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_550, torch.bfloat16);  view_550 = None
        view_551: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_343, _shape_param_729);  convert_element_type_343 = _shape_param_729 = None
        permute_206: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_342, [1, 0]);  convert_element_type_342 = None
        addmm_80: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_341, view_551, permute_206);  convert_element_type_341 = None
        view_552: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_80, _shape_param_730);  addmm_80 = _shape_param_730 = None
        view_553: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_552, _shape_param_731);  view_552 = _shape_param_731 = None
        permute_207: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_553, [2, 0, 3, 1, 4]);  view_553 = None
        unbind_20 = torch.ops.aten.unbind.int(permute_207);  permute_207 = None
        getitem_148: "bf16[512, 16, 49, 32]" = unbind_20[0]
        getitem_149: "bf16[512, 16, 49, 32]" = unbind_20[1]
        getitem_150: "bf16[512, 16, 49, 32]" = unbind_20[2];  unbind_20 = None
        mul_206: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_148, 0.1767766952966369);  getitem_148 = None
        permute_208: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_149, [0, 1, 3, 2]);  getitem_149 = None
        expand_80: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_206, _shape_param_732);  mul_206 = _shape_param_732 = None
        clone_143: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_554: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_143, _shape_param_733);  clone_143 = _shape_param_733 = None
        expand_81: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_208, _shape_param_734);  permute_208 = _shape_param_734 = None
        clone_144: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_555: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_144, _shape_param_735);  clone_144 = _shape_param_735 = None
        constant_pad_nd_80: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_554, _shape_param_736);  _shape_param_736 = None
        constant_pad_nd_81: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_555, _shape_param_737);  _shape_param_737 = None
        bmm_40: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_80, constant_pad_nd_81);  constant_pad_nd_80 = constant_pad_nd_81 = None
        slice_61: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_40, 1, 0, -7)
        slice_62: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_61, 2, 0, -7);  slice_61 = None
        view_556: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_62, _shape_param_738);  slice_62 = _shape_param_738 = None
        view_557: "i64[2401]" = torch.ops.aten.view.default(arg303_1, [-1]);  arg303_1 = None
        index_60: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg304_1, [view_557]);  arg304_1 = view_557 = None
        view_558: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_60, _shape_param_739);  index_60 = _shape_param_739 = None
        permute_209: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_558, [2, 0, 1]);  view_558 = None
        clone_145: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
        unsqueeze_40: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_145, 0);  clone_145 = None
        add_184: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_556, unsqueeze_40);  view_556 = unsqueeze_40 = None
        amax_20: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(add_184, [-1], True)
        sub_64: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(add_184, amax_20);  add_184 = None
        exp_20: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        sum_21: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_58: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = None
        convert_element_type_344: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_58, torch.bfloat16);  div_58 = None
        expand_82: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_344, _shape_param_740);  convert_element_type_344 = _shape_param_740 = None
        view_559: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_82, _shape_param_741);  expand_82 = _shape_param_741 = None
        expand_83: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_150, _shape_param_742);  getitem_150 = _shape_param_742 = None
        clone_146: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_560: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_146, _shape_param_743);  clone_146 = _shape_param_743 = None
        constant_pad_nd_82: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_559, _shape_param_744);  _shape_param_744 = None
        constant_pad_nd_83: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_560, _shape_param_745);  _shape_param_745 = None
        bmm_41: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_82, constant_pad_nd_83);  constant_pad_nd_82 = constant_pad_nd_83 = None
        slice_63: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_41, 1, 0, -7);  bmm_41 = None
        view_561: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_63, _shape_param_746);  slice_63 = _shape_param_746 = None
        permute_210: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_561, [0, 2, 1, 3]);  view_561 = None
        clone_147: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_562: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_147, _shape_param_747);  clone_147 = _shape_param_747 = None
        convert_element_type_345: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg305_1, torch.bfloat16);  arg305_1 = None
        convert_element_type_346: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg306_1, torch.bfloat16);  arg306_1 = None
        view_563: "bf16[25088, 512]" = torch.ops.aten.view.default(view_562, _shape_param_748);  view_562 = _shape_param_748 = None
        permute_211: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_346, [1, 0]);  convert_element_type_346 = None
        addmm_81: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_345, view_563, permute_211);  convert_element_type_345 = None
        view_564: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_81, _shape_param_749);  addmm_81 = _shape_param_749 = None
        view_565: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_564, _shape_param_750);  view_564 = _shape_param_750 = None
        view_566: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_565, _shape_param_751);  view_565 = _shape_param_751 = None
        permute_212: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_566, [0, 1, 3, 2, 4, 5]);  view_566 = None
        clone_148: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_567: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_148, _shape_param_752);  clone_148 = _shape_param_752 = None
        inductor_lookup_seed_38: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 38)
        inductor_random_38: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_753, inductor_lookup_seed_38, 'rand');  _shape_param_753 = inductor_lookup_seed_38 = None
        lt_38: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_38, 0.9130434766411781);  inductor_random_38 = None
        convert_element_type_347: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_38, torch.bfloat16)
        div_59: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_347, 0.9130434766411781);  convert_element_type_347 = None
        mul_207: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_567, div_59);  view_567 = div_59 = None
        add_185: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_547, mul_207);  mul_207 = None
        view_568: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_185, _shape_param_754);  add_185 = _shape_param_754 = None
        convert_element_type_348: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_568, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_348, [2], correction = 0, keepdim = True)
        getitem_151: "f32[128, 196, 1]" = var_mean_44[0]
        getitem_152: "f32[128, 196, 1]" = var_mean_44[1];  var_mean_44 = None
        add_186: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_151, 1e-05);  getitem_151 = None
        rsqrt_44: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_65: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_348, getitem_152);  convert_element_type_348 = None
        mul_208: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_44);  sub_65 = None
        mul_209: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_208, arg307_1);  mul_208 = arg307_1 = None
        add_187: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_209, arg308_1);  mul_209 = arg308_1 = None
        convert_element_type_349: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg309_1, torch.bfloat16);  arg309_1 = None
        convert_element_type_350: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg310_1, torch.bfloat16);  arg310_1 = None
        convert_element_type_351: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_187, torch.bfloat16);  add_187 = None
        view_569: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_351, _shape_param_755);  convert_element_type_351 = _shape_param_755 = None
        permute_213: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_350, [1, 0]);  convert_element_type_350 = None
        addmm_82: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_349, view_569, permute_213);  convert_element_type_349 = None
        view_570: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_82, _shape_param_756);  _shape_param_756 = None
        convert_element_type_352: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_570, torch.float32);  view_570 = None
        mul_210: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_352, 0.5)
        mul_211: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_352, 0.7071067811865476);  convert_element_type_352 = None
        erf_20: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_211);  mul_211 = None
        add_188: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_212: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_210, add_188);  mul_210 = add_188 = None
        convert_element_type_353: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_212, torch.bfloat16);  mul_212 = None
        convert_element_type_354: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg311_1, torch.bfloat16);  arg311_1 = None
        convert_element_type_355: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg312_1, torch.bfloat16);  arg312_1 = None
        view_571: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_353, _shape_param_757);  convert_element_type_353 = _shape_param_757 = None
        permute_214: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_355, [1, 0]);  convert_element_type_355 = None
        addmm_83: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_354, view_571, permute_214);  convert_element_type_354 = None
        view_572: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_83, _shape_param_758);  addmm_83 = _shape_param_758 = None
        inductor_lookup_seed_39: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 39)
        inductor_random_39: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_759, inductor_lookup_seed_39, 'rand');  _shape_param_759 = inductor_lookup_seed_39 = None
        lt_39: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_39, 0.9130434766411781);  inductor_random_39 = None
        convert_element_type_356: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_39, torch.bfloat16)
        div_60: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_356, 0.9130434766411781);  convert_element_type_356 = None
        mul_213: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_572, div_60);  view_572 = div_60 = None
        add_189: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_568, mul_213);  mul_213 = None
        view_573: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_189, _shape_param_760);  add_189 = _shape_param_760 = None
        convert_element_type_357: "f32[128, 14, 14, 512]" = torch.ops.prims.convert_element_type.default(view_573, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_357, [3], correction = 0, keepdim = True)
        getitem_153: "f32[128, 14, 14, 1]" = var_mean_45[0]
        getitem_154: "f32[128, 14, 14, 1]" = var_mean_45[1];  var_mean_45 = None
        add_190: "f32[128, 14, 14, 1]" = torch.ops.aten.add.Tensor(getitem_153, 1e-05);  getitem_153 = None
        rsqrt_45: "f32[128, 14, 14, 1]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
        sub_66: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_357, getitem_154);  convert_element_type_357 = None
        mul_214: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_45);  sub_66 = None
        mul_215: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_214, arg313_1);  mul_214 = arg313_1 = None
        add_191: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(mul_215, arg314_1);  mul_215 = arg314_1 = None
        index_61: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(add_191, [None, fmod_4]);  add_191 = None
        index_62: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_61, [None, None, fmod_4]);  index_61 = fmod_4 = None
        view_574: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.view.default(index_62, _shape_param_761);  index_62 = _shape_param_761 = None
        permute_215: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_574, [0, 1, 3, 2, 4, 5]);  view_574 = None
        clone_149: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_575: "f32[512, 7, 7, 512]" = torch.ops.aten.view.default(clone_149, _shape_param_762);  clone_149 = _shape_param_762 = None
        view_576: "f32[512, 49, 512]" = torch.ops.aten.view.default(view_575, _shape_param_763);  view_575 = _shape_param_763 = None
        convert_element_type_358: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg315_1, torch.bfloat16);  arg315_1 = None
        convert_element_type_359: "bf16[1536, 512]" = torch.ops.prims.convert_element_type.default(arg316_1, torch.bfloat16);  arg316_1 = None
        convert_element_type_360: "bf16[512, 49, 512]" = torch.ops.prims.convert_element_type.default(view_576, torch.bfloat16);  view_576 = None
        view_577: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_360, _shape_param_764);  convert_element_type_360 = _shape_param_764 = None
        permute_216: "bf16[512, 1536]" = torch.ops.aten.permute.default(convert_element_type_359, [1, 0]);  convert_element_type_359 = None
        addmm_84: "bf16[25088, 1536]" = torch.ops.aten.addmm.default(convert_element_type_358, view_577, permute_216);  convert_element_type_358 = None
        view_578: "bf16[512, 49, 1536]" = torch.ops.aten.view.default(addmm_84, _shape_param_765);  addmm_84 = _shape_param_765 = None
        view_579: "bf16[512, 49, 3, 16, 32]" = torch.ops.aten.view.default(view_578, _shape_param_766);  view_578 = _shape_param_766 = None
        permute_217: "bf16[3, 512, 16, 49, 32]" = torch.ops.aten.permute.default(view_579, [2, 0, 3, 1, 4]);  view_579 = None
        unbind_21 = torch.ops.aten.unbind.int(permute_217);  permute_217 = None
        getitem_155: "bf16[512, 16, 49, 32]" = unbind_21[0]
        getitem_156: "bf16[512, 16, 49, 32]" = unbind_21[1]
        getitem_157: "bf16[512, 16, 49, 32]" = unbind_21[2];  unbind_21 = None
        mul_216: "bf16[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_155, 0.1767766952966369);  getitem_155 = None
        permute_218: "bf16[512, 16, 32, 49]" = torch.ops.aten.permute.default(getitem_156, [0, 1, 3, 2]);  getitem_156 = None
        expand_84: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(mul_216, _shape_param_767);  mul_216 = _shape_param_767 = None
        clone_150: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_580: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_150, _shape_param_768);  clone_150 = _shape_param_768 = None
        expand_85: "bf16[512, 16, 32, 49]" = torch.ops.aten.expand.default(permute_218, _shape_param_769);  permute_218 = _shape_param_769 = None
        clone_151: "bf16[512, 16, 32, 49]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_581: "bf16[8192, 32, 49]" = torch.ops.aten.view.default(clone_151, _shape_param_770);  clone_151 = _shape_param_770 = None
        constant_pad_nd_84: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_580, _shape_param_771);  _shape_param_771 = None
        constant_pad_nd_85: "bf16[8192, 32, 56]" = torch.ops.aten.constant_pad_nd.default(view_581, _shape_param_772);  _shape_param_772 = None
        bmm_42: "bf16[8192, 56, 56]" = torch.ops.aten.bmm.default(constant_pad_nd_84, constant_pad_nd_85);  constant_pad_nd_84 = constant_pad_nd_85 = None
        slice_64: "bf16[8192, 49, 56]" = torch.ops.aten.slice.Tensor(bmm_42, 1, 0, -7)
        slice_65: "bf16[8192, 49, 49]" = torch.ops.aten.slice.Tensor(slice_64, 2, 0, -7);  slice_64 = None
        view_582: "bf16[512, 16, 49, 49]" = torch.ops.aten.view.default(slice_65, _shape_param_773);  slice_65 = _shape_param_773 = None
        view_583: "i64[2401]" = torch.ops.aten.view.default(arg317_1, [-1]);  arg317_1 = None
        index_63: "f32[2401, 16]" = torch.ops.aten.index.Tensor(arg318_1, [view_583]);  arg318_1 = view_583 = None
        view_584: "f32[49, 49, 16]" = torch.ops.aten.view.default(index_63, _shape_param_774);  index_63 = _shape_param_774 = None
        permute_219: "f32[16, 49, 49]" = torch.ops.aten.permute.default(view_584, [2, 0, 1]);  view_584 = None
        clone_152: "f32[16, 49, 49]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
        unsqueeze_41: "f32[1, 16, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_152, 0);  clone_152 = None
        add_192: "f32[512, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_582, unsqueeze_41);  view_582 = unsqueeze_41 = None
        view_585: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.view.default(add_192, _shape_param_775);  add_192 = _shape_param_775 = None
        unsqueeze_42: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(arg319_1, 1);  arg319_1 = None
        unsqueeze_43: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 0);  unsqueeze_42 = None
        add_193: "f32[128, 4, 16, 49, 49]" = torch.ops.aten.add.Tensor(view_585, unsqueeze_43);  view_585 = unsqueeze_43 = None
        view_586: "f32[512, 16, 49, 49]" = torch.ops.aten.view.default(add_193, _shape_param_776);  add_193 = _shape_param_776 = None
        amax_21: "f32[512, 16, 49, 1]" = torch.ops.aten.amax.default(view_586, [-1], True)
        sub_67: "f32[512, 16, 49, 49]" = torch.ops.aten.sub.Tensor(view_586, amax_21);  view_586 = None
        exp_21: "f32[512, 16, 49, 49]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_22: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_61: "f32[512, 16, 49, 49]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = None
        convert_element_type_361: "bf16[512, 16, 49, 49]" = torch.ops.prims.convert_element_type.default(div_61, torch.bfloat16);  div_61 = None
        expand_86: "bf16[512, 16, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_361, _shape_param_777);  convert_element_type_361 = _shape_param_777 = None
        view_587: "bf16[8192, 49, 49]" = torch.ops.aten.view.default(expand_86, _shape_param_778);  expand_86 = _shape_param_778 = None
        expand_87: "bf16[512, 16, 49, 32]" = torch.ops.aten.expand.default(getitem_157, _shape_param_779);  getitem_157 = _shape_param_779 = None
        clone_153: "bf16[512, 16, 49, 32]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_588: "bf16[8192, 49, 32]" = torch.ops.aten.view.default(clone_153, _shape_param_780);  clone_153 = _shape_param_780 = None
        constant_pad_nd_86: "bf16[8192, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_587, _shape_param_781);  _shape_param_781 = None
        constant_pad_nd_87: "bf16[8192, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_588, _shape_param_782);  _shape_param_782 = None
        bmm_43: "bf16[8192, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_86, constant_pad_nd_87);  constant_pad_nd_86 = constant_pad_nd_87 = None
        slice_66: "bf16[8192, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_43, 1, 0, -7);  bmm_43 = None
        view_589: "bf16[512, 16, 49, 32]" = torch.ops.aten.view.default(slice_66, _shape_param_783);  slice_66 = _shape_param_783 = None
        permute_220: "bf16[512, 49, 16, 32]" = torch.ops.aten.permute.default(view_589, [0, 2, 1, 3]);  view_589 = None
        clone_154: "bf16[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        view_590: "bf16[512, 49, 512]" = torch.ops.aten.view.default(clone_154, _shape_param_784);  clone_154 = _shape_param_784 = None
        convert_element_type_362: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg320_1, torch.bfloat16);  arg320_1 = None
        convert_element_type_363: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg321_1, torch.bfloat16);  arg321_1 = None
        view_591: "bf16[25088, 512]" = torch.ops.aten.view.default(view_590, _shape_param_785);  view_590 = _shape_param_785 = None
        permute_221: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_363, [1, 0]);  convert_element_type_363 = None
        addmm_85: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_362, view_591, permute_221);  convert_element_type_362 = None
        view_592: "bf16[512, 49, 512]" = torch.ops.aten.view.default(addmm_85, _shape_param_786);  addmm_85 = _shape_param_786 = None
        view_593: "bf16[512, 7, 7, 512]" = torch.ops.aten.view.default(view_592, _shape_param_787);  view_592 = _shape_param_787 = None
        view_594: "bf16[128, 2, 2, 7, 7, 512]" = torch.ops.aten.view.default(view_593, _shape_param_788);  view_593 = _shape_param_788 = None
        permute_222: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_594, [0, 1, 3, 2, 4, 5]);  view_594 = None
        clone_155: "bf16[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
        view_595: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(clone_155, _shape_param_789);  clone_155 = _shape_param_789 = None
        index_64: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_595, [None, fmod_5]);  view_595 = None
        index_65: "bf16[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_64, [None, None, fmod_5]);  index_64 = None
        inductor_lookup_seed_40: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 40)
        inductor_random_40: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_790, inductor_lookup_seed_40, 'rand');  _shape_param_790 = inductor_lookup_seed_40 = None
        lt_40: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_40, 0.9086956530809402);  inductor_random_40 = None
        convert_element_type_364: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_40, torch.bfloat16)
        div_62: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_364, 0.9086956530809402);  convert_element_type_364 = None
        mul_217: "bf16[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_65, div_62);  index_65 = div_62 = None
        add_194: "bf16[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_573, mul_217);  mul_217 = None
        view_596: "bf16[128, 196, 512]" = torch.ops.aten.view.default(add_194, _shape_param_791);  add_194 = _shape_param_791 = None
        convert_element_type_365: "f32[128, 196, 512]" = torch.ops.prims.convert_element_type.default(view_596, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_365, [2], correction = 0, keepdim = True)
        getitem_158: "f32[128, 196, 1]" = var_mean_46[0]
        getitem_159: "f32[128, 196, 1]" = var_mean_46[1];  var_mean_46 = None
        add_195: "f32[128, 196, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_46: "f32[128, 196, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
        sub_68: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_365, getitem_159);  convert_element_type_365 = None
        mul_218: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_46);  sub_68 = None
        mul_219: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_218, arg322_1);  mul_218 = arg322_1 = None
        add_196: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(mul_219, arg323_1);  mul_219 = arg323_1 = None
        convert_element_type_366: "bf16[2048]" = torch.ops.prims.convert_element_type.default(arg324_1, torch.bfloat16);  arg324_1 = None
        convert_element_type_367: "bf16[2048, 512]" = torch.ops.prims.convert_element_type.default(arg325_1, torch.bfloat16);  arg325_1 = None
        convert_element_type_368: "bf16[128, 196, 512]" = torch.ops.prims.convert_element_type.default(add_196, torch.bfloat16);  add_196 = None
        view_597: "bf16[25088, 512]" = torch.ops.aten.view.default(convert_element_type_368, _shape_param_792);  convert_element_type_368 = _shape_param_792 = None
        permute_223: "bf16[512, 2048]" = torch.ops.aten.permute.default(convert_element_type_367, [1, 0]);  convert_element_type_367 = None
        addmm_86: "bf16[25088, 2048]" = torch.ops.aten.addmm.default(convert_element_type_366, view_597, permute_223);  convert_element_type_366 = None
        view_598: "bf16[128, 196, 2048]" = torch.ops.aten.view.default(addmm_86, _shape_param_793);  _shape_param_793 = None
        convert_element_type_369: "f32[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(view_598, torch.float32);  view_598 = None
        mul_220: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_369, 0.5)
        mul_221: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_369, 0.7071067811865476);  convert_element_type_369 = None
        erf_21: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_221);  mul_221 = None
        add_197: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_222: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_220, add_197);  mul_220 = add_197 = None
        convert_element_type_370: "bf16[128, 196, 2048]" = torch.ops.prims.convert_element_type.default(mul_222, torch.bfloat16);  mul_222 = None
        convert_element_type_371: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg326_1, torch.bfloat16);  arg326_1 = None
        convert_element_type_372: "bf16[512, 2048]" = torch.ops.prims.convert_element_type.default(arg327_1, torch.bfloat16);  arg327_1 = None
        view_599: "bf16[25088, 2048]" = torch.ops.aten.view.default(convert_element_type_370, _shape_param_794);  convert_element_type_370 = _shape_param_794 = None
        permute_224: "bf16[2048, 512]" = torch.ops.aten.permute.default(convert_element_type_372, [1, 0]);  convert_element_type_372 = None
        addmm_87: "bf16[25088, 512]" = torch.ops.aten.addmm.default(convert_element_type_371, view_599, permute_224);  convert_element_type_371 = None
        view_600: "bf16[128, 196, 512]" = torch.ops.aten.view.default(addmm_87, _shape_param_795);  addmm_87 = _shape_param_795 = None
        inductor_lookup_seed_41: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 41)
        inductor_random_41: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_796, inductor_lookup_seed_41, 'rand');  _shape_param_796 = inductor_lookup_seed_41 = None
        lt_41: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_41, 0.9086956530809402);  inductor_random_41 = None
        convert_element_type_373: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_41, torch.bfloat16)
        div_63: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_373, 0.9086956530809402);  convert_element_type_373 = None
        mul_223: "bf16[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_600, div_63);  view_600 = div_63 = None
        add_198: "bf16[128, 196, 512]" = torch.ops.aten.add.Tensor(view_596, mul_223);  mul_223 = None
        view_601: "bf16[128, 14, 14, 512]" = torch.ops.aten.view.default(add_198, _shape_param_797);  add_198 = _shape_param_797 = None
        view_602: "bf16[128, 7, 2, 7, 2, 512]" = torch.ops.aten.view.default(view_601, _shape_param_798);  view_601 = _shape_param_798 = None
        permute_225: "bf16[128, 7, 7, 2, 2, 512]" = torch.ops.aten.permute.default(view_602, [0, 1, 3, 4, 2, 5]);  view_602 = None
        clone_156: "bf16[128, 7, 7, 2, 2, 512]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_603: "bf16[128, 7, 7, 2048]" = torch.ops.aten.view.default(clone_156, _shape_param_799);  clone_156 = _shape_param_799 = None
        convert_element_type_374: "f32[128, 7, 7, 2048]" = torch.ops.prims.convert_element_type.default(view_603, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_374, [3], correction = 0, keepdim = True)
        getitem_160: "f32[128, 7, 7, 1]" = var_mean_47[0]
        getitem_161: "f32[128, 7, 7, 1]" = var_mean_47[1];  var_mean_47 = None
        add_199: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_47: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        sub_69: "f32[128, 7, 7, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_374, getitem_161);  convert_element_type_374 = None
        mul_224: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_47);  sub_69 = None
        mul_225: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_224, arg328_1);  mul_224 = arg328_1 = None
        add_200: "f32[128, 7, 7, 2048]" = torch.ops.aten.add.Tensor(mul_225, arg329_1);  mul_225 = arg329_1 = None
        convert_element_type_375: "bf16[1024, 2048]" = torch.ops.prims.convert_element_type.default(arg330_1, torch.bfloat16);  arg330_1 = None
        convert_element_type_376: "bf16[128, 7, 7, 2048]" = torch.ops.prims.convert_element_type.default(add_200, torch.bfloat16);  add_200 = None
        permute_226: "bf16[2048, 1024]" = torch.ops.aten.permute.default(convert_element_type_375, [1, 0]);  convert_element_type_375 = None
        view_604: "bf16[6272, 2048]" = torch.ops.aten.view.default(convert_element_type_376, _shape_param_800);  convert_element_type_376 = _shape_param_800 = None
        mm_2: "bf16[6272, 1024]" = torch.ops.aten.mm.default(view_604, permute_226)
        view_605: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(mm_2, _shape_param_801);  _shape_param_801 = None
        convert_element_type_377: "f32[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(view_605, torch.float32)
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_377, [3], correction = 0, keepdim = True)
        getitem_162: "f32[128, 7, 7, 1]" = var_mean_48[0]
        getitem_163: "f32[128, 7, 7, 1]" = var_mean_48[1];  var_mean_48 = None
        add_201: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05);  getitem_162 = None
        rsqrt_48: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
        sub_70: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_377, getitem_163);  convert_element_type_377 = None
        mul_226: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_48);  sub_70 = None
        mul_227: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_226, arg331_1);  mul_226 = arg331_1 = None
        add_202: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_227, arg332_1);  mul_227 = arg332_1 = None
        view_606: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.view.default(add_202, _shape_param_802);  add_202 = _shape_param_802 = None
        permute_227: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_606, [0, 1, 3, 2, 4, 5]);  view_606 = None
        view_607: "f32[128, 7, 7, 1024]" = torch.ops.aten.view.default(permute_227, _shape_param_803);  permute_227 = _shape_param_803 = None
        view_608: "f32[128, 49, 1024]" = torch.ops.aten.view.default(view_607, _shape_param_804);  view_607 = _shape_param_804 = None
        convert_element_type_378: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg333_1, torch.bfloat16);  arg333_1 = None
        convert_element_type_379: "bf16[3072, 1024]" = torch.ops.prims.convert_element_type.default(arg334_1, torch.bfloat16);  arg334_1 = None
        convert_element_type_380: "bf16[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(view_608, torch.bfloat16);  view_608 = None
        view_609: "bf16[6272, 1024]" = torch.ops.aten.view.default(convert_element_type_380, _shape_param_805);  convert_element_type_380 = _shape_param_805 = None
        permute_228: "bf16[1024, 3072]" = torch.ops.aten.permute.default(convert_element_type_379, [1, 0]);  convert_element_type_379 = None
        addmm_88: "bf16[6272, 3072]" = torch.ops.aten.addmm.default(convert_element_type_378, view_609, permute_228);  convert_element_type_378 = None
        view_610: "bf16[128, 49, 3072]" = torch.ops.aten.view.default(addmm_88, _shape_param_806);  addmm_88 = _shape_param_806 = None
        view_611: "bf16[128, 49, 3, 32, 32]" = torch.ops.aten.view.default(view_610, _shape_param_807);  view_610 = _shape_param_807 = None
        permute_229: "bf16[3, 128, 32, 49, 32]" = torch.ops.aten.permute.default(view_611, [2, 0, 3, 1, 4]);  view_611 = None
        unbind_22 = torch.ops.aten.unbind.int(permute_229);  permute_229 = None
        getitem_164: "bf16[128, 32, 49, 32]" = unbind_22[0]
        getitem_165: "bf16[128, 32, 49, 32]" = unbind_22[1]
        getitem_166: "bf16[128, 32, 49, 32]" = unbind_22[2];  unbind_22 = None
        mul_228: "bf16[128, 32, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_164, 0.1767766952966369);  getitem_164 = None
        permute_230: "bf16[128, 32, 32, 49]" = torch.ops.aten.permute.default(getitem_165, [0, 1, 3, 2]);  getitem_165 = None
        expand_88: "bf16[128, 32, 49, 32]" = torch.ops.aten.expand.default(mul_228, _shape_param_808);  mul_228 = _shape_param_808 = None
        clone_157: "bf16[128, 32, 49, 32]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_612: "bf16[4096, 49, 32]" = torch.ops.aten.view.default(clone_157, _shape_param_809);  clone_157 = _shape_param_809 = None
        expand_89: "bf16[128, 32, 32, 49]" = torch.ops.aten.expand.default(permute_230, _shape_param_810);  permute_230 = _shape_param_810 = None
        clone_158: "bf16[128, 32, 32, 49]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_613: "bf16[4096, 32, 49]" = torch.ops.aten.view.default(clone_158, _shape_param_811);  clone_158 = _shape_param_811 = None
        bmm_44: "bf16[4096, 49, 49]" = torch.ops.aten.bmm.default(view_612, view_613)
        view_614: "bf16[128, 32, 49, 49]" = torch.ops.aten.view.default(bmm_44, _shape_param_812);  _shape_param_812 = None
        view_615: "i64[2401]" = torch.ops.aten.view.default(arg335_1, [-1]);  arg335_1 = None
        index_66: "f32[2401, 32]" = torch.ops.aten.index.Tensor(arg336_1, [view_615]);  arg336_1 = view_615 = None
        view_616: "f32[49, 49, 32]" = torch.ops.aten.view.default(index_66, _shape_param_813);  index_66 = _shape_param_813 = None
        permute_231: "f32[32, 49, 49]" = torch.ops.aten.permute.default(view_616, [2, 0, 1]);  view_616 = None
        clone_159: "f32[32, 49, 49]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
        unsqueeze_44: "f32[1, 32, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_159, 0);  clone_159 = None
        add_203: "f32[128, 32, 49, 49]" = torch.ops.aten.add.Tensor(view_614, unsqueeze_44);  view_614 = unsqueeze_44 = None
        amax_22: "f32[128, 32, 49, 1]" = torch.ops.aten.amax.default(add_203, [-1], True)
        sub_71: "f32[128, 32, 49, 49]" = torch.ops.aten.sub.Tensor(add_203, amax_22);  add_203 = None
        exp_22: "f32[128, 32, 49, 49]" = torch.ops.aten.exp.default(sub_71);  sub_71 = None
        sum_23: "f32[128, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_64: "f32[128, 32, 49, 49]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = None
        convert_element_type_381: "bf16[128, 32, 49, 49]" = torch.ops.prims.convert_element_type.default(div_64, torch.bfloat16);  div_64 = None
        expand_90: "bf16[128, 32, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_381, _shape_param_814);  convert_element_type_381 = _shape_param_814 = None
        view_617: "bf16[4096, 49, 49]" = torch.ops.aten.view.default(expand_90, _shape_param_815);  expand_90 = _shape_param_815 = None
        expand_91: "bf16[128, 32, 49, 32]" = torch.ops.aten.expand.default(getitem_166, _shape_param_816);  getitem_166 = _shape_param_816 = None
        clone_160: "bf16[128, 32, 49, 32]" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_618: "bf16[4096, 49, 32]" = torch.ops.aten.view.default(clone_160, _shape_param_817);  clone_160 = _shape_param_817 = None
        constant_pad_nd_88: "bf16[4096, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_617, _shape_param_818);  _shape_param_818 = None
        constant_pad_nd_89: "bf16[4096, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_618, _shape_param_819);  _shape_param_819 = None
        bmm_45: "bf16[4096, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_88, constant_pad_nd_89);  constant_pad_nd_88 = constant_pad_nd_89 = None
        slice_67: "bf16[4096, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_45, 1, 0, -7);  bmm_45 = None
        view_619: "bf16[128, 32, 49, 32]" = torch.ops.aten.view.default(slice_67, _shape_param_820);  slice_67 = _shape_param_820 = None
        permute_232: "bf16[128, 49, 32, 32]" = torch.ops.aten.permute.default(view_619, [0, 2, 1, 3]);  view_619 = None
        clone_161: "bf16[128, 49, 32, 32]" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
        view_620: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(clone_161, _shape_param_821);  clone_161 = _shape_param_821 = None
        convert_element_type_382: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg337_1, torch.bfloat16);  arg337_1 = None
        convert_element_type_383: "bf16[1024, 1024]" = torch.ops.prims.convert_element_type.default(arg338_1, torch.bfloat16);  arg338_1 = None
        view_621: "bf16[6272, 1024]" = torch.ops.aten.view.default(view_620, _shape_param_822);  view_620 = _shape_param_822 = None
        permute_233: "bf16[1024, 1024]" = torch.ops.aten.permute.default(convert_element_type_383, [1, 0]);  convert_element_type_383 = None
        addmm_89: "bf16[6272, 1024]" = torch.ops.aten.addmm.default(convert_element_type_382, view_621, permute_233);  convert_element_type_382 = None
        view_622: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(addmm_89, _shape_param_823);  addmm_89 = _shape_param_823 = None
        view_623: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(view_622, _shape_param_824);  view_622 = _shape_param_824 = None
        view_624: "bf16[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.view.default(view_623, _shape_param_825);  view_623 = _shape_param_825 = None
        permute_234: "bf16[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_624, [0, 1, 3, 2, 4, 5]);  view_624 = None
        view_625: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(permute_234, _shape_param_826);  permute_234 = _shape_param_826 = None
        inductor_lookup_seed_42: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 42)
        inductor_random_42: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_827, inductor_lookup_seed_42, 'rand');  _shape_param_827 = inductor_lookup_seed_42 = None
        lt_42: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_42, 0.9043478220701218);  inductor_random_42 = None
        convert_element_type_384: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_42, torch.bfloat16)
        div_65: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_384, 0.9043478220701218);  convert_element_type_384 = None
        mul_229: "bf16[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_625, div_65);  view_625 = div_65 = None
        add_204: "bf16[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_605, mul_229);  view_605 = mul_229 = None
        view_626: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(add_204, _shape_param_828);  add_204 = _shape_param_828 = None
        convert_element_type_385: "f32[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(view_626, torch.float32)
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_385, [2], correction = 0, keepdim = True)
        getitem_167: "f32[128, 49, 1]" = var_mean_49[0]
        getitem_168: "f32[128, 49, 1]" = var_mean_49[1];  var_mean_49 = None
        add_205: "f32[128, 49, 1]" = torch.ops.aten.add.Tensor(getitem_167, 1e-05);  getitem_167 = None
        rsqrt_49: "f32[128, 49, 1]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
        sub_72: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_385, getitem_168);  convert_element_type_385 = None
        mul_230: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_49);  sub_72 = None
        mul_231: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_230, arg339_1);  mul_230 = arg339_1 = None
        add_206: "f32[128, 49, 1024]" = torch.ops.aten.add.Tensor(mul_231, arg340_1);  mul_231 = arg340_1 = None
        convert_element_type_386: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg341_1, torch.bfloat16);  arg341_1 = None
        convert_element_type_387: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg342_1, torch.bfloat16);  arg342_1 = None
        convert_element_type_388: "bf16[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(add_206, torch.bfloat16);  add_206 = None
        view_627: "bf16[6272, 1024]" = torch.ops.aten.view.default(convert_element_type_388, _shape_param_829);  convert_element_type_388 = _shape_param_829 = None
        permute_235: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_387, [1, 0]);  convert_element_type_387 = None
        addmm_90: "bf16[6272, 4096]" = torch.ops.aten.addmm.default(convert_element_type_386, view_627, permute_235);  convert_element_type_386 = None
        view_628: "bf16[128, 49, 4096]" = torch.ops.aten.view.default(addmm_90, _shape_param_830);  _shape_param_830 = None
        convert_element_type_389: "f32[128, 49, 4096]" = torch.ops.prims.convert_element_type.default(view_628, torch.float32);  view_628 = None
        mul_232: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_389, 0.5)
        mul_233: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_389, 0.7071067811865476);  convert_element_type_389 = None
        erf_22: "f32[128, 49, 4096]" = torch.ops.aten.erf.default(mul_233);  mul_233 = None
        add_207: "f32[128, 49, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_234: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(mul_232, add_207);  mul_232 = add_207 = None
        convert_element_type_390: "bf16[128, 49, 4096]" = torch.ops.prims.convert_element_type.default(mul_234, torch.bfloat16);  mul_234 = None
        convert_element_type_391: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg343_1, torch.bfloat16);  arg343_1 = None
        convert_element_type_392: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg344_1, torch.bfloat16);  arg344_1 = None
        view_629: "bf16[6272, 4096]" = torch.ops.aten.view.default(convert_element_type_390, _shape_param_831);  convert_element_type_390 = _shape_param_831 = None
        permute_236: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_392, [1, 0]);  convert_element_type_392 = None
        addmm_91: "bf16[6272, 1024]" = torch.ops.aten.addmm.default(convert_element_type_391, view_629, permute_236);  convert_element_type_391 = None
        view_630: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(addmm_91, _shape_param_832);  addmm_91 = _shape_param_832 = None
        inductor_lookup_seed_43: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 43)
        inductor_random_43: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_833, inductor_lookup_seed_43, 'rand');  _shape_param_833 = inductor_lookup_seed_43 = None
        lt_43: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_43, 0.9043478220701218);  inductor_random_43 = None
        convert_element_type_393: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_43, torch.bfloat16)
        div_66: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_393, 0.9043478220701218);  convert_element_type_393 = None
        mul_235: "bf16[128, 49, 1024]" = torch.ops.aten.mul.Tensor(view_630, div_66);  view_630 = div_66 = None
        add_208: "bf16[128, 49, 1024]" = torch.ops.aten.add.Tensor(view_626, mul_235);  mul_235 = None
        view_631: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(add_208, _shape_param_834);  add_208 = _shape_param_834 = None
        convert_element_type_394: "f32[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(view_631, torch.float32)
        var_mean_50 = torch.ops.aten.var_mean.correction(convert_element_type_394, [3], correction = 0, keepdim = True)
        getitem_169: "f32[128, 7, 7, 1]" = var_mean_50[0]
        getitem_170: "f32[128, 7, 7, 1]" = var_mean_50[1];  var_mean_50 = None
        add_209: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_169, 1e-05);  getitem_169 = None
        rsqrt_50: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
        sub_73: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_394, getitem_170);  convert_element_type_394 = None
        mul_236: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_50);  sub_73 = None
        mul_237: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_236, arg345_1);  mul_236 = arg345_1 = None
        add_210: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_237, arg346_1);  mul_237 = arg346_1 = None
        view_632: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.view.default(add_210, _shape_param_835);  add_210 = _shape_param_835 = None
        permute_237: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_632, [0, 1, 3, 2, 4, 5]);  view_632 = None
        view_633: "f32[128, 7, 7, 1024]" = torch.ops.aten.view.default(permute_237, _shape_param_836);  permute_237 = _shape_param_836 = None
        view_634: "f32[128, 49, 1024]" = torch.ops.aten.view.default(view_633, _shape_param_837);  view_633 = _shape_param_837 = None
        convert_element_type_395: "bf16[3072]" = torch.ops.prims.convert_element_type.default(arg347_1, torch.bfloat16);  arg347_1 = None
        convert_element_type_396: "bf16[3072, 1024]" = torch.ops.prims.convert_element_type.default(arg348_1, torch.bfloat16);  arg348_1 = None
        convert_element_type_397: "bf16[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(view_634, torch.bfloat16);  view_634 = None
        view_635: "bf16[6272, 1024]" = torch.ops.aten.view.default(convert_element_type_397, _shape_param_838);  convert_element_type_397 = _shape_param_838 = None
        permute_238: "bf16[1024, 3072]" = torch.ops.aten.permute.default(convert_element_type_396, [1, 0]);  convert_element_type_396 = None
        addmm_92: "bf16[6272, 3072]" = torch.ops.aten.addmm.default(convert_element_type_395, view_635, permute_238);  convert_element_type_395 = None
        view_636: "bf16[128, 49, 3072]" = torch.ops.aten.view.default(addmm_92, _shape_param_839);  addmm_92 = _shape_param_839 = None
        view_637: "bf16[128, 49, 3, 32, 32]" = torch.ops.aten.view.default(view_636, _shape_param_840);  view_636 = _shape_param_840 = None
        permute_239: "bf16[3, 128, 32, 49, 32]" = torch.ops.aten.permute.default(view_637, [2, 0, 3, 1, 4]);  view_637 = None
        unbind_23 = torch.ops.aten.unbind.int(permute_239);  permute_239 = None
        getitem_171: "bf16[128, 32, 49, 32]" = unbind_23[0]
        getitem_172: "bf16[128, 32, 49, 32]" = unbind_23[1]
        getitem_173: "bf16[128, 32, 49, 32]" = unbind_23[2];  unbind_23 = None
        mul_238: "bf16[128, 32, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_171, 0.1767766952966369);  getitem_171 = None
        permute_240: "bf16[128, 32, 32, 49]" = torch.ops.aten.permute.default(getitem_172, [0, 1, 3, 2]);  getitem_172 = None
        expand_92: "bf16[128, 32, 49, 32]" = torch.ops.aten.expand.default(mul_238, _shape_param_841);  mul_238 = _shape_param_841 = None
        clone_162: "bf16[128, 32, 49, 32]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_638: "bf16[4096, 49, 32]" = torch.ops.aten.view.default(clone_162, _shape_param_842);  clone_162 = _shape_param_842 = None
        expand_93: "bf16[128, 32, 32, 49]" = torch.ops.aten.expand.default(permute_240, _shape_param_843);  permute_240 = _shape_param_843 = None
        clone_163: "bf16[128, 32, 32, 49]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_639: "bf16[4096, 32, 49]" = torch.ops.aten.view.default(clone_163, _shape_param_844);  clone_163 = _shape_param_844 = None
        bmm_46: "bf16[4096, 49, 49]" = torch.ops.aten.bmm.default(view_638, view_639)
        view_640: "bf16[128, 32, 49, 49]" = torch.ops.aten.view.default(bmm_46, _shape_param_845);  _shape_param_845 = None
        view_641: "i64[2401]" = torch.ops.aten.view.default(arg349_1, [-1]);  arg349_1 = None
        index_67: "f32[2401, 32]" = torch.ops.aten.index.Tensor(arg350_1, [view_641]);  arg350_1 = view_641 = None
        view_642: "f32[49, 49, 32]" = torch.ops.aten.view.default(index_67, _shape_param_846);  index_67 = _shape_param_846 = None
        permute_241: "f32[32, 49, 49]" = torch.ops.aten.permute.default(view_642, [2, 0, 1]);  view_642 = None
        clone_164: "f32[32, 49, 49]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
        unsqueeze_45: "f32[1, 32, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_164, 0);  clone_164 = None
        add_211: "f32[128, 32, 49, 49]" = torch.ops.aten.add.Tensor(view_640, unsqueeze_45);  view_640 = unsqueeze_45 = None
        amax_23: "f32[128, 32, 49, 1]" = torch.ops.aten.amax.default(add_211, [-1], True)
        sub_74: "f32[128, 32, 49, 49]" = torch.ops.aten.sub.Tensor(add_211, amax_23);  add_211 = None
        exp_23: "f32[128, 32, 49, 49]" = torch.ops.aten.exp.default(sub_74);  sub_74 = None
        sum_24: "f32[128, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_67: "f32[128, 32, 49, 49]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = None
        convert_element_type_398: "bf16[128, 32, 49, 49]" = torch.ops.prims.convert_element_type.default(div_67, torch.bfloat16);  div_67 = None
        expand_94: "bf16[128, 32, 49, 49]" = torch.ops.aten.expand.default(convert_element_type_398, _shape_param_847);  convert_element_type_398 = _shape_param_847 = None
        view_643: "bf16[4096, 49, 49]" = torch.ops.aten.view.default(expand_94, _shape_param_848);  expand_94 = _shape_param_848 = None
        expand_95: "bf16[128, 32, 49, 32]" = torch.ops.aten.expand.default(getitem_173, _shape_param_849);  getitem_173 = _shape_param_849 = None
        clone_165: "bf16[128, 32, 49, 32]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_644: "bf16[4096, 49, 32]" = torch.ops.aten.view.default(clone_165, _shape_param_850);  clone_165 = _shape_param_850 = None
        constant_pad_nd_90: "bf16[4096, 56, 56]" = torch.ops.aten.constant_pad_nd.default(view_643, _shape_param_851);  _shape_param_851 = None
        constant_pad_nd_91: "bf16[4096, 56, 32]" = torch.ops.aten.constant_pad_nd.default(view_644, _shape_param_852);  _shape_param_852 = None
        bmm_47: "bf16[4096, 56, 32]" = torch.ops.aten.bmm.default(constant_pad_nd_90, constant_pad_nd_91);  constant_pad_nd_90 = constant_pad_nd_91 = None
        slice_68: "bf16[4096, 49, 32]" = torch.ops.aten.slice.Tensor(bmm_47, 1, 0, -7);  bmm_47 = None
        view_645: "bf16[128, 32, 49, 32]" = torch.ops.aten.view.default(slice_68, _shape_param_853);  slice_68 = _shape_param_853 = None
        permute_242: "bf16[128, 49, 32, 32]" = torch.ops.aten.permute.default(view_645, [0, 2, 1, 3]);  view_645 = None
        clone_166: "bf16[128, 49, 32, 32]" = torch.ops.aten.clone.default(permute_242, memory_format = torch.contiguous_format);  permute_242 = None
        view_646: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(clone_166, _shape_param_854);  clone_166 = _shape_param_854 = None
        convert_element_type_399: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg351_1, torch.bfloat16);  arg351_1 = None
        convert_element_type_400: "bf16[1024, 1024]" = torch.ops.prims.convert_element_type.default(arg352_1, torch.bfloat16);  arg352_1 = None
        view_647: "bf16[6272, 1024]" = torch.ops.aten.view.default(view_646, _shape_param_855);  view_646 = _shape_param_855 = None
        permute_243: "bf16[1024, 1024]" = torch.ops.aten.permute.default(convert_element_type_400, [1, 0]);  convert_element_type_400 = None
        addmm_93: "bf16[6272, 1024]" = torch.ops.aten.addmm.default(convert_element_type_399, view_647, permute_243);  convert_element_type_399 = None
        view_648: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(addmm_93, _shape_param_856);  addmm_93 = _shape_param_856 = None
        view_649: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(view_648, _shape_param_857);  view_648 = _shape_param_857 = None
        view_650: "bf16[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.view.default(view_649, _shape_param_858);  view_649 = _shape_param_858 = None
        permute_244: "bf16[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_650, [0, 1, 3, 2, 4, 5]);  view_650 = None
        view_651: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(permute_244, _shape_param_859);  permute_244 = _shape_param_859 = None
        inductor_lookup_seed_44: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 44)
        inductor_random_44: "f32[128, 1, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_860, inductor_lookup_seed_44, 'rand');  _shape_param_860 = inductor_lookup_seed_44 = None
        lt_44: "b8[128, 1, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_44, 0.8999999985098839);  inductor_random_44 = None
        convert_element_type_401: "bf16[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_44, torch.bfloat16)
        div_68: "bf16[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_401, 0.8999999985098839);  convert_element_type_401 = None
        mul_239: "bf16[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_651, div_68);  view_651 = div_68 = None
        add_212: "bf16[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_631, mul_239);  mul_239 = None
        view_652: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(add_212, _shape_param_861);  add_212 = _shape_param_861 = None
        convert_element_type_402: "f32[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(view_652, torch.float32)
        var_mean_51 = torch.ops.aten.var_mean.correction(convert_element_type_402, [2], correction = 0, keepdim = True)
        getitem_174: "f32[128, 49, 1]" = var_mean_51[0]
        getitem_175: "f32[128, 49, 1]" = var_mean_51[1];  var_mean_51 = None
        add_213: "f32[128, 49, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_51: "f32[128, 49, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
        sub_75: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_402, getitem_175);  convert_element_type_402 = None
        mul_240: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_51);  sub_75 = None
        mul_241: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_240, arg353_1);  mul_240 = arg353_1 = None
        add_214: "f32[128, 49, 1024]" = torch.ops.aten.add.Tensor(mul_241, arg354_1);  mul_241 = arg354_1 = None
        convert_element_type_403: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg355_1, torch.bfloat16);  arg355_1 = None
        convert_element_type_404: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg356_1, torch.bfloat16);  arg356_1 = None
        convert_element_type_405: "bf16[128, 49, 1024]" = torch.ops.prims.convert_element_type.default(add_214, torch.bfloat16);  add_214 = None
        view_653: "bf16[6272, 1024]" = torch.ops.aten.view.default(convert_element_type_405, _shape_param_862);  convert_element_type_405 = _shape_param_862 = None
        permute_245: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_404, [1, 0]);  convert_element_type_404 = None
        addmm_94: "bf16[6272, 4096]" = torch.ops.aten.addmm.default(convert_element_type_403, view_653, permute_245);  convert_element_type_403 = None
        view_654: "bf16[128, 49, 4096]" = torch.ops.aten.view.default(addmm_94, _shape_param_863);  _shape_param_863 = None
        convert_element_type_406: "f32[128, 49, 4096]" = torch.ops.prims.convert_element_type.default(view_654, torch.float32);  view_654 = None
        mul_242: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_406, 0.5)
        mul_243: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_406, 0.7071067811865476);  convert_element_type_406 = None
        erf_23: "f32[128, 49, 4096]" = torch.ops.aten.erf.default(mul_243);  mul_243 = None
        add_215: "f32[128, 49, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_244: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(mul_242, add_215);  mul_242 = add_215 = None
        convert_element_type_407: "bf16[128, 49, 4096]" = torch.ops.prims.convert_element_type.default(mul_244, torch.bfloat16);  mul_244 = None
        convert_element_type_408: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg357_1, torch.bfloat16);  arg357_1 = None
        convert_element_type_409: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg358_1, torch.bfloat16);  arg358_1 = None
        view_655: "bf16[6272, 4096]" = torch.ops.aten.view.default(convert_element_type_407, _shape_param_864);  convert_element_type_407 = _shape_param_864 = None
        permute_246: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_409, [1, 0]);  convert_element_type_409 = None
        addmm_95: "bf16[6272, 1024]" = torch.ops.aten.addmm.default(convert_element_type_408, view_655, permute_246);  convert_element_type_408 = None
        view_656: "bf16[128, 49, 1024]" = torch.ops.aten.view.default(addmm_95, _shape_param_865);  addmm_95 = _shape_param_865 = None
        inductor_lookup_seed_45: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 45);  inductor_seeds = None
        inductor_random_45: "f32[128, 1, 1]" = torch.ops.prims.inductor_random.default(_shape_param_866, inductor_lookup_seed_45, 'rand');  _shape_param_866 = inductor_lookup_seed_45 = None
        lt_45: "b8[128, 1, 1]" = torch.ops.aten.lt.Scalar(inductor_random_45, 0.8999999985098839);  inductor_random_45 = None
        convert_element_type_410: "bf16[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_45, torch.bfloat16)
        div_69: "bf16[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_410, 0.8999999985098839);  convert_element_type_410 = None
        mul_245: "bf16[128, 49, 1024]" = torch.ops.aten.mul.Tensor(view_656, div_69);  view_656 = div_69 = None
        add_216: "bf16[128, 49, 1024]" = torch.ops.aten.add.Tensor(view_652, mul_245);  mul_245 = None
        view_657: "bf16[128, 7, 7, 1024]" = torch.ops.aten.view.default(add_216, _shape_param_867);  add_216 = _shape_param_867 = None
        convert_element_type_411: "f32[128, 7, 7, 1024]" = torch.ops.prims.convert_element_type.default(view_657, torch.float32)
        var_mean_52 = torch.ops.aten.var_mean.correction(convert_element_type_411, [3], correction = 0, keepdim = True)
        getitem_176: "f32[128, 7, 7, 1]" = var_mean_52[0]
        getitem_177: "f32[128, 7, 7, 1]" = var_mean_52[1];  var_mean_52 = None
        add_217: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_52: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
        sub_76: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_411, getitem_177);  convert_element_type_411 = None
        mul_246: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_52);  sub_76 = None
        mul_247: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_246, arg359_1);  mul_246 = arg359_1 = None
        add_218: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_247, arg360_1);  mul_247 = arg360_1 = None
        mean: "f32[128, 1024]" = torch.ops.aten.mean.dim(add_218, [1, 2]);  add_218 = None
        convert_element_type_412: "bf16[1000]" = torch.ops.prims.convert_element_type.default(arg361_1, torch.bfloat16);  arg361_1 = None
        convert_element_type_413: "bf16[1000, 1024]" = torch.ops.prims.convert_element_type.default(arg362_1, torch.bfloat16);  arg362_1 = None
        convert_element_type_414: "bf16[128, 1024]" = torch.ops.prims.convert_element_type.default(mean, torch.bfloat16);  mean = None
        permute_247: "bf16[1024, 1000]" = torch.ops.aten.permute.default(convert_element_type_413, [1, 0]);  convert_element_type_413 = None
        addmm_96: "bf16[128, 1000]" = torch.ops.aten.addmm.default(convert_element_type_412, convert_element_type_414, permute_247);  convert_element_type_412 = None
        permute_248: "bf16[1000, 1024]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
        permute_249: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
        permute_250: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_245, [1, 0]);  permute_245 = None
        permute_251: "bf16[1024, 1024]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
        permute_252: "bf16[4096, 49, 49]" = torch.ops.aten.permute.default(view_643, [0, 2, 1]);  view_643 = None
        permute_253: "bf16[4096, 32, 49]" = torch.ops.aten.permute.default(view_644, [0, 2, 1]);  view_644 = None
        permute_254: "bf16[4096, 32, 49]" = torch.ops.aten.permute.default(view_638, [0, 2, 1]);  view_638 = None
        permute_255: "bf16[4096, 49, 32]" = torch.ops.aten.permute.default(view_639, [0, 2, 1]);  view_639 = None
        permute_256: "bf16[3072, 1024]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
        permute_257: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
        permute_258: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
        permute_259: "bf16[1024, 1024]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        permute_260: "bf16[4096, 49, 49]" = torch.ops.aten.permute.default(view_617, [0, 2, 1]);  view_617 = None
        permute_261: "bf16[4096, 32, 49]" = torch.ops.aten.permute.default(view_618, [0, 2, 1]);  view_618 = None
        permute_262: "bf16[4096, 32, 49]" = torch.ops.aten.permute.default(view_612, [0, 2, 1]);  view_612 = None
        permute_263: "bf16[4096, 49, 32]" = torch.ops.aten.permute.default(view_613, [0, 2, 1]);  view_613 = None
        permute_264: "bf16[3072, 1024]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
        permute_265: "bf16[1024, 2048]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
        permute_266: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
        permute_267: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
        permute_268: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
        permute_269: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_587, [0, 2, 1]);  view_587 = None
        permute_270: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_588, [0, 2, 1]);  view_588 = None
        permute_271: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_580, [0, 2, 1]);  view_580 = None
        permute_272: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_581, [0, 2, 1]);  view_581 = None
        permute_273: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
        permute_274: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
        permute_275: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
        permute_276: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        permute_277: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_559, [0, 2, 1]);  view_559 = None
        permute_278: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_560, [0, 2, 1]);  view_560 = None
        permute_279: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_554, [0, 2, 1]);  view_554 = None
        permute_280: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_555, [0, 2, 1]);  view_555 = None
        permute_281: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
        permute_282: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_204, [1, 0]);  permute_204 = None
        permute_283: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
        permute_284: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
        permute_285: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_533, [0, 2, 1]);  view_533 = None
        permute_286: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_534, [0, 2, 1]);  view_534 = None
        permute_287: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_526, [0, 2, 1]);  view_526 = None
        permute_288: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_527, [0, 2, 1]);  view_527 = None
        permute_289: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
        permute_290: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
        permute_291: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
        permute_292: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
        permute_293: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_505, [0, 2, 1]);  view_505 = None
        permute_294: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_506, [0, 2, 1]);  view_506 = None
        permute_295: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_500, [0, 2, 1]);  view_500 = None
        permute_296: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_501, [0, 2, 1]);  view_501 = None
        permute_297: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        permute_298: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
        permute_299: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
        permute_300: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
        permute_301: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_479, [0, 2, 1]);  view_479 = None
        permute_302: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_480, [0, 2, 1]);  view_480 = None
        permute_303: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_472, [0, 2, 1]);  view_472 = None
        permute_304: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_473, [0, 2, 1]);  view_473 = None
        permute_305: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        permute_306: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        permute_307: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        permute_308: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None
        permute_309: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_451, [0, 2, 1]);  view_451 = None
        permute_310: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_452, [0, 2, 1]);  view_452 = None
        permute_311: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_446, [0, 2, 1]);  view_446 = None
        permute_312: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_447, [0, 2, 1]);  view_447 = None
        permute_313: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
        permute_314: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
        permute_315: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        permute_316: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
        permute_317: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_425, [0, 2, 1]);  view_425 = None
        permute_318: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_426, [0, 2, 1]);  view_426 = None
        permute_319: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_418, [0, 2, 1]);  view_418 = None
        permute_320: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_419, [0, 2, 1]);  view_419 = None
        permute_321: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        permute_322: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
        permute_323: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        permute_324: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        permute_325: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_397, [0, 2, 1]);  view_397 = None
        permute_326: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_398, [0, 2, 1]);  view_398 = None
        permute_327: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_392, [0, 2, 1]);  view_392 = None
        permute_328: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_393, [0, 2, 1]);  view_393 = None
        permute_329: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
        permute_330: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        permute_331: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        permute_332: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        permute_333: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_371, [0, 2, 1]);  view_371 = None
        permute_334: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_372, [0, 2, 1]);  view_372 = None
        permute_335: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_364, [0, 2, 1]);  view_364 = None
        permute_336: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None
        permute_337: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        permute_338: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        permute_339: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        permute_340: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        permute_341: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_343, [0, 2, 1]);  view_343 = None
        permute_342: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_344, [0, 2, 1]);  view_344 = None
        permute_343: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_338, [0, 2, 1]);  view_338 = None
        permute_344: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None
        permute_345: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
        permute_346: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
        permute_347: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        permute_348: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        permute_349: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_317, [0, 2, 1]);  view_317 = None
        permute_350: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_318, [0, 2, 1]);  view_318 = None
        permute_351: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_310, [0, 2, 1]);  view_310 = None
        permute_352: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None
        permute_353: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
        permute_354: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
        permute_355: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
        permute_356: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        permute_357: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_289, [0, 2, 1]);  view_289 = None
        permute_358: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_290, [0, 2, 1]);  view_290 = None
        permute_359: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_284, [0, 2, 1]);  view_284 = None
        permute_360: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None
        permute_361: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
        permute_362: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
        permute_363: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        permute_364: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        permute_365: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_263, [0, 2, 1]);  view_263 = None
        permute_366: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_264, [0, 2, 1]);  view_264 = None
        permute_367: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None
        permute_368: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_257, [0, 2, 1]);  view_257 = None
        permute_369: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        permute_370: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
        permute_371: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
        permute_372: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
        permute_373: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
        permute_374: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_236, [0, 2, 1]);  view_236 = None
        permute_375: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
        permute_376: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
        permute_377: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        permute_378: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
        permute_379: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
        permute_380: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        permute_381: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
        permute_382: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        permute_383: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_202, [0, 2, 1]);  view_202 = None
        permute_384: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None
        permute_385: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        permute_386: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        permute_387: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
        permute_388: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        permute_389: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_181, [0, 2, 1]);  view_181 = None
        permute_390: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_182, [0, 2, 1]);  view_182 = None
        permute_391: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None
        permute_392: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_177, [0, 2, 1]);  view_177 = None
        permute_393: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        permute_394: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        permute_395: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        permute_396: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
        permute_397: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
        permute_398: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_156, [0, 2, 1]);  view_156 = None
        permute_399: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_148, [0, 2, 1]);  view_148 = None
        permute_400: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_149, [0, 2, 1]);  view_149 = None
        permute_401: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        permute_402: "bf16[512, 2048]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        permute_403: "bf16[2048, 512]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        permute_404: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
        permute_405: "bf16[8192, 49, 49]" = torch.ops.aten.permute.default(view_127, [0, 2, 1]);  view_127 = None
        permute_406: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_128, [0, 2, 1]);  view_128 = None
        permute_407: "bf16[8192, 32, 49]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        permute_408: "bf16[8192, 49, 32]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        permute_409: "bf16[1536, 512]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        permute_410: "bf16[512, 1024]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        permute_411: "bf16[256, 1024]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        permute_412: "bf16[1024, 256]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        permute_413: "bf16[256, 256]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
        permute_414: "bf16[16384, 49, 49]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        permute_415: "bf16[16384, 32, 49]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
        permute_416: "bf16[16384, 32, 49]" = torch.ops.aten.permute.default(view_90, [0, 2, 1]);  view_90 = None
        permute_417: "bf16[16384, 49, 32]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
        permute_418: "bf16[768, 256]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        permute_419: "bf16[256, 1024]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        permute_420: "bf16[1024, 256]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        permute_421: "bf16[256, 256]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        permute_422: "bf16[16384, 49, 49]" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
        permute_423: "bf16[16384, 32, 49]" = torch.ops.aten.permute.default(view_70, [0, 2, 1]);  view_70 = None
        permute_424: "bf16[16384, 32, 49]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
        permute_425: "bf16[16384, 49, 32]" = torch.ops.aten.permute.default(view_65, [0, 2, 1]);  view_65 = None
        permute_426: "bf16[768, 256]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        permute_427: "bf16[256, 512]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        div_70: "f32[128, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 512);  rsqrt_5 = None
        permute_428: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        permute_429: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        div_71: "f32[128, 3136, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 128);  rsqrt_4 = None
        permute_430: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
        permute_431: "bf16[32768, 49, 49]" = torch.ops.aten.permute.default(view_39, [0, 2, 1]);  view_39 = None
        permute_432: "bf16[32768, 32, 49]" = torch.ops.aten.permute.default(view_40, [0, 2, 1]);  view_40 = None
        permute_433: "bf16[32768, 32, 49]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
        permute_434: "bf16[32768, 49, 32]" = torch.ops.aten.permute.default(view_33, [0, 2, 1]);  view_33 = None
        permute_435: "bf16[384, 128]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        div_72: "f32[128, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 128);  rsqrt_3 = None
        permute_436: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_437: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        div_73: "f32[128, 3136, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None
        permute_438: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        permute_439: "bf16[32768, 49, 49]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        permute_440: "bf16[32768, 32, 49]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_441: "bf16[32768, 32, 49]" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
        permute_442: "bf16[32768, 49, 32]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        permute_443: "bf16[384, 128]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        return (getitem_1, rsqrt, getitem_3, rsqrt_1, view_3, bmm, amax, sum_1, view_15, mul_5, view_21, addmm_2, view_23, mul_10, view_29, bmm_2, amax_1, sum_2, view_43, fmod_1, lt, mul_14, view_49, addmm_6, view_51, lt_1, mul_20, view_56, mm, getitem_19, rsqrt_6, view_61, bmm_4, amax_2, sum_3, view_73, lt_2, view_78, getitem_24, rsqrt_7, view_79, addmm_10, view_81, lt_3, view_83, getitem_26, rsqrt_8, view_87, bmm_6, amax_3, sum_4, view_101, fmod_3, lt_4, view_106, getitem_31, rsqrt_9, view_107, addmm_14, view_109, lt_5, view_113, getitem_33, rsqrt_10, view_114, mm_1, getitem_35, rsqrt_11, view_119, bmm_8, amax_4, sum_5, view_131, lt_6, view_136, getitem_40, rsqrt_12, view_137, addmm_18, view_139, lt_7, view_141, getitem_42, rsqrt_13, view_145, bmm_10, amax_5, sum_6, view_159, fmod_5, lt_8, view_164, getitem_47, rsqrt_14, view_165, addmm_22, view_167, lt_9, view_169, getitem_49, rsqrt_15, view_173, bmm_12, amax_6, sum_7, view_185, lt_10, view_190, getitem_54, rsqrt_16, view_191, addmm_26, view_193, lt_11, view_195, getitem_56, rsqrt_17, view_199, bmm_14, amax_7, sum_8, view_213, lt_12, view_218, getitem_61, rsqrt_18, view_219, addmm_30, view_221, lt_13, view_223, getitem_63, rsqrt_19, view_227, bmm_16, amax_8, sum_9, view_239, lt_14, view_244, getitem_68, rsqrt_20, view_245, addmm_34, view_247, lt_15, view_249, getitem_70, rsqrt_21, view_253, bmm_18, amax_9, sum_10, view_267, lt_16, view_272, getitem_75, rsqrt_22, view_273, addmm_38, view_275, lt_17, view_277, getitem_77, rsqrt_23, view_281, bmm_20, amax_10, sum_11, view_293, lt_18, view_298, getitem_82, rsqrt_24, view_299, addmm_42, view_301, lt_19, view_303, getitem_84, rsqrt_25, view_307, bmm_22, amax_11, sum_12, view_321, lt_20, view_326, getitem_89, rsqrt_26, view_327, addmm_46, view_329, lt_21, view_331, getitem_91, rsqrt_27, view_335, bmm_24, amax_12, sum_13, view_347, lt_22, view_352, getitem_96, rsqrt_28, view_353, addmm_50, view_355, lt_23, view_357, getitem_98, rsqrt_29, view_361, bmm_26, amax_13, sum_14, view_375, lt_24, view_380, getitem_103, rsqrt_30, view_381, addmm_54, view_383, lt_25, view_385, getitem_105, rsqrt_31, view_389, bmm_28, amax_14, sum_15, view_401, lt_26, view_406, getitem_110, rsqrt_32, view_407, addmm_58, view_409, lt_27, view_411, getitem_112, rsqrt_33, view_415, bmm_30, amax_15, sum_16, view_429, lt_28, view_434, getitem_117, rsqrt_34, view_435, addmm_62, view_437, lt_29, view_439, getitem_119, rsqrt_35, view_443, bmm_32, amax_16, sum_17, view_455, lt_30, view_460, getitem_124, rsqrt_36, view_461, addmm_66, view_463, lt_31, view_465, getitem_126, rsqrt_37, view_469, bmm_34, amax_17, sum_18, view_483, lt_32, view_488, getitem_131, rsqrt_38, view_489, addmm_70, view_491, lt_33, view_493, getitem_133, rsqrt_39, view_497, bmm_36, amax_18, sum_19, view_509, lt_34, view_514, getitem_138, rsqrt_40, view_515, addmm_74, view_517, lt_35, view_519, getitem_140, rsqrt_41, view_523, bmm_38, amax_19, sum_20, view_537, lt_36, view_542, getitem_145, rsqrt_42, view_543, addmm_78, view_545, lt_37, view_547, getitem_147, rsqrt_43, view_551, bmm_40, amax_20, sum_21, view_563, lt_38, view_568, getitem_152, rsqrt_44, view_569, addmm_82, view_571, lt_39, view_573, getitem_154, rsqrt_45, view_577, bmm_42, amax_21, sum_22, view_591, lt_40, view_596, getitem_159, rsqrt_46, view_597, addmm_86, view_599, lt_41, view_603, getitem_161, rsqrt_47, view_604, mm_2, getitem_163, rsqrt_48, view_609, bmm_44, amax_22, sum_23, view_621, lt_42, view_626, getitem_168, rsqrt_49, view_627, addmm_90, view_629, lt_43, view_631, getitem_170, rsqrt_50, view_635, bmm_46, amax_23, sum_24, view_647, lt_44, view_652, getitem_175, rsqrt_51, view_653, addmm_94, view_655, lt_45, view_657, getitem_177, rsqrt_52, convert_element_type_414, addmm_96, permute_248, permute_249, permute_250, permute_251, permute_252, permute_253, permute_254, permute_255, permute_256, permute_257, permute_258, permute_259, permute_260, permute_261, permute_262, permute_263, permute_264, permute_265, permute_266, permute_267, permute_268, permute_269, permute_270, permute_271, permute_272, permute_273, permute_274, permute_275, permute_276, permute_277, permute_278, permute_279, permute_280, permute_281, permute_282, permute_283, permute_284, permute_285, permute_286, permute_287, permute_288, permute_289, permute_290, permute_291, permute_292, permute_293, permute_294, permute_295, permute_296, permute_297, permute_298, permute_299, permute_300, permute_301, permute_302, permute_303, permute_304, permute_305, permute_306, permute_307, permute_308, permute_309, permute_310, permute_311, permute_312, permute_313, permute_314, permute_315, permute_316, permute_317, permute_318, permute_319, permute_320, permute_321, permute_322, permute_323, permute_324, permute_325, permute_326, permute_327, permute_328, permute_329, permute_330, permute_331, permute_332, permute_333, permute_334, permute_335, permute_336, permute_337, permute_338, permute_339, permute_340, permute_341, permute_342, permute_343, permute_344, permute_345, permute_346, permute_347, permute_348, permute_349, permute_350, permute_351, permute_352, permute_353, permute_354, permute_355, permute_356, permute_357, permute_358, permute_359, permute_360, permute_361, permute_362, permute_363, permute_364, permute_365, permute_366, permute_367, permute_368, permute_369, permute_370, permute_371, permute_372, permute_373, permute_374, permute_375, permute_376, permute_377, permute_378, permute_379, permute_380, permute_381, permute_382, permute_383, permute_384, permute_385, permute_386, permute_387, permute_388, permute_389, permute_390, permute_391, permute_392, permute_393, permute_394, permute_395, permute_396, permute_397, permute_398, permute_399, permute_400, permute_401, permute_402, permute_403, permute_404, permute_405, permute_406, permute_407, permute_408, permute_409, permute_410, permute_411, permute_412, permute_413, permute_414, permute_415, permute_416, permute_417, permute_418, permute_419, permute_420, permute_421, permute_422, permute_423, permute_424, permute_425, permute_426, permute_427, div_70, permute_428, permute_429, div_71, permute_430, permute_431, permute_432, permute_433, permute_434, permute_435, div_72, permute_436, permute_437, div_73, permute_438, permute_439, permute_440, permute_441, permute_442, permute_443)



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
