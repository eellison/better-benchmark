"""
Standalone repro captured via capture_hook.
Label: BertForMaskedLM
Pattern hash: f81f34c94db3
Shape hash: e3913720
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
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "i64[32, 512]", arg3_1: "bf16[32, 512, 30522]", arg4_1: "f32[16384, 1]", arg5_1: "f32[16384, 1]", arg6_1: "bf16[32, 512, 30522]", arg7_1: "bf16[30522, 768]", arg8_1: "bf16[16384, 768]", arg9_1: "f32[768]", arg10_1: "bf16[16384, 768]", arg11_1: "f32[32, 512, 1]", arg12_1: "f32[32, 512, 1]", arg13_1: "bf16[768, 768]", arg14_1: "bf16[16384, 768]", arg15_1: "f32[768]", arg16_1: "f32[32, 512, 768]", arg17_1: "f32[32, 512, 1]", arg18_1: "b8[32, 512, 768]", arg19_1: "bf16[768, 3072]", arg20_1: "bf16[16384, 3072]", arg21_1: "bf16[16384, 3072]", arg22_1: "bf16[3072, 768]", arg23_1: "bf16[16384, 768]", arg24_1: "f32[768]", arg25_1: "f32[32, 512, 768]", arg26_1: "f32[32, 512, 1]", arg27_1: "b8[32, 512, 768]", arg28_1: "bf16[768, 768]", arg29_1: "bf16[16384, 768]", arg30_1: "bf16[384, 512, 512]", arg31_1: "bf16[384, 64, 512]", arg32_1: "b8[32, 12, 512, 512]", arg33_1: "bf16[32, 12, 512, 512]", arg34_1: "bf16[384, 64, 512]", arg35_1: "bf16[384, 512, 64]", arg36_1: "bf16[768, 768]", arg37_1: "bf16[16384, 768]", arg38_1: "bf16[768, 768]", arg39_1: "bf16[768, 768]", arg40_1: "f32[768]", arg41_1: "f32[32, 512, 768]", arg42_1: "f32[32, 512, 1]", arg43_1: "b8[32, 512, 768]", arg44_1: "bf16[768, 3072]", arg45_1: "bf16[16384, 3072]", arg46_1: "bf16[16384, 3072]", arg47_1: "bf16[3072, 768]", arg48_1: "bf16[16384, 768]", arg49_1: "f32[768]", arg50_1: "f32[32, 512, 768]", arg51_1: "f32[32, 512, 1]", arg52_1: "b8[32, 512, 768]", arg53_1: "bf16[768, 768]", arg54_1: "bf16[16384, 768]", arg55_1: "bf16[384, 512, 512]", arg56_1: "bf16[384, 64, 512]", arg57_1: "b8[32, 12, 512, 512]", arg58_1: "bf16[32, 12, 512, 512]", arg59_1: "bf16[384, 64, 512]", arg60_1: "bf16[384, 512, 64]", arg61_1: "bf16[768, 768]", arg62_1: "bf16[16384, 768]", arg63_1: "bf16[768, 768]", arg64_1: "bf16[768, 768]", arg65_1: "f32[768]", arg66_1: "f32[32, 512, 768]", arg67_1: "f32[32, 512, 1]", arg68_1: "b8[32, 512, 768]", arg69_1: "bf16[768, 3072]", arg70_1: "bf16[16384, 3072]", arg71_1: "bf16[16384, 3072]", arg72_1: "bf16[3072, 768]", arg73_1: "bf16[16384, 768]", arg74_1: "f32[768]", arg75_1: "f32[32, 512, 768]", arg76_1: "f32[32, 512, 1]", arg77_1: "b8[32, 512, 768]", arg78_1: "bf16[768, 768]", arg79_1: "bf16[16384, 768]", arg80_1: "bf16[384, 512, 512]", arg81_1: "bf16[384, 64, 512]", arg82_1: "b8[32, 12, 512, 512]", arg83_1: "bf16[32, 12, 512, 512]", arg84_1: "bf16[384, 64, 512]", arg85_1: "bf16[384, 512, 64]", arg86_1: "bf16[768, 768]", arg87_1: "bf16[16384, 768]", arg88_1: "bf16[768, 768]", arg89_1: "bf16[768, 768]", arg90_1: "f32[768]", arg91_1: "f32[32, 512, 768]", arg92_1: "f32[32, 512, 1]", arg93_1: "b8[32, 512, 768]", arg94_1: "bf16[768, 3072]", arg95_1: "bf16[16384, 3072]", arg96_1: "bf16[16384, 3072]", arg97_1: "bf16[3072, 768]", arg98_1: "bf16[16384, 768]", arg99_1: "f32[768]", arg100_1: "f32[32, 512, 768]", arg101_1: "f32[32, 512, 1]", arg102_1: "b8[32, 512, 768]", arg103_1: "bf16[768, 768]", arg104_1: "bf16[16384, 768]", arg105_1: "bf16[384, 512, 512]", arg106_1: "bf16[384, 64, 512]", arg107_1: "b8[32, 12, 512, 512]", arg108_1: "bf16[32, 12, 512, 512]", arg109_1: "bf16[384, 64, 512]", arg110_1: "bf16[384, 512, 64]", arg111_1: "bf16[768, 768]", arg112_1: "bf16[16384, 768]", arg113_1: "bf16[768, 768]", arg114_1: "bf16[768, 768]", arg115_1: "f32[768]", arg116_1: "f32[32, 512, 768]", arg117_1: "f32[32, 512, 1]", arg118_1: "b8[32, 512, 768]", arg119_1: "bf16[768, 3072]", arg120_1: "bf16[16384, 3072]", arg121_1: "bf16[16384, 3072]", arg122_1: "bf16[3072, 768]", arg123_1: "bf16[16384, 768]", arg124_1: "f32[768]", arg125_1: "f32[32, 512, 768]", arg126_1: "f32[32, 512, 1]", arg127_1: "b8[32, 512, 768]", arg128_1: "bf16[768, 768]", arg129_1: "bf16[16384, 768]", arg130_1: "bf16[384, 512, 512]", arg131_1: "bf16[384, 64, 512]", arg132_1: "b8[32, 12, 512, 512]", arg133_1: "bf16[32, 12, 512, 512]", arg134_1: "bf16[384, 64, 512]", arg135_1: "bf16[384, 512, 64]", arg136_1: "bf16[768, 768]", arg137_1: "bf16[16384, 768]", arg138_1: "bf16[768, 768]", arg139_1: "bf16[768, 768]", arg140_1: "f32[768]", arg141_1: "f32[32, 512, 768]", arg142_1: "f32[32, 512, 1]", arg143_1: "b8[32, 512, 768]", arg144_1: "bf16[768, 3072]", arg145_1: "bf16[16384, 3072]", arg146_1: "bf16[16384, 3072]", arg147_1: "bf16[3072, 768]", arg148_1: "bf16[16384, 768]", arg149_1: "f32[768]", arg150_1: "f32[32, 512, 768]", arg151_1: "f32[32, 512, 1]", arg152_1: "b8[32, 512, 768]", arg153_1: "bf16[768, 768]", arg154_1: "bf16[16384, 768]", arg155_1: "bf16[384, 512, 512]", arg156_1: "bf16[384, 64, 512]", arg157_1: "b8[32, 12, 512, 512]", arg158_1: "bf16[32, 12, 512, 512]", arg159_1: "bf16[384, 64, 512]", arg160_1: "bf16[384, 512, 64]", arg161_1: "bf16[768, 768]", arg162_1: "bf16[16384, 768]", arg163_1: "bf16[768, 768]", arg164_1: "bf16[768, 768]", arg165_1: "f32[768]", arg166_1: "f32[32, 512, 768]", arg167_1: "f32[32, 512, 1]", arg168_1: "b8[32, 512, 768]", arg169_1: "bf16[768, 3072]", arg170_1: "bf16[16384, 3072]", arg171_1: "bf16[16384, 3072]", arg172_1: "bf16[3072, 768]", arg173_1: "bf16[16384, 768]", arg174_1: "f32[768]", arg175_1: "f32[32, 512, 768]", arg176_1: "f32[32, 512, 1]", arg177_1: "b8[32, 512, 768]", arg178_1: "bf16[768, 768]", arg179_1: "bf16[16384, 768]", arg180_1: "bf16[384, 512, 512]", arg181_1: "bf16[384, 64, 512]", arg182_1: "b8[32, 12, 512, 512]", arg183_1: "bf16[32, 12, 512, 512]", arg184_1: "bf16[384, 64, 512]", arg185_1: "bf16[384, 512, 64]", arg186_1: "bf16[768, 768]", arg187_1: "bf16[16384, 768]", arg188_1: "bf16[768, 768]", arg189_1: "bf16[768, 768]", arg190_1: "f32[768]", arg191_1: "f32[32, 512, 768]", arg192_1: "f32[32, 512, 1]", arg193_1: "b8[32, 512, 768]", arg194_1: "bf16[768, 3072]", arg195_1: "bf16[16384, 3072]", arg196_1: "bf16[16384, 3072]", arg197_1: "bf16[3072, 768]", arg198_1: "bf16[16384, 768]", arg199_1: "f32[768]", arg200_1: "f32[32, 512, 768]", arg201_1: "f32[32, 512, 1]", arg202_1: "b8[32, 512, 768]", arg203_1: "bf16[768, 768]", arg204_1: "bf16[16384, 768]", arg205_1: "bf16[384, 512, 512]", arg206_1: "bf16[384, 64, 512]", arg207_1: "b8[32, 12, 512, 512]", arg208_1: "bf16[32, 12, 512, 512]", arg209_1: "bf16[384, 64, 512]", arg210_1: "bf16[384, 512, 64]", arg211_1: "bf16[768, 768]", arg212_1: "bf16[16384, 768]", arg213_1: "bf16[768, 768]", arg214_1: "bf16[768, 768]", arg215_1: "f32[768]", arg216_1: "f32[32, 512, 768]", arg217_1: "f32[32, 512, 1]", arg218_1: "b8[32, 512, 768]", arg219_1: "bf16[768, 3072]", arg220_1: "bf16[16384, 3072]", arg221_1: "bf16[16384, 3072]", arg222_1: "bf16[3072, 768]", arg223_1: "bf16[16384, 768]", arg224_1: "f32[768]", arg225_1: "f32[32, 512, 768]", arg226_1: "f32[32, 512, 1]", arg227_1: "b8[32, 512, 768]", arg228_1: "bf16[768, 768]", arg229_1: "bf16[16384, 768]", arg230_1: "bf16[384, 512, 512]", arg231_1: "bf16[384, 64, 512]", arg232_1: "b8[32, 12, 512, 512]", arg233_1: "bf16[32, 12, 512, 512]", arg234_1: "bf16[384, 64, 512]", arg235_1: "bf16[384, 512, 64]", arg236_1: "bf16[768, 768]", arg237_1: "bf16[16384, 768]", arg238_1: "bf16[768, 768]", arg239_1: "bf16[768, 768]", arg240_1: "f32[768]", arg241_1: "f32[32, 512, 768]", arg242_1: "f32[32, 512, 1]", arg243_1: "b8[32, 512, 768]", arg244_1: "bf16[768, 3072]", arg245_1: "bf16[16384, 3072]", arg246_1: "bf16[16384, 3072]", arg247_1: "bf16[3072, 768]", arg248_1: "bf16[16384, 768]", arg249_1: "f32[768]", arg250_1: "f32[32, 512, 768]", arg251_1: "f32[32, 512, 1]", arg252_1: "b8[32, 512, 768]", arg253_1: "bf16[768, 768]", arg254_1: "bf16[16384, 768]", arg255_1: "bf16[384, 512, 512]", arg256_1: "bf16[384, 64, 512]", arg257_1: "b8[32, 12, 512, 512]", arg258_1: "bf16[32, 12, 512, 512]", arg259_1: "bf16[384, 64, 512]", arg260_1: "bf16[384, 512, 64]", arg261_1: "bf16[768, 768]", arg262_1: "bf16[16384, 768]", arg263_1: "bf16[768, 768]", arg264_1: "bf16[768, 768]", arg265_1: "f32[768]", arg266_1: "f32[32, 512, 768]", arg267_1: "f32[32, 512, 1]", arg268_1: "b8[32, 512, 768]", arg269_1: "bf16[768, 3072]", arg270_1: "bf16[16384, 3072]", arg271_1: "bf16[16384, 3072]", arg272_1: "bf16[3072, 768]", arg273_1: "bf16[16384, 768]", arg274_1: "f32[768]", arg275_1: "f32[32, 512, 768]", arg276_1: "f32[32, 512, 1]", arg277_1: "b8[32, 512, 768]", arg278_1: "bf16[768, 768]", arg279_1: "bf16[16384, 768]", arg280_1: "bf16[384, 512, 512]", arg281_1: "bf16[384, 64, 512]", arg282_1: "b8[32, 12, 512, 512]", arg283_1: "bf16[32, 12, 512, 512]", arg284_1: "bf16[384, 64, 512]", arg285_1: "bf16[384, 512, 64]", arg286_1: "bf16[768, 768]", arg287_1: "bf16[16384, 768]", arg288_1: "bf16[768, 768]", arg289_1: "bf16[768, 768]", arg290_1: "f32[768]", arg291_1: "f32[32, 512, 768]", arg292_1: "f32[32, 512, 1]", arg293_1: "b8[32, 512, 768]", arg294_1: "bf16[768, 3072]", arg295_1: "bf16[16384, 3072]", arg296_1: "bf16[16384, 3072]", arg297_1: "bf16[3072, 768]", arg298_1: "bf16[16384, 768]", arg299_1: "f32[768]", arg300_1: "f32[32, 512, 768]", arg301_1: "f32[32, 512, 1]", arg302_1: "b8[32, 512, 768]", arg303_1: "bf16[768, 768]", arg304_1: "bf16[16384, 768]", arg305_1: "bf16[384, 512, 512]", arg306_1: "bf16[384, 64, 512]", arg307_1: "b8[32, 12, 512, 512]", arg308_1: "bf16[384, 512, 512]", arg309_1: "f32[32, 12, 512, 1]", arg310_1: "f32[32, 12, 512, 1]", arg311_1: "b8[32, 12, 512, 1]", arg312_1: "bf16[384, 64, 512]", arg313_1: "bf16[384, 512, 64]", arg314_1: "bf16[768, 768]", arg315_1: "bf16[16384, 768]", arg316_1: "bf16[768, 768]", arg317_1: "bf16[768, 768]", arg318_1: "b8[32, 512, 768]", arg319_1: "f32[768]", arg320_1: "f32[32, 512, 768]", arg321_1: "f32[32, 512, 1]", arg322_1: "i64[1, 512]", arg323_1: "i64[1, 512]", arg324_1: "i64[32, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "i64[16384]" = torch.ops.aten.view.default(arg2_1, [-1]);  arg2_1 = None
        unsqueeze: "i64[16384, 1]" = torch.ops.aten.unsqueeze.default(view, 1);  view = None
        ne: "b8[16384, 1]" = torch.ops.aten.ne.Scalar(unsqueeze, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[16384, 1]" = torch.ops.aten.where.self(ne, unsqueeze, full);  unsqueeze = full = None
        iota: "i64[30522]" = torch.ops.prims.iota.default(30522, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_1: "i64[1, 30522]" = torch.ops.aten.view.default(iota, _shape_param_0);  iota = _shape_param_0 = None
        expand: "i64[16384, 30522]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        eq: "b8[16384, 30522]" = torch.ops.aten.eq.Tensor(expand, view_1);  expand = view_1 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[16384, 30522]" = torch.ops.aten.where.self(eq, scalar_tensor_1, scalar_tensor);  eq = scalar_tensor_1 = scalar_tensor = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[16384, 1]" = torch.ops.aten.where.self(ne, div, full_1);  ne = div = full_1 = None
        mul: "f32[16384, 30522]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        convert_element_type: "bf16[16384, 30522]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_1: "f32[16384, 30522]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        view_2: "bf16[16384, 30522]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_2: "f32[16384, 30522]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        sub: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg4_1);  convert_element_type_2 = arg4_1 = None
        sub_1: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(sub, arg5_1);  sub = arg5_1 = None
        convert_element_type_3: "bf16[16384, 30522]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_4: "f32[16384, 30522]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        exp: "f32[16384, 30522]" = torch.ops.aten.exp.default(convert_element_type_4);  convert_element_type_4 = None
        sum_1: "f32[16384, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [1], True)
        mul_1: "f32[16384, 30522]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub_2: "f32[16384, 30522]" = torch.ops.aten.sub.Tensor(convert_element_type_1, mul_1);  convert_element_type_1 = mul_1 = None
        convert_element_type_5: "bf16[16384, 30522]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        view_3: "bf16[32, 512, 30522]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_3);  convert_element_type_5 = _shape_param_3 = None
        add: "bf16[32, 512, 30522]" = torch.ops.aten.add.Tensor(arg6_1, view_3);  arg6_1 = view_3 = None
        view_4: "bf16[16384, 30522]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        constant_pad_nd: "bf16[16384, 30528]" = torch.ops.aten.constant_pad_nd.default(view_4, _shape_param_5);  _shape_param_5 = None
        constant_pad_nd_1: "bf16[30528, 768]" = torch.ops.aten.constant_pad_nd.default(arg7_1, _shape_param_6);  arg7_1 = _shape_param_6 = None
        mm: "bf16[16384, 768]" = torch.ops.aten.mm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        permute: "bf16[30522, 16384]" = torch.ops.aten.permute.default(view_4, [1, 0])
        mm_1: "bf16[30522, 768]" = torch.ops.aten.mm.default(permute, arg8_1);  permute = arg8_1 = None
        sum_2: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32);  view_4 = None
        view_5: "f32[30522]" = torch.ops.aten.view.default(sum_2, _shape_param_7);  sum_2 = _shape_param_7 = None
        convert_element_type_6: "bf16[30522]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm, _shape_param_8);  mm = _shape_param_8 = None
        convert_element_type_7: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_8: "f32[30522, 768]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_9: "f32[30522]" = torch.ops.prims.convert_element_type.default(convert_element_type_6, torch.float32);  convert_element_type_6 = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, arg9_1);  arg9_1 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, 768)
        sum_3: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        view_7: "bf16[32, 512, 768]" = torch.ops.aten.view.default(arg10_1, _shape_param_9);  arg10_1 = _shape_param_9 = None
        convert_element_type_10: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.5)
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.7071067811865476)
        erf: "f32[32, 512, 768]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, add_1);  mul_4 = None
        convert_element_type_11: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        convert_element_type_12: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_11, torch.float32);  convert_element_type_11 = None
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_12, arg11_1);  convert_element_type_12 = arg11_1 = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub_3, arg12_1);  sub_3 = None
        mul_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, mul_7);  mul_2 = None
        sum_4: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_8, [2], True);  mul_8 = None
        mul_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_7, sum_4);  sum_4 = None
        sub_4: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_3, sum_3);  mul_3 = sum_3 = None
        sub_5: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_4, mul_9);  sub_4 = mul_9 = None
        div_1: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(arg12_1, 768);  arg12_1 = None
        mul_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div_1, sub_5);  div_1 = sub_5 = None
        mul_11: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, mul_7);  mul_7 = None
        sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_11, [0, 1]);  mul_11 = None
        sum_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_7, [0, 1]);  convert_element_type_7 = None
        convert_element_type_13: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        convert_element_type_14: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_13, torch.float32);  convert_element_type_13 = None
        mul_12: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_1, 0.5);  add_1 = None
        mul_13: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, convert_element_type_10)
        mul_14: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_13, -0.5);  mul_13 = None
        exp_1: "f32[32, 512, 768]" = torch.ops.aten.exp.default(mul_14);  mul_14 = None
        mul_15: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_16: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, mul_15);  convert_element_type_10 = mul_15 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_12, mul_16);  mul_12 = mul_16 = None
        mul_17: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, add_2);  convert_element_type_14 = add_2 = None
        convert_element_type_15: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        view_8: "bf16[16384, 768]" = torch.ops.aten.view.default(convert_element_type_15, _shape_param_10);  convert_element_type_15 = _shape_param_10 = None
        mm_2: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_8, arg13_1);  arg13_1 = None
        permute_1: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_8, [1, 0])
        mm_3: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_1, arg14_1);  permute_1 = arg14_1 = None
        sum_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True, dtype = torch.float32);  view_8 = None
        view_9: "f32[768]" = torch.ops.aten.view.default(sum_7, _shape_param_11);  sum_7 = _shape_param_11 = None
        convert_element_type_16: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_9, torch.bfloat16);  view_9 = None
        view_10: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_2, _shape_param_12);  mm_2 = _shape_param_12 = None
        convert_element_type_17: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        convert_element_type_18: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_19: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_16, torch.float32);  convert_element_type_16 = None
        mul_18: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, arg15_1);  arg15_1 = None
        mul_19: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, 768)
        sum_8: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_18, [2], True)
        mul_20: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, arg16_1);  mul_18 = None
        sum_9: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True);  mul_20 = None
        mul_21: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg16_1, sum_9);  sum_9 = None
        sub_6: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_19, sum_8);  mul_19 = sum_8 = None
        sub_7: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_6, mul_21);  sub_6 = mul_21 = None
        mul_22: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg17_1, sub_7);  arg17_1 = sub_7 = None
        mul_23: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, arg16_1);  arg16_1 = None
        sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_23, [0, 1]);  mul_23 = None
        sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_17, [0, 1]);  convert_element_type_17 = None
        convert_element_type_20: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16)
        convert_element_type_21: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.bfloat16);  arg18_1 = None
        mul_24: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
        mul_25: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, mul_24);  convert_element_type_20 = mul_24 = None
        view_11: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_25, _shape_param_13);  mul_25 = _shape_param_13 = None
        mm_4: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_11, arg19_1);  arg19_1 = None
        permute_2: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_11, [1, 0])
        mm_5: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_2, arg20_1);  permute_2 = arg20_1 = None
        sum_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_11, [0], True, dtype = torch.float32);  view_11 = None
        view_12: "f32[768]" = torch.ops.aten.view.default(sum_12, _shape_param_14);  sum_12 = _shape_param_14 = None
        convert_element_type_22: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_12, torch.bfloat16);  view_12 = None
        view_13: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_4, _shape_param_15);  mm_4 = _shape_param_15 = None
        convert_element_type_23: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_24: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_22, torch.float32);  convert_element_type_22 = None
        convert_element_type_25: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        view_14: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg21_1, _shape_param_16);  arg21_1 = _shape_param_16 = None
        convert_element_type_26: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_14, torch.float32);  view_14 = None
        mul_26: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.7071067811865476)
        erf_1: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_3: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_27: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_3, 0.5);  add_3 = None
        mul_28: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_26, convert_element_type_26)
        mul_29: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_28, -0.5);  mul_28 = None
        exp_2: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_29);  mul_29 = None
        mul_30: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_31: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_26, mul_30);  convert_element_type_26 = mul_30 = None
        add_4: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_27, mul_31);  mul_27 = mul_31 = None
        mul_32: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_25, add_4);  convert_element_type_25 = add_4 = None
        convert_element_type_27: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None
        view_15: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_27, _shape_param_17);  convert_element_type_27 = _shape_param_17 = None
        mm_6: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_15, arg22_1);  arg22_1 = None
        permute_3: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_15, [1, 0])
        mm_7: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_3, arg23_1);  permute_3 = arg23_1 = None
        sum_13: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_15, [0], True, dtype = torch.float32);  view_15 = None
        view_16: "f32[3072]" = torch.ops.aten.view.default(sum_13, _shape_param_18);  sum_13 = _shape_param_18 = None
        convert_element_type_28: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_16, torch.bfloat16);  view_16 = None
        view_17: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_6, _shape_param_19);  mm_6 = _shape_param_19 = None
        convert_element_type_29: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None
        add_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_22, convert_element_type_29);  mul_22 = convert_element_type_29 = None
        convert_element_type_30: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_31: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_28, torch.float32);  convert_element_type_28 = None
        mul_33: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_5, arg24_1);  arg24_1 = None
        mul_34: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, 768)
        sum_14: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_33, [2], True)
        mul_35: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_33, arg25_1);  mul_33 = None
        sum_15: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_35, [2], True);  mul_35 = None
        mul_36: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg25_1, sum_15);  sum_15 = None
        sub_8: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_34, sum_14);  mul_34 = sum_14 = None
        sub_9: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_8, mul_36);  sub_8 = mul_36 = None
        mul_37: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg26_1, sub_9);  arg26_1 = sub_9 = None
        mul_38: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_5, arg25_1);  arg25_1 = None
        sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_38, [0, 1]);  mul_38 = None
        sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_5, [0, 1]);  add_5 = None
        convert_element_type_32: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16)
        convert_element_type_33: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg27_1, torch.bfloat16);  arg27_1 = None
        mul_39: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
        mul_40: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_32, mul_39);  convert_element_type_32 = mul_39 = None
        view_18: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_40, _shape_param_20);  mul_40 = _shape_param_20 = None
        mm_8: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_18, arg28_1);  arg28_1 = None
        permute_4: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_18, [1, 0])
        mm_9: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_4, arg29_1);  permute_4 = arg29_1 = None
        sum_18: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_18, [0], True, dtype = torch.float32);  view_18 = None
        view_19: "f32[768]" = torch.ops.aten.view.default(sum_18, _shape_param_21);  sum_18 = _shape_param_21 = None
        convert_element_type_34: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_19, torch.bfloat16);  view_19 = None
        view_20: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_8, _shape_param_22);  mm_8 = _shape_param_22 = None
        convert_element_type_35: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_36: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_34, torch.float32);  convert_element_type_34 = None
        view_21: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_20, _shape_param_23);  view_20 = _shape_param_23 = None
        permute_5: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        clone: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_22: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_24);  clone = _shape_param_24 = None
        bmm: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg30_1, view_22);  arg30_1 = None
        bmm_1: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_22, arg31_1);  view_22 = arg31_1 = None
        view_23: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm, _shape_param_25);  bmm = _shape_param_25 = None
        view_24: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_1, _shape_param_26);  bmm_1 = _shape_param_26 = None
        convert_element_type_37: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg32_1, torch.bfloat16);  arg32_1 = None
        mul_41: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
        mul_42: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_24, mul_41);  view_24 = mul_41 = None
        convert_element_type_38: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_42, torch.float32);  mul_42 = None
        convert_element_type_39: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg33_1, torch.float32);  arg33_1 = None
        mul_43: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_38, convert_element_type_39);  convert_element_type_38 = None
        sum_19: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_43, [-1], True)
        neg: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_39);  convert_element_type_39 = None
        fma: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg, sum_19, mul_43);  neg = sum_19 = mul_43 = None
        convert_element_type_40: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_25: "bf16[384, 512, 512]" = torch.ops.aten.view.default(convert_element_type_40, _shape_param_27);  convert_element_type_40 = _shape_param_27 = None
        bmm_2: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg34_1, view_25);  arg34_1 = None
        bmm_3: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_25, arg35_1);  view_25 = arg35_1 = None
        view_26: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_2, _shape_param_28);  bmm_2 = _shape_param_28 = None
        view_27: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_29);  bmm_3 = _shape_param_29 = None
        mul_44: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_26, 0.3535533905932738);  view_26 = None
        permute_6: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_44, [0, 1, 3, 2]);  mul_44 = None
        mul_45: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_27, 0.3535533905932738);  view_27 = None
        permute_7: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
        clone_1: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_28: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_1, _shape_param_30);  clone_1 = _shape_param_30 = None
        view_29: "bf16[16384, 768]" = torch.ops.aten.view.default(view_28, _shape_param_31);  view_28 = _shape_param_31 = None
        mm_10: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_29, arg36_1);  arg36_1 = None
        permute_8: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_29, [1, 0])
        mm_11: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_8, arg37_1);  permute_8 = None
        sum_20: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_29, [0], True, dtype = torch.float32);  view_29 = None
        view_30: "f32[768]" = torch.ops.aten.view.default(sum_20, _shape_param_32);  sum_20 = _shape_param_32 = None
        convert_element_type_41: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_30, torch.bfloat16);  view_30 = None
        view_31: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_10, _shape_param_33);  mm_10 = _shape_param_33 = None
        convert_element_type_42: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_31, torch.float32);  view_31 = None
        add_6: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, convert_element_type_42);  mul_37 = convert_element_type_42 = None
        convert_element_type_43: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_44: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_41, torch.float32);  convert_element_type_41 = None
        permute_9: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1, 3]);  permute_6 = None
        view_32: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_9, _shape_param_34);  permute_9 = _shape_param_34 = None
        clone_2: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_32, memory_format = torch.contiguous_format);  view_32 = None
        view_33: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_2, _shape_param_35);  clone_2 = _shape_param_35 = None
        mm_12: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_33, arg38_1);  arg38_1 = None
        permute_10: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_33, [1, 0])
        mm_13: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_10, arg37_1);  permute_10 = None
        sum_21: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_33, [0], True, dtype = torch.float32);  view_33 = None
        view_34: "f32[768]" = torch.ops.aten.view.default(sum_21, _shape_param_36);  sum_21 = _shape_param_36 = None
        convert_element_type_45: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_34, torch.bfloat16);  view_34 = None
        view_35: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_12, _shape_param_37);  mm_12 = _shape_param_37 = None
        convert_element_type_46: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_35, torch.float32);  view_35 = None
        add_7: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_6, convert_element_type_46);  add_6 = convert_element_type_46 = None
        convert_element_type_47: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_48: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_45, torch.float32);  convert_element_type_45 = None
        permute_11: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_45, [0, 2, 1, 3]);  mul_45 = None
        clone_3: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        view_36: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_3, _shape_param_38);  clone_3 = _shape_param_38 = None
        view_37: "bf16[16384, 768]" = torch.ops.aten.view.default(view_36, _shape_param_39);  view_36 = _shape_param_39 = None
        mm_14: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_37, arg39_1);  arg39_1 = None
        permute_12: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_37, [1, 0])
        mm_15: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_12, arg37_1);  permute_12 = arg37_1 = None
        sum_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_37, [0], True, dtype = torch.float32);  view_37 = None
        view_38: "f32[768]" = torch.ops.aten.view.default(sum_22, _shape_param_40);  sum_22 = _shape_param_40 = None
        convert_element_type_49: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_38, torch.bfloat16);  view_38 = None
        view_39: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_14, _shape_param_41);  mm_14 = _shape_param_41 = None
        convert_element_type_50: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_39, torch.float32);  view_39 = None
        add_8: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_7, convert_element_type_50);  add_7 = convert_element_type_50 = None
        convert_element_type_51: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_52: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_49, torch.float32);  convert_element_type_49 = None
        mul_46: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_8, arg40_1);  arg40_1 = None
        mul_47: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_46, 768)
        sum_23: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_46, [2], True)
        mul_48: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_46, arg41_1);  mul_46 = None
        sum_24: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_48, [2], True);  mul_48 = None
        mul_49: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg41_1, sum_24);  sum_24 = None
        sub_10: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_47, sum_23);  mul_47 = sum_23 = None
        sub_11: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_10, mul_49);  sub_10 = mul_49 = None
        mul_50: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg42_1, sub_11);  arg42_1 = sub_11 = None
        mul_51: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_8, arg41_1);  arg41_1 = None
        sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_51, [0, 1]);  mul_51 = None
        sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_8, [0, 1]);  add_8 = None
        convert_element_type_53: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_50, torch.bfloat16)
        convert_element_type_54: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg43_1, torch.bfloat16);  arg43_1 = None
        mul_52: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 1.1111111111111112);  convert_element_type_54 = None
        mul_53: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_53, mul_52);  convert_element_type_53 = mul_52 = None
        view_40: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_53, _shape_param_42);  mul_53 = _shape_param_42 = None
        mm_16: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_40, arg44_1);  arg44_1 = None
        permute_13: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_40, [1, 0])
        mm_17: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_13, arg45_1);  permute_13 = arg45_1 = None
        sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_40, [0], True, dtype = torch.float32);  view_40 = None
        view_41: "f32[768]" = torch.ops.aten.view.default(sum_27, _shape_param_43);  sum_27 = _shape_param_43 = None
        convert_element_type_55: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_41, torch.bfloat16);  view_41 = None
        view_42: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_16, _shape_param_44);  mm_16 = _shape_param_44 = None
        convert_element_type_56: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_57: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_55, torch.float32);  convert_element_type_55 = None
        convert_element_type_58: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_42, torch.float32);  view_42 = None
        view_43: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg46_1, _shape_param_45);  arg46_1 = _shape_param_45 = None
        convert_element_type_59: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_43, torch.float32);  view_43 = None
        mul_54: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 0.7071067811865476)
        erf_2: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_9: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_55: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
        mul_56: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_59, convert_element_type_59)
        mul_57: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_56, -0.5);  mul_56 = None
        exp_3: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_57);  mul_57 = None
        mul_58: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
        mul_59: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_59, mul_58);  convert_element_type_59 = mul_58 = None
        add_10: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_55, mul_59);  mul_55 = mul_59 = None
        mul_60: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_58, add_10);  convert_element_type_58 = add_10 = None
        convert_element_type_60: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_60, torch.bfloat16);  mul_60 = None
        view_44: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_60, _shape_param_46);  convert_element_type_60 = _shape_param_46 = None
        mm_18: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_44, arg47_1);  arg47_1 = None
        permute_14: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_44, [1, 0])
        mm_19: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_14, arg48_1);  permute_14 = arg48_1 = None
        sum_28: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_44, [0], True, dtype = torch.float32);  view_44 = None
        view_45: "f32[3072]" = torch.ops.aten.view.default(sum_28, _shape_param_47);  sum_28 = _shape_param_47 = None
        convert_element_type_61: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_45, torch.bfloat16);  view_45 = None
        view_46: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_18, _shape_param_48);  mm_18 = _shape_param_48 = None
        convert_element_type_62: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_46, torch.float32);  view_46 = None
        add_11: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_50, convert_element_type_62);  mul_50 = convert_element_type_62 = None
        convert_element_type_63: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_64: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_61, torch.float32);  convert_element_type_61 = None
        mul_61: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_11, arg49_1);  arg49_1 = None
        mul_62: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_61, 768)
        sum_29: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_61, [2], True)
        mul_63: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_61, arg50_1);  mul_61 = None
        sum_30: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_63, [2], True);  mul_63 = None
        mul_64: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg50_1, sum_30);  sum_30 = None
        sub_12: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_62, sum_29);  mul_62 = sum_29 = None
        sub_13: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_12, mul_64);  sub_12 = mul_64 = None
        mul_65: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg51_1, sub_13);  arg51_1 = sub_13 = None
        mul_66: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_11, arg50_1);  arg50_1 = None
        sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_66, [0, 1]);  mul_66 = None
        sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_11, [0, 1]);  add_11 = None
        convert_element_type_65: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16)
        convert_element_type_66: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg52_1, torch.bfloat16);  arg52_1 = None
        mul_67: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
        mul_68: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_65, mul_67);  convert_element_type_65 = mul_67 = None
        view_47: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_68, _shape_param_49);  mul_68 = _shape_param_49 = None
        mm_20: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_47, arg53_1);  arg53_1 = None
        permute_15: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_47, [1, 0])
        mm_21: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_15, arg54_1);  permute_15 = arg54_1 = None
        sum_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_47, [0], True, dtype = torch.float32);  view_47 = None
        view_48: "f32[768]" = torch.ops.aten.view.default(sum_33, _shape_param_50);  sum_33 = _shape_param_50 = None
        convert_element_type_67: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_48, torch.bfloat16);  view_48 = None
        view_49: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_20, _shape_param_51);  mm_20 = _shape_param_51 = None
        convert_element_type_68: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_69: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_67, torch.float32);  convert_element_type_67 = None
        view_50: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_49, _shape_param_52);  view_49 = _shape_param_52 = None
        permute_16: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        clone_4: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_51: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_53);  clone_4 = _shape_param_53 = None
        bmm_4: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg55_1, view_51);  arg55_1 = None
        bmm_5: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_51, arg56_1);  view_51 = arg56_1 = None
        view_52: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_4, _shape_param_54);  bmm_4 = _shape_param_54 = None
        view_53: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_5, _shape_param_55);  bmm_5 = _shape_param_55 = None
        convert_element_type_70: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg57_1, torch.bfloat16);  arg57_1 = None
        mul_69: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 1.1111111111111112);  convert_element_type_70 = None
        mul_70: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_53, mul_69);  view_53 = mul_69 = None
        convert_element_type_71: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_70, torch.float32);  mul_70 = None
        convert_element_type_72: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg58_1, torch.float32);  arg58_1 = None
        mul_71: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_71, convert_element_type_72);  convert_element_type_71 = None
        sum_34: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_71, [-1], True)
        neg_1: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_72);  convert_element_type_72 = None
        fma_1: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_1, sum_34, mul_71);  neg_1 = sum_34 = mul_71 = None
        convert_element_type_73: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None
        view_54: "bf16[384, 512, 512]" = torch.ops.aten.view.default(convert_element_type_73, _shape_param_56);  convert_element_type_73 = _shape_param_56 = None
        bmm_6: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg59_1, view_54);  arg59_1 = None
        bmm_7: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_54, arg60_1);  view_54 = arg60_1 = None
        view_55: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_6, _shape_param_57);  bmm_6 = _shape_param_57 = None
        view_56: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_58);  bmm_7 = _shape_param_58 = None
        mul_72: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_55, 0.3535533905932738);  view_55 = None
        permute_17: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_72, [0, 1, 3, 2]);  mul_72 = None
        mul_73: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_56, 0.3535533905932738);  view_56 = None
        permute_18: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        clone_5: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_57: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_5, _shape_param_59);  clone_5 = _shape_param_59 = None
        view_58: "bf16[16384, 768]" = torch.ops.aten.view.default(view_57, _shape_param_60);  view_57 = _shape_param_60 = None
        mm_22: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_58, arg61_1);  arg61_1 = None
        permute_19: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_58, [1, 0])
        mm_23: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_19, arg62_1);  permute_19 = None
        sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_58, [0], True, dtype = torch.float32);  view_58 = None
        view_59: "f32[768]" = torch.ops.aten.view.default(sum_35, _shape_param_61);  sum_35 = _shape_param_61 = None
        convert_element_type_74: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_59, torch.bfloat16);  view_59 = None
        view_60: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_22, _shape_param_62);  mm_22 = _shape_param_62 = None
        convert_element_type_75: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_60, torch.float32);  view_60 = None
        add_12: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_65, convert_element_type_75);  mul_65 = convert_element_type_75 = None
        convert_element_type_76: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_77: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_74, torch.float32);  convert_element_type_74 = None
        permute_20: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_17, [0, 2, 1, 3]);  permute_17 = None
        view_61: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_20, _shape_param_63);  permute_20 = _shape_param_63 = None
        clone_6: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_61, memory_format = torch.contiguous_format);  view_61 = None
        view_62: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_6, _shape_param_64);  clone_6 = _shape_param_64 = None
        mm_24: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_62, arg63_1);  arg63_1 = None
        permute_21: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_62, [1, 0])
        mm_25: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_21, arg62_1);  permute_21 = None
        sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_62, [0], True, dtype = torch.float32);  view_62 = None
        view_63: "f32[768]" = torch.ops.aten.view.default(sum_36, _shape_param_65);  sum_36 = _shape_param_65 = None
        convert_element_type_78: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_63, torch.bfloat16);  view_63 = None
        view_64: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_24, _shape_param_66);  mm_24 = _shape_param_66 = None
        convert_element_type_79: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_64, torch.float32);  view_64 = None
        add_13: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_12, convert_element_type_79);  add_12 = convert_element_type_79 = None
        convert_element_type_80: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_81: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_78, torch.float32);  convert_element_type_78 = None
        permute_22: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_73, [0, 2, 1, 3]);  mul_73 = None
        clone_7: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
        view_65: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_7, _shape_param_67);  clone_7 = _shape_param_67 = None
        view_66: "bf16[16384, 768]" = torch.ops.aten.view.default(view_65, _shape_param_68);  view_65 = _shape_param_68 = None
        mm_26: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_66, arg64_1);  arg64_1 = None
        permute_23: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_66, [1, 0])
        mm_27: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_23, arg62_1);  permute_23 = arg62_1 = None
        sum_37: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_66, [0], True, dtype = torch.float32);  view_66 = None
        view_67: "f32[768]" = torch.ops.aten.view.default(sum_37, _shape_param_69);  sum_37 = _shape_param_69 = None
        convert_element_type_82: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_67, torch.bfloat16);  view_67 = None
        view_68: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_26, _shape_param_70);  mm_26 = _shape_param_70 = None
        convert_element_type_83: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_68, torch.float32);  view_68 = None
        add_14: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_13, convert_element_type_83);  add_13 = convert_element_type_83 = None
        convert_element_type_84: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_85: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_82, torch.float32);  convert_element_type_82 = None
        mul_74: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_14, arg65_1);  arg65_1 = None
        mul_75: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, 768)
        sum_38: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_74, [2], True)
        mul_76: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, arg66_1);  mul_74 = None
        sum_39: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_76, [2], True);  mul_76 = None
        mul_77: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg66_1, sum_39);  sum_39 = None
        sub_14: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_75, sum_38);  mul_75 = sum_38 = None
        sub_15: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_14, mul_77);  sub_14 = mul_77 = None
        mul_78: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg67_1, sub_15);  arg67_1 = sub_15 = None
        mul_79: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_14, arg66_1);  arg66_1 = None
        sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_79, [0, 1]);  mul_79 = None
        sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_14, [0, 1]);  add_14 = None
        convert_element_type_86: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_78, torch.bfloat16)
        convert_element_type_87: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg68_1, torch.bfloat16);  arg68_1 = None
        mul_80: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_87, 1.1111111111111112);  convert_element_type_87 = None
        mul_81: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_86, mul_80);  convert_element_type_86 = mul_80 = None
        view_69: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_81, _shape_param_71);  mul_81 = _shape_param_71 = None
        mm_28: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_69, arg69_1);  arg69_1 = None
        permute_24: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_69, [1, 0])
        mm_29: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_24, arg70_1);  permute_24 = arg70_1 = None
        sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_69, [0], True, dtype = torch.float32);  view_69 = None
        view_70: "f32[768]" = torch.ops.aten.view.default(sum_42, _shape_param_72);  sum_42 = _shape_param_72 = None
        convert_element_type_88: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_70, torch.bfloat16);  view_70 = None
        view_71: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_28, _shape_param_73);  mm_28 = _shape_param_73 = None
        convert_element_type_89: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_90: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_88, torch.float32);  convert_element_type_88 = None
        convert_element_type_91: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_71, torch.float32);  view_71 = None
        view_72: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg71_1, _shape_param_74);  arg71_1 = _shape_param_74 = None
        convert_element_type_92: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_72, torch.float32);  view_72 = None
        mul_82: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_92, 0.7071067811865476)
        erf_3: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_15: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_83: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_15, 0.5);  add_15 = None
        mul_84: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_92, convert_element_type_92)
        mul_85: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_84, -0.5);  mul_84 = None
        exp_4: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_85);  mul_85 = None
        mul_86: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_87: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_92, mul_86);  convert_element_type_92 = mul_86 = None
        add_16: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_83, mul_87);  mul_83 = mul_87 = None
        mul_88: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_91, add_16);  convert_element_type_91 = add_16 = None
        convert_element_type_93: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_88, torch.bfloat16);  mul_88 = None
        view_73: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_93, _shape_param_75);  convert_element_type_93 = _shape_param_75 = None
        mm_30: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_73, arg72_1);  arg72_1 = None
        permute_25: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_73, [1, 0])
        mm_31: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_25, arg73_1);  permute_25 = arg73_1 = None
        sum_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_73, [0], True, dtype = torch.float32);  view_73 = None
        view_74: "f32[3072]" = torch.ops.aten.view.default(sum_43, _shape_param_76);  sum_43 = _shape_param_76 = None
        convert_element_type_94: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_74, torch.bfloat16);  view_74 = None
        view_75: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_30, _shape_param_77);  mm_30 = _shape_param_77 = None
        convert_element_type_95: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_75, torch.float32);  view_75 = None
        add_17: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_78, convert_element_type_95);  mul_78 = convert_element_type_95 = None
        convert_element_type_96: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_97: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_94, torch.float32);  convert_element_type_94 = None
        mul_89: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_17, arg74_1);  arg74_1 = None
        mul_90: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_89, 768)
        sum_44: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_89, [2], True)
        mul_91: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_89, arg75_1);  mul_89 = None
        sum_45: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_91, [2], True);  mul_91 = None
        mul_92: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg75_1, sum_45);  sum_45 = None
        sub_16: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_90, sum_44);  mul_90 = sum_44 = None
        sub_17: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_16, mul_92);  sub_16 = mul_92 = None
        mul_93: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg76_1, sub_17);  arg76_1 = sub_17 = None
        mul_94: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_17, arg75_1);  arg75_1 = None
        sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_94, [0, 1]);  mul_94 = None
        sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_17, [0, 1]);  add_17 = None
        convert_element_type_98: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16)
        convert_element_type_99: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg77_1, torch.bfloat16);  arg77_1 = None
        mul_95: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_99, 1.1111111111111112);  convert_element_type_99 = None
        mul_96: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_98, mul_95);  convert_element_type_98 = mul_95 = None
        view_76: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_96, _shape_param_78);  mul_96 = _shape_param_78 = None
        mm_32: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_76, arg78_1);  arg78_1 = None
        permute_26: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_76, [1, 0])
        mm_33: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_26, arg79_1);  permute_26 = arg79_1 = None
        sum_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_76, [0], True, dtype = torch.float32);  view_76 = None
        view_77: "f32[768]" = torch.ops.aten.view.default(sum_48, _shape_param_79);  sum_48 = _shape_param_79 = None
        convert_element_type_100: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_77, torch.bfloat16);  view_77 = None
        view_78: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_32, _shape_param_80);  mm_32 = _shape_param_80 = None
        convert_element_type_101: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_102: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_100, torch.float32);  convert_element_type_100 = None
        view_79: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_78, _shape_param_81);  view_78 = _shape_param_81 = None
        permute_27: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None
        clone_8: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_80: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_82);  clone_8 = _shape_param_82 = None
        bmm_8: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg80_1, view_80);  arg80_1 = None
        bmm_9: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_80, arg81_1);  view_80 = arg81_1 = None
        view_81: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_8, _shape_param_83);  bmm_8 = _shape_param_83 = None
        view_82: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_9, _shape_param_84);  bmm_9 = _shape_param_84 = None
        convert_element_type_103: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg82_1, torch.bfloat16);  arg82_1 = None
        mul_97: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_103, 1.1111111111111112);  convert_element_type_103 = None
        mul_98: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_82, mul_97);  view_82 = mul_97 = None
        convert_element_type_104: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_98, torch.float32);  mul_98 = None
        convert_element_type_105: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg83_1, torch.float32);  arg83_1 = None
        mul_99: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_104, convert_element_type_105);  convert_element_type_104 = None
        sum_49: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_99, [-1], True)
        neg_2: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_105);  convert_element_type_105 = None
        fma_2: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_2, sum_49, mul_99);  neg_2 = sum_49 = mul_99 = None
        convert_element_type_106: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None
        view_83: "bf16[384, 512, 512]" = torch.ops.aten.view.default(convert_element_type_106, _shape_param_85);  convert_element_type_106 = _shape_param_85 = None
        bmm_10: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg84_1, view_83);  arg84_1 = None
        bmm_11: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_83, arg85_1);  view_83 = arg85_1 = None
        view_84: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_10, _shape_param_86);  bmm_10 = _shape_param_86 = None
        view_85: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_87);  bmm_11 = _shape_param_87 = None
        mul_100: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_84, 0.3535533905932738);  view_84 = None
        permute_28: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_100, [0, 1, 3, 2]);  mul_100 = None
        mul_101: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_85, 0.3535533905932738);  view_85 = None
        permute_29: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None
        clone_9: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_86: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_9, _shape_param_88);  clone_9 = _shape_param_88 = None
        view_87: "bf16[16384, 768]" = torch.ops.aten.view.default(view_86, _shape_param_89);  view_86 = _shape_param_89 = None
        mm_34: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_87, arg86_1);  arg86_1 = None
        permute_30: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_87, [1, 0])
        mm_35: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_30, arg87_1);  permute_30 = None
        sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_87, [0], True, dtype = torch.float32);  view_87 = None
        view_88: "f32[768]" = torch.ops.aten.view.default(sum_50, _shape_param_90);  sum_50 = _shape_param_90 = None
        convert_element_type_107: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_88, torch.bfloat16);  view_88 = None
        view_89: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_34, _shape_param_91);  mm_34 = _shape_param_91 = None
        convert_element_type_108: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_89, torch.float32);  view_89 = None
        add_18: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_93, convert_element_type_108);  mul_93 = convert_element_type_108 = None
        convert_element_type_109: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_110: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_107, torch.float32);  convert_element_type_107 = None
        permute_31: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_28, [0, 2, 1, 3]);  permute_28 = None
        view_90: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_31, _shape_param_92);  permute_31 = _shape_param_92 = None
        clone_10: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_90, memory_format = torch.contiguous_format);  view_90 = None
        view_91: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_10, _shape_param_93);  clone_10 = _shape_param_93 = None
        mm_36: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_91, arg88_1);  arg88_1 = None
        permute_32: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_91, [1, 0])
        mm_37: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_32, arg87_1);  permute_32 = None
        sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_91, [0], True, dtype = torch.float32);  view_91 = None
        view_92: "f32[768]" = torch.ops.aten.view.default(sum_51, _shape_param_94);  sum_51 = _shape_param_94 = None
        convert_element_type_111: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_92, torch.bfloat16);  view_92 = None
        view_93: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_36, _shape_param_95);  mm_36 = _shape_param_95 = None
        convert_element_type_112: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        add_19: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_18, convert_element_type_112);  add_18 = convert_element_type_112 = None
        convert_element_type_113: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_114: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_111, torch.float32);  convert_element_type_111 = None
        permute_33: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_101, [0, 2, 1, 3]);  mul_101 = None
        clone_11: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        view_94: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_11, _shape_param_96);  clone_11 = _shape_param_96 = None
        view_95: "bf16[16384, 768]" = torch.ops.aten.view.default(view_94, _shape_param_97);  view_94 = _shape_param_97 = None
        mm_38: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_95, arg89_1);  arg89_1 = None
        permute_34: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_95, [1, 0])
        mm_39: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_34, arg87_1);  permute_34 = arg87_1 = None
        sum_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_95, [0], True, dtype = torch.float32);  view_95 = None
        view_96: "f32[768]" = torch.ops.aten.view.default(sum_52, _shape_param_98);  sum_52 = _shape_param_98 = None
        convert_element_type_115: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_96, torch.bfloat16);  view_96 = None
        view_97: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_38, _shape_param_99);  mm_38 = _shape_param_99 = None
        convert_element_type_116: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_97, torch.float32);  view_97 = None
        add_20: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_19, convert_element_type_116);  add_19 = convert_element_type_116 = None
        convert_element_type_117: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_118: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_115, torch.float32);  convert_element_type_115 = None
        mul_102: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_20, arg90_1);  arg90_1 = None
        mul_103: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, 768)
        sum_53: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_102, [2], True)
        mul_104: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_102, arg91_1);  mul_102 = None
        sum_54: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_104, [2], True);  mul_104 = None
        mul_105: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg91_1, sum_54);  sum_54 = None
        sub_18: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_103, sum_53);  mul_103 = sum_53 = None
        sub_19: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_18, mul_105);  sub_18 = mul_105 = None
        mul_106: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg92_1, sub_19);  arg92_1 = sub_19 = None
        mul_107: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_20, arg91_1);  arg91_1 = None
        sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_107, [0, 1]);  mul_107 = None
        sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_20, [0, 1]);  add_20 = None
        convert_element_type_119: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_106, torch.bfloat16)
        convert_element_type_120: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg93_1, torch.bfloat16);  arg93_1 = None
        mul_108: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_120, 1.1111111111111112);  convert_element_type_120 = None
        mul_109: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_119, mul_108);  convert_element_type_119 = mul_108 = None
        view_98: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_109, _shape_param_100);  mul_109 = _shape_param_100 = None
        mm_40: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_98, arg94_1);  arg94_1 = None
        permute_35: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_98, [1, 0])
        mm_41: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_35, arg95_1);  permute_35 = arg95_1 = None
        sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_98, [0], True, dtype = torch.float32);  view_98 = None
        view_99: "f32[768]" = torch.ops.aten.view.default(sum_57, _shape_param_101);  sum_57 = _shape_param_101 = None
        convert_element_type_121: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_99, torch.bfloat16);  view_99 = None
        view_100: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_40, _shape_param_102);  mm_40 = _shape_param_102 = None
        convert_element_type_122: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_123: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_121, torch.float32);  convert_element_type_121 = None
        convert_element_type_124: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_100, torch.float32);  view_100 = None
        view_101: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg96_1, _shape_param_103);  arg96_1 = _shape_param_103 = None
        convert_element_type_125: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_101, torch.float32);  view_101 = None
        mul_110: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_125, 0.7071067811865476)
        erf_4: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_110);  mul_110 = None
        add_21: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_111: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_21, 0.5);  add_21 = None
        mul_112: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_125, convert_element_type_125)
        mul_113: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_112, -0.5);  mul_112 = None
        exp_5: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_113);  mul_113 = None
        mul_114: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
        mul_115: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_125, mul_114);  convert_element_type_125 = mul_114 = None
        add_22: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_111, mul_115);  mul_111 = mul_115 = None
        mul_116: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_124, add_22);  convert_element_type_124 = add_22 = None
        convert_element_type_126: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_116, torch.bfloat16);  mul_116 = None
        view_102: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_126, _shape_param_104);  convert_element_type_126 = _shape_param_104 = None
        mm_42: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_102, arg97_1);  arg97_1 = None
        permute_36: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_102, [1, 0])
        mm_43: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_36, arg98_1);  permute_36 = arg98_1 = None
        sum_58: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_102, [0], True, dtype = torch.float32);  view_102 = None
        view_103: "f32[3072]" = torch.ops.aten.view.default(sum_58, _shape_param_105);  sum_58 = _shape_param_105 = None
        convert_element_type_127: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_103, torch.bfloat16);  view_103 = None
        view_104: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_42, _shape_param_106);  mm_42 = _shape_param_106 = None
        convert_element_type_128: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_104, torch.float32);  view_104 = None
        add_23: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_106, convert_element_type_128);  mul_106 = convert_element_type_128 = None
        convert_element_type_129: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_130: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_127, torch.float32);  convert_element_type_127 = None
        mul_117: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_23, arg99_1);  arg99_1 = None
        mul_118: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_117, 768)
        sum_59: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_117, [2], True)
        mul_119: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_117, arg100_1);  mul_117 = None
        sum_60: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_119, [2], True);  mul_119 = None
        mul_120: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg100_1, sum_60);  sum_60 = None
        sub_20: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_118, sum_59);  mul_118 = sum_59 = None
        sub_21: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_20, mul_120);  sub_20 = mul_120 = None
        mul_121: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg101_1, sub_21);  arg101_1 = sub_21 = None
        mul_122: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_23, arg100_1);  arg100_1 = None
        sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_122, [0, 1]);  mul_122 = None
        sum_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_23, [0, 1]);  add_23 = None
        convert_element_type_131: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_121, torch.bfloat16)
        convert_element_type_132: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg102_1, torch.bfloat16);  arg102_1 = None
        mul_123: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_132, 1.1111111111111112);  convert_element_type_132 = None
        mul_124: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_131, mul_123);  convert_element_type_131 = mul_123 = None
        view_105: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_124, _shape_param_107);  mul_124 = _shape_param_107 = None
        mm_44: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_105, arg103_1);  arg103_1 = None
        permute_37: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_105, [1, 0])
        mm_45: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_37, arg104_1);  permute_37 = arg104_1 = None
        sum_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_105, [0], True, dtype = torch.float32);  view_105 = None
        view_106: "f32[768]" = torch.ops.aten.view.default(sum_63, _shape_param_108);  sum_63 = _shape_param_108 = None
        convert_element_type_133: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_106, torch.bfloat16);  view_106 = None
        view_107: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_44, _shape_param_109);  mm_44 = _shape_param_109 = None
        convert_element_type_134: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_135: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_133, torch.float32);  convert_element_type_133 = None
        view_108: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_107, _shape_param_110);  view_107 = _shape_param_110 = None
        permute_38: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
        clone_12: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_109: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_111);  clone_12 = _shape_param_111 = None
        bmm_12: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg105_1, view_109);  arg105_1 = None
        bmm_13: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_109, arg106_1);  view_109 = arg106_1 = None
        view_110: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_12, _shape_param_112);  bmm_12 = _shape_param_112 = None
        view_111: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_13, _shape_param_113);  bmm_13 = _shape_param_113 = None
        convert_element_type_136: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg107_1, torch.bfloat16);  arg107_1 = None
        mul_125: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_136, 1.1111111111111112);  convert_element_type_136 = None
        mul_126: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_111, mul_125);  view_111 = mul_125 = None
        convert_element_type_137: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_126, torch.float32);  mul_126 = None
        convert_element_type_138: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg108_1, torch.float32);  arg108_1 = None
        mul_127: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_137, convert_element_type_138);  convert_element_type_137 = None
        sum_64: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_127, [-1], True)
        neg_3: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_138);  convert_element_type_138 = None
        fma_3: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_3, sum_64, mul_127);  neg_3 = sum_64 = mul_127 = None
        convert_element_type_139: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None
        view_112: "bf16[384, 512, 512]" = torch.ops.aten.view.default(convert_element_type_139, _shape_param_114);  convert_element_type_139 = _shape_param_114 = None
        bmm_14: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg109_1, view_112);  arg109_1 = None
        bmm_15: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_112, arg110_1);  view_112 = arg110_1 = None
        view_113: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_14, _shape_param_115);  bmm_14 = _shape_param_115 = None
        view_114: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_116);  bmm_15 = _shape_param_116 = None
        mul_128: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_113, 0.3535533905932738);  view_113 = None
        permute_39: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_128, [0, 1, 3, 2]);  mul_128 = None
        mul_129: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_114, 0.3535533905932738);  view_114 = None
        permute_40: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
        clone_13: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_115: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_13, _shape_param_117);  clone_13 = _shape_param_117 = None
        view_116: "bf16[16384, 768]" = torch.ops.aten.view.default(view_115, _shape_param_118);  view_115 = _shape_param_118 = None
        mm_46: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_116, arg111_1);  arg111_1 = None
        permute_41: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_116, [1, 0])
        mm_47: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_41, arg112_1);  permute_41 = None
        sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_116, [0], True, dtype = torch.float32);  view_116 = None
        view_117: "f32[768]" = torch.ops.aten.view.default(sum_65, _shape_param_119);  sum_65 = _shape_param_119 = None
        convert_element_type_140: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_117, torch.bfloat16);  view_117 = None
        view_118: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_46, _shape_param_120);  mm_46 = _shape_param_120 = None
        convert_element_type_141: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_118, torch.float32);  view_118 = None
        add_24: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_121, convert_element_type_141);  mul_121 = convert_element_type_141 = None
        convert_element_type_142: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_143: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_140, torch.float32);  convert_element_type_140 = None
        permute_42: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_39, [0, 2, 1, 3]);  permute_39 = None
        view_119: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_42, _shape_param_121);  permute_42 = _shape_param_121 = None
        clone_14: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_119, memory_format = torch.contiguous_format);  view_119 = None
        view_120: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_14, _shape_param_122);  clone_14 = _shape_param_122 = None
        mm_48: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_120, arg113_1);  arg113_1 = None
        permute_43: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_120, [1, 0])
        mm_49: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_43, arg112_1);  permute_43 = None
        sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_120, [0], True, dtype = torch.float32);  view_120 = None
        view_121: "f32[768]" = torch.ops.aten.view.default(sum_66, _shape_param_123);  sum_66 = _shape_param_123 = None
        convert_element_type_144: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_121, torch.bfloat16);  view_121 = None
        view_122: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_48, _shape_param_124);  mm_48 = _shape_param_124 = None
        convert_element_type_145: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_122, torch.float32);  view_122 = None
        add_25: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_24, convert_element_type_145);  add_24 = convert_element_type_145 = None
        convert_element_type_146: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_147: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_144, torch.float32);  convert_element_type_144 = None
        permute_44: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_129, [0, 2, 1, 3]);  mul_129 = None
        clone_15: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_123: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_15, _shape_param_125);  clone_15 = _shape_param_125 = None
        view_124: "bf16[16384, 768]" = torch.ops.aten.view.default(view_123, _shape_param_126);  view_123 = _shape_param_126 = None
        mm_50: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_124, arg114_1);  arg114_1 = None
        permute_45: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_124, [1, 0])
        mm_51: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_45, arg112_1);  permute_45 = arg112_1 = None
        sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_124, [0], True, dtype = torch.float32);  view_124 = None
        view_125: "f32[768]" = torch.ops.aten.view.default(sum_67, _shape_param_127);  sum_67 = _shape_param_127 = None
        convert_element_type_148: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_125, torch.bfloat16);  view_125 = None
        view_126: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_50, _shape_param_128);  mm_50 = _shape_param_128 = None
        convert_element_type_149: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_126, torch.float32);  view_126 = None
        add_26: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_25, convert_element_type_149);  add_25 = convert_element_type_149 = None
        convert_element_type_150: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_151: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_148, torch.float32);  convert_element_type_148 = None
        mul_130: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_26, arg115_1);  arg115_1 = None
        mul_131: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_130, 768)
        sum_68: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_130, [2], True)
        mul_132: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_130, arg116_1);  mul_130 = None
        sum_69: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_132, [2], True);  mul_132 = None
        mul_133: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg116_1, sum_69);  sum_69 = None
        sub_22: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_131, sum_68);  mul_131 = sum_68 = None
        sub_23: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_22, mul_133);  sub_22 = mul_133 = None
        mul_134: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg117_1, sub_23);  arg117_1 = sub_23 = None
        mul_135: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_26, arg116_1);  arg116_1 = None
        sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_135, [0, 1]);  mul_135 = None
        sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_26, [0, 1]);  add_26 = None
        convert_element_type_152: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_134, torch.bfloat16)
        convert_element_type_153: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg118_1, torch.bfloat16);  arg118_1 = None
        mul_136: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_153, 1.1111111111111112);  convert_element_type_153 = None
        mul_137: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_152, mul_136);  convert_element_type_152 = mul_136 = None
        view_127: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_137, _shape_param_129);  mul_137 = _shape_param_129 = None
        mm_52: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_127, arg119_1);  arg119_1 = None
        permute_46: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_127, [1, 0])
        mm_53: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_46, arg120_1);  permute_46 = arg120_1 = None
        sum_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_127, [0], True, dtype = torch.float32);  view_127 = None
        view_128: "f32[768]" = torch.ops.aten.view.default(sum_72, _shape_param_130);  sum_72 = _shape_param_130 = None
        convert_element_type_154: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_128, torch.bfloat16);  view_128 = None
        view_129: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_52, _shape_param_131);  mm_52 = _shape_param_131 = None
        convert_element_type_155: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_156: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_154, torch.float32);  convert_element_type_154 = None
        convert_element_type_157: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        view_130: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg121_1, _shape_param_132);  arg121_1 = _shape_param_132 = None
        convert_element_type_158: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_130, torch.float32);  view_130 = None
        mul_138: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_158, 0.7071067811865476)
        erf_5: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_138);  mul_138 = None
        add_27: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_139: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_27, 0.5);  add_27 = None
        mul_140: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_158, convert_element_type_158)
        mul_141: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_140, -0.5);  mul_140 = None
        exp_6: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_141);  mul_141 = None
        mul_142: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_143: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_158, mul_142);  convert_element_type_158 = mul_142 = None
        add_28: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_139, mul_143);  mul_139 = mul_143 = None
        mul_144: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_157, add_28);  convert_element_type_157 = add_28 = None
        convert_element_type_159: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_144, torch.bfloat16);  mul_144 = None
        view_131: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_159, _shape_param_133);  convert_element_type_159 = _shape_param_133 = None
        mm_54: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_131, arg122_1);  arg122_1 = None
        permute_47: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_131, [1, 0])
        mm_55: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_47, arg123_1);  permute_47 = arg123_1 = None
        sum_73: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_131, [0], True, dtype = torch.float32);  view_131 = None
        view_132: "f32[3072]" = torch.ops.aten.view.default(sum_73, _shape_param_134);  sum_73 = _shape_param_134 = None
        convert_element_type_160: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_132, torch.bfloat16);  view_132 = None
        view_133: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_54, _shape_param_135);  mm_54 = _shape_param_135 = None
        convert_element_type_161: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_133, torch.float32);  view_133 = None
        add_29: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_134, convert_element_type_161);  mul_134 = convert_element_type_161 = None
        convert_element_type_162: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_163: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_160, torch.float32);  convert_element_type_160 = None
        mul_145: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_29, arg124_1);  arg124_1 = None
        mul_146: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_145, 768)
        sum_74: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True)
        mul_147: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_145, arg125_1);  mul_145 = None
        sum_75: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [2], True);  mul_147 = None
        mul_148: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg125_1, sum_75);  sum_75 = None
        sub_24: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_146, sum_74);  mul_146 = sum_74 = None
        sub_25: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_24, mul_148);  sub_24 = mul_148 = None
        mul_149: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg126_1, sub_25);  arg126_1 = sub_25 = None
        mul_150: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_29, arg125_1);  arg125_1 = None
        sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_150, [0, 1]);  mul_150 = None
        sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_29, [0, 1]);  add_29 = None
        convert_element_type_164: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_149, torch.bfloat16)
        convert_element_type_165: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg127_1, torch.bfloat16);  arg127_1 = None
        mul_151: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_165, 1.1111111111111112);  convert_element_type_165 = None
        mul_152: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_164, mul_151);  convert_element_type_164 = mul_151 = None
        view_134: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_152, _shape_param_136);  mul_152 = _shape_param_136 = None
        mm_56: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_134, arg128_1);  arg128_1 = None
        permute_48: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_134, [1, 0])
        mm_57: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_48, arg129_1);  permute_48 = arg129_1 = None
        sum_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_134, [0], True, dtype = torch.float32);  view_134 = None
        view_135: "f32[768]" = torch.ops.aten.view.default(sum_78, _shape_param_137);  sum_78 = _shape_param_137 = None
        convert_element_type_166: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_135, torch.bfloat16);  view_135 = None
        view_136: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_56, _shape_param_138);  mm_56 = _shape_param_138 = None
        convert_element_type_167: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_168: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_166, torch.float32);  convert_element_type_166 = None
        view_137: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_136, _shape_param_139);  view_136 = _shape_param_139 = None
        permute_49: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        clone_16: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_138: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_140);  clone_16 = _shape_param_140 = None
        bmm_16: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg130_1, view_138);  arg130_1 = None
        bmm_17: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_138, arg131_1);  view_138 = arg131_1 = None
        view_139: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_16, _shape_param_141);  bmm_16 = _shape_param_141 = None
        view_140: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_17, _shape_param_142);  bmm_17 = _shape_param_142 = None
        convert_element_type_169: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg132_1, torch.bfloat16);  arg132_1 = None
        mul_153: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_169, 1.1111111111111112);  convert_element_type_169 = None
        mul_154: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_140, mul_153);  view_140 = mul_153 = None
        convert_element_type_170: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_154, torch.float32);  mul_154 = None
        convert_element_type_171: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg133_1, torch.float32);  arg133_1 = None
        mul_155: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_170, convert_element_type_171);  convert_element_type_170 = None
        sum_79: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_155, [-1], True)
        neg_4: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_171);  convert_element_type_171 = None
        fma_4: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_4, sum_79, mul_155);  neg_4 = sum_79 = mul_155 = None
        convert_element_type_172: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None
        view_141: "bf16[384, 512, 512]" = torch.ops.aten.view.default(convert_element_type_172, _shape_param_143);  convert_element_type_172 = _shape_param_143 = None
        bmm_18: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg134_1, view_141);  arg134_1 = None
        bmm_19: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_141, arg135_1);  view_141 = arg135_1 = None
        view_142: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_18, _shape_param_144);  bmm_18 = _shape_param_144 = None
        view_143: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_145);  bmm_19 = _shape_param_145 = None
        mul_156: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_142, 0.3535533905932738);  view_142 = None
        permute_50: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_156, [0, 1, 3, 2]);  mul_156 = None
        mul_157: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_143, 0.3535533905932738);  view_143 = None
        permute_51: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
        clone_17: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_144: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_17, _shape_param_146);  clone_17 = _shape_param_146 = None
        view_145: "bf16[16384, 768]" = torch.ops.aten.view.default(view_144, _shape_param_147);  view_144 = _shape_param_147 = None
        mm_58: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_145, arg136_1);  arg136_1 = None
        permute_52: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_145, [1, 0])
        mm_59: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_52, arg137_1);  permute_52 = None
        sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_145, [0], True, dtype = torch.float32);  view_145 = None
        view_146: "f32[768]" = torch.ops.aten.view.default(sum_80, _shape_param_148);  sum_80 = _shape_param_148 = None
        convert_element_type_173: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_146, torch.bfloat16);  view_146 = None
        view_147: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_58, _shape_param_149);  mm_58 = _shape_param_149 = None
        convert_element_type_174: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_147, torch.float32);  view_147 = None
        add_30: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_149, convert_element_type_174);  mul_149 = convert_element_type_174 = None
        convert_element_type_175: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_176: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_173, torch.float32);  convert_element_type_173 = None
        permute_53: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_50, [0, 2, 1, 3]);  permute_50 = None
        view_148: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_53, _shape_param_150);  permute_53 = _shape_param_150 = None
        clone_18: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_148, memory_format = torch.contiguous_format);  view_148 = None
        view_149: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_18, _shape_param_151);  clone_18 = _shape_param_151 = None
        mm_60: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_149, arg138_1);  arg138_1 = None
        permute_54: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_149, [1, 0])
        mm_61: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_54, arg137_1);  permute_54 = None
        sum_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_149, [0], True, dtype = torch.float32);  view_149 = None
        view_150: "f32[768]" = torch.ops.aten.view.default(sum_81, _shape_param_152);  sum_81 = _shape_param_152 = None
        convert_element_type_177: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_150, torch.bfloat16);  view_150 = None
        view_151: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_60, _shape_param_153);  mm_60 = _shape_param_153 = None
        convert_element_type_178: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        add_31: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_30, convert_element_type_178);  add_30 = convert_element_type_178 = None
        convert_element_type_179: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_180: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_177, torch.float32);  convert_element_type_177 = None
        permute_55: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_157, [0, 2, 1, 3]);  mul_157 = None
        clone_19: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_152: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_19, _shape_param_154);  clone_19 = _shape_param_154 = None
        view_153: "bf16[16384, 768]" = torch.ops.aten.view.default(view_152, _shape_param_155);  view_152 = _shape_param_155 = None
        mm_62: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_153, arg139_1);  arg139_1 = None
        permute_56: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_153, [1, 0])
        mm_63: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_56, arg137_1);  permute_56 = arg137_1 = None
        sum_82: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_153, [0], True, dtype = torch.float32);  view_153 = None
        view_154: "f32[768]" = torch.ops.aten.view.default(sum_82, _shape_param_156);  sum_82 = _shape_param_156 = None
        convert_element_type_181: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_154, torch.bfloat16);  view_154 = None
        view_155: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_62, _shape_param_157);  mm_62 = _shape_param_157 = None
        convert_element_type_182: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_155, torch.float32);  view_155 = None
        add_32: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_31, convert_element_type_182);  add_31 = convert_element_type_182 = None
        convert_element_type_183: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_184: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_181, torch.float32);  convert_element_type_181 = None
        mul_158: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_32, arg140_1);  arg140_1 = None
        mul_159: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_158, 768)
        sum_83: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_158, [2], True)
        mul_160: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_158, arg141_1);  mul_158 = None
        sum_84: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_160, [2], True);  mul_160 = None
        mul_161: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg141_1, sum_84);  sum_84 = None
        sub_26: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_159, sum_83);  mul_159 = sum_83 = None
        sub_27: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_26, mul_161);  sub_26 = mul_161 = None
        mul_162: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg142_1, sub_27);  arg142_1 = sub_27 = None
        mul_163: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_32, arg141_1);  arg141_1 = None
        sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_163, [0, 1]);  mul_163 = None
        sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_32, [0, 1]);  add_32 = None
        convert_element_type_185: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_162, torch.bfloat16)
        convert_element_type_186: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg143_1, torch.bfloat16);  arg143_1 = None
        mul_164: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_186, 1.1111111111111112);  convert_element_type_186 = None
        mul_165: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_185, mul_164);  convert_element_type_185 = mul_164 = None
        view_156: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_165, _shape_param_158);  mul_165 = _shape_param_158 = None
        mm_64: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_156, arg144_1);  arg144_1 = None
        permute_57: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_156, [1, 0])
        mm_65: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_57, arg145_1);  permute_57 = arg145_1 = None
        sum_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_156, [0], True, dtype = torch.float32);  view_156 = None
        view_157: "f32[768]" = torch.ops.aten.view.default(sum_87, _shape_param_159);  sum_87 = _shape_param_159 = None
        convert_element_type_187: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_157, torch.bfloat16);  view_157 = None
        view_158: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_64, _shape_param_160);  mm_64 = _shape_param_160 = None
        convert_element_type_188: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_189: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_187, torch.float32);  convert_element_type_187 = None
        convert_element_type_190: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_158, torch.float32);  view_158 = None
        view_159: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg146_1, _shape_param_161);  arg146_1 = _shape_param_161 = None
        convert_element_type_191: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_159, torch.float32);  view_159 = None
        mul_166: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_191, 0.7071067811865476)
        erf_6: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_166);  mul_166 = None
        add_33: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_167: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_33, 0.5);  add_33 = None
        mul_168: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_191, convert_element_type_191)
        mul_169: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_168, -0.5);  mul_168 = None
        exp_7: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_169);  mul_169 = None
        mul_170: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
        mul_171: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_191, mul_170);  convert_element_type_191 = mul_170 = None
        add_34: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_167, mul_171);  mul_167 = mul_171 = None
        mul_172: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_190, add_34);  convert_element_type_190 = add_34 = None
        convert_element_type_192: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_172, torch.bfloat16);  mul_172 = None
        view_160: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_192, _shape_param_162);  convert_element_type_192 = _shape_param_162 = None
        mm_66: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_160, arg147_1);  arg147_1 = None
        permute_58: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_160, [1, 0])
        mm_67: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_58, arg148_1);  permute_58 = arg148_1 = None
        sum_88: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_160, [0], True, dtype = torch.float32);  view_160 = None
        view_161: "f32[3072]" = torch.ops.aten.view.default(sum_88, _shape_param_163);  sum_88 = _shape_param_163 = None
        convert_element_type_193: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_161, torch.bfloat16);  view_161 = None
        view_162: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_66, _shape_param_164);  mm_66 = _shape_param_164 = None
        convert_element_type_194: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_162, torch.float32);  view_162 = None
        add_35: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_162, convert_element_type_194);  mul_162 = convert_element_type_194 = None
        convert_element_type_195: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_196: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_193, torch.float32);  convert_element_type_193 = None
        mul_173: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_35, arg149_1);  arg149_1 = None
        mul_174: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_173, 768)
        sum_89: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [2], True)
        mul_175: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_173, arg150_1);  mul_173 = None
        sum_90: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_175, [2], True);  mul_175 = None
        mul_176: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg150_1, sum_90);  sum_90 = None
        sub_28: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_174, sum_89);  mul_174 = sum_89 = None
        sub_29: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_28, mul_176);  sub_28 = mul_176 = None
        mul_177: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg151_1, sub_29);  arg151_1 = sub_29 = None
        mul_178: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_35, arg150_1);  arg150_1 = None
        sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_178, [0, 1]);  mul_178 = None
        sum_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_35, [0, 1]);  add_35 = None
        convert_element_type_197: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_177, torch.bfloat16)
        convert_element_type_198: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg152_1, torch.bfloat16);  arg152_1 = None
        mul_179: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_198, 1.1111111111111112);  convert_element_type_198 = None
        mul_180: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_197, mul_179);  convert_element_type_197 = mul_179 = None
        view_163: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_180, _shape_param_165);  mul_180 = _shape_param_165 = None
        mm_68: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_163, arg153_1);  arg153_1 = None
        permute_59: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_163, [1, 0])
        mm_69: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_59, arg154_1);  permute_59 = arg154_1 = None
        sum_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_163, [0], True, dtype = torch.float32);  view_163 = None
        view_164: "f32[768]" = torch.ops.aten.view.default(sum_93, _shape_param_166);  sum_93 = _shape_param_166 = None
        convert_element_type_199: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_164, torch.bfloat16);  view_164 = None
        view_165: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_68, _shape_param_167);  mm_68 = _shape_param_167 = None
        convert_element_type_200: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_201: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_199, torch.float32);  convert_element_type_199 = None
        view_166: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_165, _shape_param_168);  view_165 = _shape_param_168 = None
        permute_60: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        clone_20: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_167: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_169);  clone_20 = _shape_param_169 = None
        bmm_20: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg155_1, view_167);  arg155_1 = None
        bmm_21: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_167, arg156_1);  view_167 = arg156_1 = None
        view_168: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_20, _shape_param_170);  bmm_20 = _shape_param_170 = None
        view_169: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_21, _shape_param_171);  bmm_21 = _shape_param_171 = None
        convert_element_type_202: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg157_1, torch.bfloat16);  arg157_1 = None
        mul_181: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_202, 1.1111111111111112);  convert_element_type_202 = None
        mul_182: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_169, mul_181);  view_169 = mul_181 = None
        convert_element_type_203: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_182, torch.float32);  mul_182 = None
        convert_element_type_204: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg158_1, torch.float32);  arg158_1 = None
        mul_183: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_203, convert_element_type_204);  convert_element_type_203 = None
        sum_94: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_183, [-1], True)
        neg_5: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_204);  convert_element_type_204 = None
        fma_5: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_5, sum_94, mul_183);  neg_5 = sum_94 = mul_183 = None
        convert_element_type_205: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None
        view_170: "bf16[384, 512, 512]" = torch.ops.aten.view.default(convert_element_type_205, _shape_param_172);  convert_element_type_205 = _shape_param_172 = None
        bmm_22: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg159_1, view_170);  arg159_1 = None
        bmm_23: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_170, arg160_1);  view_170 = arg160_1 = None
        view_171: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_22, _shape_param_173);  bmm_22 = _shape_param_173 = None
        view_172: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_174);  bmm_23 = _shape_param_174 = None
        mul_184: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_171, 0.3535533905932738);  view_171 = None
        permute_61: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_184, [0, 1, 3, 2]);  mul_184 = None
        mul_185: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_172, 0.3535533905932738);  view_172 = None
        permute_62: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_21: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_173: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_21, _shape_param_175);  clone_21 = _shape_param_175 = None
        view_174: "bf16[16384, 768]" = torch.ops.aten.view.default(view_173, _shape_param_176);  view_173 = _shape_param_176 = None
        mm_70: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_174, arg161_1);  arg161_1 = None
        permute_63: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_174, [1, 0])
        mm_71: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_63, arg162_1);  permute_63 = None
        sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_174, [0], True, dtype = torch.float32);  view_174 = None
        view_175: "f32[768]" = torch.ops.aten.view.default(sum_95, _shape_param_177);  sum_95 = _shape_param_177 = None
        convert_element_type_206: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_175, torch.bfloat16);  view_175 = None
        view_176: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_70, _shape_param_178);  mm_70 = _shape_param_178 = None
        convert_element_type_207: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_176, torch.float32);  view_176 = None
        add_36: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_177, convert_element_type_207);  mul_177 = convert_element_type_207 = None
        convert_element_type_208: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_209: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_206, torch.float32);  convert_element_type_206 = None
        permute_64: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_61, [0, 2, 1, 3]);  permute_61 = None
        view_177: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_64, _shape_param_179);  permute_64 = _shape_param_179 = None
        clone_22: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_177, memory_format = torch.contiguous_format);  view_177 = None
        view_178: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_22, _shape_param_180);  clone_22 = _shape_param_180 = None
        mm_72: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_178, arg163_1);  arg163_1 = None
        permute_65: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_178, [1, 0])
        mm_73: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_65, arg162_1);  permute_65 = None
        sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_178, [0], True, dtype = torch.float32);  view_178 = None
        view_179: "f32[768]" = torch.ops.aten.view.default(sum_96, _shape_param_181);  sum_96 = _shape_param_181 = None
        convert_element_type_210: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_179, torch.bfloat16);  view_179 = None
        view_180: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_72, _shape_param_182);  mm_72 = _shape_param_182 = None
        convert_element_type_211: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_180, torch.float32);  view_180 = None
        add_37: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_36, convert_element_type_211);  add_36 = convert_element_type_211 = None
        convert_element_type_212: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_213: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_210, torch.float32);  convert_element_type_210 = None
        permute_66: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_185, [0, 2, 1, 3]);  mul_185 = None
        clone_23: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
        view_181: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_23, _shape_param_183);  clone_23 = _shape_param_183 = None
        view_182: "bf16[16384, 768]" = torch.ops.aten.view.default(view_181, _shape_param_184);  view_181 = _shape_param_184 = None
        mm_74: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_182, arg164_1);  arg164_1 = None
        permute_67: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_182, [1, 0])
        mm_75: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_67, arg162_1);  permute_67 = arg162_1 = None
        sum_97: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_182, [0], True, dtype = torch.float32);  view_182 = None
        view_183: "f32[768]" = torch.ops.aten.view.default(sum_97, _shape_param_185);  sum_97 = _shape_param_185 = None
        convert_element_type_214: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_183, torch.bfloat16);  view_183 = None
        view_184: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_74, _shape_param_186);  mm_74 = _shape_param_186 = None
        convert_element_type_215: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_184, torch.float32);  view_184 = None
        add_38: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_37, convert_element_type_215);  add_37 = convert_element_type_215 = None
        convert_element_type_216: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_217: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_214, torch.float32);  convert_element_type_214 = None
        mul_186: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_38, arg165_1);  arg165_1 = None
        mul_187: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_186, 768)
        sum_98: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_186, [2], True)
        mul_188: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_186, arg166_1);  mul_186 = None
        sum_99: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_188, [2], True);  mul_188 = None
        mul_189: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg166_1, sum_99);  sum_99 = None
        sub_30: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_187, sum_98);  mul_187 = sum_98 = None
        sub_31: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_30, mul_189);  sub_30 = mul_189 = None
        mul_190: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg167_1, sub_31);  arg167_1 = sub_31 = None
        mul_191: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_38, arg166_1);  arg166_1 = None
        sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_191, [0, 1]);  mul_191 = None
        sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_38, [0, 1]);  add_38 = None
        convert_element_type_218: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_190, torch.bfloat16)
        convert_element_type_219: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg168_1, torch.bfloat16);  arg168_1 = None
        mul_192: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_219, 1.1111111111111112);  convert_element_type_219 = None
        mul_193: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_218, mul_192);  convert_element_type_218 = mul_192 = None
        view_185: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_193, _shape_param_187);  mul_193 = _shape_param_187 = None
        mm_76: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_185, arg169_1);  arg169_1 = None
        permute_68: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_185, [1, 0])
        mm_77: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_68, arg170_1);  permute_68 = arg170_1 = None
        sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_185, [0], True, dtype = torch.float32);  view_185 = None
        view_186: "f32[768]" = torch.ops.aten.view.default(sum_102, _shape_param_188);  sum_102 = _shape_param_188 = None
        convert_element_type_220: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_186, torch.bfloat16);  view_186 = None
        view_187: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_76, _shape_param_189);  mm_76 = _shape_param_189 = None
        convert_element_type_221: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_222: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_220, torch.float32);  convert_element_type_220 = None
        convert_element_type_223: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_187, torch.float32);  view_187 = None
        view_188: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg171_1, _shape_param_190);  arg171_1 = _shape_param_190 = None
        convert_element_type_224: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_188, torch.float32);  view_188 = None
        mul_194: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_224, 0.7071067811865476)
        erf_7: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_194);  mul_194 = None
        add_39: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_195: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_39, 0.5);  add_39 = None
        mul_196: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_224, convert_element_type_224)
        mul_197: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_196, -0.5);  mul_196 = None
        exp_8: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_197);  mul_197 = None
        mul_198: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_199: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_224, mul_198);  convert_element_type_224 = mul_198 = None
        add_40: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_195, mul_199);  mul_195 = mul_199 = None
        mul_200: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_223, add_40);  convert_element_type_223 = add_40 = None
        convert_element_type_225: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_200, torch.bfloat16);  mul_200 = None
        view_189: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_225, _shape_param_191);  convert_element_type_225 = _shape_param_191 = None
        mm_78: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_189, arg172_1);  arg172_1 = None
        permute_69: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_189, [1, 0])
        mm_79: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_69, arg173_1);  permute_69 = arg173_1 = None
        sum_103: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_189, [0], True, dtype = torch.float32);  view_189 = None
        view_190: "f32[3072]" = torch.ops.aten.view.default(sum_103, _shape_param_192);  sum_103 = _shape_param_192 = None
        convert_element_type_226: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_190, torch.bfloat16);  view_190 = None
        view_191: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_78, _shape_param_193);  mm_78 = _shape_param_193 = None
        convert_element_type_227: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_191, torch.float32);  view_191 = None
        add_41: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_190, convert_element_type_227);  mul_190 = convert_element_type_227 = None
        convert_element_type_228: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_229: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_226, torch.float32);  convert_element_type_226 = None
        mul_201: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_41, arg174_1);  arg174_1 = None
        mul_202: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_201, 768)
        sum_104: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_201, [2], True)
        mul_203: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_201, arg175_1);  mul_201 = None
        sum_105: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [2], True);  mul_203 = None
        mul_204: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg175_1, sum_105);  sum_105 = None
        sub_32: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_202, sum_104);  mul_202 = sum_104 = None
        sub_33: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_32, mul_204);  sub_32 = mul_204 = None
        mul_205: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg176_1, sub_33);  arg176_1 = sub_33 = None
        mul_206: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_41, arg175_1);  arg175_1 = None
        sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_206, [0, 1]);  mul_206 = None
        sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_41, [0, 1]);  add_41 = None
        convert_element_type_230: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_205, torch.bfloat16)
        convert_element_type_231: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg177_1, torch.bfloat16);  arg177_1 = None
        mul_207: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_231, 1.1111111111111112);  convert_element_type_231 = None
        mul_208: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_230, mul_207);  convert_element_type_230 = mul_207 = None
        view_192: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_208, _shape_param_194);  mul_208 = _shape_param_194 = None
        mm_80: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_192, arg178_1);  arg178_1 = None
        permute_70: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_192, [1, 0])
        mm_81: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_70, arg179_1);  permute_70 = arg179_1 = None
        sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_192, [0], True, dtype = torch.float32);  view_192 = None
        view_193: "f32[768]" = torch.ops.aten.view.default(sum_108, _shape_param_195);  sum_108 = _shape_param_195 = None
        convert_element_type_232: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_193, torch.bfloat16);  view_193 = None
        view_194: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_80, _shape_param_196);  mm_80 = _shape_param_196 = None
        convert_element_type_233: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_234: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_232, torch.float32);  convert_element_type_232 = None
        view_195: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_194, _shape_param_197);  view_194 = _shape_param_197 = None
        permute_71: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None
        clone_24: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_196: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_198);  clone_24 = _shape_param_198 = None
        bmm_24: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg180_1, view_196);  arg180_1 = None
        bmm_25: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_196, arg181_1);  view_196 = arg181_1 = None
        view_197: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_24, _shape_param_199);  bmm_24 = _shape_param_199 = None
        view_198: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_25, _shape_param_200);  bmm_25 = _shape_param_200 = None
        convert_element_type_235: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg182_1, torch.bfloat16);  arg182_1 = None
        mul_209: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_235, 1.1111111111111112);  convert_element_type_235 = None
        mul_210: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_198, mul_209);  view_198 = mul_209 = None
        convert_element_type_236: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_210, torch.float32);  mul_210 = None
        convert_element_type_237: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg183_1, torch.float32);  arg183_1 = None
        mul_211: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_236, convert_element_type_237);  convert_element_type_236 = None
        sum_109: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [-1], True)
        neg_6: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_237);  convert_element_type_237 = None
        fma_6: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_6, sum_109, mul_211);  neg_6 = sum_109 = mul_211 = None
        convert_element_type_238: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None
        view_199: "bf16[384, 512, 512]" = torch.ops.aten.view.default(convert_element_type_238, _shape_param_201);  convert_element_type_238 = _shape_param_201 = None
        bmm_26: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg184_1, view_199);  arg184_1 = None
        bmm_27: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_199, arg185_1);  view_199 = arg185_1 = None
        view_200: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_26, _shape_param_202);  bmm_26 = _shape_param_202 = None
        view_201: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_27, _shape_param_203);  bmm_27 = _shape_param_203 = None
        mul_212: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_200, 0.3535533905932738);  view_200 = None
        permute_72: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_212, [0, 1, 3, 2]);  mul_212 = None
        mul_213: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_201, 0.3535533905932738);  view_201 = None
        permute_73: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_197, [0, 2, 1, 3]);  view_197 = None
        clone_25: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_202: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_25, _shape_param_204);  clone_25 = _shape_param_204 = None
        view_203: "bf16[16384, 768]" = torch.ops.aten.view.default(view_202, _shape_param_205);  view_202 = _shape_param_205 = None
        mm_82: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_203, arg186_1);  arg186_1 = None
        permute_74: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_203, [1, 0])
        mm_83: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_74, arg187_1);  permute_74 = None
        sum_110: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_203, [0], True, dtype = torch.float32);  view_203 = None
        view_204: "f32[768]" = torch.ops.aten.view.default(sum_110, _shape_param_206);  sum_110 = _shape_param_206 = None
        convert_element_type_239: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_204, torch.bfloat16);  view_204 = None
        view_205: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_82, _shape_param_207);  mm_82 = _shape_param_207 = None
        convert_element_type_240: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_205, torch.float32);  view_205 = None
        add_42: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_205, convert_element_type_240);  mul_205 = convert_element_type_240 = None
        convert_element_type_241: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_242: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_239, torch.float32);  convert_element_type_239 = None
        permute_75: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_72, [0, 2, 1, 3]);  permute_72 = None
        view_206: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_75, _shape_param_208);  permute_75 = _shape_param_208 = None
        clone_26: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_206, memory_format = torch.contiguous_format);  view_206 = None
        view_207: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_26, _shape_param_209);  clone_26 = _shape_param_209 = None
        mm_84: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_207, arg188_1);  arg188_1 = None
        permute_76: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_207, [1, 0])
        mm_85: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_76, arg187_1);  permute_76 = None
        sum_111: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_207, [0], True, dtype = torch.float32);  view_207 = None
        view_208: "f32[768]" = torch.ops.aten.view.default(sum_111, _shape_param_210);  sum_111 = _shape_param_210 = None
        convert_element_type_243: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_208, torch.bfloat16);  view_208 = None
        view_209: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_84, _shape_param_211);  mm_84 = _shape_param_211 = None
        convert_element_type_244: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_209, torch.float32);  view_209 = None
        add_43: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_42, convert_element_type_244);  add_42 = convert_element_type_244 = None
        convert_element_type_245: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_246: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_243, torch.float32);  convert_element_type_243 = None
        permute_77: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_213, [0, 2, 1, 3]);  mul_213 = None
        clone_27: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
        view_210: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_27, _shape_param_212);  clone_27 = _shape_param_212 = None
        view_211: "bf16[16384, 768]" = torch.ops.aten.view.default(view_210, _shape_param_213);  view_210 = _shape_param_213 = None
        mm_86: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_211, arg189_1);  arg189_1 = None
        permute_78: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_211, [1, 0])
        mm_87: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_78, arg187_1);  permute_78 = arg187_1 = None
        sum_112: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_211, [0], True, dtype = torch.float32);  view_211 = None
        view_212: "f32[768]" = torch.ops.aten.view.default(sum_112, _shape_param_214);  sum_112 = _shape_param_214 = None
        convert_element_type_247: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_212, torch.bfloat16);  view_212 = None
        view_213: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_86, _shape_param_215);  mm_86 = _shape_param_215 = None
        convert_element_type_248: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_213, torch.float32);  view_213 = None
        add_44: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_43, convert_element_type_248);  add_43 = convert_element_type_248 = None
        convert_element_type_249: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_250: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_247, torch.float32);  convert_element_type_247 = None
        mul_214: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_44, arg190_1);  arg190_1 = None
        mul_215: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_214, 768)
        sum_113: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_214, [2], True)
        mul_216: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_214, arg191_1);  mul_214 = None
        sum_114: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_216, [2], True);  mul_216 = None
        mul_217: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg191_1, sum_114);  sum_114 = None
        sub_34: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_215, sum_113);  mul_215 = sum_113 = None
        sub_35: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_34, mul_217);  sub_34 = mul_217 = None
        mul_218: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg192_1, sub_35);  arg192_1 = sub_35 = None
        mul_219: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_44, arg191_1);  arg191_1 = None
        sum_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_219, [0, 1]);  mul_219 = None
        sum_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_44, [0, 1]);  add_44 = None
        convert_element_type_251: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_218, torch.bfloat16)
        convert_element_type_252: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg193_1, torch.bfloat16);  arg193_1 = None
        mul_220: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_252, 1.1111111111111112);  convert_element_type_252 = None
        mul_221: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_251, mul_220);  convert_element_type_251 = mul_220 = None
        view_214: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_221, _shape_param_216);  mul_221 = _shape_param_216 = None
        mm_88: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_214, arg194_1);  arg194_1 = None
        permute_79: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_214, [1, 0])
        mm_89: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_79, arg195_1);  permute_79 = arg195_1 = None
        sum_117: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_214, [0], True, dtype = torch.float32);  view_214 = None
        view_215: "f32[768]" = torch.ops.aten.view.default(sum_117, _shape_param_217);  sum_117 = _shape_param_217 = None
        convert_element_type_253: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_215, torch.bfloat16);  view_215 = None
        view_216: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_88, _shape_param_218);  mm_88 = _shape_param_218 = None
        convert_element_type_254: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_255: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_253, torch.float32);  convert_element_type_253 = None
        convert_element_type_256: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_216, torch.float32);  view_216 = None
        view_217: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg196_1, _shape_param_219);  arg196_1 = _shape_param_219 = None
        convert_element_type_257: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_222: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_257, 0.7071067811865476)
        erf_8: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_222);  mul_222 = None
        add_45: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_223: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_45, 0.5);  add_45 = None
        mul_224: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_257, convert_element_type_257)
        mul_225: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_224, -0.5);  mul_224 = None
        exp_9: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_225);  mul_225 = None
        mul_226: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
        mul_227: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_257, mul_226);  convert_element_type_257 = mul_226 = None
        add_46: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_223, mul_227);  mul_223 = mul_227 = None
        mul_228: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_256, add_46);  convert_element_type_256 = add_46 = None
        convert_element_type_258: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_228, torch.bfloat16);  mul_228 = None
        view_218: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_258, _shape_param_220);  convert_element_type_258 = _shape_param_220 = None
        mm_90: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_218, arg197_1);  arg197_1 = None
        permute_80: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_218, [1, 0])
        mm_91: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_80, arg198_1);  permute_80 = arg198_1 = None
        sum_118: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True, dtype = torch.float32);  view_218 = None
        view_219: "f32[3072]" = torch.ops.aten.view.default(sum_118, _shape_param_221);  sum_118 = _shape_param_221 = None
        convert_element_type_259: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_219, torch.bfloat16);  view_219 = None
        view_220: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_90, _shape_param_222);  mm_90 = _shape_param_222 = None
        convert_element_type_260: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_220, torch.float32);  view_220 = None
        add_47: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_218, convert_element_type_260);  mul_218 = convert_element_type_260 = None
        convert_element_type_261: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_262: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_259, torch.float32);  convert_element_type_259 = None
        mul_229: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_47, arg199_1);  arg199_1 = None
        mul_230: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_229, 768)
        sum_119: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True)
        mul_231: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_229, arg200_1);  mul_229 = None
        sum_120: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_231, [2], True);  mul_231 = None
        mul_232: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg200_1, sum_120);  sum_120 = None
        sub_36: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_230, sum_119);  mul_230 = sum_119 = None
        sub_37: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_36, mul_232);  sub_36 = mul_232 = None
        mul_233: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg201_1, sub_37);  arg201_1 = sub_37 = None
        mul_234: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_47, arg200_1);  arg200_1 = None
        sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_234, [0, 1]);  mul_234 = None
        sum_122: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_47, [0, 1]);  add_47 = None
        convert_element_type_263: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_233, torch.bfloat16)
        convert_element_type_264: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg202_1, torch.bfloat16);  arg202_1 = None
        mul_235: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_264, 1.1111111111111112);  convert_element_type_264 = None
        mul_236: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_263, mul_235);  convert_element_type_263 = mul_235 = None
        view_221: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_236, _shape_param_223);  mul_236 = _shape_param_223 = None
        mm_92: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_221, arg203_1);  arg203_1 = None
        permute_81: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_221, [1, 0])
        mm_93: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_81, arg204_1);  permute_81 = arg204_1 = None
        sum_123: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_221, [0], True, dtype = torch.float32);  view_221 = None
        view_222: "f32[768]" = torch.ops.aten.view.default(sum_123, _shape_param_224);  sum_123 = _shape_param_224 = None
        convert_element_type_265: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_222, torch.bfloat16);  view_222 = None
        view_223: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_92, _shape_param_225);  mm_92 = _shape_param_225 = None
        convert_element_type_266: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_267: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_265, torch.float32);  convert_element_type_265 = None
        view_224: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_223, _shape_param_226);  view_223 = _shape_param_226 = None
        permute_82: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
        clone_28: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_225: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_227);  clone_28 = _shape_param_227 = None
        bmm_28: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg205_1, view_225);  arg205_1 = None
        bmm_29: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_225, arg206_1);  view_225 = arg206_1 = None
        view_226: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_28, _shape_param_228);  bmm_28 = _shape_param_228 = None
        view_227: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_29, _shape_param_229);  bmm_29 = _shape_param_229 = None
        convert_element_type_268: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg207_1, torch.bfloat16);  arg207_1 = None
        mul_237: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_268, 1.1111111111111112);  convert_element_type_268 = None
        mul_238: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_227, mul_237);  view_227 = mul_237 = None
        convert_element_type_269: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_238, torch.float32);  mul_238 = None
        convert_element_type_270: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg208_1, torch.float32);  arg208_1 = None
        mul_239: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_269, convert_element_type_270);  convert_element_type_269 = None
        sum_124: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [-1], True)
        neg_7: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_270);  convert_element_type_270 = None
        fma_7: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_7, sum_124, mul_239);  neg_7 = sum_124 = mul_239 = None
        convert_element_type_271: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None
        view_228: "bf16[384, 512, 512]" = torch.ops.aten.view.default(convert_element_type_271, _shape_param_230);  convert_element_type_271 = _shape_param_230 = None
        bmm_30: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg209_1, view_228);  arg209_1 = None
        bmm_31: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_228, arg210_1);  view_228 = arg210_1 = None
        view_229: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_30, _shape_param_231);  bmm_30 = _shape_param_231 = None
        view_230: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_31, _shape_param_232);  bmm_31 = _shape_param_232 = None
        mul_240: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_229, 0.3535533905932738);  view_229 = None
        permute_83: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_240, [0, 1, 3, 2]);  mul_240 = None
        mul_241: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_230, 0.3535533905932738);  view_230 = None
        permute_84: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_29: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_231: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_29, _shape_param_233);  clone_29 = _shape_param_233 = None
        view_232: "bf16[16384, 768]" = torch.ops.aten.view.default(view_231, _shape_param_234);  view_231 = _shape_param_234 = None
        mm_94: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_232, arg211_1);  arg211_1 = None
        permute_85: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_232, [1, 0])
        mm_95: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_85, arg212_1);  permute_85 = None
        sum_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_232, [0], True, dtype = torch.float32);  view_232 = None
        view_233: "f32[768]" = torch.ops.aten.view.default(sum_125, _shape_param_235);  sum_125 = _shape_param_235 = None
        convert_element_type_272: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_233, torch.bfloat16);  view_233 = None
        view_234: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_94, _shape_param_236);  mm_94 = _shape_param_236 = None
        convert_element_type_273: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_234, torch.float32);  view_234 = None
        add_48: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_233, convert_element_type_273);  mul_233 = convert_element_type_273 = None
        convert_element_type_274: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_275: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_272, torch.float32);  convert_element_type_272 = None
        permute_86: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_83, [0, 2, 1, 3]);  permute_83 = None
        view_235: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_86, _shape_param_237);  permute_86 = _shape_param_237 = None
        clone_30: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_235, memory_format = torch.contiguous_format);  view_235 = None
        view_236: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_30, _shape_param_238);  clone_30 = _shape_param_238 = None
        mm_96: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_236, arg213_1);  arg213_1 = None
        permute_87: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_236, [1, 0])
        mm_97: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_87, arg212_1);  permute_87 = None
        sum_126: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_236, [0], True, dtype = torch.float32);  view_236 = None
        view_237: "f32[768]" = torch.ops.aten.view.default(sum_126, _shape_param_239);  sum_126 = _shape_param_239 = None
        convert_element_type_276: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_237, torch.bfloat16);  view_237 = None
        view_238: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_96, _shape_param_240);  mm_96 = _shape_param_240 = None
        convert_element_type_277: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_238, torch.float32);  view_238 = None
        add_49: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_48, convert_element_type_277);  add_48 = convert_element_type_277 = None
        convert_element_type_278: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_279: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_276, torch.float32);  convert_element_type_276 = None
        permute_88: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_241, [0, 2, 1, 3]);  mul_241 = None
        clone_31: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
        view_239: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_31, _shape_param_241);  clone_31 = _shape_param_241 = None
        view_240: "bf16[16384, 768]" = torch.ops.aten.view.default(view_239, _shape_param_242);  view_239 = _shape_param_242 = None
        mm_98: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_240, arg214_1);  arg214_1 = None
        permute_89: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_240, [1, 0])
        mm_99: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_89, arg212_1);  permute_89 = arg212_1 = None
        sum_127: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_240, [0], True, dtype = torch.float32);  view_240 = None
        view_241: "f32[768]" = torch.ops.aten.view.default(sum_127, _shape_param_243);  sum_127 = _shape_param_243 = None
        convert_element_type_280: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_241, torch.bfloat16);  view_241 = None
        view_242: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_98, _shape_param_244);  mm_98 = _shape_param_244 = None
        convert_element_type_281: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_242, torch.float32);  view_242 = None
        add_50: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_49, convert_element_type_281);  add_49 = convert_element_type_281 = None
        convert_element_type_282: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_283: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_280, torch.float32);  convert_element_type_280 = None
        mul_242: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_50, arg215_1);  arg215_1 = None
        mul_243: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_242, 768)
        sum_128: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_242, [2], True)
        mul_244: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_242, arg216_1);  mul_242 = None
        sum_129: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_244, [2], True);  mul_244 = None
        mul_245: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg216_1, sum_129);  sum_129 = None
        sub_38: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_243, sum_128);  mul_243 = sum_128 = None
        sub_39: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_38, mul_245);  sub_38 = mul_245 = None
        mul_246: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg217_1, sub_39);  arg217_1 = sub_39 = None
        mul_247: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_50, arg216_1);  arg216_1 = None
        sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_247, [0, 1]);  mul_247 = None
        sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_50, [0, 1]);  add_50 = None
        convert_element_type_284: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_246, torch.bfloat16)
        convert_element_type_285: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg218_1, torch.bfloat16);  arg218_1 = None
        mul_248: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_285, 1.1111111111111112);  convert_element_type_285 = None
        mul_249: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_284, mul_248);  convert_element_type_284 = mul_248 = None
        view_243: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_249, _shape_param_245);  mul_249 = _shape_param_245 = None
        mm_100: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_243, arg219_1);  arg219_1 = None
        permute_90: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_243, [1, 0])
        mm_101: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_90, arg220_1);  permute_90 = arg220_1 = None
        sum_132: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_243, [0], True, dtype = torch.float32);  view_243 = None
        view_244: "f32[768]" = torch.ops.aten.view.default(sum_132, _shape_param_246);  sum_132 = _shape_param_246 = None
        convert_element_type_286: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_244, torch.bfloat16);  view_244 = None
        view_245: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_100, _shape_param_247);  mm_100 = _shape_param_247 = None
        convert_element_type_287: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_288: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_286, torch.float32);  convert_element_type_286 = None
        convert_element_type_289: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_245, torch.float32);  view_245 = None
        view_246: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg221_1, _shape_param_248);  arg221_1 = _shape_param_248 = None
        convert_element_type_290: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_246, torch.float32);  view_246 = None
        mul_250: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_290, 0.7071067811865476)
        erf_9: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_250);  mul_250 = None
        add_51: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_251: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_51, 0.5);  add_51 = None
        mul_252: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_290, convert_element_type_290)
        mul_253: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_252, -0.5);  mul_252 = None
        exp_10: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_253);  mul_253 = None
        mul_254: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_255: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_290, mul_254);  convert_element_type_290 = mul_254 = None
        add_52: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_251, mul_255);  mul_251 = mul_255 = None
        mul_256: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_289, add_52);  convert_element_type_289 = add_52 = None
        convert_element_type_291: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_256, torch.bfloat16);  mul_256 = None
        view_247: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_291, _shape_param_249);  convert_element_type_291 = _shape_param_249 = None
        mm_102: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_247, arg222_1);  arg222_1 = None
        permute_91: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_247, [1, 0])
        mm_103: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_91, arg223_1);  permute_91 = arg223_1 = None
        sum_133: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_247, [0], True, dtype = torch.float32);  view_247 = None
        view_248: "f32[3072]" = torch.ops.aten.view.default(sum_133, _shape_param_250);  sum_133 = _shape_param_250 = None
        convert_element_type_292: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_248, torch.bfloat16);  view_248 = None
        view_249: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_102, _shape_param_251);  mm_102 = _shape_param_251 = None
        convert_element_type_293: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_249, torch.float32);  view_249 = None
        add_53: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_246, convert_element_type_293);  mul_246 = convert_element_type_293 = None
        convert_element_type_294: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_295: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_292, torch.float32);  convert_element_type_292 = None
        mul_257: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_53, arg224_1);  arg224_1 = None
        mul_258: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_257, 768)
        sum_134: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True)
        mul_259: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_257, arg225_1);  mul_257 = None
        sum_135: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_259, [2], True);  mul_259 = None
        mul_260: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg225_1, sum_135);  sum_135 = None
        sub_40: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_258, sum_134);  mul_258 = sum_134 = None
        sub_41: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_40, mul_260);  sub_40 = mul_260 = None
        mul_261: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg226_1, sub_41);  arg226_1 = sub_41 = None
        mul_262: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_53, arg225_1);  arg225_1 = None
        sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_262, [0, 1]);  mul_262 = None
        sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_53, [0, 1]);  add_53 = None
        convert_element_type_296: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_261, torch.bfloat16)
        convert_element_type_297: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg227_1, torch.bfloat16);  arg227_1 = None
        mul_263: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_297, 1.1111111111111112);  convert_element_type_297 = None
        mul_264: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_296, mul_263);  convert_element_type_296 = mul_263 = None
        view_250: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_264, _shape_param_252);  mul_264 = _shape_param_252 = None
        mm_104: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_250, arg228_1);  arg228_1 = None
        permute_92: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_250, [1, 0])
        mm_105: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_92, arg229_1);  permute_92 = arg229_1 = None
        sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_250, [0], True, dtype = torch.float32);  view_250 = None
        view_251: "f32[768]" = torch.ops.aten.view.default(sum_138, _shape_param_253);  sum_138 = _shape_param_253 = None
        convert_element_type_298: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_251, torch.bfloat16);  view_251 = None
        view_252: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_104, _shape_param_254);  mm_104 = _shape_param_254 = None
        convert_element_type_299: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None
        convert_element_type_300: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_298, torch.float32);  convert_element_type_298 = None
        view_253: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_252, _shape_param_255);  view_252 = _shape_param_255 = None
        permute_93: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
        clone_32: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_254: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_32, _shape_param_256);  clone_32 = _shape_param_256 = None
        bmm_32: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg230_1, view_254);  arg230_1 = None
        bmm_33: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_254, arg231_1);  view_254 = arg231_1 = None
        view_255: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_32, _shape_param_257);  bmm_32 = _shape_param_257 = None
        view_256: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_33, _shape_param_258);  bmm_33 = _shape_param_258 = None
        convert_element_type_301: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg232_1, torch.bfloat16);  arg232_1 = None
        mul_265: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_301, 1.1111111111111112);  convert_element_type_301 = None
        mul_266: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_256, mul_265);  view_256 = mul_265 = None
        convert_element_type_302: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_266, torch.float32);  mul_266 = None
        convert_element_type_303: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg233_1, torch.float32);  arg233_1 = None
        mul_267: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_302, convert_element_type_303);  convert_element_type_302 = None
        sum_139: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [-1], True)
        neg_8: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_303);  convert_element_type_303 = None
        fma_8: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_8, sum_139, mul_267);  neg_8 = sum_139 = mul_267 = None
        convert_element_type_304: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None
        view_257: "bf16[384, 512, 512]" = torch.ops.aten.view.default(convert_element_type_304, _shape_param_259);  convert_element_type_304 = _shape_param_259 = None
        bmm_34: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg234_1, view_257);  arg234_1 = None
        bmm_35: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_257, arg235_1);  view_257 = arg235_1 = None
        view_258: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_34, _shape_param_260);  bmm_34 = _shape_param_260 = None
        view_259: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_35, _shape_param_261);  bmm_35 = _shape_param_261 = None
        mul_268: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_258, 0.3535533905932738);  view_258 = None
        permute_94: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_268, [0, 1, 3, 2]);  mul_268 = None
        mul_269: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_259, 0.3535533905932738);  view_259 = None
        permute_95: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
        clone_33: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_260: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_33, _shape_param_262);  clone_33 = _shape_param_262 = None
        view_261: "bf16[16384, 768]" = torch.ops.aten.view.default(view_260, _shape_param_263);  view_260 = _shape_param_263 = None
        mm_106: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_261, arg236_1);  arg236_1 = None
        permute_96: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_261, [1, 0])
        mm_107: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_96, arg237_1);  permute_96 = None
        sum_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_261, [0], True, dtype = torch.float32);  view_261 = None
        view_262: "f32[768]" = torch.ops.aten.view.default(sum_140, _shape_param_264);  sum_140 = _shape_param_264 = None
        convert_element_type_305: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_262, torch.bfloat16);  view_262 = None
        view_263: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_106, _shape_param_265);  mm_106 = _shape_param_265 = None
        convert_element_type_306: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None
        add_54: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_261, convert_element_type_306);  mul_261 = convert_element_type_306 = None
        convert_element_type_307: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None
        convert_element_type_308: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_305, torch.float32);  convert_element_type_305 = None
        permute_97: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_94, [0, 2, 1, 3]);  permute_94 = None
        view_264: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_97, _shape_param_266);  permute_97 = _shape_param_266 = None
        clone_34: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_264, memory_format = torch.contiguous_format);  view_264 = None
        view_265: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_34, _shape_param_267);  clone_34 = _shape_param_267 = None
        mm_108: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_265, arg238_1);  arg238_1 = None
        permute_98: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_265, [1, 0])
        mm_109: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_98, arg237_1);  permute_98 = None
        sum_141: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_265, [0], True, dtype = torch.float32);  view_265 = None
        view_266: "f32[768]" = torch.ops.aten.view.default(sum_141, _shape_param_268);  sum_141 = _shape_param_268 = None
        convert_element_type_309: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_266, torch.bfloat16);  view_266 = None
        view_267: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_108, _shape_param_269);  mm_108 = _shape_param_269 = None
        convert_element_type_310: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_267, torch.float32);  view_267 = None
        add_55: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_54, convert_element_type_310);  add_54 = convert_element_type_310 = None
        convert_element_type_311: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        convert_element_type_312: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_309, torch.float32);  convert_element_type_309 = None
        permute_99: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_269, [0, 2, 1, 3]);  mul_269 = None
        clone_35: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
        view_268: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_35, _shape_param_270);  clone_35 = _shape_param_270 = None
        view_269: "bf16[16384, 768]" = torch.ops.aten.view.default(view_268, _shape_param_271);  view_268 = _shape_param_271 = None
        mm_110: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_269, arg239_1);  arg239_1 = None
        permute_100: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_269, [1, 0])
        mm_111: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_100, arg237_1);  permute_100 = arg237_1 = None
        sum_142: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True, dtype = torch.float32);  view_269 = None
        view_270: "f32[768]" = torch.ops.aten.view.default(sum_142, _shape_param_272);  sum_142 = _shape_param_272 = None
        convert_element_type_313: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_270, torch.bfloat16);  view_270 = None
        view_271: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_110, _shape_param_273);  mm_110 = _shape_param_273 = None
        convert_element_type_314: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_271, torch.float32);  view_271 = None
        add_56: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_55, convert_element_type_314);  add_55 = convert_element_type_314 = None
        convert_element_type_315: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_316: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_313, torch.float32);  convert_element_type_313 = None
        mul_270: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_56, arg240_1);  arg240_1 = None
        mul_271: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_270, 768)
        sum_143: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_270, [2], True)
        mul_272: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_270, arg241_1);  mul_270 = None
        sum_144: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True);  mul_272 = None
        mul_273: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg241_1, sum_144);  sum_144 = None
        sub_42: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_271, sum_143);  mul_271 = sum_143 = None
        sub_43: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_42, mul_273);  sub_42 = mul_273 = None
        mul_274: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg242_1, sub_43);  arg242_1 = sub_43 = None
        mul_275: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_56, arg241_1);  arg241_1 = None
        sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_275, [0, 1]);  mul_275 = None
        sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_56, [0, 1]);  add_56 = None
        convert_element_type_317: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_274, torch.bfloat16)
        convert_element_type_318: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg243_1, torch.bfloat16);  arg243_1 = None
        mul_276: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_318, 1.1111111111111112);  convert_element_type_318 = None
        mul_277: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_317, mul_276);  convert_element_type_317 = mul_276 = None
        view_272: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_277, _shape_param_274);  mul_277 = _shape_param_274 = None
        mm_112: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_272, arg244_1);  arg244_1 = None
        permute_101: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_272, [1, 0])
        mm_113: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_101, arg245_1);  permute_101 = arg245_1 = None
        sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_272, [0], True, dtype = torch.float32);  view_272 = None
        view_273: "f32[768]" = torch.ops.aten.view.default(sum_147, _shape_param_275);  sum_147 = _shape_param_275 = None
        convert_element_type_319: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_273, torch.bfloat16);  view_273 = None
        view_274: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_112, _shape_param_276);  mm_112 = _shape_param_276 = None
        convert_element_type_320: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_321: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_319, torch.float32);  convert_element_type_319 = None
        convert_element_type_322: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_274, torch.float32);  view_274 = None
        view_275: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg246_1, _shape_param_277);  arg246_1 = _shape_param_277 = None
        convert_element_type_323: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_275, torch.float32);  view_275 = None
        mul_278: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_323, 0.7071067811865476)
        erf_10: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_278);  mul_278 = None
        add_57: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_279: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_57, 0.5);  add_57 = None
        mul_280: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_323, convert_element_type_323)
        mul_281: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_280, -0.5);  mul_280 = None
        exp_11: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_281);  mul_281 = None
        mul_282: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
        mul_283: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_323, mul_282);  convert_element_type_323 = mul_282 = None
        add_58: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_279, mul_283);  mul_279 = mul_283 = None
        mul_284: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_322, add_58);  convert_element_type_322 = add_58 = None
        convert_element_type_324: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_284, torch.bfloat16);  mul_284 = None
        view_276: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_324, _shape_param_278);  convert_element_type_324 = _shape_param_278 = None
        mm_114: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_276, arg247_1);  arg247_1 = None
        permute_102: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_276, [1, 0])
        mm_115: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_102, arg248_1);  permute_102 = arg248_1 = None
        sum_148: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True, dtype = torch.float32);  view_276 = None
        view_277: "f32[3072]" = torch.ops.aten.view.default(sum_148, _shape_param_279);  sum_148 = _shape_param_279 = None
        convert_element_type_325: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_277, torch.bfloat16);  view_277 = None
        view_278: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_114, _shape_param_280);  mm_114 = _shape_param_280 = None
        convert_element_type_326: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_278, torch.float32);  view_278 = None
        add_59: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_274, convert_element_type_326);  mul_274 = convert_element_type_326 = None
        convert_element_type_327: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_328: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_325, torch.float32);  convert_element_type_325 = None
        mul_285: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_59, arg249_1);  arg249_1 = None
        mul_286: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_285, 768)
        sum_149: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_285, [2], True)
        mul_287: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_285, arg250_1);  mul_285 = None
        sum_150: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_287, [2], True);  mul_287 = None
        mul_288: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg250_1, sum_150);  sum_150 = None
        sub_44: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_286, sum_149);  mul_286 = sum_149 = None
        sub_45: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_44, mul_288);  sub_44 = mul_288 = None
        mul_289: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg251_1, sub_45);  arg251_1 = sub_45 = None
        mul_290: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_59, arg250_1);  arg250_1 = None
        sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_290, [0, 1]);  mul_290 = None
        sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_59, [0, 1]);  add_59 = None
        convert_element_type_329: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_289, torch.bfloat16)
        convert_element_type_330: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg252_1, torch.bfloat16);  arg252_1 = None
        mul_291: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_330, 1.1111111111111112);  convert_element_type_330 = None
        mul_292: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_329, mul_291);  convert_element_type_329 = mul_291 = None
        view_279: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_292, _shape_param_281);  mul_292 = _shape_param_281 = None
        mm_116: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_279, arg253_1);  arg253_1 = None
        permute_103: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_279, [1, 0])
        mm_117: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_103, arg254_1);  permute_103 = arg254_1 = None
        sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True, dtype = torch.float32);  view_279 = None
        view_280: "f32[768]" = torch.ops.aten.view.default(sum_153, _shape_param_282);  sum_153 = _shape_param_282 = None
        convert_element_type_331: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_280, torch.bfloat16);  view_280 = None
        view_281: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_116, _shape_param_283);  mm_116 = _shape_param_283 = None
        convert_element_type_332: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        convert_element_type_333: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_331, torch.float32);  convert_element_type_331 = None
        view_282: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_281, _shape_param_284);  view_281 = _shape_param_284 = None
        permute_104: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
        clone_36: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_283: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_285);  clone_36 = _shape_param_285 = None
        bmm_36: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg255_1, view_283);  arg255_1 = None
        bmm_37: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_283, arg256_1);  view_283 = arg256_1 = None
        view_284: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_36, _shape_param_286);  bmm_36 = _shape_param_286 = None
        view_285: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_37, _shape_param_287);  bmm_37 = _shape_param_287 = None
        convert_element_type_334: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg257_1, torch.bfloat16);  arg257_1 = None
        mul_293: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_334, 1.1111111111111112);  convert_element_type_334 = None
        mul_294: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_285, mul_293);  view_285 = mul_293 = None
        convert_element_type_335: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_294, torch.float32);  mul_294 = None
        convert_element_type_336: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg258_1, torch.float32);  arg258_1 = None
        mul_295: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_335, convert_element_type_336);  convert_element_type_335 = None
        sum_154: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_295, [-1], True)
        neg_9: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_336);  convert_element_type_336 = None
        fma_9: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_9, sum_154, mul_295);  neg_9 = sum_154 = mul_295 = None
        convert_element_type_337: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None
        view_286: "bf16[384, 512, 512]" = torch.ops.aten.view.default(convert_element_type_337, _shape_param_288);  convert_element_type_337 = _shape_param_288 = None
        bmm_38: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg259_1, view_286);  arg259_1 = None
        bmm_39: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_286, arg260_1);  view_286 = arg260_1 = None
        view_287: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_38, _shape_param_289);  bmm_38 = _shape_param_289 = None
        view_288: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_39, _shape_param_290);  bmm_39 = _shape_param_290 = None
        mul_296: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_287, 0.3535533905932738);  view_287 = None
        permute_105: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_296, [0, 1, 3, 2]);  mul_296 = None
        mul_297: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_288, 0.3535533905932738);  view_288 = None
        permute_106: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None
        clone_37: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_289: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_37, _shape_param_291);  clone_37 = _shape_param_291 = None
        view_290: "bf16[16384, 768]" = torch.ops.aten.view.default(view_289, _shape_param_292);  view_289 = _shape_param_292 = None
        mm_118: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_290, arg261_1);  arg261_1 = None
        permute_107: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_290, [1, 0])
        mm_119: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_107, arg262_1);  permute_107 = None
        sum_155: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_290, [0], True, dtype = torch.float32);  view_290 = None
        view_291: "f32[768]" = torch.ops.aten.view.default(sum_155, _shape_param_293);  sum_155 = _shape_param_293 = None
        convert_element_type_338: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_291, torch.bfloat16);  view_291 = None
        view_292: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_118, _shape_param_294);  mm_118 = _shape_param_294 = None
        convert_element_type_339: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_292, torch.float32);  view_292 = None
        add_60: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_289, convert_element_type_339);  mul_289 = convert_element_type_339 = None
        convert_element_type_340: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        convert_element_type_341: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_338, torch.float32);  convert_element_type_338 = None
        permute_108: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_105, [0, 2, 1, 3]);  permute_105 = None
        view_293: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_108, _shape_param_295);  permute_108 = _shape_param_295 = None
        clone_38: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_293, memory_format = torch.contiguous_format);  view_293 = None
        view_294: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_38, _shape_param_296);  clone_38 = _shape_param_296 = None
        mm_120: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_294, arg263_1);  arg263_1 = None
        permute_109: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_294, [1, 0])
        mm_121: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_109, arg262_1);  permute_109 = None
        sum_156: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True, dtype = torch.float32);  view_294 = None
        view_295: "f32[768]" = torch.ops.aten.view.default(sum_156, _shape_param_297);  sum_156 = _shape_param_297 = None
        convert_element_type_342: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_295, torch.bfloat16);  view_295 = None
        view_296: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_120, _shape_param_298);  mm_120 = _shape_param_298 = None
        convert_element_type_343: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_296, torch.float32);  view_296 = None
        add_61: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_60, convert_element_type_343);  add_60 = convert_element_type_343 = None
        convert_element_type_344: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        convert_element_type_345: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_342, torch.float32);  convert_element_type_342 = None
        permute_110: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_297, [0, 2, 1, 3]);  mul_297 = None
        clone_39: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_297: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_39, _shape_param_299);  clone_39 = _shape_param_299 = None
        view_298: "bf16[16384, 768]" = torch.ops.aten.view.default(view_297, _shape_param_300);  view_297 = _shape_param_300 = None
        mm_122: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_298, arg264_1);  arg264_1 = None
        permute_111: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_298, [1, 0])
        mm_123: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_111, arg262_1);  permute_111 = arg262_1 = None
        sum_157: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True, dtype = torch.float32);  view_298 = None
        view_299: "f32[768]" = torch.ops.aten.view.default(sum_157, _shape_param_301);  sum_157 = _shape_param_301 = None
        convert_element_type_346: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_299, torch.bfloat16);  view_299 = None
        view_300: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_122, _shape_param_302);  mm_122 = _shape_param_302 = None
        convert_element_type_347: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_300, torch.float32);  view_300 = None
        add_62: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_61, convert_element_type_347);  add_61 = convert_element_type_347 = None
        convert_element_type_348: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_349: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_346, torch.float32);  convert_element_type_346 = None
        mul_298: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_62, arg265_1);  arg265_1 = None
        mul_299: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_298, 768)
        sum_158: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True)
        mul_300: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_298, arg266_1);  mul_298 = None
        sum_159: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True);  mul_300 = None
        mul_301: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg266_1, sum_159);  sum_159 = None
        sub_46: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_299, sum_158);  mul_299 = sum_158 = None
        sub_47: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_46, mul_301);  sub_46 = mul_301 = None
        mul_302: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg267_1, sub_47);  arg267_1 = sub_47 = None
        mul_303: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_62, arg266_1);  arg266_1 = None
        sum_160: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 1]);  mul_303 = None
        sum_161: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_62, [0, 1]);  add_62 = None
        convert_element_type_350: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_302, torch.bfloat16)
        convert_element_type_351: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg268_1, torch.bfloat16);  arg268_1 = None
        mul_304: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_351, 1.1111111111111112);  convert_element_type_351 = None
        mul_305: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_350, mul_304);  convert_element_type_350 = mul_304 = None
        view_301: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_305, _shape_param_303);  mul_305 = _shape_param_303 = None
        mm_124: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_301, arg269_1);  arg269_1 = None
        permute_112: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_301, [1, 0])
        mm_125: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_112, arg270_1);  permute_112 = arg270_1 = None
        sum_162: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_301, [0], True, dtype = torch.float32);  view_301 = None
        view_302: "f32[768]" = torch.ops.aten.view.default(sum_162, _shape_param_304);  sum_162 = _shape_param_304 = None
        convert_element_type_352: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_302, torch.bfloat16);  view_302 = None
        view_303: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_124, _shape_param_305);  mm_124 = _shape_param_305 = None
        convert_element_type_353: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_354: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_352, torch.float32);  convert_element_type_352 = None
        convert_element_type_355: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_303, torch.float32);  view_303 = None
        view_304: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg271_1, _shape_param_306);  arg271_1 = _shape_param_306 = None
        convert_element_type_356: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_304, torch.float32);  view_304 = None
        mul_306: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_356, 0.7071067811865476)
        erf_11: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_306);  mul_306 = None
        add_63: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_307: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_63, 0.5);  add_63 = None
        mul_308: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_356, convert_element_type_356)
        mul_309: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_308, -0.5);  mul_308 = None
        exp_12: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_309);  mul_309 = None
        mul_310: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_311: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_356, mul_310);  convert_element_type_356 = mul_310 = None
        add_64: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_307, mul_311);  mul_307 = mul_311 = None
        mul_312: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_355, add_64);  convert_element_type_355 = add_64 = None
        convert_element_type_357: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_312, torch.bfloat16);  mul_312 = None
        view_305: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_357, _shape_param_307);  convert_element_type_357 = _shape_param_307 = None
        mm_126: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_305, arg272_1);  arg272_1 = None
        permute_113: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_127: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_113, arg273_1);  permute_113 = arg273_1 = None
        sum_163: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True, dtype = torch.float32);  view_305 = None
        view_306: "f32[3072]" = torch.ops.aten.view.default(sum_163, _shape_param_308);  sum_163 = _shape_param_308 = None
        convert_element_type_358: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_306, torch.bfloat16);  view_306 = None
        view_307: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_126, _shape_param_309);  mm_126 = _shape_param_309 = None
        convert_element_type_359: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_307, torch.float32);  view_307 = None
        add_65: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_302, convert_element_type_359);  mul_302 = convert_element_type_359 = None
        convert_element_type_360: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_361: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_358, torch.float32);  convert_element_type_358 = None
        mul_313: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_65, arg274_1);  arg274_1 = None
        mul_314: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_313, 768)
        sum_164: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True)
        mul_315: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_313, arg275_1);  mul_313 = None
        sum_165: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [2], True);  mul_315 = None
        mul_316: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg275_1, sum_165);  sum_165 = None
        sub_48: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_314, sum_164);  mul_314 = sum_164 = None
        sub_49: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_316);  sub_48 = mul_316 = None
        mul_317: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg276_1, sub_49);  arg276_1 = sub_49 = None
        mul_318: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_65, arg275_1);  arg275_1 = None
        sum_166: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 1]);  mul_318 = None
        sum_167: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_65, [0, 1]);  add_65 = None
        convert_element_type_362: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_317, torch.bfloat16)
        convert_element_type_363: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg277_1, torch.bfloat16);  arg277_1 = None
        mul_319: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_363, 1.1111111111111112);  convert_element_type_363 = None
        mul_320: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_362, mul_319);  convert_element_type_362 = mul_319 = None
        view_308: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_320, _shape_param_310);  mul_320 = _shape_param_310 = None
        mm_128: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_308, arg278_1);  arg278_1 = None
        permute_114: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_308, [1, 0])
        mm_129: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_114, arg279_1);  permute_114 = arg279_1 = None
        sum_168: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True, dtype = torch.float32);  view_308 = None
        view_309: "f32[768]" = torch.ops.aten.view.default(sum_168, _shape_param_311);  sum_168 = _shape_param_311 = None
        convert_element_type_364: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_309, torch.bfloat16);  view_309 = None
        view_310: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_128, _shape_param_312);  mm_128 = _shape_param_312 = None
        convert_element_type_365: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        convert_element_type_366: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_364, torch.float32);  convert_element_type_364 = None
        view_311: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_310, _shape_param_313);  view_310 = _shape_param_313 = None
        permute_115: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_311, [0, 2, 1, 3]);  view_311 = None
        clone_40: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_312: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_314);  clone_40 = _shape_param_314 = None
        bmm_40: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg280_1, view_312);  arg280_1 = None
        bmm_41: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_312, arg281_1);  view_312 = arg281_1 = None
        view_313: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_40, _shape_param_315);  bmm_40 = _shape_param_315 = None
        view_314: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_41, _shape_param_316);  bmm_41 = _shape_param_316 = None
        convert_element_type_367: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg282_1, torch.bfloat16);  arg282_1 = None
        mul_321: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_367, 1.1111111111111112);  convert_element_type_367 = None
        mul_322: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_314, mul_321);  view_314 = mul_321 = None
        convert_element_type_368: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_322, torch.float32);  mul_322 = None
        convert_element_type_369: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg283_1, torch.float32);  arg283_1 = None
        mul_323: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_368, convert_element_type_369);  convert_element_type_368 = None
        sum_169: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [-1], True)
        neg_10: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_369);  convert_element_type_369 = None
        fma_10: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_10, sum_169, mul_323);  neg_10 = sum_169 = mul_323 = None
        convert_element_type_370: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None
        view_315: "bf16[384, 512, 512]" = torch.ops.aten.view.default(convert_element_type_370, _shape_param_317);  convert_element_type_370 = _shape_param_317 = None
        bmm_42: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg284_1, view_315);  arg284_1 = None
        bmm_43: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_315, arg285_1);  view_315 = arg285_1 = None
        view_316: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_42, _shape_param_318);  bmm_42 = _shape_param_318 = None
        view_317: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_43, _shape_param_319);  bmm_43 = _shape_param_319 = None
        mul_324: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_316, 0.3535533905932738);  view_316 = None
        permute_116: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_324, [0, 1, 3, 2]);  mul_324 = None
        mul_325: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_317, 0.3535533905932738);  view_317 = None
        permute_117: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None
        clone_41: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_318: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_41, _shape_param_320);  clone_41 = _shape_param_320 = None
        view_319: "bf16[16384, 768]" = torch.ops.aten.view.default(view_318, _shape_param_321);  view_318 = _shape_param_321 = None
        mm_130: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_319, arg286_1);  arg286_1 = None
        permute_118: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_319, [1, 0])
        mm_131: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_118, arg287_1);  permute_118 = None
        sum_170: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_319, [0], True, dtype = torch.float32);  view_319 = None
        view_320: "f32[768]" = torch.ops.aten.view.default(sum_170, _shape_param_322);  sum_170 = _shape_param_322 = None
        convert_element_type_371: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_320, torch.bfloat16);  view_320 = None
        view_321: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_130, _shape_param_323);  mm_130 = _shape_param_323 = None
        convert_element_type_372: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_321, torch.float32);  view_321 = None
        add_66: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_317, convert_element_type_372);  mul_317 = convert_element_type_372 = None
        convert_element_type_373: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        convert_element_type_374: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_371, torch.float32);  convert_element_type_371 = None
        permute_119: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_116, [0, 2, 1, 3]);  permute_116 = None
        view_322: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_119, _shape_param_324);  permute_119 = _shape_param_324 = None
        clone_42: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_322, memory_format = torch.contiguous_format);  view_322 = None
        view_323: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_42, _shape_param_325);  clone_42 = _shape_param_325 = None
        mm_132: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_323, arg288_1);  arg288_1 = None
        permute_120: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_323, [1, 0])
        mm_133: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_120, arg287_1);  permute_120 = None
        sum_171: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_323, [0], True, dtype = torch.float32);  view_323 = None
        view_324: "f32[768]" = torch.ops.aten.view.default(sum_171, _shape_param_326);  sum_171 = _shape_param_326 = None
        convert_element_type_375: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_324, torch.bfloat16);  view_324 = None
        view_325: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_132, _shape_param_327);  mm_132 = _shape_param_327 = None
        convert_element_type_376: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_325, torch.float32);  view_325 = None
        add_67: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_66, convert_element_type_376);  add_66 = convert_element_type_376 = None
        convert_element_type_377: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        convert_element_type_378: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_375, torch.float32);  convert_element_type_375 = None
        permute_121: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_325, [0, 2, 1, 3]);  mul_325 = None
        clone_43: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
        view_326: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_43, _shape_param_328);  clone_43 = _shape_param_328 = None
        view_327: "bf16[16384, 768]" = torch.ops.aten.view.default(view_326, _shape_param_329);  view_326 = _shape_param_329 = None
        mm_134: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_327, arg289_1);  arg289_1 = None
        permute_122: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_327, [1, 0])
        mm_135: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_122, arg287_1);  permute_122 = arg287_1 = None
        sum_172: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True, dtype = torch.float32);  view_327 = None
        view_328: "f32[768]" = torch.ops.aten.view.default(sum_172, _shape_param_330);  sum_172 = _shape_param_330 = None
        convert_element_type_379: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_328, torch.bfloat16);  view_328 = None
        view_329: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_134, _shape_param_331);  mm_134 = _shape_param_331 = None
        convert_element_type_380: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_329, torch.float32);  view_329 = None
        add_68: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_67, convert_element_type_380);  add_67 = convert_element_type_380 = None
        convert_element_type_381: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_382: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_379, torch.float32);  convert_element_type_379 = None
        mul_326: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_68, arg290_1);  arg290_1 = None
        mul_327: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_326, 768)
        sum_173: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [2], True)
        mul_328: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_326, arg291_1);  mul_326 = None
        sum_174: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_328, [2], True);  mul_328 = None
        mul_329: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg291_1, sum_174);  sum_174 = None
        sub_50: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_327, sum_173);  mul_327 = sum_173 = None
        sub_51: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_50, mul_329);  sub_50 = mul_329 = None
        mul_330: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg292_1, sub_51);  arg292_1 = sub_51 = None
        mul_331: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_68, arg291_1);  arg291_1 = None
        sum_175: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_331, [0, 1]);  mul_331 = None
        sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_68, [0, 1]);  add_68 = None
        convert_element_type_383: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_330, torch.bfloat16)
        convert_element_type_384: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg293_1, torch.bfloat16);  arg293_1 = None
        mul_332: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_384, 1.1111111111111112);  convert_element_type_384 = None
        mul_333: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_383, mul_332);  convert_element_type_383 = mul_332 = None
        view_330: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_333, _shape_param_332);  mul_333 = _shape_param_332 = None
        mm_136: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_330, arg294_1);  arg294_1 = None
        permute_123: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_330, [1, 0])
        mm_137: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_123, arg295_1);  permute_123 = arg295_1 = None
        sum_177: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True, dtype = torch.float32);  view_330 = None
        view_331: "f32[768]" = torch.ops.aten.view.default(sum_177, _shape_param_333);  sum_177 = _shape_param_333 = None
        convert_element_type_385: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_331, torch.bfloat16);  view_331 = None
        view_332: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_136, _shape_param_334);  mm_136 = _shape_param_334 = None
        convert_element_type_386: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        convert_element_type_387: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_385, torch.float32);  convert_element_type_385 = None
        convert_element_type_388: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_332, torch.float32);  view_332 = None
        view_333: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg296_1, _shape_param_335);  arg296_1 = _shape_param_335 = None
        convert_element_type_389: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_333, torch.float32);  view_333 = None
        mul_334: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_389, 0.7071067811865476)
        erf_12: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_334);  mul_334 = None
        add_69: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_335: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None
        mul_336: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_389, convert_element_type_389)
        mul_337: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_336, -0.5);  mul_336 = None
        exp_13: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_337);  mul_337 = None
        mul_338: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
        mul_339: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_389, mul_338);  convert_element_type_389 = mul_338 = None
        add_70: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_335, mul_339);  mul_335 = mul_339 = None
        mul_340: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_388, add_70);  convert_element_type_388 = add_70 = None
        convert_element_type_390: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_340, torch.bfloat16);  mul_340 = None
        view_334: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_390, _shape_param_336);  convert_element_type_390 = _shape_param_336 = None
        mm_138: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_334, arg297_1);  arg297_1 = None
        permute_124: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_334, [1, 0])
        mm_139: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_124, arg298_1);  permute_124 = arg298_1 = None
        sum_178: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True, dtype = torch.float32);  view_334 = None
        view_335: "f32[3072]" = torch.ops.aten.view.default(sum_178, _shape_param_337);  sum_178 = _shape_param_337 = None
        convert_element_type_391: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_335, torch.bfloat16);  view_335 = None
        view_336: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_138, _shape_param_338);  mm_138 = _shape_param_338 = None
        convert_element_type_392: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_336, torch.float32);  view_336 = None
        add_71: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_330, convert_element_type_392);  mul_330 = convert_element_type_392 = None
        convert_element_type_393: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_394: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_391, torch.float32);  convert_element_type_391 = None
        mul_341: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_71, arg299_1);  arg299_1 = None
        mul_342: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_341, 768)
        sum_179: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True)
        mul_343: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_341, arg300_1);  mul_341 = None
        sum_180: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
        mul_344: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg300_1, sum_180);  sum_180 = None
        sub_52: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_342, sum_179);  mul_342 = sum_179 = None
        sub_53: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_52, mul_344);  sub_52 = mul_344 = None
        mul_345: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg301_1, sub_53);  arg301_1 = sub_53 = None
        mul_346: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_71, arg300_1);  arg300_1 = None
        sum_181: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_346, [0, 1]);  mul_346 = None
        sum_182: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_71, [0, 1]);  add_71 = None
        convert_element_type_395: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_345, torch.bfloat16)
        convert_element_type_396: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg302_1, torch.bfloat16);  arg302_1 = None
        mul_347: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_396, 1.1111111111111112);  convert_element_type_396 = None
        mul_348: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_395, mul_347);  convert_element_type_395 = mul_347 = None
        view_337: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_348, _shape_param_339);  mul_348 = _shape_param_339 = None
        mm_140: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_337, arg303_1);  arg303_1 = None
        permute_125: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_337, [1, 0])
        mm_141: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_125, arg304_1);  permute_125 = arg304_1 = None
        sum_183: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_337, [0], True, dtype = torch.float32);  view_337 = None
        view_338: "f32[768]" = torch.ops.aten.view.default(sum_183, _shape_param_340);  sum_183 = _shape_param_340 = None
        convert_element_type_397: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_338, torch.bfloat16);  view_338 = None
        view_339: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_140, _shape_param_341);  mm_140 = _shape_param_341 = None
        convert_element_type_398: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        convert_element_type_399: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_397, torch.float32);  convert_element_type_397 = None
        view_340: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_339, _shape_param_342);  view_339 = _shape_param_342 = None
        permute_126: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
        clone_44: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_341: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_44, _shape_param_343);  clone_44 = _shape_param_343 = None
        bmm_44: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg305_1, view_341);  arg305_1 = None
        bmm_45: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_341, arg306_1);  view_341 = arg306_1 = None
        view_342: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_44, _shape_param_344);  bmm_44 = _shape_param_344 = None
        view_343: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_45, _shape_param_345);  bmm_45 = _shape_param_345 = None
        convert_element_type_400: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg307_1, torch.bfloat16);  arg307_1 = None
        mul_349: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_400, 1.1111111111111112);  convert_element_type_400 = None
        mul_350: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_343, mul_349);  view_343 = mul_349 = None
        convert_element_type_401: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_350, torch.float32);  mul_350 = None
        view_344: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(arg308_1, _shape_param_346);  arg308_1 = _shape_param_346 = None
        convert_element_type_402: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_344, torch.float32);  view_344 = None
        sub_54: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_402, arg309_1);  convert_element_type_402 = arg309_1 = None
        exp_14: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(sub_54);  sub_54 = None
        div_2: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_14, arg310_1);  exp_14 = arg310_1 = None
        convert_element_type_403: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        full_2: "bf16[32, 12, 512, 512]" = torch.ops.aten.full.default(_shape_param_347, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_347 = None
        where_3: "bf16[32, 12, 512, 512]" = torch.ops.aten.where.self(arg311_1, full_2, convert_element_type_403);  arg311_1 = full_2 = convert_element_type_403 = None
        convert_element_type_404: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        mul_351: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_401, convert_element_type_404);  convert_element_type_401 = None
        sum_184: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_351, [-1], True)
        neg_11: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(convert_element_type_404);  convert_element_type_404 = None
        fma_11: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_11, sum_184, mul_351);  neg_11 = sum_184 = mul_351 = None
        convert_element_type_405: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None
        view_345: "bf16[384, 512, 512]" = torch.ops.aten.view.default(convert_element_type_405, _shape_param_348);  convert_element_type_405 = _shape_param_348 = None
        bmm_46: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg312_1, view_345);  arg312_1 = None
        bmm_47: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_345, arg313_1);  view_345 = arg313_1 = None
        view_346: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_46, _shape_param_349);  bmm_46 = _shape_param_349 = None
        view_347: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_47, _shape_param_350);  bmm_47 = _shape_param_350 = None
        mul_352: "bf16[32, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_346, 0.3535533905932738);  view_346 = None
        permute_127: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(mul_352, [0, 1, 3, 2]);  mul_352 = None
        mul_353: "bf16[32, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_347, 0.3535533905932738);  view_347 = None
        permute_128: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None
        clone_45: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_348: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_45, _shape_param_351);  clone_45 = _shape_param_351 = None
        view_349: "bf16[16384, 768]" = torch.ops.aten.view.default(view_348, _shape_param_352);  view_348 = _shape_param_352 = None
        mm_142: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_349, arg314_1);  arg314_1 = None
        permute_129: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_349, [1, 0])
        mm_143: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_129, arg315_1);  permute_129 = None
        sum_185: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_349, [0], True, dtype = torch.float32);  view_349 = None
        view_350: "f32[768]" = torch.ops.aten.view.default(sum_185, _shape_param_353);  sum_185 = _shape_param_353 = None
        convert_element_type_406: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_350, torch.bfloat16);  view_350 = None
        view_351: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_142, _shape_param_354);  mm_142 = _shape_param_354 = None
        convert_element_type_407: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_351, torch.float32);  view_351 = None
        add_72: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_345, convert_element_type_407);  mul_345 = convert_element_type_407 = None
        convert_element_type_408: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        convert_element_type_409: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_406, torch.float32);  convert_element_type_406 = None
        permute_130: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_127, [0, 2, 1, 3]);  permute_127 = None
        view_352: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_130, _shape_param_355);  permute_130 = _shape_param_355 = None
        clone_46: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_352, memory_format = torch.contiguous_format);  view_352 = None
        view_353: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_46, _shape_param_356);  clone_46 = _shape_param_356 = None
        mm_144: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_353, arg316_1);  arg316_1 = None
        permute_131: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_353, [1, 0])
        mm_145: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_131, arg315_1);  permute_131 = None
        sum_186: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True, dtype = torch.float32);  view_353 = None
        view_354: "f32[768]" = torch.ops.aten.view.default(sum_186, _shape_param_357);  sum_186 = _shape_param_357 = None
        convert_element_type_410: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_354, torch.bfloat16);  view_354 = None
        view_355: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_144, _shape_param_358);  mm_144 = _shape_param_358 = None
        convert_element_type_411: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_355, torch.float32);  view_355 = None
        add_73: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_72, convert_element_type_411);  add_72 = convert_element_type_411 = None
        convert_element_type_412: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        convert_element_type_413: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_410, torch.float32);  convert_element_type_410 = None
        permute_132: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(mul_353, [0, 2, 1, 3]);  mul_353 = None
        clone_47: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
        view_356: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_47, _shape_param_359);  clone_47 = _shape_param_359 = None
        view_357: "bf16[16384, 768]" = torch.ops.aten.view.default(view_356, _shape_param_360);  view_356 = _shape_param_360 = None
        mm_146: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_357, arg317_1);  arg317_1 = None
        permute_133: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_357, [1, 0])
        mm_147: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_133, arg315_1);  permute_133 = arg315_1 = None
        sum_187: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_357, [0], True, dtype = torch.float32);  view_357 = None
        view_358: "f32[768]" = torch.ops.aten.view.default(sum_187, _shape_param_361);  sum_187 = _shape_param_361 = None
        convert_element_type_414: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_358, torch.bfloat16);  view_358 = None
        view_359: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_146, _shape_param_362);  mm_146 = _shape_param_362 = None
        convert_element_type_415: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_359, torch.float32);  view_359 = None
        add_74: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_73, convert_element_type_415);  add_73 = convert_element_type_415 = None
        convert_element_type_416: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_417: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_414, torch.float32);  convert_element_type_414 = None
        convert_element_type_418: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg318_1, torch.float32);  arg318_1 = None
        mul_354: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_418, 1.1111111111111112);  convert_element_type_418 = None
        mul_355: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_74, mul_354);  add_74 = mul_354 = None
        mul_356: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_355, arg319_1);  arg319_1 = None
        mul_357: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_356, 768)
        sum_188: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True)
        mul_358: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_356, arg320_1);  mul_356 = None
        sum_189: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True);  mul_358 = None
        mul_359: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg320_1, sum_189);  sum_189 = None
        sub_55: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_357, sum_188);  mul_357 = sum_188 = None
        sub_56: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_55, mul_359);  sub_55 = mul_359 = None
        mul_360: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg321_1, sub_56);  arg321_1 = sub_56 = None
        mul_361: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_355, arg320_1);  arg320_1 = None
        sum_190: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_361, [0, 1]);  mul_361 = None
        sum_191: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_355, [0, 1]);  mul_355 = None
        sum_192: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_360, [0], True, dtype = torch.float32)
        ge: "b8[1, 512]" = torch.ops.aten.ge.Scalar(arg322_1, 0)
        lt: "b8[1, 512]" = torch.ops.aten.lt.Scalar(arg322_1, 512)
        bitwise_and: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne_1: "b8[1, 512]" = torch.ops.aten.ne.Scalar(arg322_1, -1)
        bitwise_and_1: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne_1);  bitwise_and = ne_1 = None
        unsqueeze_1: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_3: "f32[512, 768]" = torch.ops.aten.full.default(_shape_param_363, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_363 = None
        _unsafe_masked_index_put_accumulate: "f32[512, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_3, unsqueeze_1, [arg322_1], sum_192);  full_3 = unsqueeze_1 = arg322_1 = sum_192 = None
        expand_1: "i64[32, 512]" = torch.ops.aten.expand.default(arg323_1, _shape_param_364);  arg323_1 = _shape_param_364 = None
        ge_1: "b8[32, 512]" = torch.ops.aten.ge.Scalar(expand_1, 0)
        lt_1: "b8[32, 512]" = torch.ops.aten.lt.Scalar(expand_1, 2)
        bitwise_and_2: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_2: "b8[32, 512]" = torch.ops.aten.ne.Scalar(expand_1, -1)
        bitwise_and_3: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_2);  bitwise_and_2 = ne_2 = None
        unsqueeze_2: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_4: "f32[2, 768]" = torch.ops.aten.full.default(_shape_param_365, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_365 = None
        _unsafe_masked_index_put_accumulate_1: "f32[2, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_4, unsqueeze_2, [expand_1], mul_360);  full_4 = unsqueeze_2 = expand_1 = None
        ge_2: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg324_1, 0)
        lt_2: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg324_1, 30522)
        bitwise_and_4: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_2, lt_2);  ge_2 = lt_2 = None
        ne_3: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg324_1, 0)
        bitwise_and_5: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, ne_3);  bitwise_and_4 = ne_3 = None
        unsqueeze_3: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_5, -1);  bitwise_and_5 = None
        full_5: "f32[30522, 768]" = torch.ops.aten.full.default(_shape_param_366, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_366 = None
        _unsafe_masked_index_put_accumulate_2: "f32[30522, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_5, unsqueeze_3, [arg324_1], mul_360);  full_5 = unsqueeze_3 = arg324_1 = mul_360 = None
        add_75: "f32[30522, 768]" = torch.ops.aten.add.Tensor(convert_element_type_8, _unsafe_masked_index_put_accumulate_2);  convert_element_type_8 = _unsafe_masked_index_put_accumulate_2 = None
        return (convert_element_type_9, sum_5, sum_6, convert_element_type_18, convert_element_type_19, sum_10, sum_11, convert_element_type_23, convert_element_type_24, convert_element_type_30, convert_element_type_31, sum_16, sum_17, convert_element_type_35, convert_element_type_36, convert_element_type_43, convert_element_type_44, convert_element_type_47, convert_element_type_48, convert_element_type_51, convert_element_type_52, sum_25, sum_26, convert_element_type_56, convert_element_type_57, convert_element_type_63, convert_element_type_64, sum_31, sum_32, convert_element_type_68, convert_element_type_69, convert_element_type_76, convert_element_type_77, convert_element_type_80, convert_element_type_81, convert_element_type_84, convert_element_type_85, sum_40, sum_41, convert_element_type_89, convert_element_type_90, convert_element_type_96, convert_element_type_97, sum_46, sum_47, convert_element_type_101, convert_element_type_102, convert_element_type_109, convert_element_type_110, convert_element_type_113, convert_element_type_114, convert_element_type_117, convert_element_type_118, sum_55, sum_56, convert_element_type_122, convert_element_type_123, convert_element_type_129, convert_element_type_130, sum_61, sum_62, convert_element_type_134, convert_element_type_135, convert_element_type_142, convert_element_type_143, convert_element_type_146, convert_element_type_147, convert_element_type_150, convert_element_type_151, sum_70, sum_71, convert_element_type_155, convert_element_type_156, convert_element_type_162, convert_element_type_163, sum_76, sum_77, convert_element_type_167, convert_element_type_168, convert_element_type_175, convert_element_type_176, convert_element_type_179, convert_element_type_180, convert_element_type_183, convert_element_type_184, sum_85, sum_86, convert_element_type_188, convert_element_type_189, convert_element_type_195, convert_element_type_196, sum_91, sum_92, convert_element_type_200, convert_element_type_201, convert_element_type_208, convert_element_type_209, convert_element_type_212, convert_element_type_213, convert_element_type_216, convert_element_type_217, sum_100, sum_101, convert_element_type_221, convert_element_type_222, convert_element_type_228, convert_element_type_229, sum_106, sum_107, convert_element_type_233, convert_element_type_234, convert_element_type_241, convert_element_type_242, convert_element_type_245, convert_element_type_246, convert_element_type_249, convert_element_type_250, sum_115, sum_116, convert_element_type_254, convert_element_type_255, convert_element_type_261, convert_element_type_262, sum_121, sum_122, convert_element_type_266, convert_element_type_267, convert_element_type_274, convert_element_type_275, convert_element_type_278, convert_element_type_279, convert_element_type_282, convert_element_type_283, sum_130, sum_131, convert_element_type_287, convert_element_type_288, convert_element_type_294, convert_element_type_295, sum_136, sum_137, convert_element_type_299, convert_element_type_300, convert_element_type_307, convert_element_type_308, convert_element_type_311, convert_element_type_312, convert_element_type_315, convert_element_type_316, sum_145, sum_146, convert_element_type_320, convert_element_type_321, convert_element_type_327, convert_element_type_328, sum_151, sum_152, convert_element_type_332, convert_element_type_333, convert_element_type_340, convert_element_type_341, convert_element_type_344, convert_element_type_345, convert_element_type_348, convert_element_type_349, sum_160, sum_161, convert_element_type_353, convert_element_type_354, convert_element_type_360, convert_element_type_361, sum_166, sum_167, convert_element_type_365, convert_element_type_366, convert_element_type_373, convert_element_type_374, convert_element_type_377, convert_element_type_378, convert_element_type_381, convert_element_type_382, sum_175, sum_176, convert_element_type_386, convert_element_type_387, convert_element_type_393, convert_element_type_394, sum_181, sum_182, convert_element_type_398, convert_element_type_399, convert_element_type_408, convert_element_type_409, convert_element_type_412, convert_element_type_413, convert_element_type_416, convert_element_type_417, sum_190, sum_191, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1, add_75)



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
